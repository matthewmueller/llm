package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"log/slog"
	"reflect"
	"sort"
	"strings"

	"golang.org/x/sync/errgroup"
)

// Message represents a chat message
type Message struct {
	Role       string
	Content    string
	Thinking   string // For chain-of-thought / thinking content
	ToolCallID string // For tool results, the ID of the tool call being responded to
}

// Model represents an available model
type Model struct {
	Provider string
	Name     string
}

// ToolInfo defines a tool's JSON schema specification
type ToolInfo struct {
	Type     string
	Function ToolFunction
}

// ToolFunction defines the function details for a tool
type ToolFunction struct {
	Name        string
	Description string
	Parameters  ToolFunctionParameters
}

// ToolFunctionParameters defines the parameters schema for a tool
type ToolFunctionParameters struct {
	Type       string
	Properties map[string]ToolProperty
	Required   []string
}

// ToolProperty defines a single property in the tool schema
type ToolProperty struct {
	Type        string
	Description string
	Enum        []string
}

// Thinking represents the level of extended thinking/reasoning
type Thinking string

const (
	ThinkingLow    Thinking = "low"    // Low thinking budget
	ThinkingMedium Thinking = "medium" // Medium thinking budget (default)
	ThinkingHigh   Thinking = "high"   // High thinking budget
)

// ChatRequest represents a request to the chat API
type ChatRequest struct {
	Model    string
	Messages []*Message
	Tools    []*ToolInfo
	Thinking Thinking // Extended thinking level (default: medium)
}

// ChatResponse represents a streaming response from the chat API
type ChatResponse struct {
	Role     string
	Content  string
	Thinking string // Thinking/reasoning content (shown dim in CLI)
	Tool     *ToolCall
	Done     bool // True when response is complete
}

// ToolCall represents a tool invocation from the model
type ToolCall struct {
	ID        string
	Name      string
	Arguments json.RawMessage
}

// Provider interface
type Provider interface {
	Name() string
	Models(ctx context.Context) ([]*Model, error)
	Chat(ctx context.Context, req *ChatRequest) iter.Seq2[*ChatResponse, error]
}

// Tool interface - high-level typed tool definition
type Tool interface {
	Info() *ToolInfo
	Run(ctx context.Context, args json.RawMessage) (json.RawMessage, error)
}

// Client manages providers
type Client struct {
	log       *slog.Logger
	providers []Provider
}

// New creates a new Client
func New(log *slog.Logger, providers ...Provider) *Client {
	return &Client{log, providers}
}

func findModel(models []*Model, name string) (*Model, bool) {
	for _, m := range models {
		if m.Name == name {
			return m, true
		}
	}
	if len(models) > 0 {
		return models[0], true
	}
	return nil, false
}

func findProvider(providers []Provider, name string) (Provider, bool) {
	for _, p := range providers {
		if p.Name() == name {
			return p, true
		}
	}
	return nil, false
}

// Chat sends a chat request to the appropriate provider
func (c *Client) Chat(ctx context.Context, req *ChatRequest) iter.Seq2[*ChatResponse, error] {
	models, err := c.Models(ctx)
	if err != nil {
		return func(yield func(*ChatResponse, error) bool) {
			yield(nil, fmt.Errorf("llm: unable to list models: %w", err))
		}
	}

	model, ok := findModel(models, req.Model)
	if !ok {
		return func(yield func(*ChatResponse, error) bool) {
			yield(nil, fmt.Errorf("llm: model %q not found", req.Model))
		}
	}

	provider, ok := findProvider(c.providers, model.Provider)
	if !ok {
		return func(yield func(*ChatResponse, error) bool) {
			yield(nil, fmt.Errorf("llm: provider %q not found", model.Provider))
		}
	}

	return provider.Chat(ctx, req)
}

// Models returns all available models from all providers
func (c *Client) Models(ctx context.Context) (models []*Model, err error) {
	eg, ctx := errgroup.WithContext(ctx)
	for _, provider := range c.providers {
		eg.Go(func() error {
			m, err := provider.Models(ctx)
			if err != nil {
				return err
			}
			// TODO: dedupe
			models = append(models, m...)
			return nil
		})
	}
	if err := eg.Wait(); err != nil {
		return nil, err
	}
	sort.Slice(models, func(i, j int) bool {
		if models[i].Provider == models[j].Provider {
			return models[i].Name < models[j].Name
		}
		return models[i].Provider < models[j].Provider
	})
	return models, nil
}

// Function creates a typed tool with automatic JSON marshaling
func Function[In, Out any](name, description string, run func(ctx context.Context, in In) (Out, error)) Tool {
	return &typedTool[In, Out]{
		name:        name,
		description: description,
		run:         run,
	}
}

// typedTool wraps a typed function as a Tool
type typedTool[In, Out any] struct {
	name        string
	description string
	run         func(ctx context.Context, in In) (Out, error)
}

func (t *typedTool[In, Out]) Name() string        { return t.name }
func (t *typedTool[In, Out]) Description() string { return t.description }

func (t *typedTool[In, Out]) Info() *ToolInfo {
	var in In
	return &ToolInfo{
		Type: "function",
		Function: ToolFunction{
			Name:        t.name,
			Description: t.description,
			Parameters:  generateSchema(in),
		},
	}
}

func (t *typedTool[In, Out]) Run(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	var in In
	if len(args) > 0 {
		if err := json.Unmarshal(args, &in); err != nil {
			return nil, fmt.Errorf("tool %s: unmarshaling input: %w", t.name, err)
		}
	}
	out, err := t.run(ctx, in)
	if err != nil {
		return nil, err
	}
	return json.Marshal(out)
}

// generateSchema creates ToolFunctionParameters from a struct type
// Supported struct tags:
//   - `json:"fieldname"` - JSON field name
//   - `description:"text"` - field description for the schema
//   - `enums:"a,b,c"` - allowed values (comma-separated)
//   - `is:"required"` - marks field as required (presence only, no value)
func generateSchema(v any) ToolFunctionParameters {
	params := ToolFunctionParameters{
		Type:       "object",
		Properties: make(map[string]ToolProperty),
		Required:   []string{},
	}

	t := reflect.TypeOf(v)
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	if t.Kind() != reflect.Struct {
		return params
	}

	for i := range t.NumField() {
		field := t.Field(i)
		if !field.IsExported() {
			continue
		}

		// Get JSON field name
		name := field.Name
		if jsonTag := field.Tag.Get("json"); jsonTag != "" {
			parts := strings.Split(jsonTag, ",")
			if parts[0] != "" && parts[0] != "-" {
				name = parts[0]
			}
		}

		// Get description
		description := field.Tag.Get("description")

		// Get enums
		var enums []string
		if enumTag := field.Tag.Get("enums"); enumTag != "" {
			enums = strings.Split(enumTag, ",")
		}

		// Determine type
		propType := "string"
		switch field.Type.Kind() {
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
			reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			propType = "integer"
		case reflect.Float32, reflect.Float64:
			propType = "number"
		case reflect.Bool:
			propType = "boolean"
		case reflect.Slice, reflect.Array:
			propType = "array"
		case reflect.Struct, reflect.Map:
			propType = "object"
		}

		params.Properties[name] = ToolProperty{
			Type:        propType,
			Description: description,
			Enum:        enums,
		}

		// Check if required
		if field.Tag.Get("is") == "required" {
			params.Required = append(params.Required, name)
		}
	}

	return params
}

// Event represents a streaming chunk or final response.
// During streaming: partial Content/Thinking deltas.
// When Done: complete accumulated response.
type Event struct {
	Content  string    // Content delta (streaming) or complete (when Done)
	Thinking string    // Thinking delta (streaming) or complete (when Done)
	Tool     *ToolCall // Non-nil when a tool is being called
	Done     bool      // True on final event with complete response
}

// Agent handles interactive sessions
type Agent struct {
	client   *Client
	model    string
	thinking Thinking // Extended thinking level
	tools    []Tool
	messages []*Message
}

// AgentOption configures an Agent
type AgentOption func(*Agent)

// WithModel sets the model for the agent
func WithModel(model string) AgentOption {
	return func(a *Agent) {
		a.model = model
	}
}

// WithThinking sets the extended thinking level.
// Supported values: ThinkingLow, ThinkingMedium, ThinkingHigh.
// Default is ThinkingMedium if not specified.
func WithThinking(level Thinking) AgentOption {
	return func(a *Agent) {
		a.thinking = level
	}
}

// WithTool adds a tool to the agent
func WithTool(t Tool) AgentOption {
	return func(a *Agent) {
		a.tools = append(a.tools, t)
	}
}

// WithMessages sets initial conversation history
func WithMessages(msgs []*Message) AgentOption {
	return func(a *Agent) {
		a.messages = msgs
	}
}

// Agent creates a new Agent with the given options
func (c *Client) Agent(opts ...AgentOption) *Agent {
	a := &Agent{
		client:   c,
		messages: []*Message{},
	}
	for _, opt := range opts {
		opt(a)
	}
	return a
}

// Send sends a message and returns a streaming iterator.
// Handles tool loop internally. Builds conversation history automatically.
// The final event has Done=true with complete Content/Thinking.
func (a *Agent) Send(ctx context.Context, content string) iter.Seq2[*Event, error] {
	return func(yield func(*Event, error) bool) {
		a.messages = append(a.messages, &Message{
			Role:    "user",
			Content: content,
		})

		// Build tool specs if we have tools
		var toolSpecs []*ToolInfo
		toolMap := make(map[string]Tool)
		for _, t := range a.tools {
			info := t.Info()
			toolSpecs = append(toolSpecs, info)
			toolMap[info.Function.Name] = t
		}

		for {
			req := &ChatRequest{
				Model:    a.model,
				Messages: a.messages,
				Tools:    toolSpecs,
				Thinking: a.thinking,
			}

			var assistantContent strings.Builder
			var assistantThinking strings.Builder
			var toolCall *ToolCall

			for resp, err := range a.client.Chat(ctx, req) {
				if err != nil {
					yield(nil, err)
					return
				}

				// Yield streaming events for thinking and content
				if resp.Thinking != "" {
					assistantThinking.WriteString(resp.Thinking)
					if !yield(&Event{Thinking: resp.Thinking}, nil) {
						return
					}
				}

				if resp.Content != "" {
					assistantContent.WriteString(resp.Content)
					if !yield(&Event{Content: resp.Content}, nil) {
						return
					}
				}

				// Handle tool calls
				if resp.Tool != nil {
					toolCall = resp.Tool
				}
			}

			// Add assistant message to history
			a.messages = append(a.messages, &Message{
				Role:     "assistant",
				Content:  assistantContent.String(),
				Thinking: assistantThinking.String(),
			})

			// If there's a tool call, execute it and continue the loop
			if toolCall != nil {
				// Yield tool event
				if !yield(&Event{Tool: toolCall}, nil) {
					return
				}

				tool, ok := toolMap[toolCall.Name]
				if !ok {
					yield(nil, fmt.Errorf("llm: unknown tool %q", toolCall.Name))
					return
				}

				result, err := tool.Run(ctx, toolCall.Arguments)
				if err != nil {
					// Add error as tool result
					a.messages = append(a.messages, &Message{
						Role:       "tool",
						Content:    fmt.Sprintf("Error: %v", err),
						ToolCallID: toolCall.ID,
					})
				} else {
					// Add tool result to messages
					a.messages = append(a.messages, &Message{
						Role:       "tool",
						Content:    string(result),
						ToolCallID: toolCall.ID,
					})
				}
				continue
			}

			// Yield final event with complete content
			yield(&Event{
				Content:  assistantContent.String(),
				Thinking: assistantThinking.String(),
				Done:     true,
			}, nil)
			return
		}
	}
}

// Clear resets the conversation history.
func (a *Agent) Clear() {
	a.messages = []*Message{}
}
