package llm

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"log/slog"
	"reflect"
	"sort"
	"strings"

	"golang.org/x/sync/errgroup"
)

// Message represents a chat message
type Message struct {
	Role     string
	Content  string
	Thinking string // For chain-of-thought / thinking content
}

// Model represents an available model
type Model struct {
	Provider string
	Name     string
}

// ToolSpec defines a tool's JSON schema specification
type ToolSpec struct {
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

// ChatRequest represents a request to the chat API
type ChatRequest struct {
	Model    string
	Messages []*Message
	Tools    []*ToolSpec
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
	Name() string
	Description() string
	Spec() *ToolSpec
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

// typedTool wraps a typed function as a Tool
type typedTool[In, Out any] struct {
	name        string
	description string
	run         func(ctx context.Context, in In) (Out, error)
}

// Function creates a typed tool with automatic JSON marshaling
func Function[In, Out any](name, description string, run func(ctx context.Context, in In) (Out, error)) Tool {
	return &typedTool[In, Out]{
		name:        name,
		description: description,
		run:         run,
	}
}

func (t *typedTool[In, Out]) Name() string        { return t.name }
func (t *typedTool[In, Out]) Description() string { return t.description }

func (t *typedTool[In, Out]) Spec() *ToolSpec {
	var in In
	return &ToolSpec{
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

type Writer interface {
	Write(p []byte) (n int, err error)
	Think(p []byte) (n int, err error)
}

// Agent handles interactive sessions
type Agent struct {
	client   *Client
	model    string
	reader   io.Reader
	writer   Writer
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

// WithReader sets the input reader for the agent
func WithReader(r io.Reader) AgentOption {
	return func(a *Agent) {
		a.reader = r
	}
}

// WithWriter sets the output writer for the agent
func WithWriter(w Writer) AgentOption {
	return func(a *Agent) {
		a.writer = w
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

// Run starts the agent loop reading from reader
func (a *Agent) Run(ctx context.Context) error {
	if a.reader == nil {
		return errors.New("llm: agent requires a reader for Run()")
	}
	if a.writer == nil {
		return errors.New("llm: agent requires a writer for Run()")
	}

	scanner := bufio.NewScanner(a.reader)
	for {
		fmt.Fprint(a.writer, "> ")
		if !scanner.Scan() {
			break
		}
		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}
		if input == "exit" || input == "quit" {
			break
		}

		_, err := a.Send(ctx, input)
		if err != nil {
			return err
		}
		fmt.Fprintln(a.writer)
	}
	return scanner.Err()
}

// Response from Send - includes thinking for programmatic access
type Response struct {
	Content  string
	Thinking string
}

// Send sends a single message and returns the response
func (a *Agent) Send(ctx context.Context, content string) (*Response, error) {
	a.messages = append(a.messages, &Message{
		Role:    "user",
		Content: content,
	})

	// Build tool specs if we have tools
	var toolSpecs []*ToolSpec
	toolMap := make(map[string]Tool)
	for _, t := range a.tools {
		toolSpecs = append(toolSpecs, t.Spec())
		toolMap[t.Name()] = t
	}

	for {
		req := &ChatRequest{
			Model:    a.model,
			Messages: a.messages,
			Tools:    toolSpecs,
		}

		var response Response
		var assistantContent strings.Builder
		var assistantThinking strings.Builder
		var toolCall *ToolCall

		for resp, err := range a.client.Chat(ctx, req) {
			if err != nil {
				return nil, err
			}

			// Handle thinking content
			if resp.Thinking != "" {
				assistantThinking.WriteString(resp.Thinking)
				if a.writer != nil {
					a.writer.Think([]byte(resp.Thinking))
					// if tw, ok := a.writer.(Writer); ok {
					// } else {
					// 	// Write thinking with dim ANSI codes
					// 	fmt.Fprintf(a.writer, "\033[2m%s\033[0m", resp.Thinking)
					// }
				}
			}

			// Handle regular content
			if resp.Content != "" {
				assistantContent.WriteString(resp.Content)
				if a.writer != nil {
					fmt.Fprint(a.writer, resp.Content)
				}
			}

			// Handle tool calls
			if resp.Tool != nil {
				toolCall = resp.Tool
			}
		}

		response.Content = assistantContent.String()
		response.Thinking = assistantThinking.String()

		// Add assistant message to history
		a.messages = append(a.messages, &Message{
			Role:     "assistant",
			Content:  response.Content,
			Thinking: response.Thinking,
		})

		// If there's a tool call, execute it and continue the loop
		if toolCall != nil {
			tool, ok := toolMap[toolCall.Name]
			if !ok {
				return nil, fmt.Errorf("llm: unknown tool %q", toolCall.Name)
			}

			result, err := tool.Run(ctx, toolCall.Arguments)
			if err != nil {
				// Add error as tool result
				a.messages = append(a.messages, &Message{
					Role:    "tool",
					Content: fmt.Sprintf("Error: %v", err),
				})
			} else {
				// Add tool result to messages
				a.messages = append(a.messages, &Message{
					Role:    "tool",
					Content: string(result),
				})
			}
			continue
		}

		return &response, nil
	}
}

// Messages returns the conversation history
func (a *Agent) Messages() []*Message {
	return a.messages
}
