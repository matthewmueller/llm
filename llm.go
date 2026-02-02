package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"log/slog"
	"sort"

	"github.com/matthewmueller/llm/internal/batch"
	"golang.org/x/sync/errgroup"
)

// Message represents a chat message
type Message struct {
	Role       string
	Content    string
	Thinking   string    // For chain-of-thought / thinking content
	ToolCall   *ToolCall // For assistant messages that invoke a tool
	ToolCallID string    // For tool results, the ID of the tool call being responded to
}

// Model represents an available model
type Model struct {
	Provider string
	Name     string
}

// ToolSchema defines a tool's JSON schema specification
type ToolSchema struct {
	Type     string
	Function *ToolFunction
}

// ToolFunction defines the function details for a tool
type ToolFunction struct {
	Name        string
	Description string
	Parameters  *ToolFunctionParameters
}

// ToolFunctionParameters defines the parameters schema for a tool
type ToolFunctionParameters struct {
	Type       string
	Properties map[string]*ToolProperty
	Required   []string
}

// ToolProperty defines a single property in the tool schema
type ToolProperty struct {
	Type        string
	Description string
	Enum        []string
}

type ChatRequest struct {
	Model    string
	Thinking Thinking
	Tools    []*ToolSchema
	Messages []*Message
}

// Provider interface
type Provider interface {
	Name() string
	Models(ctx context.Context) ([]*Model, error)
	Chat(ctx context.Context, req *ChatRequest) iter.Seq2[*ChatResponse, error]
}

// ChatResponse represents a streaming response from the chat API
type ChatResponse struct {
	Role     string    `json:"role,omitempty"`
	Content  string    `json:"content,omitempty"`  // Content chunk
	Thinking string    `json:"thinking,omitempty"` // Thinking/reasoning content (if any)
	ToolCall *ToolCall `json:"tool_call,omitempty"`
	Done     bool      `json:"done,omitempty"` // True when response is complete
}

// Tool interface - high-level typed tool definition
type Tool interface {
	Schema() *ToolSchema
	Run(ctx context.Context, in json.RawMessage) (out []byte, err error)
}

// ToolCall represents a tool invocation from the model
type ToolCall struct {
	ID               string
	Name             string
	Arguments        json.RawMessage
	ThoughtSignature []byte
}

// Thinking represents the level of extended thinking/reasoning
type Thinking string

const (
	ThinkingNone   Thinking = "none"   // Disable thinking
	ThinkingLow    Thinking = "low"    // Low thinking budget
	ThinkingMedium Thinking = "medium" // Medium thinking budget
	ThinkingHigh   Thinking = "high"   // High thinking budget
)

type Option func(*Config)

type Config struct {
	Log *slog.Logger
	// Provider string
	Model    string
	Thinking Thinking
	Tools    []Tool
	Messages []*Message
	MaxSteps int
}

// func WithProvider(name string) Option {
// 	return func(c *Config) {
// 		c.Provider = name
// 	}
// }

// WithModel sets the model for the agent
func WithModel(model string) Option {
	return func(c *Config) {
		c.Model = model
	}
}

// WithThinking sets the extended thinking level.
// Supported values: ThinkingLow, ThinkingMedium, ThinkingHigh.
// Default is ThinkingMedium if not specified.
func WithThinking(level Thinking) Option {
	return func(c *Config) {
		c.Thinking = level
	}
}

// WithTool adds a tool to the agent
func WithTool(tools ...Tool) Option {
	return func(c *Config) {
		c.Tools = append(c.Tools, tools...)
	}
}

// WithMessages sets initial conversation history
func WithMessage(messages ...*Message) Option {
	return func(c *Config) {
		c.Messages = append(c.Messages, messages...)
	}
}

// WithMaxSteps sets the maximum number of steps in a turn
func WithMaxSteps(max int) Option {
	return func(c *Config) {
		c.MaxSteps = max
	}
}

// SystemMessage creates a system message
func SystemMessage(content string) *Message {
	return &Message{
		Role:    "system",
		Content: content,
	}
}

// UserMessage creates a user message
func UserMessage(content string) *Message {
	return &Message{
		Role:    "user",
		Content: content,
	}
}

// AssistantMessage creates an assistant message
func AssistantMessage(content string) *Message {
	return &Message{
		Role:    "assistant",
		Content: content,
	}
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

func (c *Client) findProvider(name string) (Provider, error) {
	for _, p := range c.providers {
		if p.Name() == name {
			return p, nil
		}
	}
	return nil, fmt.Errorf("llm: provider %q not found", name)
}

func toolSchemas(tools []Tool) []*ToolSchema {
	schemas := []*ToolSchema{}
	for _, t := range tools {
		schemas = append(schemas, t.Schema())
	}
	return schemas
}

// Chat sends a chat request to the appropriate provider
func (c *Client) Chat(ctx context.Context, provider string, options ...Option) iter.Seq2[*ChatResponse, error] {
	return func(yield func(*ChatResponse, error) bool) {
		config := &Config{
			Thinking: ThinkingMedium,
		}
		for _, option := range options {
			option(config)
		}

		provider, err := c.findProvider(provider)
		if err != nil {
			yield(nil, err)
			return
		}

		toolbox := map[string]Tool{}
		for _, tool := range config.Tools {
			schema := tool.Schema()
			toolbox[schema.Function.Name] = tool
		}

		messages := append([]*Message{}, config.Messages...)

	turn:
		for steps := 0; steps < config.MaxSteps || config.MaxSteps == 0; steps++ {
			req := &ChatRequest{
				Model:    config.Model,
				Thinking: config.Thinking,
				Tools:    toolSchemas(config.Tools),
				Messages: messages,
			}

			batch, ctx := batch.New[*Message](ctx)
			hasContent := false
			isThinking := false

			// Make a request to the LLM and stream back the response
			for res, err := range provider.Chat(ctx, req) {
				if err != nil {
					if !yield(res, err) {
						break turn
					}
					continue
				}

				messages = append(messages, &Message{
					Role:     res.Role,
					Content:  res.Content,
					Thinking: res.Thinking,
					ToolCall: res.ToolCall,
				})

				// We've got a tool call to handle
				if res.ToolCall != nil {
					// Yield response back to caller
					if !yield(res, err) {
						break turn
					}

					tool, ok := toolbox[res.ToolCall.Name]
					if !ok {
						if !yield(nil, fmt.Errorf("llm: unknown tool %q called by model", res.ToolCall.Name)) {
							break turn
						}
						continue
					}
					// Run tool in a goroutine
					batch.Go(func() (*Message, error) {
						result, err := tool.Run(ctx, res.ToolCall.Arguments)
						if err != nil {
							return nil, fmt.Errorf("llm: running tool %q: %w", res.ToolCall.Name, err)
						}
						return &Message{
							Role:       "tool",
							Content:    string(result),
							ToolCallID: res.ToolCall.ID,
						}, nil
					})
				}

				// Stop yielding further messages if we have tool calls to process
				if batch.Size() > 0 {
					continue
				}

				// Track if we're in thinking mode
				if res.Thinking != "" {
					isThinking = true
				}

				// If we're switching between thinking and content, add a small separator
				if isThinking && res.Thinking == "" && res.Content != "" {
					isThinking = false
					if !yield(&ChatResponse{
						Role:    res.Role,
						Content: "\n\n",
					}, nil) {
						break turn
					}
				}

				// We're going to send content
				if res.Content != "" {
					hasContent = true
				}

				// Yield response back to caller
				if !yield(res, err) {
					break
				}
			}

			// Wait for tool calls to complete
			toolResults, err := batch.Wait()
			if err != nil {
				if !yield(nil, err) {
					break
				}
			}

			// If there are no tool results, we're done this turn
			if len(toolResults) == 0 {
				break turn
			}

			// Append tool results to messages and continue the loop
			messages = append(messages, toolResults...)

			// Add some artificial spacing to separate tool results from next LLM response
			if hasContent {
				if !yield(&ChatResponse{
					Role:    "assistant",
					Content: "\n\n",
				}, nil) {
					break turn
				}
			}
		}
	}
}

type ErrMultipleModels struct {
	Provider string
	Name     string
	Matches  []*Model
}

func (e *ErrMultipleModels) Error() string {
	matchStr := ""
	for _, m := range e.Matches {
		matchStr += fmt.Sprintf("- Provider: %q, Model: %q\n", m.Provider, m.Name)
	}
	if e.Provider == "" {
		return fmt.Sprintf("llm: multiple models found for %q:\n%s", e.Name, matchStr)
	}
	return fmt.Sprintf("llm: multiple models found for %q from provider %q:\n%s", e.Name, e.Provider, matchStr)
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
