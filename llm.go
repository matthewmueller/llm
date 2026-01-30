package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"log/slog"
	"sort"

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
	ThinkingNone   Thinking = "none"   // Disable thinking
	ThinkingLow    Thinking = "low"    // Low thinking budget
	ThinkingMedium Thinking = "medium" // Medium thinking budget
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
