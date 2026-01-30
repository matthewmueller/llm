package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"log/slog"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/matthewmueller/llm"
	"github.com/matthewmueller/llm/internal/cache"
)

// New creates a new Anthropic client
func New(log *slog.Logger, apiKey string) *Client {
	ac := anthropic.NewClient(option.WithAPIKey(apiKey))
	return &Client{
		ac:  &ac,
		log: log,
		models: cache.Models(func(ctx context.Context) (models []*llm.Model, err error) {
			acmodels, err := ac.Models.List(ctx, anthropic.ModelListParams{})
			if err != nil {
				return nil, fmt.Errorf("anthropic: listing models: %w", err)
			}
			for _, model := range acmodels.Data {
				models = append(models, &llm.Model{
					Provider: "anthropic",
					Name:     model.ID,
				})
			}
			return models, nil
		}),
	}
}

// Client implements the llm.Provider interface for Anthropic
type Client struct {
	ac     *anthropic.Client
	log    *slog.Logger
	models func(ctx context.Context) ([]*llm.Model, error)
}

var _ llm.Provider = (*Client)(nil)

// thinkingBudget maps thinking levels to token budgets
func thinkingBudget(level llm.Thinking) int64 {
	switch level {
	case llm.ThinkingLow:
		return 4000
	case llm.ThinkingMedium:
		return 10000
	case llm.ThinkingHigh:
		return 32000
	default:
		return 10000 // default to medium
	}
}

func (c *Client) Name() string {
	return "anthropic"
}

// Models lists available models
func (c *Client) Models(ctx context.Context) (models []*llm.Model, err error) {
	return c.models(ctx)
}

// Chat sends a chat request to Anthropic
func (c *Client) Chat(ctx context.Context, req *llm.ChatRequest) iter.Seq2[*llm.ChatResponse, error] {
	return func(yield func(*llm.ChatResponse, error) bool) {
		model := req.Model
		if model == "" {
			yield(nil, fmt.Errorf("anthropic: required model is empty"))
			return
		}

		// Convert messages, extracting system message if present
		var systemBlocks []anthropic.TextBlockParam
		var messages []anthropic.MessageParam
		for _, m := range req.Messages {
			switch m.Role {
			case "system":
				systemBlocks = append(systemBlocks, anthropic.TextBlockParam{Text: m.Content})
			case "user":
				messages = append(messages, anthropic.NewUserMessage(anthropic.NewTextBlock(m.Content)))
			case "assistant":
				messages = append(messages, anthropic.NewAssistantMessage(anthropic.NewTextBlock(m.Content)))
			case "tool":
				// Tool results - add as user message with tool result block
				messages = append(messages, anthropic.NewUserMessage(anthropic.NewToolResultBlock(m.ToolCallID, m.Content, false)))
			}
		}

		// Convert tools
		var tools []anthropic.ToolUnionParam
		for _, t := range req.Tools {
			props := make(map[string]any)
			for name, prop := range t.Function.Parameters.Properties {
				p := map[string]any{
					"type":        prop.Type,
					"description": prop.Description,
				}
				if len(prop.Enum) > 0 {
					p["enum"] = prop.Enum
				}
				props[name] = p
			}

			tools = append(tools, anthropic.ToolUnionParam{
				OfTool: &anthropic.ToolParam{
					Name:        t.Function.Name,
					Description: anthropic.String(t.Function.Description),
					InputSchema: anthropic.ToolInputSchemaParam{
						Properties: props,
					},
				},
			})
		}

		params := anthropic.MessageNewParams{
			Model:     anthropic.Model(model),
			MaxTokens: 4096,
			Messages:  messages,
		}

		if len(systemBlocks) > 0 {
			params.System = systemBlocks
		}

		if len(tools) > 0 {
			params.Tools = tools
		}

		// Enable extended thinking based on level
		if budget := thinkingBudget(req.Thinking); budget > 0 {
			params.Thinking = anthropic.ThinkingConfigParamOfEnabled(budget)
			// Extended thinking requires higher max tokens
			if params.MaxTokens < budget+1000 {
				params.MaxTokens = budget + 1000
			}
		}

		stream := c.ac.Messages.NewStreaming(ctx, params)

		// Track tool use blocks being built
		var currentToolUse *llm.ToolCall
		var toolInput string

		for stream.Next() {
			event := stream.Current()

			switch evt := event.AsAny().(type) {
			case anthropic.ContentBlockDeltaEvent:
				chatResp := &llm.ChatResponse{
					Role: "assistant",
				}

				switch delta := evt.Delta.AsAny().(type) {
				case anthropic.TextDelta:
					chatResp.Content = delta.Text
				case anthropic.ThinkingDelta:
					chatResp.Thinking = delta.Thinking
				case anthropic.InputJSONDelta:
					// Accumulate tool input JSON
					toolInput += delta.PartialJSON
					continue // Don't yield yet
				}

				if chatResp.Content != "" || chatResp.Thinking != "" {
					if !yield(chatResp, nil) {
						return
					}
				}

			case anthropic.ContentBlockStartEvent:
				// Check if this is a tool use block
				if toolUse, ok := evt.ContentBlock.AsAny().(anthropic.ToolUseBlock); ok {
					currentToolUse = &llm.ToolCall{
						ID:   toolUse.ID,
						Name: toolUse.Name,
					}
					toolInput = ""
				}

			case anthropic.ContentBlockStopEvent:
				// If we were building a tool use, emit it now
				if currentToolUse != nil {
					currentToolUse.Arguments = json.RawMessage(toolInput)
					chatResp := &llm.ChatResponse{
						Role: "assistant",
						Tool: currentToolUse,
					}
					if !yield(chatResp, nil) {
						return
					}
					currentToolUse = nil
					toolInput = ""
				}

			case anthropic.MessageDeltaEvent:
				// Message finished
				if evt.Delta.StopReason != "" {
					chatResp := &llm.ChatResponse{
						Role: "assistant",
						Done: true,
					}
					if !yield(chatResp, nil) {
						return
					}
				}
			}
		}

		if err := stream.Err(); err != nil {
			yield(nil, fmt.Errorf("anthropic: streaming: %w", err))
		}
	}
}
