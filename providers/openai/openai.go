package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"log/slog"

	"github.com/matthewmueller/llm"
	"github.com/matthewmueller/llm/internal/cache"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// New creates a new OpenAI client
func New(log *slog.Logger, apiKey string) *Client {
	oc := openai.NewClient(option.WithAPIKey(apiKey))
	return &Client{
		&oc,
		log,
		cache.Models(func(ctx context.Context) ([]*llm.Model, error) {
			page, err := oc.Models.List(ctx)
			if err != nil {
				return nil, fmt.Errorf("openai: listing models: %w", err)
			}
			var models []*llm.Model
			for _, m := range page.Data {
				models = append(models, &llm.Model{
					Provider: "openai",
					Name:     m.ID,
				})
			}
			return models, nil
		}),
	}
}

// Client implements the llm.Provider interface for OpenAI
type Client struct {
	oc     *openai.Client
	log    *slog.Logger
	models func(ctx context.Context) ([]*llm.Model, error)
}

var _ llm.Provider = (*Client)(nil)

func (c *Client) Name() string {
	return "openai"
}

// Models lists available models
func (c *Client) Models(ctx context.Context) ([]*llm.Model, error) {
	return c.models(ctx)
}

// Chat sends a chat request to OpenAI
func (c *Client) Chat(ctx context.Context, req *llm.ChatRequest) iter.Seq2[*llm.ChatResponse, error] {
	return func(yield func(*llm.ChatResponse, error) bool) {
		model := req.Model
		if model == "" {
			yield(nil, fmt.Errorf("openai: required model is empty"))
			return
		}

		// Convert messages
		messages := make([]openai.ChatCompletionMessageParamUnion, 0, len(req.Messages))
		for _, m := range req.Messages {
			switch m.Role {
			case "user":
				messages = append(messages, openai.UserMessage(m.Content))
			case "assistant":
				messages = append(messages, openai.AssistantMessage(m.Content))
			case "system":
				messages = append(messages, openai.SystemMessage(m.Content))
			case "tool":
				// Tool results need special handling - for now add as user message
				messages = append(messages, openai.UserMessage(m.Content))
			}
		}

		// Convert tools
		var tools []openai.ChatCompletionToolParam
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

			tools = append(tools, openai.ChatCompletionToolParam{
				Function: openai.FunctionDefinitionParam{
					Name:        t.Function.Name,
					Description: openai.String(t.Function.Description),
					Parameters: openai.FunctionParameters{
						"type":       t.Function.Parameters.Type,
						"properties": props,
						"required":   t.Function.Parameters.Required,
					},
				},
			})
		}

		params := openai.ChatCompletionNewParams{
			Model:    openai.ChatModel(model),
			Messages: messages,
		}

		if len(tools) > 0 {
			params.Tools = tools
		}

		stream := c.oc.Chat.Completions.NewStreaming(ctx, params)

		for stream.Next() {
			chunk := stream.Current()

			for _, choice := range chunk.Choices {
				chatResp := &llm.ChatResponse{
					Role: "assistant",
				}

				// Handle content delta
				if choice.Delta.Content != "" {
					chatResp.Content = choice.Delta.Content
				}

				// Note: reasoning content for o1/o3 models is not streamed
				// and would need to be handled differently if needed

				// Handle tool calls
				if len(choice.Delta.ToolCalls) > 0 {
					tc := choice.Delta.ToolCalls[0]
					if tc.Function.Name != "" {
						chatResp.Tool = &llm.ToolCall{
							ID:        tc.ID,
							Name:      tc.Function.Name,
							Arguments: json.RawMessage(tc.Function.Arguments),
						}
					}
				}

				// Check if this choice is finished
				if choice.FinishReason != "" {
					chatResp.Done = true
				}

				if chatResp.Content != "" || chatResp.Thinking != "" || chatResp.Tool != nil || chatResp.Done {
					if !yield(chatResp, nil) {
						return
					}
				}
			}
		}

		if err := stream.Err(); err != nil {
			yield(nil, fmt.Errorf("openai: streaming: %w", err))
		}
	}
}
