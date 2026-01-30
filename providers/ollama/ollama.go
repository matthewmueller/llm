package ollama

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"log/slog"
	"net/http"
	"net/url"

	"github.com/matthewmueller/llm"
	"github.com/matthewmueller/llm/internal/cache"
	ollama "github.com/ollama/ollama/api"
)

func Default(log *slog.Logger) *Client {
	return New(log, &url.URL{
		Scheme: "http",
		Host:   "localhost:11434",
	})
}

// New creates a new Ollama client
func New(log *slog.Logger, url *url.URL) *Client {
	oc := ollama.NewClient(url, http.DefaultClient)
	return &Client{
		oc,
		log,
		cache.Models(func(ctx context.Context) ([]*llm.Model, error) {
			res, err := oc.List(ctx)
			if err != nil {
				return nil, fmt.Errorf("ollama: listing models: %w", err)
			}

			models := make([]*llm.Model, len(res.Models))
			for i, m := range res.Models {
				models[i] = &llm.Model{
					Provider: "ollama",
					Name:     m.Name,
				}
			}
			return models, nil
		}),
	}
}

// Client implements the llm.Provider interface for Ollama
type Client struct {
	oc     *ollama.Client
	log    *slog.Logger
	models func(ctx context.Context) ([]*llm.Model, error)
}

var _ llm.Provider = (*Client)(nil)

func (c *Client) Name() string {
	return "ollama"
}

// Models lists available models
func (c *Client) Models(ctx context.Context) ([]*llm.Model, error) {
	return c.models(ctx)
}

// Chat sends a chat request to Ollama
func (c *Client) Chat(ctx context.Context, req *llm.ChatRequest) iter.Seq2[*llm.ChatResponse, error] {
	return func(yield func(*llm.ChatResponse, error) bool) {
		model := req.Model
		if model == "" {
			yield(nil, fmt.Errorf("ollama: required model is empty"))
			return
		}

		// Convert messages
		messages := make([]ollama.Message, len(req.Messages))
		for i, m := range req.Messages {
			messages[i] = ollama.Message{
				Role:    m.Role,
				Content: m.Content,
			}
		}

		// Convert tools
		var tools ollama.Tools
		for _, t := range req.Tools {
			props := ollama.NewToolPropertiesMap()
			for name, prop := range t.Function.Parameters.Properties {
				p := ollama.ToolProperty{
					Type:        ollama.PropertyType{prop.Type},
					Description: prop.Description,
				}
				if len(prop.Enum) > 0 {
					enums := make([]any, len(prop.Enum))
					for i, e := range prop.Enum {
						enums[i] = e
					}
					p.Enum = enums
				}
				props.Set(name, p)
			}

			tools = append(tools, ollama.Tool{
				Type: t.Type,
				Function: ollama.ToolFunction{
					Name:        t.Function.Name,
					Description: t.Function.Description,
					Parameters: ollama.ToolFunctionParameters{
						Type:       t.Function.Parameters.Type,
						Properties: props,
						Required:   t.Function.Parameters.Required,
					},
				},
			})
		}

		stream := true
		chatReq := &ollama.ChatRequest{
			Model:    model,
			Messages: messages,
			Tools:    tools,
			Stream:   &stream,
		}

		// Enable thinking if set
		if req.Thinking != "" {
			chatReq.Options = map[string]any{
				"think": true,
			}
		}

		err := c.oc.Chat(ctx, chatReq, func(resp ollama.ChatResponse) error {
			chatResp := &llm.ChatResponse{
				Role:    resp.Message.Role,
				Content: resp.Message.Content,
				Done:    resp.Done,
			}

			// Handle thinking content if present
			if resp.Message.Thinking != "" {
				chatResp.Thinking = resp.Message.Thinking
			}

			// Handle tool calls
			if len(resp.Message.ToolCalls) > 0 {
				tc := resp.Message.ToolCalls[0]
				args, err := json.Marshal(tc.Function.Arguments)
				if err != nil {
					return fmt.Errorf("ollama: marshaling tool arguments: %w", err)
				}
				chatResp.Tool = &llm.ToolCall{
					Name:      tc.Function.Name,
					Arguments: args,
				}
			}

			if !yield(chatResp, nil) {
				return context.Canceled
			}
			return nil
		})

		if err != nil {
			yield(nil, fmt.Errorf("ollama: chat: %w", err))
		}
	}
}
