package gemini

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"log/slog"

	"github.com/matthewmueller/llm"
	"github.com/matthewmueller/llm/internal/cache"
	"google.golang.org/genai"
)

const DefaultModel = "gemini-2.0-flash"

// Config for the Gemini provider
type Config struct {
	APIKey string
	Log    *slog.Logger
}

// New creates a new Gemini client
func New(log *slog.Logger, apiKey string) *Client {
	gc, _ := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})
	return &Client{
		gc,
		log,
		cache.Models(func(ctx context.Context) (models []*llm.Model, err error) {
			for model, err := range gc.Models.All(ctx) {
				if err != nil {
					return nil, fmt.Errorf("gemini: listing models: %w", err)
				}
				models = append(models, &llm.Model{
					Provider: "gemini",
					Name:     model.Name,
				})
			}
			return models, nil
		}),
	}
}

// Client implements the llm.Provider interface for Gemini
type Client struct {
	gc     *genai.Client
	log    *slog.Logger
	models func(ctx context.Context) ([]*llm.Model, error)
}

var _ llm.Provider = (*Client)(nil)

func (c *Client) Name() string {
	return "gemini"
}

// Models lists available models
func (c *Client) Models(ctx context.Context) (models []*llm.Model, err error) {
	return c.models(ctx)
}

// Chat sends a chat request to Gemini
func (c *Client) Chat(ctx context.Context, req *llm.ChatRequest) iter.Seq2[*llm.ChatResponse, error] {
	return func(yield func(*llm.ChatResponse, error) bool) {
		model := req.Model
		if model == "" {
			model = DefaultModel
		}

		// Convert messages to Gemini format
		var contents []*genai.Content
		var systemInstruction *genai.Content

		for _, m := range req.Messages {
			switch m.Role {
			case "system":
				systemInstruction = &genai.Content{
					Parts: []*genai.Part{{Text: m.Content}},
					Role:  genai.RoleUser, // System uses user role internally
				}
			case "user":
				contents = append(contents, &genai.Content{
					Parts: []*genai.Part{{Text: m.Content}},
					Role:  genai.RoleUser,
				})
			case "assistant":
				contents = append(contents, &genai.Content{
					Parts: []*genai.Part{{Text: m.Content}},
					Role:  genai.RoleModel,
				})
			case "tool":
				// Tool results
				contents = append(contents, &genai.Content{
					Parts: []*genai.Part{{Text: m.Content}},
					Role:  genai.RoleUser,
				})
			}
		}

		// Build config
		config := &genai.GenerateContentConfig{}

		if systemInstruction != nil {
			config.SystemInstruction = systemInstruction
		}

		// Convert tools
		if len(req.Tools) > 0 {
			var funcs []*genai.FunctionDeclaration
			for _, t := range req.Tools {
				props := make(map[string]*genai.Schema)
				for name, prop := range t.Function.Parameters.Properties {
					schema := &genai.Schema{
						Type:        genai.Type(prop.Type),
						Description: prop.Description,
					}
					if len(prop.Enum) > 0 {
						schema.Enum = prop.Enum
					}
					props[name] = schema
				}

				funcs = append(funcs, &genai.FunctionDeclaration{
					Name:        t.Function.Name,
					Description: t.Function.Description,
					Parameters: &genai.Schema{
						Type:       genai.TypeObject,
						Properties: props,
						Required:   t.Function.Parameters.Required,
					},
				})
			}

			config.Tools = []*genai.Tool{
				{FunctionDeclarations: funcs},
			}
		}

		// Stream response
		stream := c.gc.Models.GenerateContentStream(ctx, model, contents, config)

		for resp, err := range stream {
			if err != nil {
				yield(nil, fmt.Errorf("gemini: streaming: %w", err))
				return
			}

			for _, candidate := range resp.Candidates {
				if candidate.Content == nil {
					continue
				}

				for _, part := range candidate.Content.Parts {
					chatResp := &llm.ChatResponse{
						Role: "assistant",
					}

					// Handle text content
					if part.Text != "" {
						chatResp.Content = part.Text
					}

					// Handle thinking content (for thinking models)
					if part.Thought {
						chatResp.Thinking = part.Text
						chatResp.Content = "" // Move to thinking
					}

					// Handle function calls
					if part.FunctionCall != nil {
						args, err := json.Marshal(part.FunctionCall.Args)
						if err != nil {
							yield(nil, fmt.Errorf("gemini: marshaling function args: %w", err))
							return
						}
						chatResp.Tool = &llm.ToolCall{
							Name:      part.FunctionCall.Name,
							Arguments: args,
						}
					}

					// Check finish reason
					if candidate.FinishReason != "" {
						chatResp.Done = true
					}

					if chatResp.Content != "" || chatResp.Thinking != "" || chatResp.Tool != nil || chatResp.Done {
						if !yield(chatResp, nil) {
							return
						}
					}
				}
			}
		}
	}
}
