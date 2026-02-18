package gemini

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"log/slog"

	"github.com/matthewmueller/llm"
	"google.golang.org/genai"
)

// Config for the Gemini provider
type Config struct {
	APIKey string
	Log    *slog.Logger
}

// New creates a new Gemini client
func New(apiKey string) *Client {
	gc, _ := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})
	return &Client{
		gc,
	}
}

// Client implements the llm.Provider interface for Gemini
type Client struct {
	gc *genai.Client
}

var _ llm.Provider = (*Client)(nil)

// thinkingBudget maps thinking levels to token budgets
func thinkingBudget(level llm.Thinking) int {
	switch level {
	case llm.ThinkingNone, "":
		return 0
	case llm.ThinkingLow:
		return 4000
	case llm.ThinkingMedium:
		return 10000
	case llm.ThinkingHigh:
		return 32000
	default:
		return 0
	}
}

func (c *Client) Name() string {
	return "gemini"
}

func toUsage(usage *genai.GenerateContentResponseUsageMetadata) *llm.Usage {
	if usage == nil {
		return nil
	}
	total := int(usage.TotalTokenCount)
	input := int(usage.PromptTokenCount + usage.ToolUsePromptTokenCount)
	output := int(usage.CandidatesTokenCount + usage.ThoughtsTokenCount)
	if total == 0 && input == 0 && output == 0 {
		return nil
	}
	if total == 0 {
		total = input + output
	}
	return &llm.Usage{
		InputTokens:  input,
		OutputTokens: output,
		TotalTokens:  total,
	}
}

func toGeminiSchema(prop *llm.ToolProperty) *genai.Schema {
	schema := &genai.Schema{
		Type:        genai.Type(prop.Type),
		Description: prop.Description,
	}
	if len(prop.Enum) > 0 {
		schema.Enum = prop.Enum
	}
	if prop.Items != nil {
		schema.Items = toGeminiSchema(prop.Items)
	}
	return schema
}

// Chat sends a chat request to Gemini
func (c *Client) Chat(ctx context.Context, req *llm.ChatRequest) iter.Seq2[*llm.ChatResponse, error] {
	return func(yield func(*llm.ChatResponse, error) bool) {
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
				var parts []*genai.Part
				if m.Content != "" {
					parts = append(parts, &genai.Part{Text: m.Content})
				}
				// Include function call if present
				if m.ToolCall != nil {
					var args map[string]any
					if len(m.ToolCall.Arguments) > 0 {
						json.Unmarshal(m.ToolCall.Arguments, &args)
					}
					part := &genai.Part{
						FunctionCall: &genai.FunctionCall{
							Name: m.ToolCall.Name,
							Args: args,
						},
					}
					if len(m.ToolCall.ThoughtSignature) > 0 {
						part.ThoughtSignature = m.ToolCall.ThoughtSignature
					}
					parts = append(parts, part)
				}
				if len(parts) > 0 {
					contents = append(contents, &genai.Content{
						Parts: parts,
						Role:  genai.RoleModel,
					})
				}
			case "tool":
				// Tool results as function response
				// Parse the content as JSON to pass as response data
				var responseData map[string]any
				if err := json.Unmarshal([]byte(m.Content), &responseData); err != nil {
					// If not valid JSON, wrap in a result field
					responseData = map[string]any{"result": m.Content}
				}
				contents = append(contents, &genai.Content{
					Parts: []*genai.Part{{
						FunctionResponse: &genai.FunctionResponse{
							Name:     m.ToolCallID, // Gemini uses function name, not call ID
							Response: responseData,
						},
					}},
					Role: genai.RoleUser,
				})
			}
		}

		// Build config
		config := &genai.GenerateContentConfig{}

		if systemInstruction != nil {
			config.SystemInstruction = systemInstruction
		}

		// Enable thinking if set
		if budget := thinkingBudget(req.Thinking); budget > 0 {
			b := int32(budget)
			config.ThinkingConfig = &genai.ThinkingConfig{
				ThinkingBudget:  &b,
				IncludeThoughts: true,
			}
		}

		// Convert tools
		if len(req.Tools) > 0 {
			var funcs []*genai.FunctionDeclaration
			for _, t := range req.Tools {
				props := make(map[string]*genai.Schema)
				for name, prop := range t.Function.Parameters.Properties {
					props[name] = toGeminiSchema(prop)
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
		stream := c.gc.Models.GenerateContentStream(ctx, req.Model, contents, config)

		for resp, err := range stream {
			if err != nil {
				yield(nil, fmt.Errorf("gemini: streaming: %w", err))
				return
			}
			usage := toUsage(resp.UsageMetadata)

			for _, candidate := range resp.Candidates {
				if candidate.Content == nil {
					continue
				}

				var lastThoughtSignature []byte

				for _, part := range candidate.Content.Parts {
					chatResp := &llm.ChatResponse{
						Role:  "assistant",
						Usage: usage,
					}

					// Handle text content
					if part.Text != "" {
						chatResp.Content = part.Text
					}

					// Handle thinking content (for thinking models)
					if part.Thought {
						chatResp.Thinking = part.Text
						chatResp.Content = "" // Move to thinking
						if len(part.ThoughtSignature) > 0 {
							lastThoughtSignature = part.ThoughtSignature
						}
					}

					// Handle function calls
					if part.FunctionCall != nil {
						args, err := json.Marshal(part.FunctionCall.Args)
						if err != nil {
							yield(nil, fmt.Errorf("gemini: marshaling function args: %w", err))
							return
						}
						thoughtSignature := part.ThoughtSignature
						if len(thoughtSignature) == 0 {
							thoughtSignature = lastThoughtSignature
						}
						chatResp.ToolCall = &llm.ToolCall{
							ID:               part.FunctionCall.Name, // Gemini uses function name for correlation
							Name:             part.FunctionCall.Name,
							Arguments:        args,
							ThoughtSignature: thoughtSignature,
						}
					}

					// Check finish reason
					if candidate.FinishReason != "" {
						chatResp.Done = true
					}

					if chatResp.Content != "" || chatResp.Thinking != "" || chatResp.ToolCall != nil || chatResp.Done {
						if !yield(chatResp, nil) {
							return
						}
					}
				}
			}
		}
	}
}
