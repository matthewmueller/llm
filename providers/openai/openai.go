package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"strings"

	"github.com/matthewmueller/llm"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared"
)

// New creates a new OpenAI client
func New(apiKey string) *Client {
	oc := openai.NewClient(option.WithAPIKey(apiKey))
	return &Client{
		&oc,
	}
}

// Client implements the llm.Provider interface for OpenAI
type Client struct {
	oc *openai.Client
}

var _ llm.Provider = (*Client)(nil)

// reasoningEffort maps thinking levels to OpenAI reasoning effort values
func reasoningEffort(level llm.Thinking) shared.ReasoningEffort {
	switch level {
	case llm.ThinkingLow:
		return shared.ReasoningEffortLow
	case llm.ThinkingMedium:
		return shared.ReasoningEffortMedium
	case llm.ThinkingHigh:
		return shared.ReasoningEffortHigh
	default:
		return shared.ReasoningEffortMedium
	}
}

func (c *Client) Name() string {
	return "openai"
}

func toOpenAISchema(prop *llm.ToolProperty) map[string]any {
	p := map[string]any{
		"type":        prop.Type,
		"description": prop.Description,
	}
	if len(prop.Enum) > 0 {
		p["enum"] = prop.Enum
	}
	if prop.Items != nil {
		p["items"] = toOpenAISchema(prop.Items)
	}
	return p
}

// Chat sends a chat request to OpenAI using the Responses API
func (c *Client) Chat(ctx context.Context, req *llm.ChatRequest) iter.Seq2[*llm.ChatResponse, error] {
	return func(yield func(*llm.ChatResponse, error) bool) {
		model := req.Model
		if model == "" {
			yield(nil, fmt.Errorf("openai: required model is empty"))
			return
		}

		// Convert messages to Responses API input format
		var input []responses.ResponseInputItemUnionParam
		for _, m := range req.Messages {
			switch m.Role {
			case "user":
				input = append(input, responses.ResponseInputItemParamOfMessage(m.Content, responses.EasyInputMessageRoleUser))
			case "assistant":
				if m.Content != "" {
					input = append(input, responses.ResponseInputItemParamOfMessage(m.Content, responses.EasyInputMessageRoleAssistant))
				}
				// Include function call if present
				if m.ToolCall != nil {
					input = append(input, responses.ResponseInputItemUnionParam{
						OfFunctionCall: &responses.ResponseFunctionToolCallParam{
							CallID:    m.ToolCall.ID,
							Name:      m.ToolCall.Name,
							Arguments: string(m.ToolCall.Arguments),
						},
					})
				}
			case "system":
				input = append(input, responses.ResponseInputItemParamOfMessage(m.Content, responses.EasyInputMessageRoleSystem))
			case "tool":
				// Tool results use function call output
				input = append(input, responses.ResponseInputItemParamOfFunctionCallOutput(m.ToolCallID, m.Content))
			}
		}

		// Convert tools to Responses API format
		var tools []responses.ToolUnionParam
		for _, t := range req.Tools {
			props := make(map[string]any)
			for name, prop := range t.Function.Parameters.Properties {
				props[name] = toOpenAISchema(prop)
			}

			tool := responses.ToolParamOfFunction(
				t.Function.Name,
				map[string]any{
					"type":       t.Function.Parameters.Type,
					"properties": props,
					"required":   t.Function.Parameters.Required,
				},
				false,
			)
			tool.OfFunction.Description = openai.String(t.Function.Description)
			tools = append(tools, tool)
		}

		params := responses.ResponseNewParams{
			Model: shared.ResponsesModel(model),
			Input: responses.ResponseNewParamsInputUnion{
				OfInputItemList: input,
			},
		}

		if len(tools) > 0 {
			params.Tools = tools
		}

		// Configure reasoning for o-series models
		if req.Thinking != "" {
			params.Reasoning = shared.ReasoningParam{
				Effort:  reasoningEffort(req.Thinking),
				Summary: shared.ReasoningSummaryDetailed,
			}
		}

		stream := c.oc.Responses.NewStreaming(ctx, params)

		// Track function call state across streaming events
		var currentFunctionCall *llm.ToolCall
		var functionArgs strings.Builder

		for stream.Next() {
			event := stream.Current()

			switch event.Type {
			case "response.output_text.delta":
				// Text content delta
				delta := event.AsResponseOutputTextDelta()
				if delta.Delta != "" {
					if !yield(&llm.ChatResponse{
						Role:    "assistant",
						Content: delta.Delta,
					}, nil) {
						return
					}
				}

			case "response.reasoning_summary_text.delta":
				// Reasoning/thinking content delta
				delta := event.AsResponseReasoningSummaryTextDelta()
				if delta.Delta != "" {
					if !yield(&llm.ChatResponse{
						Role:     "assistant",
						Thinking: delta.Delta,
					}, nil) {
						return
					}
				}

			case "response.output_item.added":
				// New output item - check if it's a function call
				added := event.AsResponseOutputItemAdded()
				if added.Item.Type == "function_call" {
					currentFunctionCall = &llm.ToolCall{
						ID:   added.Item.CallID,
						Name: added.Item.Name,
					}
					functionArgs.Reset()
				}

			case "response.function_call_arguments.delta":
				// Function call arguments delta
				delta := event.AsResponseFunctionCallArgumentsDelta()
				functionArgs.WriteString(delta.Delta)

			case "response.output_item.done":
				// Output item completed - if function call, emit it
				done := event.AsResponseOutputItemDone()
				if done.Item.Type == "function_call" && currentFunctionCall != nil {
					currentFunctionCall.Arguments = json.RawMessage(functionArgs.String())
					if !yield(&llm.ChatResponse{
						Role:     "assistant",
						ToolCall: currentFunctionCall,
					}, nil) {
						return
					}
					currentFunctionCall = nil
				}

			case "response.completed":
				// Response complete
				if !yield(&llm.ChatResponse{
					Role: "assistant",
					Done: true,
				}, nil) {
					return
				}

			case "response.failed":
				// Handle failure
				failed := event.AsResponseFailed()
				yield(nil, fmt.Errorf("openai: response failed: %s", failed.Response.Status))
				return
			}
		}

		if err := stream.Err(); err != nil {
			yield(nil, fmt.Errorf("openai: streaming: %w", err))
		}
	}
}
