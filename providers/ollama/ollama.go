package ollama

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"net/http"
	"net/url"
	"time"

	"github.com/matthewmueller/llm"
	ollama "github.com/ollama/ollama/api"
)

func Default() *Client {
	return New(&url.URL{
		Scheme: "http",
		Host:   "localhost:11434",
	})
}

// New creates a new Ollama client
func New(url *url.URL) *Client {
	oc := ollama.NewClient(url, http.DefaultClient)
	return &Client{
		oc,
	}
}

// Client implements the llm.Provider interface for Ollama
type Client struct {
	oc *ollama.Client
}

var _ llm.Provider = (*Client)(nil)

func (c *Client) Name() string {
	return "ollama"
}

func toUsage(resp ollama.ChatResponse) *llm.Usage {
	if resp.PromptEvalCount == 0 && resp.EvalCount == 0 {
		return nil
	}
	return &llm.Usage{
		InputTokens:  resp.PromptEvalCount,
		OutputTokens: resp.EvalCount,
		TotalTokens:  resp.PromptEvalCount + resp.EvalCount,
	}
}

func defaultOptions() map[string]any {
	// Popular runtime knobs from Ollama's official PARAMETER docs:
	// temperature, top_k, top_p, num_predict, num_ctx, repeat_last_n, repeat_penalty.
	// We source defaults from the Ollama SDK to stay aligned with server behavior.
	opts := ollama.DefaultOptions()
	return map[string]any{
		"num_ctx":        opts.NumCtx,
		"num_predict":    opts.NumPredict,
		"temperature":    opts.Temperature,
		"top_k":          opts.TopK,
		"top_p":          opts.TopP,
		"repeat_last_n":  opts.RepeatLastN,
		"repeat_penalty": opts.RepeatPenalty,
	}
}

func toThink(level llm.Thinking) *ollama.ThinkValue {
	switch level {
	case llm.ThinkingNone:
		return &ollama.ThinkValue{Value: false}
	case llm.ThinkingLow:
		return &ollama.ThinkValue{Value: "low"}
	case llm.ThinkingMedium:
		return &ollama.ThinkValue{Value: true}
	case llm.ThinkingHigh:
		return &ollama.ThinkValue{Value: "high"}
	default:
		return nil
	}
}

func toOllamaSchema(prop *llm.ToolProperty) ollama.ToolProperty {
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
	if prop.Items != nil {
		p.Items = toOllamaSchema(prop.Items)
	}
	return p
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
				props.Set(name, toOllamaSchema(prop))
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
			Options:  defaultOptions(),
			Think:    toThink(req.Thinking),
			// TODO: make this configurable on the ollama provider.
			KeepAlive: &ollama.Duration{
				Duration: 30 * time.Second,
			},
		}

		err := c.oc.Chat(ctx, chatReq, func(resp ollama.ChatResponse) error {
			chatResp := &llm.ChatResponse{
				Role:    resp.Message.Role,
				Content: resp.Message.Content,
				Usage:   toUsage(resp),
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
				chatResp.ToolCall = &llm.ToolCall{
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
