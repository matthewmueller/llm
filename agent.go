package llm

import (
	"context"
	"fmt"
	"iter"
	"strings"
	"sync"
)

// Event represents a streaming chunk or final response.
// During streaming: partial Content/Thinking deltas.
// When Done: complete accumulated response.
type Event struct {
	Content  string    // Content delta (streaming) or complete (when Done)
	Thinking string    // Thinking delta (streaming) or complete (when Done)
	Tool     *ToolCall // Non-nil when a tool is being called
	Done     bool      // True on final event with complete response
}

// Agent handles interactive sessions
type Agent struct {
	client   *Client
	model    string
	thinking Thinking // Extended thinking level
	tools    []Tool
	messages *messages
}

type messages struct {
	mu   sync.RWMutex
	msgs []*Message
}

func (m *messages) Add(msgs ...*Message) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.msgs = append(m.msgs, msgs...)
}

func (m *messages) List() []*Message {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.msgs
}

func (m *messages) Clear() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.msgs = nil
}

// AgentOption configures an Agent
type AgentOption func(*Agent)

// WithModel sets the model for the agent
// TODO: consolidate with Client struct inputs
func WithModel(model string) AgentOption {
	return func(a *Agent) {
		a.model = model
	}
}

// WithThinking sets the extended thinking level.
// Supported values: ThinkingLow, ThinkingMedium, ThinkingHigh.
// Default is ThinkingMedium if not specified.
func WithThinking(level Thinking) AgentOption {
	return func(a *Agent) {
		a.thinking = level
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
		a.messages.Add(msgs...)
	}
}

// Agent creates a new Agent with the given options
func (c *Client) Agent(opts ...AgentOption) *Agent {
	a := &Agent{
		client:   c,
		messages: &messages{},
	}
	for _, opt := range opts {
		opt(a)
	}
	return a
}

// Chat sends a message and returns a streaming iterator.
// Handles tool loop internally. Builds conversation history automatically.
// The final event has Done=true with complete Content/Thinking.
func (a *Agent) Chat(ctx context.Context, content string) iter.Seq2[*Event, error] {
	return func(yield func(*Event, error) bool) {
		a.messages.Add(&Message{
			Role:    "user",
			Content: content,
		})

		// Build tool specs if we have tools
		var toolSpecs []*ToolInfo
		toolMap := make(map[string]Tool)
		for _, t := range a.tools {
			info := t.Info()
			toolSpecs = append(toolSpecs, info)
			toolMap[info.Function.Name] = t
		}

		for {
			req := &ChatRequest{
				Model:    a.model,
				Messages: a.messages.List(),
				Tools:    toolSpecs,
				Thinking: a.thinking,
			}

			var assistantContent strings.Builder
			var assistantThinking strings.Builder
			var toolCall *ToolCall

			for resp, err := range a.client.Chat(ctx, req) {
				if err != nil {
					yield(nil, err)
					return
				}

				// Yield streaming events for thinking and content
				if resp.Thinking != "" {
					assistantThinking.WriteString(resp.Thinking)
					if !yield(&Event{Thinking: resp.Thinking}, nil) {
						return
					}
				}

				if resp.Content != "" {
					assistantContent.WriteString(resp.Content)
					if !yield(&Event{Content: resp.Content}, nil) {
						return
					}
				}

				// Handle tool calls
				if resp.Tool != nil {
					toolCall = resp.Tool
				}
			}

			// Add assistant message to history
			a.messages.Add(&Message{
				Role:     "assistant",
				Content:  assistantContent.String(),
				Thinking: assistantThinking.String(),
				ToolCall: toolCall,
			})

			// If there's a tool call, execute it and continue the loop
			if toolCall != nil {
				// Yield tool event
				if !yield(&Event{Tool: toolCall}, nil) {
					return
				}

				tool, ok := toolMap[toolCall.Name]
				if !ok {
					yield(nil, fmt.Errorf("llm: unknown tool %q", toolCall.Name))
					return
				}

				result, err := tool.Run(ctx, toolCall.Arguments)
				if err != nil {
					// Add error as tool result
					a.messages.Add(&Message{
						Role:       "tool",
						Content:    fmt.Sprintf("Error: %v", err),
						ToolCallID: toolCall.ID,
					})
				} else {
					// Add tool result to messages
					a.messages.Add(&Message{
						Role:       "tool",
						Content:    string(result),
						ToolCallID: toolCall.ID,
					})
				}
				continue
			}

			// Yield final event with complete content
			yield(&Event{
				Content:  assistantContent.String(),
				Thinking: assistantThinking.String(),
				Done:     true,
			}, nil)
			return
		}
	}
}

// Clear resets the conversation history.
func (a *Agent) Clear() {
	a.messages.Clear()
}
