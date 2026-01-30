package claudecode

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"iter"
	"log/slog"
	"os/exec"
	"strings"

	"github.com/matthewmueller/llm"
)

// Config holds configuration for the Claude CLI provider
type flags struct {
	Mode      string   // Permission mode (--permission-mode)
	Dirs      []string // Additional directories (--add-dir)
	SessionID string   // Session continuation (--session-id)
	Dangerous bool     // Skip permissions (--dangerously-skip-permissions)
	MaxTurns  int      // Limit turns (--max-turns)
}

// Client implements the llm.Provider interface for Claude CLI
type Client struct {
	log    *slog.Logger
	flags  flags
	models func(ctx context.Context) ([]*llm.Model, error)
}

var _ llm.Provider = (*Client)(nil)

// New creates a new Claude CLI client
func New(log *slog.Logger, flags ...string) *Client {
	return &Client{
		log:   log,
		flags: parseFlags(flags...),
		models: func(ctx context.Context) ([]*llm.Model, error) {
			// CLI uses its own models, return static list
			return []*llm.Model{
				{Provider: "claudecode", Name: "sonnet"},
				{Provider: "claudecode", Name: "opus"},
				{Provider: "claudecode", Name: "haiku"},
			}, nil
		},
	}
}

// stringSlice implements flag.Value for repeated string flags
type stringSlice []string

func (s *stringSlice) String() string { return strings.Join(*s, ",") }
func (s *stringSlice) Set(v string) error {
	*s = append(*s, v)
	return nil
}

func parseFlags(input ...string) flags {
	var f flags
	if len(input) == 0 {
		return f
	}

	args := strings.Fields(strings.Join(input, " "))
	fs := flag.NewFlagSet("claudecode", flag.ContinueOnError)
	fs.StringVar(&f.Mode, "permission-mode", "", "")
	fs.Var((*stringSlice)(&f.Dirs), "add-dir", "")
	fs.StringVar(&f.SessionID, "session-id", "", "")
	fs.BoolVar(&f.Dangerous, "dangerously-skip-permissions", false, "")
	fs.IntVar(&f.MaxTurns, "max-turns", 0, "")
	fs.Parse(args)
	return f
}

// Name returns the provider name
func (c *Client) Name() string {
	return "claudecode"
}

// Models returns available models
func (c *Client) Models(ctx context.Context) ([]*llm.Model, error) {
	return c.models(ctx)
}

// Chat sends a chat request via the Claude CLI
func (c *Client) Chat(ctx context.Context, req *llm.ChatRequest) iter.Seq2[*llm.ChatResponse, error] {
	return func(yield func(*llm.ChatResponse, error) bool) {
		// Build prompt from messages
		prompt := c.buildPrompt(req.Messages)

		// Build command args
		args := c.buildArgs(req)
		args = append(args, prompt)

		cmd := exec.CommandContext(ctx, "claude", args...)

		var stderr bytes.Buffer
		cmd.Stderr = &stderr

		stdout, err := cmd.StdoutPipe()
		if err != nil {
			yield(nil, fmt.Errorf("claudecode: creating stdout pipe: %w", err))
			return
		}

		if err := cmd.Start(); err != nil {
			yield(nil, fmt.Errorf("claudecode: starting command: %w", err))
			return
		}

		// Track tool use state for accumulating input_json_delta
		var currentToolUse *llm.ToolCall
		var toolInput strings.Builder

		scanner := bufio.NewScanner(stdout)
		// Increase scanner buffer for potentially large JSON lines
		buf := make([]byte, 0, 64*1024)
		scanner.Buffer(buf, 1024*1024)

		for scanner.Scan() {
			line := scanner.Bytes()
			if len(line) == 0 {
				continue
			}

			// Parse base event to determine type
			var base Event
			if err := json.Unmarshal(line, &base); err != nil {
				c.log.Debug("claudecode: failed to parse event", slog.Any("error", err))
				continue
			}

			switch base.Type {
			case "stream_event":
				var streamEvt StreamEvent
				if err := json.Unmarshal(line, &streamEvt); err != nil {
					continue
				}
				resp := c.handleStreamEvent(&streamEvt, &currentToolUse, &toolInput)
				if resp != nil {
					if !yield(resp, nil) {
						cmd.Process.Kill()
						return
					}
				}

			case "assistant":
				var msgEvt MessageEvent
				if err := json.Unmarshal(line, &msgEvt); err != nil {
					continue
				}
				resps := c.handleMessageEvent(&msgEvt)
				for _, resp := range resps {
					if !yield(resp, nil) {
						cmd.Process.Kill()
						return
					}
				}

			case "result":
				var resultEvt ResultEvent
				if err := json.Unmarshal(line, &resultEvt); err != nil {
					continue
				}
				resp := &llm.ChatResponse{
					Role: "assistant",
					Done: true,
				}
				if !yield(resp, nil) {
					cmd.Process.Kill()
					return
				}
			}
		}

		if err := scanner.Err(); err != nil {
			yield(nil, fmt.Errorf("claudecode: scanning output: %w", err))
			return
		}

		if err := cmd.Wait(); err != nil {
			errMsg := strings.TrimSpace(stderr.String())
			if errMsg != "" {
				yield(nil, fmt.Errorf("claudecode: command failed: %s: %w", errMsg, err))
			} else {
				yield(nil, fmt.Errorf("claudecode: command failed: %w", err))
			}
		}
	}
}

// buildPrompt concatenates messages into a single prompt string
func (c *Client) buildPrompt(messages []*llm.Message) string {
	var parts []string
	for _, m := range messages {
		switch m.Role {
		case "system":
			parts = append(parts, fmt.Sprintf("[System]: %s", m.Content))
		case "user":
			parts = append(parts, fmt.Sprintf("[User]: %s", m.Content))
		case "assistant":
			parts = append(parts, fmt.Sprintf("[Assistant]: %s", m.Content))
		case "tool":
			parts = append(parts, fmt.Sprintf("[Tool Result]: %s", m.Content))
		}
	}
	return strings.Join(parts, "\n\n")
}

// buildArgs constructs command line arguments from config and request
func (c *Client) buildArgs(req *llm.ChatRequest) []string {
	args := []string{
		"--print",
		"--verbose",
		"--output-format", "stream-json",
		"--include-partial-messages",
		"--strict-mcp-config",
	}

	// Add model if specified in request or config
	model := req.Model
	if model != "" {
		args = append(args, "--model", model)
	}

	// Add config options
	if c.flags.Mode != "" {
		args = append(args, "--permission-mode", c.flags.Mode)
	}
	for _, dir := range c.flags.Dirs {
		args = append(args, "--add-dir", dir)
	}
	if c.flags.SessionID != "" {
		args = append(args, "--session-id", c.flags.SessionID)
	}
	if c.flags.Dangerous {
		args = append(args, "--dangerously-skip-permissions")
	}
	if c.flags.MaxTurns > 0 {
		args = append(args, "--max-turns", fmt.Sprintf("%d", c.flags.MaxTurns))
	}

	return args
}

// handleStreamEvent processes streaming events and returns a ChatResponse if applicable
func (c *Client) handleStreamEvent(evt *StreamEvent, currentToolUse **llm.ToolCall, toolInput *strings.Builder) *llm.ChatResponse {
	inner := evt.Event

	switch inner.Type {
	case "content_block_start":
		if inner.ContentBlock != nil && inner.ContentBlock.Type == "tool_use" {
			*currentToolUse = &llm.ToolCall{
				ID:   inner.ContentBlock.ID,
				Name: inner.ContentBlock.Name,
			}
			toolInput.Reset()
		}
		return nil

	case "content_block_delta":
		if inner.Delta == nil {
			return nil
		}
		switch inner.Delta.Type {
		case "text_delta":
			if inner.Delta.Text != "" {
				return &llm.ChatResponse{
					Role:    "assistant",
					Content: inner.Delta.Text,
				}
			}
		case "thinking_delta":
			if inner.Delta.Thinking != "" {
				return &llm.ChatResponse{
					Role:     "assistant",
					Thinking: inner.Delta.Thinking,
				}
			}
		case "input_json_delta":
			// Accumulate tool input
			toolInput.WriteString(inner.Delta.PartialJSON)
		}
		return nil

	case "content_block_stop":
		// Emit completed tool call if we were building one
		if *currentToolUse != nil {
			(*currentToolUse).Arguments = json.RawMessage(toolInput.String())
			resp := &llm.ChatResponse{
				Role: "assistant",
				Tool: *currentToolUse,
			}
			*currentToolUse = nil
			toolInput.Reset()
			return resp
		}
		return nil
	}

	return nil
}

// handleMessageEvent processes complete message events
func (c *Client) handleMessageEvent(evt *MessageEvent) []*llm.ChatResponse {
	var resps []*llm.ChatResponse

	for _, item := range evt.Message.Content {
		switch item.Type {
		case "text":
			if item.Text != "" {
				resps = append(resps, &llm.ChatResponse{
					Role:    "assistant",
					Content: item.Text,
				})
			}
		case "thinking":
			if item.Thinking != "" {
				resps = append(resps, &llm.ChatResponse{
					Role:     "assistant",
					Thinking: item.Thinking,
				})
			}
		case "tool_use":
			resps = append(resps, &llm.ChatResponse{
				Role: "assistant",
				Tool: &llm.ToolCall{
					ID:        item.ID,
					Name:      item.Name,
					Arguments: item.Input,
				},
			})
		}
	}

	return resps
}
