package claudecode

import "encoding/json"

// Event is the base event structure from CLI output
type Event struct {
	Type    string `json:"type"`    // "system", "stream_event", "assistant", "user", "result"
	Subtype string `json:"subtype"` // "init", "success", "error"
}

// StreamEvent represents streaming events with --include-partial-messages
type StreamEvent struct {
	Type  string     `json:"type"`
	Event InnerEvent `json:"event"`
}

// InnerEvent contains the streaming event details
type InnerEvent struct {
	Type         string        `json:"type"` // "content_block_delta", "content_block_start", "content_block_stop"
	Index        int           `json:"index"`
	Delta        *Delta        `json:"delta"`
	ContentBlock *ContentBlock `json:"content_block"`
}

// Delta represents incremental content updates
type Delta struct {
	Type        string `json:"type"`         // "text_delta", "thinking_delta", "input_json_delta"
	Text        string `json:"text"`         // For text_delta
	Thinking    string `json:"thinking"`     // For thinking_delta
	PartialJSON string `json:"partial_json"` // For input_json_delta
}

// ContentBlock represents a content block in streaming
type ContentBlock struct {
	Type string `json:"type"` // "text", "thinking", "tool_use"
	ID   string `json:"id"`   // For tool_use blocks
	Name string `json:"name"` // For tool_use blocks
}

// MessageEvent represents assistant/user message events
type MessageEvent struct {
	Type    string  `json:"type"`
	Message Message `json:"message"`
}

// Message represents a complete message
type Message struct {
	Role    string        `json:"role"`
	Content []ContentItem `json:"content"`
}

// ContentItem represents a single content item in a message
type ContentItem struct {
	Type      string          `json:"type"` // "text", "thinking", "tool_use", "tool_result"
	Text      string          `json:"text"`
	Thinking  string          `json:"thinking"`
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Input     json.RawMessage `json:"input"`
	ToolUseID string          `json:"tool_use_id"`
}

// ResultEvent represents the final result
type ResultEvent struct {
	Type    string `json:"type"`
	Subtype string `json:"subtype"` // "success" or "error"
}
