package tools

import (
	"context"
	"fmt"

	"github.com/matthewmueller/llm"
	"github.com/matthewmueller/llm/internal/ask"
)

// AskInput defines the input parameters for the Ask tool.
type AskInput struct {
	Question string   `json:"question" is:"required" description:"Question to ask the user"`
	Choices  []string `json:"choices" description:"Optional list of choices to present"`
}

// AskOutput defines the output of the Ask tool.
type AskOutput struct {
	Response string `json:"response"`
}

// Ask creates a tool for asking the user questions interactively.
func Ask(a ask.Asker) llm.Tool {
	return llm.Func("tool_ask",
		"Ask the user a question and wait for their response. Use this when you need clarification, confirmation, or input from the user before proceeding.",
		func(ctx context.Context, in AskInput) (AskOutput, error) {
			response, err := a.Ask(ctx, in.Question, in.Choices)
			if err != nil {
				return AskOutput{}, fmt.Errorf("ask: failed to get user response: %w", err)
			}
			return AskOutput{
				Response: response,
			}, nil
		},
	)
}
