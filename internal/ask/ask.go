package ask

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/Bowery/prompt"
)

// Asker is an interface for asking the user questions interactively.
type Asker interface {
	Ask(ctx context.Context, question string, choices []string) (string, error)
}

// Default returns the default asker implementation using bowery/prompt.
func Default() Asker {
	return &defaultAsker{}
}

// defaultAsker implements the Asker interface using bowery/prompt.
type defaultAsker struct{}

// Ask prompts the user with a question and returns their response.
func (a *defaultAsker) Ask(ctx context.Context, question string, choices []string) (string, error) {
	if len(choices) > 0 {
		// Print the question and choices
		fmt.Println(question)
		for i, choice := range choices {
			fmt.Printf("  %d. %s\n", i+1, choice)
		}

		response, err := prompt.Basic("Enter choice number or custom response: ", false)
		if err != nil {
			return "", fmt.Errorf("prompt: %w", err)
		}

		response = strings.TrimSpace(response)
		if num, err := strconv.Atoi(response); err == nil {
			if num >= 1 && num <= len(choices) {
				return choices[num-1], nil
			}
		}
		return response, nil
	}

	response, err := prompt.Basic(question+" ", false)
	if err != nil {
		return "", fmt.Errorf("prompt: %w", err)
	}
	return response, nil
}
