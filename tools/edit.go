package tools

import (
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/matthewmueller/llm"
	"github.com/matthewmueller/virt"
)

// EditInput defines the input parameters for the Edit tool.
type EditInput struct {
	Path      string `json:"path" is:"required" description:"Absolute path to the file to edit"`
	OldString string `json:"old_string" is:"required" description:"The exact string to find and replace"`
	NewString string `json:"new_string" is:"required" description:"The string to replace it with"`
}

// EditOutput defines the output of the Edit tool.
type EditOutput struct {
	Success      bool `json:"success"`
	Replacements int  `json:"replacements"`
}

// Edit creates a tool for making targeted string replacements in files.
func Edit(fsys virt.FS) llm.Tool {
	return llm.Func("tool_edit",
		"Make a targeted string replacement in a file. Use this to modify specific parts of a file without rewriting the entire contents. The old_string must match exactly. Replaces only the first occurrence for safety.",
		func(ctx context.Context, in EditInput) (EditOutput, error) {
			// Read the current file content
			file, err := fsys.Open(in.Path)
			if err != nil {
				return EditOutput{}, fmt.Errorf("edit: unable to open file: %w", err)
			}

			content, err := io.ReadAll(file)
			file.Close()
			if err != nil {
				return EditOutput{}, fmt.Errorf("edit: unable to read file: %w", err)
			}

			contentStr := string(content)

			// Check if old_string exists
			if !strings.Contains(contentStr, in.OldString) {
				return EditOutput{}, fmt.Errorf("edit: old_string not found in file")
			}

			// Replace first occurrence only (for safety)
			newContent := strings.Replace(contentStr, in.OldString, in.NewString, 1)

			// Write the modified content back
			if err := fsys.WriteFile(in.Path, []byte(newContent), 0644); err != nil {
				return EditOutput{}, fmt.Errorf("edit: unable to write file: %w", err)
			}

			return EditOutput{
				Success:      true,
				Replacements: 1,
			}, nil
		},
	)
}
