package tools

import (
	"context"
	"fmt"
	"path/filepath"

	"github.com/matthewmueller/llm"
	"github.com/matthewmueller/virt"
)

// WriteInput defines the input parameters for the Write tool.
type WriteInput struct {
	Path    string `json:"path" is:"required" description:"Absolute path to write the file"`
	Content string `json:"content" is:"required" description:"Content to write to the file"`
}

// WriteOutput defines the output of the Write tool.
type WriteOutput struct {
	BytesWritten int `json:"bytes_written"`
}

// Write creates a tool for creating or overwriting files.
func Write(fsys virt.FS) llm.Tool {
	return llm.Func("tool_write",
		"Create or overwrite a file with the given content. Use this to create new files or completely replace existing file contents.",
		func(ctx context.Context, in WriteInput) (WriteOutput, error) {
			// Create parent directories if they don't exist
			dir := filepath.Dir(in.Path)
			if err := fsys.MkdirAll(dir, 0755); err != nil {
				return WriteOutput{}, fmt.Errorf("write: unable to create parent directories: %w", err)
			}

			// Write the file
			data := []byte(in.Content)
			if err := fsys.WriteFile(in.Path, data, 0644); err != nil {
				return WriteOutput{}, fmt.Errorf("write: unable to write file: %w", err)
			}

			return WriteOutput{
				BytesWritten: len(data),
			}, nil
		},
	)
}
