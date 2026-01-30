package tools

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/matthewmueller/llm"
	"github.com/matthewmueller/virt"
)

// ReadInput defines the input parameters for the Read tool.
type ReadInput struct {
	Path   string `json:"path" is:"required" description:"Absolute path to the file to read"`
	Offset int    `json:"offset" description:"Line number to start reading from (1-indexed, default: 1)"`
	Limit  int    `json:"limit" description:"Maximum number of lines to read (default: all lines)"`
}

// ReadOutput defines the output of the Read tool.
type ReadOutput struct {
	Content    string `json:"content"`
	TotalLines int    `json:"total_lines"`
	StartLine  int    `json:"start_line"`
	EndLine    int    `json:"end_line"`
}

// Read creates a tool for reading file contents with optional line range.
func Read(fsys virt.FS) llm.Tool {
	return llm.Func("tool_read",
		"Read the contents of a file from the filesystem. Use this to examine source code, configuration files, or any text file.",
		func(ctx context.Context, in ReadInput) (ReadOutput, error) {
			file, err := fsys.Open(in.Path)
			if err != nil {
				return ReadOutput{}, fmt.Errorf("read: unable to open file: %w", err)
			}
			defer file.Close()

			// Check if it's a directory
			stat, err := file.Stat()
			if err != nil {
				return ReadOutput{}, fmt.Errorf("read: unable to stat file: %w", err)
			}
			if stat.IsDir() {
				return ReadOutput{}, fmt.Errorf("read: path is a directory, not a file")
			}

			// Read the file content
			var buf bytes.Buffer
			scanner := bufio.NewScanner(file)
			scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // 1MB buffer for long lines
			var lines []string
			for scanner.Scan() {
				lines = append(lines, scanner.Text())
			}
			if err := scanner.Err(); err != nil {
				return ReadOutput{}, fmt.Errorf("read: scanning file: %w", err)
			}

			totalLines := len(lines)

			// Check for binary content (first 8KB)
			if totalLines > 0 {
				sample := strings.Join(lines[:min(totalLines, 100)], "\n")
				if len(sample) > 8192 {
					sample = sample[:8192]
				}
				if !utf8.ValidString(sample) || containsBinaryBytes([]byte(sample)) {
					return ReadOutput{}, fmt.Errorf("read: file appears to be binary")
				}
			}

			// Apply offset and limit
			offset := min(max(in.Offset, 1), totalLines)

			limit := in.Limit
			if limit <= 0 {
				limit = totalLines
			}

			startIdx := offset - 1
			endIdx := min(startIdx+limit, totalLines)

			// Build output with line numbers
			for i := startIdx; i < endIdx; i++ {
				lineNum := i + 1
				fmt.Fprintf(&buf, "%4d | %s\n", lineNum, lines[i])
			}

			return ReadOutput{
				Content:    buf.String(),
				TotalLines: totalLines,
				StartLine:  offset,
				EndLine:    startIdx + (endIdx - startIdx),
			}, nil
		},
	)
}

// containsBinaryBytes checks if the byte slice contains null bytes or other binary indicators.
func containsBinaryBytes(data []byte) bool {
	return slices.Contains(data, 0)
}
