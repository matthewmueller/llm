package tools

import (
	"bufio"
	"context"
	"fmt"
	"io/fs"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/matthewmueller/llm"
	"github.com/matthewmueller/virt"
)

const maxGrepMatches = 100

// GrepInput defines the input parameters for the Grep tool.
type GrepInput struct {
	Pattern string `json:"pattern" is:"required" description:"Regular expression pattern to search for"`
	Path    string `json:"path" is:"required" description:"File or directory path to search"`
	Glob    string `json:"glob" description:"File glob pattern to filter files (e.g., '*.go')"`
	Context int    `json:"context" description:"Number of context lines before and after match"`
}

// GrepOutput defines the output of the Grep tool.
type GrepOutput struct {
	Matches []GrepMatch `json:"matches"`
	Total   int         `json:"total"`
}

// GrepMatch represents a single match from the grep search.
type GrepMatch struct {
	File    string `json:"file"`
	Line    int    `json:"line"`
	Content string `json:"content"`
}

// Grep creates a tool for searching files using regular expressions.
func Grep(fsys virt.FS) llm.Tool {
	return llm.Func("tool_grep",
		"Search for a pattern within files using regular expressions. Use this to find code, function definitions, usages, or any text pattern across the codebase.",
		func(ctx context.Context, in GrepInput) (GrepOutput, error) {
			re, err := regexp.Compile(in.Pattern)
			if err != nil {
				return GrepOutput{}, fmt.Errorf("grep: invalid regex pattern: %w", err)
			}

			var matches []GrepMatch

			// Check if path is a file or directory
			stat, err := fsys.Stat(in.Path)
			if err != nil {
				return GrepOutput{}, fmt.Errorf("grep: unable to stat path: %w", err)
			}

			if !stat.IsDir() {
				// Search single file
				fileMatches, err := grepFile(fsys, in.Path, re, in.Context)
				if err != nil {
					return GrepOutput{}, err
				}
				matches = append(matches, fileMatches...)
			} else {
				// Walk directory
				err := fs.WalkDir(fsys, in.Path, func(path string, d fs.DirEntry, err error) error {
					if err != nil {
						return nil // Skip files with errors
					}
					if d.IsDir() {
						// Skip hidden directories
						if strings.HasPrefix(d.Name(), ".") && path != in.Path {
							return fs.SkipDir
						}
						return nil
					}

					// Apply glob filter if specified
					if in.Glob != "" {
						matched, err := filepath.Match(in.Glob, d.Name())
						if err != nil || !matched {
							return nil
						}
					}

					// Skip hidden files
					if strings.HasPrefix(d.Name(), ".") {
						return nil
					}

					fileMatches, err := grepFile(fsys, path, re, in.Context)
					if err != nil {
						return nil // Skip files with errors
					}
					matches = append(matches, fileMatches...)

					// Limit total matches
					if len(matches) >= maxGrepMatches {
						return fs.SkipAll
					}
					return nil
				})
				if err != nil {
					return GrepOutput{}, fmt.Errorf("grep: walking directory: %w", err)
				}
			}

			// Limit results
			total := len(matches)
			if len(matches) > maxGrepMatches {
				matches = matches[:maxGrepMatches]
			}

			return GrepOutput{
				Matches: matches,
				Total:   total,
			}, nil
		},
	)
}

// grepFile searches a single file for the pattern.
func grepFile(fsys fs.FS, path string, re *regexp.Regexp, contextLines int) ([]GrepMatch, error) {
	file, err := fsys.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var matches []GrepMatch
	var lines []string

	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // 1MB buffer
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}

	for i, line := range lines {
		if re.MatchString(line) {
			content := line
			if contextLines > 0 {
				// Add context lines
				start := max(0, i-contextLines)
				end := min(len(lines), i+contextLines+1)
				var contextContent strings.Builder
				for j := start; j < end; j++ {
					if j == i {
						contextContent.WriteString(fmt.Sprintf("> %s\n", lines[j]))
					} else {
						contextContent.WriteString(fmt.Sprintf("  %s\n", lines[j]))
					}
				}
				content = contextContent.String()
			}

			matches = append(matches, GrepMatch{
				File:    path,
				Line:    i + 1,
				Content: content,
			})
		}
	}

	return matches, nil
}
