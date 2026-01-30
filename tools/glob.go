package tools

import (
	"context"
	"fmt"
	"io/fs"
	"path/filepath"
	"sort"
	"strings"

	"github.com/matthewmueller/llm"
	"github.com/matthewmueller/virt"
)

const maxGlobFiles = 500

// GlobInput defines the input parameters for the Glob tool.
type GlobInput struct {
	Pattern string `json:"pattern" is:"required" description:"Glob pattern to match files (e.g., '**/*.go', 'src/*.ts')"`
	Path    string `json:"path" description:"Base directory to search from (default: current directory)"`
}

// GlobOutput defines the output of the Glob tool.
type GlobOutput struct {
	Files []string `json:"files"`
	Total int      `json:"total"`
}

// Glob creates a tool for finding files by pattern.
func Glob(fsys virt.FS) llm.Tool {
	return llm.Func("tool_glob",
		"Find files matching a glob pattern. Use this to discover files by name pattern, file extension, or directory structure. Supports ** for recursive matching.",
		func(ctx context.Context, in GlobInput) (GlobOutput, error) {
			basePath := in.Path
			if basePath == "" {
				basePath = "."
			}

			var files []string

			err := fs.WalkDir(fsys, basePath, func(path string, d fs.DirEntry, err error) error {
				if err != nil {
					return nil // Skip files with errors
				}

				// Skip hidden directories
				if d.IsDir() && strings.HasPrefix(d.Name(), ".") && path != basePath {
					return fs.SkipDir
				}

				if d.IsDir() {
					return nil
				}

				// Skip hidden files
				if strings.HasPrefix(d.Name(), ".") {
					return nil
				}

				// Match against pattern
				matched, err := matchGlob(in.Pattern, path)
				if err != nil {
					return nil
				}
				if matched {
					files = append(files, path)
				}

				// Limit results
				if len(files) >= maxGlobFiles {
					return fs.SkipAll
				}
				return nil
			})
			if err != nil {
				return GlobOutput{}, fmt.Errorf("glob: walking directory: %w", err)
			}

			// Sort files alphabetically
			sort.Strings(files)

			total := len(files)
			if len(files) > maxGlobFiles {
				files = files[:maxGlobFiles]
			}

			return GlobOutput{
				Files: files,
				Total: total,
			}, nil
		},
	)
}

// matchGlob matches a path against a glob pattern with ** support.
func matchGlob(pattern, path string) (bool, error) {
	// Handle ** patterns
	if strings.Contains(pattern, "**") {
		return matchDoublestar(pattern, path)
	}

	// Simple glob match against filename
	return filepath.Match(pattern, filepath.Base(path))
}

// matchDoublestar handles ** glob patterns.
func matchDoublestar(pattern, path string) (bool, error) {
	// Split pattern by **
	parts := strings.Split(pattern, "**")

	if len(parts) == 1 {
		// No **, use simple match
		return filepath.Match(pattern, path)
	}

	// Handle pattern like "**/*.go"
	if parts[0] == "" && len(parts) == 2 {
		// Pattern starts with **
		suffix := strings.TrimPrefix(parts[1], "/")
		if suffix == "" {
			return true, nil
		}
		// Match suffix against path or any suffix of path
		if matched, _ := filepath.Match(suffix, filepath.Base(path)); matched {
			return true, nil
		}
		// Try matching full pattern against the path
		if matched, _ := filepath.Match("*"+suffix, path); matched {
			return true, nil
		}
		return false, nil
	}

	// Handle pattern like "src/**/*.go"
	if len(parts) == 2 {
		prefix := strings.TrimSuffix(parts[0], "/")
		suffix := strings.TrimPrefix(parts[1], "/")

		// Check if path starts with prefix
		if prefix != "" && !strings.HasPrefix(path, prefix) {
			return false, nil
		}

		// If suffix is empty, match any path with prefix
		if suffix == "" {
			return true, nil
		}

		// Match suffix against the filename
		if matched, _ := filepath.Match(suffix, filepath.Base(path)); matched {
			return true, nil
		}
	}

	return false, nil
}
