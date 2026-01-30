package tools

import (
	"context"
	"fmt"
	"io/fs"
	"sort"

	"github.com/matthewmueller/llm"
	"github.com/matthewmueller/virt"
)

// ReadDirInput defines the input parameters for the ReadDir tool.
type ReadDirInput struct {
	Path string `json:"path" is:"required" description:"Directory path to list"`
}

// ReadDirEntry represents a single entry in a directory listing.
type ReadDirEntry struct {
	Name  string `json:"name"`
	IsDir bool   `json:"is_dir"`
	Size  int64  `json:"size"`
}

// ReadDirOutput defines the output of the ReadDir tool.
type ReadDirOutput struct {
	Entries []ReadDirEntry `json:"entries"`
}

// ReadDir creates a tool for listing directory contents.
func ReadDir(fsys virt.FS) llm.Tool {
	return llm.Func("tool_read_dir",
		"List the contents of a directory. Use this to see what files and subdirectories exist in a path.",
		func(ctx context.Context, in ReadDirInput) (ReadDirOutput, error) {
			// Check if path exists and is a directory
			stat, err := fsys.Stat(in.Path)
			if err != nil {
				return ReadDirOutput{}, fmt.Errorf("readdir: unable to stat path: %w", err)
			}
			if !stat.IsDir() {
				return ReadDirOutput{}, fmt.Errorf("readdir: path is not a directory")
			}

			// Read directory entries
			entries, err := fs.ReadDir(fsys, in.Path)
			if err != nil {
				return ReadDirOutput{}, fmt.Errorf("readdir: unable to read directory: %w", err)
			}

			var result []ReadDirEntry
			for _, entry := range entries {
				info, err := entry.Info()
				var size int64
				if err == nil {
					size = info.Size()
				}

				result = append(result, ReadDirEntry{
					Name:  entry.Name(),
					IsDir: entry.IsDir(),
					Size:  size,
				})
			}

			// Sort entries: directories first, then alphabetically
			sort.Slice(result, func(i, j int) bool {
				if result[i].IsDir != result[j].IsDir {
					return result[i].IsDir
				}
				return result[i].Name < result[j].Name
			})

			return ReadDirOutput{
				Entries: result,
			}, nil
		},
	)
}
