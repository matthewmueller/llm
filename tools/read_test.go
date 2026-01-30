package tools_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/matryer/is"
	"github.com/matthewmueller/llm/tools"
	"github.com/matthewmueller/virt"
)

func TestRead(t *testing.T) {
	is := is.New(t)
	fsys := virt.Tree{
		"test.txt": &virt.File{Data: []byte("line 1\nline 2\nline 3\nline 4\nline 5")},
	}

	tool := tools.Read(fsys)
	is.Equal(tool.Info().Function.Name, "tool_read")

	args, _ := json.Marshal(map[string]any{"path": "test.txt"})
	result, err := tool.Run(context.Background(), args)
	is.NoErr(err)

	var output struct {
		Content    string `json:"content"`
		TotalLines int    `json:"total_lines"`
	}
	is.NoErr(json.Unmarshal(result, &output))
	is.Equal(output.TotalLines, 5)
	is.True(strings.Contains(output.Content, "line 1"))
}

func TestReadWithOffsetAndLimit(t *testing.T) {
	is := is.New(t)
	fsys := virt.Tree{
		"test.txt": &virt.File{Data: []byte("line 1\nline 2\nline 3\nline 4\nline 5")},
	}

	tool := tools.Read(fsys)
	args, _ := json.Marshal(map[string]any{"path": "test.txt", "offset": 2, "limit": 2})
	result, err := tool.Run(context.Background(), args)
	is.NoErr(err)

	var output struct {
		Content   string `json:"content"`
		StartLine int    `json:"start_line"`
	}
	is.NoErr(json.Unmarshal(result, &output))
	is.Equal(output.StartLine, 2)
	is.True(strings.Contains(output.Content, "line 2"))
	is.True(!strings.Contains(output.Content, "line 1"))
}
