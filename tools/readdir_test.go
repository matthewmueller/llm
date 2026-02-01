package tools_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/matryer/is"
	"github.com/matthewmueller/llm/tools"
	"github.com/matthewmueller/virt"
)

func TestReadDir(t *testing.T) {
	is := is.New(t)
	fsys := virt.Tree{
		"src/main.go":    &virt.File{Data: []byte("package main")},
		"src/util.go":    &virt.File{Data: []byte("package main")},
		"docs/readme.md": &virt.File{Data: []byte("# Docs")},
	}

	tool := tools.ReadDir(fsys)
	is.Equal(tool.Schema().Function.Name, "tool_read_dir")

	args, _ := json.Marshal(map[string]any{"path": "."})
	result, err := tool.Run(context.Background(), args)
	is.NoErr(err)

	var output struct {
		Entries []struct {
			Name  string `json:"name"`
			IsDir bool   `json:"is_dir"`
		} `json:"entries"`
	}
	is.NoErr(json.Unmarshal(result, &output))
	is.Equal(len(output.Entries), 2)
}
