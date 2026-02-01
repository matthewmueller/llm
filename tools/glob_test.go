package tools_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/matryer/is"
	"github.com/matthewmueller/llm/tools"
	"github.com/matthewmueller/virt"
)

func TestGlob(t *testing.T) {
	is := is.New(t)
	fsys := virt.Tree{
		"src/main.go":       &virt.File{Data: []byte("package main")},
		"src/util.go":       &virt.File{Data: []byte("package main")},
		"test/main_test.go": &virt.File{Data: []byte("package main")},
		"README.md":         &virt.File{Data: []byte("# README")},
	}

	tool := tools.Glob(fsys)
	is.Equal(tool.Schema().Function.Name, "tool_glob")

	args, _ := json.Marshal(map[string]any{"pattern": "**/*.go", "path": "."})
	result, err := tool.Run(context.Background(), args)
	is.NoErr(err)

	var output struct {
		Files []string `json:"files"`
		Total int      `json:"total"`
	}
	is.NoErr(json.Unmarshal(result, &output))
	is.Equal(output.Total, 3)
}
