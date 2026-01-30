package tools_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/matryer/is"
	"github.com/matthewmueller/llm/tools"
	"github.com/matthewmueller/virt"
)

func TestGrep(t *testing.T) {
	is := is.New(t)
	fsys := virt.Tree{
		"test.go": &virt.File{Data: []byte("package main\n\nfunc main() {\n\tfmt.Println(\"hello\")\n}")},
	}

	tool := tools.Grep(fsys)
	is.Equal(tool.Info().Function.Name, "tool_grep")

	args, _ := json.Marshal(map[string]any{"pattern": "func.*main", "path": "test.go"})
	result, err := tool.Run(context.Background(), args)
	is.NoErr(err)

	var output struct {
		Matches []struct {
			File    string `json:"file"`
			Line    int    `json:"line"`
			Content string `json:"content"`
		} `json:"matches"`
		Total int `json:"total"`
	}
	is.NoErr(json.Unmarshal(result, &output))
	is.Equal(output.Total, 1)
	is.Equal(len(output.Matches), 1)
	is.Equal(output.Matches[0].Line, 3)
}
