package tools_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/matryer/is"
	"github.com/matthewmueller/llm/tools"
	"github.com/matthewmueller/virt"
)

func TestEdit(t *testing.T) {
	is := is.New(t)
	fsys := virt.Tree{
		"test.txt": &virt.File{Data: []byte("hello world")},
	}

	tool := tools.Edit(fsys)
	is.Equal(tool.Schema().Function.Name, "tool_edit")

	args, _ := json.Marshal(map[string]any{
		"path":       "test.txt",
		"old_string": "world",
		"new_string": "universe",
	})
	result, err := tool.Run(context.Background(), args)
	is.NoErr(err)

	var output struct {
		Success      bool `json:"success"`
		Replacements int  `json:"replacements"`
	}
	is.NoErr(json.Unmarshal(result, &output))
	is.True(output.Success)
	is.Equal(output.Replacements, 1)
}

func TestEditNotFound(t *testing.T) {
	is := is.New(t)
	fsys := virt.Tree{
		"test.txt": &virt.File{Data: []byte("hello world")},
	}

	tool := tools.Edit(fsys)
	args, _ := json.Marshal(map[string]any{
		"path":       "test.txt",
		"old_string": "nonexistent",
		"new_string": "replacement",
	})
	_, err := tool.Run(context.Background(), args)
	is.True(err != nil)
}
