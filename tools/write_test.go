package tools_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/matryer/is"
	"github.com/matthewmueller/llm/tools"
	"github.com/matthewmueller/virt"
)

func TestWrite(t *testing.T) {
	is := is.New(t)
	fsys := virt.Tree{}

	tool := tools.Write(fsys)
	is.Equal(tool.Info().Function.Name, "tool_write")

	args, _ := json.Marshal(map[string]any{"path": "new.txt", "content": "hello world"})
	result, err := tool.Run(context.Background(), args)
	is.NoErr(err)

	var output struct {
		BytesWritten int `json:"bytes_written"`
	}
	is.NoErr(json.Unmarshal(result, &output))
	is.Equal(output.BytesWritten, 11)

	// Verify file was written
	file, err := fsys.Open("new.txt")
	is.NoErr(err)
	file.Close()
}
