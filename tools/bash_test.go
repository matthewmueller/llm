package tools_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/matryer/is"
	"github.com/matthewmueller/llm/tools"
)

func TestBash(t *testing.T) {
	is := is.New(t)
	exec := &tools.DefaultExecutor{}

	tool := tools.Bash(exec)
	is.Equal(tool.Info().Function.Name, "tool_bash")

	args, _ := json.Marshal(map[string]any{"command": "echo hello"})
	result, err := tool.Run(context.Background(), args)
	is.NoErr(err)

	var output struct {
		Stdout   string `json:"stdout"`
		Stderr   string `json:"stderr"`
		ExitCode int    `json:"exit_code"`
	}
	is.NoErr(json.Unmarshal(result, &output))
	is.Equal(strings.TrimSpace(output.Stdout), "hello")
	is.Equal(output.ExitCode, 0)
}

func TestBashNonZeroExit(t *testing.T) {
	is := is.New(t)
	exec := &tools.DefaultExecutor{}

	tool := tools.Bash(exec)
	args, _ := json.Marshal(map[string]any{"command": "exit 42"})
	result, err := tool.Run(context.Background(), args)
	is.NoErr(err)

	var output struct {
		ExitCode int `json:"exit_code"`
	}
	is.NoErr(json.Unmarshal(result, &output))
	is.Equal(output.ExitCode, 42)
}
