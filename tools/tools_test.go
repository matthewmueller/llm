package tools_test

import (
	"net/http"
	"testing"

	"github.com/matryer/is"
	"github.com/matthewmueller/llm/internal/ask"
	"github.com/matthewmueller/llm/tools"
	"github.com/matthewmueller/virt"
)

func TestAll(t *testing.T) {
	is := is.New(t)
	asker := ask.Default()
	fetcher := http.DefaultClient
	fsys := virt.Tree{}
	exec := &tools.DefaultExecutor{}

	all := tools.All(asker, fetcher, fsys, exec)
	is.Equal(len(all), 9)

	names := make(map[string]bool)
	for _, tool := range all {
		names[tool.Info().Function.Name] = true
	}

	expected := []string{
		"tool_read", "tool_write", "tool_edit",
		"tool_grep", "tool_glob", "tool_read_dir",
		"tool_bash", "tool_ask", "tool_fetch",
	}
	for _, name := range expected {
		is.True(names[name])
	}
}
