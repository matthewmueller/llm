package tools_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/matryer/is"
	"github.com/matthewmueller/llm/internal/ask"
	"github.com/matthewmueller/llm/tools"
)

func TestAsk(t *testing.T) {
	is := is.New(t)

	asker := ask.Mock("my response")
	tool := tools.Ask(asker)
	is.Equal(tool.Schema().Function.Name, "tool_ask")

	args, _ := json.Marshal(map[string]any{"question": "What is your name?"})
	result, err := tool.Run(context.Background(), args)
	is.NoErr(err)

	var askOutput struct {
		Response string `json:"response"`
	}
	is.NoErr(json.Unmarshal(result, &askOutput))
	is.Equal(askOutput.Response, "my response")
}

func TestAskWithChoices(t *testing.T) {
	is := is.New(t)

	asker := ask.Mock("Option B")
	tool := tools.Ask(asker)

	args, _ := json.Marshal(map[string]any{
		"question": "Pick one:",
		"choices":  []string{"Option A", "Option B", "Option C"},
	})
	result, err := tool.Run(context.Background(), args)
	is.NoErr(err)

	var askOutput struct {
		Response string `json:"response"`
	}
	is.NoErr(json.Unmarshal(result, &askOutput))
	is.Equal(askOutput.Response, "Option B")
}
