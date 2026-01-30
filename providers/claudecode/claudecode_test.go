package claudecode_test

import (
	"context"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/matryer/is"
	"github.com/matthewmueller/llm"
	"github.com/matthewmueller/llm/providers/claudecode"
	"github.com/matthewmueller/logs"
)

func checkClaudeCLI(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("claude"); err != nil {
		t.Skip("claude CLI not installed")
	}
}

func testContext(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	t.Cleanup(cancel)
	return ctx
}

func TestModels(t *testing.T) {
	checkClaudeCLI(t)
	is := is.New(t)
	ctx := testContext(t)
	log := logs.Default()

	provider := claudecode.New(log)
	models, err := provider.Models(ctx)
	is.NoErr(err)
	is.Equal(len(models), 3)

	for _, m := range models {
		is.Equal(m.Provider, "claudecode")
		is.True(m.Name != "")
	}
}

func TestSimpleChat(t *testing.T) {
	checkClaudeCLI(t)
	is := is.New(t)
	ctx := testContext(t)
	log := logs.Default()

	provider := claudecode.New(log, "--dangerously-skip-permissions", "--max-turns=1")
	client := llm.New(log, provider)
	agent := client.Agent(llm.WithModel("haiku"))

	var content string
	var gotDone bool
	for event, err := range agent.Chat(ctx, "What is 2+2? Reply with just the number.") {
		is.NoErr(err)
		if event.Content != "" {
			content += event.Content
		}
		if event.Done {
			gotDone = true
		}
	}

	is.True(gotDone)
	is.True(strings.Contains(content, "4"))
}
