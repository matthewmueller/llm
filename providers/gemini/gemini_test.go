package gemini_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/matryer/is"
	"github.com/matthewmueller/llm"
	"github.com/matthewmueller/llm/internal/env"
	"github.com/matthewmueller/llm/providers/gemini"
	"github.com/matthewmueller/logs"
)

func loadEnv(t *testing.T) *env.Env {
	t.Helper()
	e, err := env.Load()
	if err != nil {
		t.Fatalf("gemini: loading env: %v", err)
	}
	if e.GeminiKey == "" {
		t.Fatal("GEMINI_API_KEY not set")
	}
	return e
}

func testContext(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	t.Cleanup(cancel)
	return ctx
}

func collectResponse(t *testing.T, agent *llm.Agent, ctx context.Context, prompt string) string {
	t.Helper()
	is := is.New(t)
	var content string
	for event, err := range agent.Chat(ctx, prompt) {
		is.NoErr(err)
		if event.Done {
			content = event.Content
		}
	}
	return content
}

func TestSimpleChat(t *testing.T) {
	e := loadEnv(t)
	is := is.New(t)
	ctx := testContext(t)
	log := logs.Default()

	provider := gemini.New(log, e.GeminiKey)
	client := llm.New(log, provider)
	agent := client.Agent(llm.WithModel("gemini-2.0-flash"))

	content := collectResponse(t, agent, ctx, "What is 2+2? Reply with just the number.")
	is.True(strings.Contains(content, "4"))
}

func TestConversation(t *testing.T) {
	e := loadEnv(t)
	is := is.New(t)
	ctx := testContext(t)
	log := logs.Default()

	provider := gemini.New(log, e.GeminiKey)
	client := llm.New(log, provider)
	agent := client.Agent(llm.WithModel("gemini-2.0-flash"))

	// First turn
	content1 := collectResponse(t, agent, ctx, "My favorite fruit is mango. Remember that.")
	is.True(len(content1) > 0)

	// Second turn - should remember context
	content2 := collectResponse(t, agent, ctx, "What is my favorite fruit?")
	is.True(strings.Contains(strings.ToLower(content2), "mango"))
}

func TestModels(t *testing.T) {
	e := loadEnv(t)
	is := is.New(t)
	ctx := testContext(t)
	log := logs.Default()

	provider := gemini.New(log, e.GeminiKey)
	models, err := provider.Models(ctx)
	is.NoErr(err)
	is.True(len(models) > 0)

	for _, m := range models {
		is.Equal(m.Provider, "gemini")
		is.True(m.Name != "")
	}
}

func TestToolSingleCall(t *testing.T) {
	e := loadEnv(t)
	is := is.New(t)
	ctx := testContext(t)
	log := logs.Default()

	subtractTool := llm.Func("subtract", "Subtract two numbers", func(ctx context.Context, in struct {
		A int `json:"a" description:"First number" is:"required"`
		B int `json:"b" description:"Second number to subtract from first" is:"required"`
	}) (int, error) {
		return in.A - in.B, nil
	})

	provider := gemini.New(log, e.GeminiKey)
	client := llm.New(log, provider)
	agent := client.Agent(
		llm.WithModel("gemini-2.0-flash"),
		llm.WithTool(subtractTool),
	)

	var sawToolCall bool
	var content string
	for event, err := range agent.Chat(ctx, "Use the subtract tool to subtract 8 from 50, then tell me the result.") {
		is.NoErr(err)
		if event.Tool != nil {
			sawToolCall = true
			is.Equal(event.Tool.Name, "subtract")
		}
		if event.Done {
			content = event.Content
		}
	}

	is.True(sawToolCall)
	is.True(strings.Contains(content, "42"))
}

func TestToolMultipleCalls(t *testing.T) {
	e := loadEnv(t)
	is := is.New(t)
	ctx := testContext(t)
	log := logs.Default()

	addTool := llm.Func("add", "Add two numbers together", func(ctx context.Context, in struct {
		A int `json:"a" description:"First number" is:"required"`
		B int `json:"b" description:"Second number" is:"required"`
	}) (int, error) {
		return in.A + in.B, nil
	})

	multiplyTool := llm.Func("multiply", "Multiply two numbers together", func(ctx context.Context, in struct {
		A int `json:"a" description:"First number" is:"required"`
		B int `json:"b" description:"Second number" is:"required"`
	}) (int, error) {
		return in.A * in.B, nil
	})

	provider := gemini.New(log, e.GeminiKey)
	client := llm.New(log, provider)
	agent := client.Agent(
		llm.WithModel("gemini-2.0-flash"),
		llm.WithTool(addTool),
		llm.WithTool(multiplyTool),
	)

	toolCalls := make(map[string]bool)
	var content string
	for event, err := range agent.Chat(ctx, "First use the add tool to add 10 and 5. Then use the multiply tool to multiply the result by 2. Tell me the final answer.") {
		is.NoErr(err)
		if event.Tool != nil {
			toolCalls[event.Tool.Name] = true
		}
		if event.Done {
			content = event.Content
		}
	}

	is.True(toolCalls["add"])
	is.True(toolCalls["multiply"])
	is.True(strings.Contains(content, "30"))
}

func TestToolMultiTurnGathering(t *testing.T) {
	e := loadEnv(t)
	is := is.New(t)
	ctx := testContext(t)
	log := logs.Default()

	// A tool that requires gathering information across turns
	var gatheredName, gatheredCity string
	gatherNameTool := llm.Func("gather_name", "Ask the user for their name", func(ctx context.Context, in struct {
		Question string `json:"question" description:"Question to ask about name" is:"required"`
	}) (string, error) {
		gatheredName = "Charlie"
		return "User's name is Charlie", nil
	})

	gatherCityTool := llm.Func("gather_city", "Ask the user for their city", func(ctx context.Context, in struct {
		Question string `json:"question" description:"Question to ask about city" is:"required"`
	}) (string, error) {
		gatheredCity = "Denver"
		return "User's city is Denver", nil
	})

	createGreetingTool := llm.Func("create_greeting", "Create a personalized greeting", func(ctx context.Context, in struct {
		Name string `json:"name" description:"Person's name" is:"required"`
		City string `json:"city" description:"Person's city" is:"required"`
	}) (string, error) {
		return "Hello " + in.Name + " from " + in.City + "!", nil
	})

	provider := gemini.New(log, e.GeminiKey)
	client := llm.New(log, provider)
	agent := client.Agent(
		llm.WithModel("gemini-2.0-flash"),
		llm.WithTool(gatherNameTool),
		llm.WithTool(gatherCityTool),
		llm.WithTool(createGreetingTool),
	)

	toolCalls := make(map[string]int)
	var content string
	for event, err := range agent.Chat(ctx, "Use the gather_name tool to get the user's name, then use the gather_city tool to get their city, then use create_greeting to make a personalized greeting. Tell me the greeting.") {
		is.NoErr(err)
		if event.Tool != nil {
			toolCalls[event.Tool.Name]++
		}
		if event.Done {
			content = event.Content
		}
	}

	is.True(toolCalls["gather_name"] >= 1)
	is.True(toolCalls["gather_city"] >= 1)
	is.True(toolCalls["create_greeting"] >= 1)
	is.True(gatheredName == "Charlie")
	is.True(gatheredCity == "Denver")
	is.True(strings.Contains(content, "Charlie"))
	is.True(strings.Contains(content, "Denver"))
}
