package openai_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/matryer/is"
	"github.com/matthewmueller/llm"
	"github.com/matthewmueller/llm/internal/env"
	"github.com/matthewmueller/llm/providers/openai"
	"github.com/matthewmueller/logs"
)

const testModel = `gpt-5-mini-2025-08-07`

func loadEnv(t *testing.T) *env.Env {
	t.Helper()
	e, err := env.Load()
	if err != nil {
		t.Fatalf("openai: loading env: %v", err)
	}
	if e.OpenAIKey == "" {
		t.Fatal("OPENAI_API_KEY not set")
	}
	return e
}

func testContext(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	t.Cleanup(cancel)
	return ctx
}

func TestSimpleChat(t *testing.T) {
	e := loadEnv(t)
	is := is.New(t)
	ctx := testContext(t)
	log := logs.Default()

	provider := openai.New(log, e.OpenAIKey)
	client := llm.New(log, provider)
	var content strings.Builder
	for event, err := range client.Chat(ctx,
		provider.Name(),
		llm.WithModel(testModel),
		llm.WithMessage(llm.UserMessage("What is 2+2? Reply with just the number.")),
	) {
		is.NoErr(err)
		content.WriteString(event.Content)
	}
	is.True(strings.Contains(content.String(), "4"))
}

func TestModels(t *testing.T) {
	e := loadEnv(t)
	is := is.New(t)
	ctx := testContext(t)
	log := logs.Default()

	provider := openai.New(log, e.OpenAIKey)
	models, err := provider.Models(ctx)
	is.NoErr(err)
	is.True(len(models) > 0)

	for _, m := range models {
		is.Equal(m.Provider, "openai")
		is.True(m.ID != "")
		is.True(m.Name == "")
	}
}

var addTool = llm.Func("add", "Add two numbers together", func(ctx context.Context, in struct {
	A int `json:"a" description:"First number" is:"required"`
	B int `json:"b" description:"Second number" is:"required"`
}) (int, error) {
	return in.A + in.B, nil
})

var multiplyTool = llm.Func("multiply", "Multiply two numbers together", func(ctx context.Context, in struct {
	A int `json:"a" description:"First number" is:"required"`
	B int `json:"b" description:"Second number" is:"required"`
}) (int, error) {
	return in.A * in.B, nil
})

func TestToolSingleCall(t *testing.T) {
	e := loadEnv(t)
	is := is.New(t)
	ctx := testContext(t)
	log := logs.Default()

	provider := openai.New(log, e.OpenAIKey)
	lc := llm.New(log, provider)

	content := new(strings.Builder)
	for event, err := range lc.Chat(ctx,
		provider.Name(),
		llm.WithModel(testModel),
		llm.WithMessage(llm.UserMessage("Use the multiply tool to multiply 6 and 7, then tell me the result.")),
		llm.WithTool(multiplyTool),
	) {
		is.NoErr(err)
		content.WriteString(event.Content)
	}
	is.True(strings.Contains(content.String(), "42"))
}

func TestToolMultipleParallel(t *testing.T) {
	e := loadEnv(t)
	is := is.New(t)
	ctx := testContext(t)
	log := logs.Default()

	provider := openai.New(log, e.OpenAIKey)
	client := llm.New(log, provider)

	content := new(strings.Builder)
	for res, err := range client.Chat(ctx,
		provider.Name(),
		llm.WithModel(testModel),
		llm.WithMessage(
			llm.UserMessage("Write a short poem and then call the add tool to add 10 and 5, and the multiply tool to multiply 3 and 4. Give me both results."),
		),
		llm.WithTool(addTool),
		llm.WithTool(multiplyTool),
	) {
		is.NoErr(err)
		content.WriteString(res.Content)
	}

	is.True(strings.Contains(content.String(), "15"))
	is.True(strings.Contains(content.String(), "12"))
}

func TestToolMultipleSerial(t *testing.T) {
	e := loadEnv(t)
	is := is.New(t)
	ctx := testContext(t)
	log := logs.Default()

	provider := openai.New(log, e.OpenAIKey)
	client := llm.New(log, provider)

	content := new(strings.Builder)
	for res, err := range client.Chat(ctx,
		provider.Name(),
		llm.WithModel(testModel),
		llm.WithMessage(
			llm.UserMessage("First use the add tool to add 10 and 5. Then use the multiply tool to multiply the result by 2. Tell me the final answer."),
		),
		llm.WithTool(addTool),
		llm.WithTool(multiplyTool),
	) {
		is.NoErr(err)
		content.WriteString(res.Content)
	}

	is.True(strings.Contains(content.String(), "30"))
}

func TestToolMultiTurnGathering(t *testing.T) {
	e := loadEnv(t)
	is := is.New(t)
	ctx := testContext(t)
	log := logs.Default()

	gatherNameTool := llm.Func("gather_name", "Ask the user for their name", func(ctx context.Context, in struct {
		Question string `json:"question" description:"Question to ask about name" is:"required"`
	}) (string, error) {
		return "User's name is Bob", nil
	})

	gatherCityTool := llm.Func("gather_city", "Ask the user for their city", func(ctx context.Context, in struct {
		Question string `json:"question" description:"Question to ask about city" is:"required"`
	}) (string, error) {
		return "User's city is Portland", nil
	})

	called := false
	createGreetingTool := llm.Func("create_greeting", "Create a personalized greeting", func(ctx context.Context, in struct {
		Name string `json:"name" description:"Person's name" is:"required"`
		City string `json:"city" description:"Person's city" is:"required"`
	}) (string, error) {
		called = true
		is.Equal(in.Name, "Bob")
		is.Equal(in.City, "Portland")
		return "Hello " + in.Name + " from " + in.City + "!", nil
	})

	provider := openai.New(log, e.OpenAIKey)
	client := llm.New(log, provider)

	content := new(strings.Builder)
	for res, err := range client.Chat(ctx,
		provider.Name(),
		llm.WithModel(testModel),
		llm.WithMessage(
			llm.UserMessage("Use the gather_name tool to get the user's name, then use the gather_city tool to get their city, then use create_greeting to make a personalized greeting. Tell me the greeting."),
		),
		llm.WithTool(gatherNameTool),
		llm.WithTool(gatherCityTool),
		llm.WithTool(createGreetingTool),
	) {
		is.NoErr(err)
		content.WriteString(res.Content)
	}

	is.True(called)
	is.True(strings.Contains(content.String(), "Bob"))
	is.True(strings.Contains(content.String(), "Portland"))
}
