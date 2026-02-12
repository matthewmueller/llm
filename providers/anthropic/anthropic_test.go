package anthropic_test

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/matryer/is"
	"github.com/matthewmueller/llm"
	"github.com/matthewmueller/llm/internal/env"
	"github.com/matthewmueller/llm/providers/anthropic"
)

const testModel = `claude-haiku-4-5-20251001`

func loadEnv(t *testing.T) *env.Env {
	t.Helper()
	e, err := env.Load()
	if err != nil {
		t.Fatalf("anthropic: loading env: %v", err)
	}
	if e.AnthropicKey == "" {
		t.Fatal("ANTHROPIC_API_KEY not set")
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

	provider := anthropic.New(e.AnthropicKey)
	lc := llm.New(provider)
	content := new(strings.Builder)
	for event, err := range lc.Chat(ctx,
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

	provider := anthropic.New(e.AnthropicKey)
	models, err := provider.Models(ctx)
	is.NoErr(err)
	is.True(len(models) > 0)

	for _, m := range models {
		is.Equal(m.Provider, "anthropic")
		is.True(m.ID != "")
		is.True(m.Name != "")
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

	provider := anthropic.New(e.AnthropicKey)
	lc := llm.New(provider)

	content := new(strings.Builder)
	for event, err := range lc.Chat(ctx,
		provider.Name(),
		llm.WithModel(testModel),
		llm.WithMessage(llm.UserMessage("Use the add tool to add 17 and 25, then tell me the result.")),
		llm.WithTool(addTool),
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

	provider := anthropic.New(e.AnthropicKey)
	client := llm.New(provider)

	content := new(strings.Builder)
	for res, err := range client.Chat(ctx,
		provider.Name(),
		llm.WithModel(testModel),
		llm.WithMessage(
			llm.UserMessage("Write a short poem and then call the add_tool to add 10 and 5, and the multiply_tool to multiply 3 and 4. Give me both results."),
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

	provider := anthropic.New(e.AnthropicKey)
	client := llm.New(provider)

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

	gatherNameTool := llm.Func("gather_name", "Ask the user for their name", func(ctx context.Context, in struct {
		Question string `json:"question" description:"Question to ask about name" is:"required"`
	}) (string, error) {
		return "User's name is Alice", nil
	})

	gatherCityTool := llm.Func("gather_city", "Ask the user for their city", func(ctx context.Context, in struct {
		Question string `json:"question" description:"Question to ask about city" is:"required"`
	}) (string, error) {
		return "User's city is Seattle", nil
	})

	called := false
	createGreetingTool := llm.Func("create_greeting", "Create a personalized greeting", func(ctx context.Context, in struct {
		Name string `json:"name" description:"Person's name" is:"required"`
		City string `json:"city" description:"Person's city" is:"required"`
	}) (string, error) {
		called = true
		is.Equal(in.Name, "Alice")
		is.Equal(in.City, "Seattle")
		return "Hello " + in.Name + " from " + in.City + "!", nil
	})

	provider := anthropic.New(e.AnthropicKey)
	client := llm.New(provider)

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
	is.True(strings.Contains(content.String(), "Alice"))
	is.True(strings.Contains(content.String(), "Seattle"))
}

func secretWord() llm.Tool {
	type in struct {
		Secret *string `json:"secret" description:"An optional secret word that is used to verify that tools are properly disabled."`
	}
	type out struct {
		Secret string `json:"secret"`
	}
	return llm.Func("secret_word", "Returns the secret word.", func(ctx context.Context, in in) (*out, error) {
		if in.Secret == nil || *in.Secret != "noodles" {
			return nil, fmt.Errorf("invalid: the secret is noodles")
		}
		return &out{Secret: *in.Secret}, nil
	})
}

func TestToolFailOnce(t *testing.T) {
	e := loadEnv(t)
	is := is.New(t)
	ctx := testContext(t)

	provider := anthropic.New(e.AnthropicKey)
	lc := llm.New(provider)

	content := new(strings.Builder)
	for event, err := range lc.Chat(ctx,
		provider.Name(),
		llm.WithModel(testModel),
		llm.WithMessage(llm.UserMessage(`Use the secret_word tool to return the secret word`)),
		llm.WithTool(secretWord()),
	) {
		is.NoErr(err)
		content.WriteString(event.Content)
	}
	is.True(strings.Contains(content.String(), "noodles"))
}
