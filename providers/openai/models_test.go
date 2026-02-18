package openai_test

import (
	"testing"

	"github.com/matryer/is"
	"github.com/matthewmueller/llm/providers/openai"
)

func TestModels(t *testing.T) {
	e := loadEnv(t)
	is := is.New(t)
	ctx := testContext(t)

	provider := openai.New(e.OpenAIKey)
	models, err := provider.Models(ctx)
	is.NoErr(err)
	is.True(len(models) > 0)

	for _, m := range models {
		is.Equal(m.Provider, "openai")
		is.True(m.ID != "")
	}
}

func TestModel(t *testing.T) {
	e := loadEnv(t)
	is := is.New(t)
	ctx := testContext(t)

	provider := openai.New(e.OpenAIKey)
	m, err := provider.Model(ctx, testModel)
	is.NoErr(err)
	is.Equal(m.Provider, "openai")
	is.Equal(m.ID, testModel)
	is.True(m.Meta != nil)
	is.Equal(m.Meta.DisplayName, "GPT-5 mini")
}
