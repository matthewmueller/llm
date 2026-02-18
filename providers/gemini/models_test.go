package gemini_test

import (
	"testing"

	"github.com/matryer/is"
	"github.com/matthewmueller/llm/providers/gemini"
)

func TestModels(t *testing.T) {
	e := loadEnv(t)
	is := is.New(t)
	ctx := testContext(t)

	provider := gemini.New(e.GeminiKey)
	models, err := provider.Models(ctx)
	is.NoErr(err)
	is.True(len(models) > 0)

	for _, m := range models {
		is.Equal(m.Provider, "gemini")
		is.True(m.ID != "")
	}
}

func TestModel(t *testing.T) {
	e := loadEnv(t)
	is := is.New(t)
	ctx := testContext(t)

	provider := gemini.New(e.GeminiKey)
	m, err := provider.Model(ctx, testModel)
	is.NoErr(err)
	is.Equal(m.Provider, "gemini")
	is.True(m.ID == testModel || m.ID == "gemini-2.5-flash")
	is.True(m.Meta != nil)
	is.Equal(m.Meta.DisplayName, "Gemini 2.5 Flash")
}
