package anthropic_test

import (
	"testing"

	"github.com/matryer/is"
	"github.com/matthewmueller/llm/providers/anthropic"
)

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
	}
}

func TestModel(t *testing.T) {
	e := loadEnv(t)
	is := is.New(t)
	ctx := testContext(t)

	provider := anthropic.New(e.AnthropicKey)
	m, err := provider.Model(ctx, testModel)
	is.NoErr(err)
	is.Equal(m.Provider, "anthropic")
	is.Equal(m.ID, testModel)
	is.True(m.Meta != nil)
	is.Equal(m.Meta.DisplayName, "Claude Haiku 4.5")
}
