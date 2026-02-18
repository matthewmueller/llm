package ollama_test

import (
	"testing"

	"github.com/matryer/is"
	"github.com/matthewmueller/llm"
	"github.com/matthewmueller/llm/providers/ollama"
)

func TestModels(t *testing.T) {
	host := loadHost(t)
	is := is.New(t)
	ctx := testContext(t)

	provider := ollama.New(host)
	lc := llm.New(provider)
	models, err := lc.Models(ctx)
	is.NoErr(err)
	is.True(len(models) > 0)

	for _, m := range models {
		is.Equal(m.Provider, "ollama")
		is.True(m.ID != "")
	}
}

func TestModel(t *testing.T) {
	host := loadHost(t)
	is := is.New(t)
	ctx := testContext(t)

	provider := ollama.New(host)
	m, err := provider.Model(ctx, testModel)
	is.NoErr(err)
	is.Equal(m.Provider, "ollama")
	is.Equal(m.ID, testModel)
	is.True(m.Meta != nil)
	is.Equal(m.Meta.DisplayName, "GLM-4.7-Flash")
}
