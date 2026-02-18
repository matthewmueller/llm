package ollama

import (
	"context"
	"fmt"
	"time"

	"github.com/matthewmueller/llm"
)

// https://llm-stats.com/
// TODO: figure out a good way to keep this up to date
var meta = map[string]*llm.ModelMeta{
	"glm-4.7-flash:latest": model("GLM-4.7-Flash", time.Time{}, 128_000, 0, true),
}

func model(displayName string, knowledgeCutoff time.Time, contextWindow int, maxOutputTokens int, hasReasoning bool) *llm.ModelMeta {
	return &llm.ModelMeta{
		DisplayName:     displayName,
		KnowledgeCutoff: knowledgeCutoff,
		ContextWindow:   contextWindow,
		MaxOutputTokens: maxOutputTokens,
		HasReasoning:    hasReasoning,
	}
}

// Models lists available models
func (c *Client) Models(ctx context.Context) ([]*llm.Model, error) {
	res, err := c.oc.List(ctx)
	if err != nil {
		return nil, fmt.Errorf("ollama: listing models: %w", err)
	}

	models := make([]*llm.Model, len(res.Models))
	for i, m := range res.Models {
		models[i] = &llm.Model{
			Provider: "ollama",
			ID:       m.Model,
			Meta:     meta[m.Model],
		}
	}
	return models, nil
}
