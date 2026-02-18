package openai

import (
	"context"
	"fmt"
	"time"

	"github.com/matthewmueller/llm"
)

// https://developers.openai.com/api/docs/models
var meta = map[string]*llm.ModelMeta{
	// GPT-5.2
	"gpt-5.2":            model("GPT-5.2", date(2025, time.August, 31), 400_000, 128_000, true),
	"gpt-5.2-2025-12-11": model("GPT-5.2", date(2025, time.August, 31), 400_000, 128_000, true),

	// GPT-5 mini
	"gpt-5-mini":            model("GPT-5 mini", date(2024, time.May, 31), 400_000, 128_000, true),
	"gpt-5-mini-2025-08-07": model("GPT-5 mini", date(2024, time.May, 31), 400_000, 128_000, true),

	// GPT-5 nano
	"gpt-5-nano":            model("GPT-5 nano", date(2024, time.May, 31), 400_000, 128_000, true),
	"gpt-5-nano-2025-08-07": model("GPT-5 nano", date(2024, time.May, 31), 400_000, 128_000, true),

	// GPT-5.2 pro
	"gpt-5.2-pro":            model("GPT-5.2 pro", date(2025, time.August, 31), 400_000, 128_000, true),
	"gpt-5.2-pro-2025-12-11": model("GPT-5.2 pro", date(2025, time.August, 31), 400_000, 128_000, true),

	// GPT-5
	"gpt-5":            model("GPT-5", date(2024, time.September, 30), 400_000, 128_000, true),
	"gpt-5-2025-08-07": model("GPT-5", date(2024, time.September, 30), 400_000, 128_000, true),

	// GPT-4.1
	"gpt-4.1":            model("GPT-4.1", date(2024, time.June, 1), 1_047_576, 32_768, false),
	"gpt-4.1-2025-04-14": model("GPT-4.1", date(2024, time.June, 1), 1_047_576, 32_768, false),
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

func date(year int, month time.Month, day int) time.Time {
	return time.Date(year, month, day, 0, 0, 0, 0, time.UTC)
}

// Model retrieves a specific model
func (c *Client) Model(ctx context.Context, id string) (*llm.Model, error) {
	m, err := c.oc.Models.Get(ctx, id)
	if err != nil {
		return nil, fmt.Errorf("openai: getting model %q: %w", id, err)
	}
	return &llm.Model{
		Provider: "openai",
		ID:       m.ID,
		Meta:     meta[m.ID],
	}, nil
}

// Models lists available models
func (c *Client) Models(ctx context.Context) ([]*llm.Model, error) {
	page, err := c.oc.Models.List(ctx)
	if err != nil {
		return nil, fmt.Errorf("openai: listing models: %w", err)
	}
	var models []*llm.Model
	for _, m := range page.Data {
		models = append(models, &llm.Model{
			Provider: "openai",
			ID:       m.ID,
			Meta:     meta[m.ID],
		})
	}
	return models, nil
}
