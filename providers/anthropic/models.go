package anthropic

import (
	"context"
	"fmt"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/matthewmueller/llm"
)

// https://platform.claude.com/docs/en/about-claude/models/overview
var meta = map[string]*llm.ModelMeta{
	// Latest models
	"claude-opus-4-6":           model("Claude Opus 4.6", date(2025, time.May, 31), 200_000, 128_000, true),
	"claude-sonnet-4-6":         model("Claude Sonnet 4.6", date(2025, time.August, 31), 200_000, 64_000, true),
	"claude-haiku-4-5":          model("Claude Haiku 4.5", date(2025, time.February, 28), 200_000, 64_000, true),
	"claude-haiku-4-5-20251001": model("Claude Haiku 4.5", date(2025, time.February, 28), 200_000, 64_000, true),

	// Legacy models
	"claude-sonnet-4-5":          model("Claude Sonnet 4.5", date(2025, time.January, 31), 200_000, 64_000, true),
	"claude-sonnet-4-5-20250929": model("Claude Sonnet 4.5", date(2025, time.January, 31), 200_000, 64_000, true),

	"claude-opus-4-5":          model("Claude Opus 4.5", date(2025, time.May, 31), 200_000, 64_000, true),
	"claude-opus-4-5-20251101": model("Claude Opus 4.5", date(2025, time.May, 31), 200_000, 64_000, true),

	"claude-opus-4-1":          model("Claude Opus 4.1", date(2025, time.January, 31), 200_000, 32_000, true),
	"claude-opus-4-1-20250805": model("Claude Opus 4.1", date(2025, time.January, 31), 200_000, 32_000, true),

	"claude-sonnet-4-0":        model("Claude Sonnet 4", date(2025, time.January, 31), 200_000, 64_000, true),
	"claude-sonnet-4-20250514": model("Claude Sonnet 4", date(2025, time.January, 31), 200_000, 64_000, true),

	"claude-3-7-sonnet-latest":   model("Claude Sonnet 3.7", date(2024, time.October, 31), 200_000, 64_000, true),
	"claude-3-7-sonnet-20250219": model("Claude Sonnet 3.7", date(2024, time.October, 31), 200_000, 64_000, true),

	"claude-opus-4-0":        model("Claude Opus 4", date(2025, time.January, 31), 200_000, 32_000, true),
	"claude-opus-4-20250514": model("Claude Opus 4", date(2025, time.January, 31), 200_000, 32_000, true),

	// Anthropic notes a single cutoff date for some Haiku models; we use that date as knowledge cutoff.
	"claude-3-haiku-20240307": model("Claude Haiku 3", date(2023, time.August, 31), 200_000, 4_000, false),
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

// Models lists available models
func (c *Client) Models(ctx context.Context) (models []*llm.Model, err error) {
	acmodels, err := c.ac.Models.List(ctx, anthropic.ModelListParams{})
	if err != nil {
		return nil, fmt.Errorf("anthropic: listing models: %w", err)
	}
	for _, model := range acmodels.Data {
		models = append(models, &llm.Model{
			Provider: "anthropic",
			ID:       model.ID,
			Meta:     meta[model.ID],
		})
	}
	return models, nil
}
