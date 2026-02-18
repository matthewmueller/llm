package gemini

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/matthewmueller/llm"
)

// https://ai.google.dev/gemini-api/docs/models
var meta = map[string]*llm.ModelMeta{
	// Gemini 3 (preview)
	"gemini-3-pro-preview":       model("Gemini 3 Pro Preview", date(2025, time.January, 31), 1_048_576, 65_536, true),
	"gemini-3-pro-image-preview": model("Gemini 3 Pro Image Preview", date(2025, time.January, 31), 65_536, 32_768, true),
	"gemini-3-flash-preview":     model("Gemini 3 Flash Preview", date(2025, time.January, 31), 1_048_576, 65_536, true),

	// Gemini 2.5 Flash
	"gemini-2.5-flash":                              model("Gemini 2.5 Flash", date(2025, time.January, 31), 1_048_576, 65_536, true),
	"gemini-2.5-flash-preview-09-2025":              model("Gemini 2.5 Flash Preview", date(2025, time.January, 31), 1_048_576, 65_536, true),
	"gemini-2.5-flash-image":                        model("Gemini 2.5 Flash Image", date(2025, time.June, 30), 65_536, 32_768, false),
	"gemini-2.5-flash-image-preview":                model("Gemini 2.5 Flash Image Preview", date(2025, time.June, 30), 65_536, 32_768, false),
	"gemini-2.5-flash-native-audio-preview-12-2025": model("Gemini 2.5 Flash Live", date(2025, time.January, 31), 131_072, 8_192, true),
	"gemini-2.5-flash-native-audio-preview-09-2025": model("Gemini 2.5 Flash Live Preview", date(2025, time.January, 31), 131_072, 8_192, true),
	// Google does not list a knowledge cutoff in this table row.
	"gemini-2.5-flash-preview-tts": model("Gemini 2.5 Flash TTS", time.Time{}, 8_192, 16_384, false),

	// Gemini 2.5 Flash-Lite
	"gemini-2.5-flash-lite":                 model("Gemini 2.5 Flash-Lite", date(2025, time.January, 31), 1_048_576, 65_536, true),
	"gemini-2.5-flash-lite-preview-09-2025": model("Gemini 2.5 Flash-Lite Preview", date(2025, time.January, 31), 1_048_576, 65_536, true),

	// Gemini 2.5 Pro
	"gemini-2.5-pro": model("Gemini 2.5 Pro", date(2025, time.January, 31), 1_048_576, 65_536, true),
	// Google does not list a knowledge cutoff in this table row.
	"gemini-2.5-pro-preview-tts": model("Gemini 2.5 Pro TTS", time.Time{}, 8_192, 16_384, false),

	// Gemini 2.0
	"gemini-2.0-flash":          model("Gemini 2.0 Flash", date(2024, time.August, 31), 1_048_576, 8_192, true), // Thinking is marked experimental.
	"gemini-2.0-flash-001":      model("Gemini 2.0 Flash", date(2024, time.August, 31), 1_048_576, 8_192, true),
	"gemini-2.0-flash-exp":      model("Gemini 2.0 Flash Experimental", date(2024, time.August, 31), 1_048_576, 8_192, true),
	"gemini-2.0-flash-lite":     model("Gemini 2.0 Flash-Lite", date(2024, time.August, 31), 1_048_576, 8_192, false),
	"gemini-2.0-flash-lite-001": model("Gemini 2.0 Flash-Lite", date(2024, time.August, 31), 1_048_576, 8_192, false),
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

func lookupMeta(id string) *llm.ModelMeta {
	if m := meta[id]; m != nil {
		return m
	}
	if after, ok := strings.CutPrefix(id, "models/"); ok {
		return meta[after]
	}
	return nil
}

// Models lists available models
func (c *Client) Models(ctx context.Context) (models []*llm.Model, err error) {
	for model, err := range c.gc.Models.All(ctx) {
		if err != nil {
			return nil, fmt.Errorf("gemini: listing models: %w", err)
		}
		models = append(models, &llm.Model{
			Provider: "gemini",
			ID:       model.Name,
			Meta:     lookupMeta(model.Name),
		})
	}
	return models, nil
}
