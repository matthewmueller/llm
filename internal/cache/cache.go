package cache

import (
	"context"
	"sync"

	"github.com/matthewmueller/llm"
)

func Models(fn func(ctx context.Context) ([]*llm.Model, error)) func(ctx context.Context) ([]*llm.Model, error) {
	var cached []*llm.Model
	var mu sync.RWMutex
	return func(ctx context.Context) ([]*llm.Model, error) {
		mu.RLock()
		if cached != nil {
			models := append([]*llm.Model{}, cached...)
			mu.RUnlock()
			return models, nil
		}
		mu.RUnlock()

		models, err := fn(ctx)
		if err != nil {
			return nil, err
		}

		mu.Lock()
		cached = append([]*llm.Model{}, models...)
		mu.Unlock()

		return cached, nil
	}
}
