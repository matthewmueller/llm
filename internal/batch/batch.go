package batch

import (
	"context"
	"sync"

	"golang.org/x/sync/errgroup"
)

func New[B any](ctx context.Context) (*Batch[B], context.Context) {
	eg, ctx := errgroup.WithContext(ctx)
	return &Batch[B]{eg: eg}, ctx
}

type Batch[B any] struct {
	eg   *errgroup.Group
	mu   sync.RWMutex
	next int
	out  []B
}

func (b *Batch[B]) Go(fn func() (B, error)) {
	b.mu.Lock()
	idx := b.next
	b.next++
	b.out = append(b.out, *new(B)) // reserve slot
	b.mu.Unlock()

	b.eg.Go(func() error {
		result, err := fn()
		if err != nil {
			return err
		}
		b.mu.Lock()
		b.out[idx] = result
		b.mu.Unlock()
		return nil
	})
}

func (b *Batch[B]) Size() int {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return len(b.out)
}

func (b *Batch[B]) Wait() ([]B, error) {
	if err := b.eg.Wait(); err != nil {
		return nil, err
	}
	return b.out, nil
}
