package ask

import (
	"context"
)

// mockAsker implements asker.Asker for testing
func Mock(response string) Asker {
	return &mockAsker{response: response}
}

type mockAsker struct {
	response string
}

var _ Asker = (*mockAsker)(nil)

func (m *mockAsker) Ask(ctx context.Context, question string, choices []string) (string, error) {
	return m.response, nil
}
