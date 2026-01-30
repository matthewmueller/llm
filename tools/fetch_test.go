package tools_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/matryer/is"
	"github.com/matthewmueller/llm/tools"
)

func TestFetch(t *testing.T) {
	is := is.New(t)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		w.Write([]byte("Hello from server"))
	}))
	defer server.Close()

	tool := tools.Fetch(server.Client())
	is.Equal(tool.Info().Function.Name, "tool_fetch")

	args, _ := json.Marshal(map[string]any{"url": server.URL})
	result, err := tool.Run(context.Background(), args)
	is.NoErr(err)

	var output struct {
		Content     string `json:"content"`
		StatusCode  int    `json:"status_code"`
		ContentType string `json:"content_type"`
	}
	is.NoErr(json.Unmarshal(result, &output))
	is.Equal(output.StatusCode, 200)
	is.Equal(output.Content, "Hello from server")
}
