package tools

import (
	"context"
	"fmt"
	"io"
	"net/http"

	"github.com/matthewmueller/llm"
)

const maxFetchSize = 1024 * 1024 // 1MB

// FetchInput defines the input parameters for the Fetch tool.
type FetchInput struct {
	URL string `json:"url" is:"required" description:"URL to fetch content from"`
}

// FetchOutput defines the output of the Fetch tool.
type FetchOutput struct {
	Content     string `json:"content"`
	StatusCode  int    `json:"status_code"`
	ContentType string `json:"content_type"`
}

// Fetch creates a tool for fetching content from URLs.
func Fetch(client *http.Client) llm.Tool {
	return llm.Func("tool_fetch",
		"Fetch content from a URL. Use this to retrieve documentation, API responses, or web page content.",
		func(ctx context.Context, in FetchInput) (FetchOutput, error) {
			req, err := http.NewRequestWithContext(ctx, http.MethodGet, in.URL, nil)
			if err != nil {
				return FetchOutput{}, fmt.Errorf("fetch: invalid URL: %w", err)
			}

			// Set a reasonable user agent
			req.Header.Set("User-Agent", "llm-tools/1.0")

			resp, err := client.Do(req)
			if err != nil {
				return FetchOutput{}, fmt.Errorf("fetch: request failed: %w", err)
			}
			defer resp.Body.Close()

			// Read response body with size limit
			limitedReader := io.LimitReader(resp.Body, maxFetchSize+1)
			body, err := io.ReadAll(limitedReader)
			if err != nil {
				return FetchOutput{}, fmt.Errorf("fetch: reading response: %w", err)
			}

			content := string(body)
			if len(body) > maxFetchSize {
				content = content[:maxFetchSize] + "\n... [content truncated]"
			}

			return FetchOutput{
				Content:     content,
				StatusCode:  resp.StatusCode,
				ContentType: resp.Header.Get("Content-Type"),
			}, nil
		},
	)
}
