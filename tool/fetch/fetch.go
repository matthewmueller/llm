package fetch

import (
	"context"
	"fmt"
	"net/http"

	"github.com/matthewmueller/llm"

	htmltomarkdown "github.com/JohannesKaufmann/html-to-markdown/v2"
)

const description = `
- Fetches the URL content, converting HTML to markdown
- Use this tool when you need to retrieve and analyze the latest web content
`

type In struct {
	URL string `json:"url" is:"required" description:"The URL to fetch content from"`
}

type Out struct {
	Status  int    `json:"status"`
	Content string `json:"content"`
}

func New(hc *http.Client) llm.Tool {
	return llm.Func("Fetch", description, func(ctx context.Context, input In) (*Out, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, input.URL, nil)
		if err != nil {
			return nil, fmt.Errorf("fetch: failed to create request: %w", err)
		}

		res, err := hc.Do(req)
		if err != nil {
			return nil, fmt.Errorf("fetch: request failed: %w", err)
		}
		defer res.Body.Close()

		markdown, err := htmltomarkdown.ConvertReader(res.Body)
		if err != nil {
			return nil, fmt.Errorf("fetch: failed to convert HTML to markdown: %w", err)
		}

		return &Out{
			Status:  res.StatusCode,
			Content: string(markdown),
		}, nil
	})
}
