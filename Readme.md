# llm

`llm` is a small Go library and CLI for building powerful agents across providers.

You can think of this library as a pluggable [agent harness](https://x.com/tobi/status/2018506396321419760) that performs the actions coming back from the models. Conceptually similar to Claude Code or Codex, but easily runnable across models and in server-side environments.

It's based on the **Agent Loop** described in [Unrolling the Codex agent loop](https://openai.com/index/unrolling-the-codex-agent-loop/).

## Features

- Providers: OpenAI, Anthropic, Gemini, Ollama (more welcome!)
- Streaming responses
- High-level, recursive, concurrent tool calling
- Thinking/reasoning controls (`none`, `low`, `medium`, `high`)

## Install

CLI:

```sh
go install github.com/matthewmueller/llm/cmd/llm@latest
```

Library:

```sh
go get github.com/matthewmueller/llm
```

## Programmatic Usage

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/matthewmueller/llm"
	"github.com/matthewmueller/llm/providers/openai"
	"github.com/matthewmueller/llm/providers/anthropic"
	"github.com/matthewmueller/logs"
)

func main() {
	ctx := context.Background()
	log := logs.Default()

	client := llm.New(
		log,
		openai.New(log, os.Getenv("OPENAI_API_KEY")),
		anthropic.New(log, os.Getenv("ANTHROPIC_API_KEY")),
	)

	add := llm.Func("add", "Add two numbers", func(ctx context.Context, in struct {
		A int `json:"a" description:"First number" is:"required"`
		B int `json:"b" description:"Second number" is:"required"`
	}) (int, error) {
		return in.A + in.B, nil
	})

	for event, err := range client.Chat(
		ctx,
		provider.Name(),
		llm.WithModel("gpt-5-mini-2025-08-07"),
		llm.WithThinking(llm.ThinkingLow),
		llm.WithMessage(llm.UserMessage("Use add to add 20 and 22, then answer briefly.")),
		llm.WithTool(add),
	) {
		if err != nil {
			log.Fatal(err)
		}
    if event.Thinking {
      fmt.Print(event.Thinking)
    }
		fmt.Print(event.Content)
	}
}
```

For testing purposes, `llm` also ships with a CLI.

## CLI Usage

`--provider` (or `LLM_PROVIDER`) is required.

Set up one provider:

```sh
export LLM_PROVIDER=openai
export OPENAI_API_KEY=...
export LLM_MODEL=gpt-5-mini-2025-08-07
```

List models:

```sh
llm models
```

One-shot prompt:

```sh
llm "Explain CAP theorem in 3 bullets"
```

Interactive chat:

```sh
llm
```

Thinking level:

```sh
llm -t low "Plan a weekend trip to Portland"
```

Provider env vars:

- `openai`: `OPENAI_API_KEY`
- `anthropic`: `ANTHROPIC_API_KEY`
- `gemini`: `GEMINI_API_KEY`
- `ollama`: `OLLAMA_HOST` (defaults to `http://localhost:11434`)
