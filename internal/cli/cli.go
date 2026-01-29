package cli

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/url"
	"os"
	"strings"

	"github.com/livebud/cli"
	"github.com/matthewmueller/llm"
	"github.com/matthewmueller/llm/internal/env"
	"github.com/matthewmueller/llm/providers/anthropic"
	"github.com/matthewmueller/llm/providers/gemini"
	"github.com/matthewmueller/llm/providers/ollama"
	"github.com/matthewmueller/llm/providers/openai"
)

func New(log *slog.Logger) *CLI {
	return &CLI{
		log:    log,
		Stdin:  os.Stdin,
		Stdout: os.Stdout,
		Stderr: os.Stderr,
		Env:    os.Environ(),
		Dir:    ".",
	}
}

type CLI struct {
	log    *slog.Logger
	Stdin  io.Reader
	Stdout io.Writer
	Stderr io.Writer
	Env    []string
	Dir    string
}

func (c *CLI) Parse(ctx context.Context, args ...string) error {
	cmd := &Chat{Log: c.log}
	cli := cli.New("llm", "chat with large language models")
	cli.Flag("model", "model to use").Short('m').Optional().String(&cmd.Model)
	cli.Args("prompt", "prompt to send to the model").Optional().Strings(&cmd.Prompt)
	cli.Flag("format", "output format").Enum(&cmd.Format, "text", "json").Default("text")
	cli.Run(func(ctx context.Context) error {
		return c.Chat(ctx, cmd)
	})

	{ // $ llm models
		cmd := &Models{Log: c.log}
		cli := cli.Command("models", "list available models")
		cli.Run(func(ctx context.Context) error {
			return c.Models(ctx, cmd)
		})
	}

	return cli.Parse(ctx, args...)
}

// thinkingWriter wraps a writer to show thinking in dim ANSI
type thinkingWriter struct {
	w io.Writer
}

func (tw *thinkingWriter) Think(p []byte) (int, error) {
	// Write with dim ANSI escape codes
	fmt.Fprintf(tw.w, "\033[2m%s\033[0m", p)
	return len(p), nil
}

func (tw *thinkingWriter) Write(p []byte) (int, error) {
	return tw.w.Write(p)
}

var _ llm.Writer = (*thinkingWriter)(nil)

type Chat struct {
	Log    *slog.Logger
	Model  *string
	Prompt []string
	Format string
}

// Chat with the LLM
func (c *CLI) Chat(ctx context.Context, in *Chat) error {
	client, err := c.initClient()
	if err != nil {
		return fmt.Errorf("cli: %w", err)
	}

	model := ""
	if in.Model != nil {
		model = *in.Model
	}

	// Create thinking writer wrapper
	writer := &thinkingWriter{w: c.Stdout}

	if len(in.Prompt) > 0 {
		// Single prompt mode - send and exit
		agent := client.Agent(
			llm.WithModel(model),
			llm.WithWriter(writer),
		)
		_, err := agent.Send(ctx, strings.Join(in.Prompt, " "))
		if err != nil {
			return err
		}
		fmt.Fprintln(c.Stdout)
		return nil
	}

	// Interactive mode
	agent := client.Agent(
		llm.WithModel(model),
		llm.WithReader(c.Stdin),
		llm.WithWriter(writer),
	)
	return agent.Run(ctx)
}

type Models struct {
	Log    *slog.Logger
	Format string
}

// Models lists available models
func (c *CLI) Models(ctx context.Context, in *Models) error {
	client, err := c.initClient()
	if err != nil {
		return fmt.Errorf("cli: %w", err)
	}

	models, err := client.Models(ctx)
	if err != nil {
		return fmt.Errorf("cli: listing models: %w", err)
	}

	for _, m := range models {
		fmt.Fprintln(c.Stdout, m.Name)
	}
	return nil
}

func (c *CLI) initClient() (*llm.Client, error) {
	e, err := env.Load()
	if err != nil {
		return nil, fmt.Errorf("loading env: %w", err)
	}

	var providers []llm.Provider
	if e.AnthropicKey != "" {
		providers = append(providers, anthropic.New(c.log, e.AnthropicKey))
	}
	if e.OpenAIKey != "" {
		providers = append(providers, openai.New(c.log, e.OpenAIKey))
	}
	if e.GeminiKey != "" {
		providers = append(providers, gemini.New(c.log, e.GeminiKey))
	}
	if e.OllamaHost != "" {
		host, err := url.Parse(e.OllamaHost)
		if err != nil {
			return nil, fmt.Errorf("parsing ollama host: %w", err)
		}
		providers = append(providers, ollama.New(c.log, host))
	}

	return llm.New(c.log, providers...), nil
}
