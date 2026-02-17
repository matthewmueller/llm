package cli

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/url"
	"os"
	"strings"

	"github.com/Bowery/prompt"
	"github.com/livebud/cli"
	"github.com/livebud/color"
	"github.com/matthewmueller/llm"
	"github.com/matthewmueller/llm/internal/env"
	"github.com/matthewmueller/llm/providers/anthropic"
	"github.com/matthewmueller/llm/providers/gemini"
	"github.com/matthewmueller/llm/providers/ollama"
	"github.com/matthewmueller/llm/providers/openai"
	"github.com/matthewmueller/llm/sandbox/container"
	"github.com/matthewmueller/llm/tool/shell"
)

func New(log *slog.Logger) *CLI {
	return &CLI{
		log:    log,
		Stdout: os.Stdout,
		Stderr: os.Stderr,
		Env:    os.Environ(),
		Dir:    ".",
	}
}

type CLI struct {
	log    *slog.Logger
	Stdout io.Writer
	Stderr io.Writer
	Env    []string
	Dir    string
}

func (c *CLI) Parse(ctx context.Context, args ...string) error {
	cmd := &Chat{Log: c.log}
	cli := cli.New("llm", "chat with large language models")
	cli.Flag("model", "model to use").Short('m').Env("LLM_MODEL").Optional().String(&cmd.Model)
	cli.Flag("provider", "provider to use").Short('p').Env("LLM_PROVIDER").Optional().String(&cmd.Provider)
	cli.Flag("thinking", "thinking level: low, medium, high").Short('t').Enum(&cmd.Thinking, "none", "low", "medium", "high").Default("medium")
	cli.Args("prompt", "prompt to send to the model").Optional().Strings(&cmd.Prompt)
	cli.Flag("format", "output format").Enum(&cmd.Format, "text", "json").Default("text")
	cli.Run(func(ctx context.Context) error {
		return c.Chat(ctx, cmd)
	})

	{ // $ llm models
		cli := cli.Command("models", "list available models")
		cli.Run(func(ctx context.Context) error {
			return c.Models(ctx, &Models{
				Log:      c.log,
				Provider: cmd.Provider,
				Format:   cmd.Format,
			})
		})
	}

	return cli.Parse(ctx, args...)
}

type Chat struct {
	Dir      string
	Log      *slog.Logger
	Provider *string
	Model    *string
	Thinking string
	Prompt   []string
	Format   string
}

func (c *CLI) providers(env *env.Env) (providers []llm.Provider, err error) {
	if env.AnthropicKey != "" {
		providers = append(providers, anthropic.New(env.AnthropicKey))
	}
	if env.OpenAIKey != "" {
		providers = append(providers, openai.New(env.OpenAIKey))
	}
	if env.GeminiKey != "" {
		providers = append(providers, gemini.New(env.GeminiKey))
	}
	if env.OllamaHost != "" {
		host, err := url.Parse(env.OllamaHost)
		if err != nil {
			return nil, fmt.Errorf("cli: unable to parse ollama host: %w", err)
		}
		providers = append(providers, ollama.New(host))
	}
	return providers, nil
}

func (c *CLI) provider(providers []llm.Provider, name *string) (provider llm.Provider, err error) {
	if name == nil {
		if len(providers) == 0 {
			return nil, fmt.Errorf("cli: no providers configured")
		}
		if len(providers) > 1 {
			return nil, fmt.Errorf("cli: multiple providers configured, please specify one with --provider")
		}
		return providers[0], nil
	}
	for _, p := range providers {
		if p.Name() == *name {
			return p, nil
		}
	}
	return nil, fmt.Errorf("cli: provider not found: %s", *name)
}

// Chat with the LLM
func (c *CLI) Chat(ctx context.Context, in *Chat) error {
	// TODO: can we just pick the most recent model as a default?
	if in.Model == nil {
		return fmt.Errorf("cli: model is required")
	}

	env, err := env.Load()
	if err != nil {
		return fmt.Errorf("cli: unable to load env: %w", err)
	}

	providers, err := c.providers(env)
	if err != nil {
		return fmt.Errorf("cli: unable to load providers: %w", err)
	}

	provider, err := c.provider(providers, in.Provider)
	if err != nil {
		return fmt.Errorf("cli: unable to find provider: %w", err)
	}

	lc := llm.New(providers...)

	// TODO: move this into the provider interface
	if _, err := lc.Model(ctx, provider.Name(), *in.Model); err != nil {
		return fmt.Errorf("cli: unable to find model: %w", err)
	}

	// Local sandbox in the configured directory for tools
	// TODO support sandboxing
	sandbox := container.New("alpine",
		container.WithWorkDir("/app"),
		container.WithVolume("./app", "/app"),
	)

	options := []llm.Option{
		llm.WithModel(*in.Model),
		llm.WithThinking(llm.Thinking(in.Thinking)),
		llm.WithTool(shell.New(c.log, sandbox)),
	}

	// Log the provider and model we're using
	fmt.Fprintln(c.Stderr, color.Dim(provider.Name()+" "+*in.Model))

	if len(in.Prompt) > 0 {
		options = append(options,
			llm.WithMessage(
				llm.UserMessage(strings.Join(in.Prompt, " ")),
			),
		)
		for res, err := range lc.Chat(ctx, provider.Name(), options...) {
			if err != nil {
				return err
			}
			if res.Thinking != "" {
				fmt.Fprint(c.Stderr, color.Dim(res.Thinking))
			}
			if res.Content != "" {
				fmt.Fprint(c.Stdout, res.Content)
			}
		}
		return nil
	}

	messages := []*llm.Message{}

	// Interactive mode
	for {
		input, err := prompt.Basic(">", true)
		if err != nil {
			if err == prompt.ErrEOF || err == prompt.ErrCTRLC {
				return nil
			}
			return err
		}
		input = strings.TrimSpace(input)
		if input == "" {
			continue
		}
		messages = append(messages, llm.UserMessage(input))
		turnOptions := append(options,
			llm.WithMessage(messages...),
		)
		for res, err := range lc.Chat(ctx, provider.Name(), turnOptions...) {
			if err != nil {
				return err
			}
			if res.Thinking != "" {
				fmt.Fprint(c.Stderr, color.Dim(res.Thinking))
			}
			if res.Content != "" {
				fmt.Fprint(c.Stdout, res.Content)
			}
			messages = append(messages, &llm.Message{
				Role:    res.Role,
				Content: res.Content,
			})
		}
		fmt.Fprintln(c.Stdout)
	}
}

type Models struct {
	Log      *slog.Logger
	Provider *string
	Format   string
}

// Models lists available models
func (c *CLI) Models(ctx context.Context, in *Models) error {
	env, err := env.Load()
	if err != nil {
		return fmt.Errorf("cli: unable to load env: %w", err)
	}

	providers, err := c.providers(env)
	if err != nil {
		return fmt.Errorf("cli: unable to load providers: %w", err)
	}

	lc := llm.New(providers...)

	filter := []string{}
	if in.Provider != nil {
		filter = append(filter, *in.Provider)
	}

	models, err := lc.Models(ctx, filter...)
	if err != nil {
		return fmt.Errorf("cli: listing models: %w", err)
	}

	for _, m := range models {
		fmt.Fprint(c.Stdout, m.ID)
		if m.Name != "" {
			fmt.Fprintf(c.Stdout, " (%s)", m.Name)
		}
		fmt.Fprintln(c.Stdout)
	}

	return nil
}
