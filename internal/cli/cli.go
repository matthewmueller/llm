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
	cli.Flag("provider", "provider to use").Short('p').Optional().String(&cmd.Provider)
	cli.Flag("thinking", "thinking level: low, medium, high").Short('t').Enum(&cmd.Thinking, "none", "low", "medium", "high").Default("medium")
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

type Chat struct {
	Dir      string
	Log      *slog.Logger
	Provider *string
	Model    *string
	Thinking string
	Prompt   []string
	Format   string
}

func (c *CLI) llm(env *env.Env) (*llm.Client, error) {
	var providers []llm.Provider
	if env.AnthropicKey != "" {
		providers = append(providers, anthropic.New(c.log, env.AnthropicKey))
	}
	if env.OpenAIKey != "" {
		providers = append(providers, openai.New(c.log, env.OpenAIKey))
	}
	if env.GeminiKey != "" {
		providers = append(providers, gemini.New(c.log, env.GeminiKey))
	}
	if env.OllamaHost != "" {
		host, err := url.Parse(env.OllamaHost)
		if err != nil {
			return nil, fmt.Errorf("cli: unable to parse ollama host: %w", err)
		}
		providers = append(providers, ollama.New(c.log, host))
	}
	return llm.New(c.log, providers...), nil
}

// Chat with the LLM
func (c *CLI) Chat(ctx context.Context, in *Chat) error {
	env, err := env.Load()
	if err != nil {
		return fmt.Errorf("cli: unable to load env: %w", err)
	}

	lc, err := c.llm(env)
	if err != nil {
		return fmt.Errorf("cli: unable to load llm: %w", err)
	}

	if in.Model == nil {
		return fmt.Errorf("cli: model is required")
	}

	options := []llm.Option{
		llm.WithModel(*in.Model),
		llm.WithThinking(llm.Thinking(in.Thinking)),
	}
	if in.Provider != nil {
		options = append(options, llm.WithProvider(*in.Provider))
	}

	if len(in.Prompt) > 0 {
		options = append(options,
			llm.WithMessage(
				llm.UserMessage(strings.Join(in.Prompt, " ")),
			),
		)
		for res, err := range lc.Chat(ctx, options...) {
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
		input, err := prompt.Basic("> ", false)
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
		for res, err := range lc.Chat(ctx, turnOptions...) {
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
	Log    *slog.Logger
	Format string
}

// Models lists available models
func (c *CLI) Models(ctx context.Context, in *Models) error {
	env, err := env.Load()
	if err != nil {
		return fmt.Errorf("cli: unable to load env: %w", err)
	}

	client, err := c.llm(env)
	if err != nil {
		return fmt.Errorf("cli: unable to load llm: %w", err)
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
