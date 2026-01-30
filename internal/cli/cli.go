package cli

import (
	"bufio"
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
	cli.Flag("thinking", "thinking level: low, medium, high").Short('t').Enum(&cmd.Thinking, "low", "medium", "high").Default("medium")
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
	Log      *slog.Logger
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

	client, err := c.llm(env)
	if err != nil {
		return fmt.Errorf("cli: unable to load llm: %w", err)
	}

	model := ""
	if in.Model != nil {
		model = *in.Model
	}

	var opts []llm.AgentOption
	opts = append(opts, llm.WithModel(model))
	opts = append(opts, llm.WithThinking(llm.Thinking(in.Thinking)))

	agent := client.Agent(opts...)

	if len(in.Prompt) > 0 {
		// Single prompt mode - send and exit
		for ev, err := range agent.Send(ctx, strings.Join(in.Prompt, " ")) {
			if err != nil {
				return err
			}
			if ev.Thinking != "" {
				fmt.Fprintf(c.Stdout, "\033[2m%s\033[0m", ev.Thinking)
			}
			if ev.Content != "" {
				fmt.Fprint(c.Stdout, ev.Content)
			}
		}
		fmt.Fprintln(c.Stdout)
		return nil
	}

	// Interactive mode
	scanner := bufio.NewScanner(c.Stdin)
	for {
		fmt.Fprint(c.Stdout, "> ")
		if !scanner.Scan() {
			break
		}
		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}
		if input == "exit" || input == "quit" {
			break
		}

		for ev, err := range agent.Send(ctx, input) {
			if err != nil {
				return err
			}
			if ev.Thinking != "" {
				fmt.Fprintf(c.Stdout, "\033[2m%s\033[0m", ev.Thinking)
			}
			if ev.Content != "" {
				fmt.Fprint(c.Stdout, ev.Content)
			}
		}
		fmt.Fprintln(c.Stdout)
	}
	return scanner.Err()
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
