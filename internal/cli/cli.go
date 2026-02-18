package cli

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"text/tabwriter"

	"github.com/livebud/cli"
	"github.com/livebud/color"
	"github.com/matthewmueller/llm"
	"github.com/matthewmueller/llm/internal/env"
	"github.com/matthewmueller/llm/providers/anthropic"
	"github.com/matthewmueller/llm/providers/gemini"
	"github.com/matthewmueller/llm/providers/ollama"
	"github.com/matthewmueller/llm/providers/openai"
	"github.com/matthewmueller/llm/sandbox/container"
	"github.com/matthewmueller/llm/tool/fetch"
	"github.com/matthewmueller/llm/tool/shell"
	"github.com/matthewmueller/prompt"
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
	model, err := lc.Model(ctx, provider.Name(), *in.Model)
	if err != nil {
		return fmt.Errorf("cli: unable to find model: %w", err)
	}

	// Local sandbox in the configured directory for tools
	// TODO: support session ids and caching instead of random temp dirs
	tmpDir, err := os.MkdirTemp("", "llm-cli-sandbox-*")
	if err != nil {
		return fmt.Errorf("cli: unable to create temp dir for sandbox: %w", err)
	}
	sandbox := container.New("alpine",
		container.WithWorkDir("/app"),
		container.WithVolume(tmpDir, "/app"),
	)

	options := []llm.Option{
		llm.WithModel(*in.Model),
		llm.WithThinking(llm.Thinking(in.Thinking)),
		llm.WithTool(
			shell.New(sandbox),
			fetch.New(http.DefaultClient),
		),
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
	var lastUsage *llm.Usage

	// Interactive mode
	for {
		input, err := prompt.Ask(ctx, "$")
		if err != nil {
			if err == prompt.ErrInterrupted {
				return nil
			}
			return err
		}
		input = strings.TrimSpace(input)
		if input == "" {
			continue
		}
		if c.handleREPLCommand(input, model, messages, lastUsage) {
			continue
		}
		messages = append(messages, llm.UserMessage(input))
		turnOptions := append(options,
			llm.WithMessage(messages...),
		)
		assistant := &llm.Message{
			Role: "assistant",
		}
		hasNewline := true
		isThinking := true
		var turnUsage *llm.Usage
		for res, err := range lc.Chat(ctx, provider.Name(), turnOptions...) {
			if err != nil {
				return err
			}
			if res.Usage != nil {
				turnUsage = res.Usage
			}
			if res.Thinking != "" {
				fmt.Fprint(c.Stderr, color.Dim(res.Thinking))
				hasNewline = strings.HasSuffix(res.Thinking, "\n")
			}
			if res.ToolCall != nil {
				if !hasNewline {
					fmt.Fprintln(c.Stderr)
					hasNewline = true
				}
				c.log.Info("tool call", "name", res.ToolCall.Name, "args", string(res.ToolCall.Arguments), "id", res.ToolCall.ID)
				messages = append(messages, &llm.Message{
					Role:     res.Role,
					ToolCall: res.ToolCall,
				})
				continue
			}
			if res.ToolCallID != "" {
				if !hasNewline {
					fmt.Fprintln(c.Stderr)
					hasNewline = true
				}
				c.log.Info("tool result", "id", res.ToolCallID, "result", res.Content)
				messages = append(messages, &llm.Message{
					Role:       res.Role,
					Content:    res.Content,
					ToolCallID: res.ToolCallID,
				})
				continue
			}
			if res.Content != "" {
				if !hasNewline && isThinking {
					fmt.Fprintln(c.Stderr)
				}
				fmt.Fprint(c.Stdout, res.Content)
				assistant.Content += res.Content
				isThinking = false
				hasNewline = strings.HasSuffix(res.Content, "\n")
			}
		}

		// Save the assistant message for this turn
		if assistant.Content != "" {
			messages = append(messages, assistant)
		}
		if turnUsage != nil {
			lastUsage = turnUsage
		}

		// Add a newline after each turn for readability
		fmt.Fprintln(c.Stdout)
	}
}

const maxContextSnippet = 72

func (c *CLI) handleREPLCommand(input string, model *llm.Model, messages []*llm.Message, usage *llm.Usage) bool {
	fields := strings.Fields(strings.TrimSpace(input))
	if len(fields) == 0 || !strings.HasPrefix(fields[0], "/") {
		return false
	}
	switch fields[0] {
	case "/context":
		fmt.Fprintln(c.Stdout, formatContextSummary(model, messages, usage))
	default:
		fmt.Fprintf(c.Stderr, "unknown command: %s\n", fields[0])
	}
	return true
}

func formatContextSummary(model *llm.Model, messages []*llm.Message, usage *llm.Usage) string {
	contextWindow := 0
	if model != nil && model.Meta != nil {
		contextWindow = model.Meta.ContextWindow
	}

	entries := contextEntries(messages)

	var b strings.Builder
	if contextWindow > 0 && usage != nil && usage.InputTokens > 0 {
		fmt.Fprintf(&b, "context: %s/%s used (%s)\n",
			formatInt(usage.InputTokens),
			formatInt(contextWindow),
			formatPercent((float64(usage.InputTokens)/float64(contextWindow))*100),
		)
	} else if contextWindow > 0 {
		fmt.Fprintf(&b, "context: unknown/%s used, %d messages\n", formatInt(contextWindow), len(messages))
	} else {
		fmt.Fprintf(&b, "context: unknown/window_unknown, %d messages\n", len(messages))
	}

	if len(entries) == 0 {
		return strings.TrimRight(b.String(), "\n")
	}
	var table strings.Builder
	tw := tabwriter.NewWriter(&table, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "share\tchars\trole\tsnippet")
	for _, entry := range entries {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\n",
			formatPercent(entry.Share),
			formatInt(entry.Chars),
			shorten(entry.Label, 24),
			entry.Preview,
		)
	}
	tw.Flush()
	b.WriteString(table.String())
	return strings.TrimRight(b.String(), "\n")
}

type contextEntry struct {
	Label   string
	Preview string
	Chars   int
	Share   float64
}

func formatPercent(pct float64) string {
	if pct >= 10 {
		return fmt.Sprintf("%.1f%%", pct)
	}
	return fmt.Sprintf("%.2f%%", pct)
}

func contextEntries(messages []*llm.Message) (entries []contextEntry) {
	totalChars := 0
	for _, message := range messages {
		label, text := summarizeMessage(message)
		chars := len(message.Content) + len(message.Thinking)
		if message.ToolCall != nil {
			chars += len(message.ToolCall.Arguments)
		}
		totalChars += chars
		if text == "" {
			text = "(empty)"
		}
		entries = append(entries, contextEntry{
			Label:   label,
			Preview: shorten(text, maxContextSnippet),
			Chars:   chars,
		})
	}
	if totalChars == 0 {
		return entries
	}
	for i := range entries {
		entries[i].Share = (float64(entries[i].Chars) / float64(totalChars)) * 100
	}
	return entries
}

func summarizeMessage(message *llm.Message) (label, text string) {
	if message.ToolCall != nil {
		return "assistant[" + message.ToolCall.Name + "]", string(message.ToolCall.Arguments)
	}
	if message.Role == "tool" {
		if message.ToolCallID != "" {
			return "tool[" + message.ToolCallID + "]", message.Content
		}
		return "tool", message.Content
	}
	if message.Role == "assistant" && message.Content == "" && message.Thinking != "" {
		return "assistant(thinking)", message.Thinking
	}
	return message.Role, message.Content
}

func shorten(input string, limit int) string {
	clean := strings.Join(strings.Fields(strings.TrimSpace(input)), " ")
	if len(clean) <= limit {
		return clean
	}
	if limit <= 3 {
		return clean[:limit]
	}
	return clean[:limit-3] + "..."
}

func formatInt(n int) string {
	s := strconv.Itoa(n)
	negative := n < 0
	if negative {
		s = s[1:]
	}
	for i := len(s) - 3; i > 0; i -= 3 {
		s = s[:i] + "," + s[i:]
	}
	if negative {
		return "-" + s
	}
	return s
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
		fmt.Fprintln(c.Stdout)
	}

	return nil
}
