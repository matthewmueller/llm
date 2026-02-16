package sprite

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"

	"github.com/matthewmueller/llm/sandbox"
)

const defaultExecURL = "https://sprites.dev/api/sprites/exec"

// Sandbox executes commands via the sprites.dev exec API.
type Sandbox struct {
	client *http.Client
	url    string
	apiKey string
}

var _ sandbox.Sandbox = (*Sandbox)(nil)

// Option configures the sprite sandbox.
type Option func(*Sandbox)

// WithHTTPClient sets the HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(s *Sandbox) {
		s.client = client
	}
}

// WithExecURL overrides the exec API URL.
func WithExecURL(execURL string) Option {
	return func(s *Sandbox) {
		s.url = execURL
	}
}

// New creates a sprite sandbox.
func New(apiKey string, options ...Option) *Sandbox {
	s := &Sandbox{
		client: http.DefaultClient,
		url:    defaultExecURL,
		apiKey: apiKey,
	}
	for _, option := range options {
		option(s)
	}
	return s
}

// CommandContext builds a sprite command handle.
func (s *Sandbox) CommandContext(ctx context.Context, cmd string, args ...string) sandbox.Cmd {
	return &command{
		ctx:      ctx,
		sandbox:  s,
		name:     cmd,
		args:     args,
		exitCode: -1,
	}
}

// Execute calls the sprites exec API and buffers output.
func (s *Sandbox) Execute(ctx context.Context, cmd string, args ...string) (sandbox.Result, error) {
	return sandbox.Execute(ctx, s, cmd, args...)
}

type command struct {
	ctx     context.Context
	sandbox *Sandbox
	name    string
	args    []string
	dir     string
	tty     bool

	stdoutR *io.PipeReader
	stdoutW *io.PipeWriter
	stderrR *io.PipeReader
	stderrW *io.PipeWriter
	stdinR  *io.PipeReader
	stdinW  *io.PipeWriter

	done     chan error
	started  bool
	exitCode int

	mu sync.Mutex
}

var _ sandbox.Cmd = (*command)(nil)

func (c *command) SetDir(dir string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.dir = dir
}

func (c *command) SetTTY(tty bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.tty = tty
}

func (c *command) StdinPipe() (io.WriteCloser, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.started {
		return nil, fmt.Errorf("sprite sandbox: StdinPipe after command started")
	}
	if c.stdinW != nil {
		return nil, fmt.Errorf("sprite sandbox: StdinPipe already called")
	}
	c.stdinR, c.stdinW = io.Pipe()
	go func() {
		_, _ = io.Copy(io.Discard, c.stdinR)
		_ = c.stdinR.Close()
	}()
	return c.stdinW, nil
}

func (c *command) StdoutPipe() (io.ReadCloser, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.started {
		return nil, fmt.Errorf("sprite sandbox: StdoutPipe after command started")
	}
	if c.stdoutR != nil {
		return nil, fmt.Errorf("sprite sandbox: StdoutPipe already called")
	}
	c.stdoutR, c.stdoutW = io.Pipe()
	return c.stdoutR, nil
}

func (c *command) StderrPipe() (io.ReadCloser, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.started {
		return nil, fmt.Errorf("sprite sandbox: StderrPipe after command started")
	}
	if c.stderrR != nil {
		return nil, fmt.Errorf("sprite sandbox: StderrPipe already called")
	}
	c.stderrR, c.stderrW = io.Pipe()
	return c.stderrR, nil
}

func (c *command) Start() error {
	c.mu.Lock()
	if c.started {
		c.mu.Unlock()
		return fmt.Errorf("sprite sandbox: command already started")
	}
	c.started = true
	c.done = make(chan error, 1)
	var stdoutW io.WriteCloser = c.stdoutW
	var stderrW io.WriteCloser = c.stderrW
	c.mu.Unlock()

	if stdoutW == nil {
		stdoutW = nopWriteCloser{Writer: io.Discard}
	}
	if stderrW == nil {
		stderrW = nopWriteCloser{Writer: io.Discard}
	}

	go func() {
		err := c.run(stdoutW, stderrW)
		_ = stdoutW.Close()
		_ = stderrW.Close()
		c.done <- err
		close(c.done)
	}()
	return nil
}

func (c *command) Wait() error {
	c.mu.Lock()
	done := c.done
	c.mu.Unlock()
	if done == nil {
		return fmt.Errorf("sprite sandbox: command not started")
	}
	err := <-done
	if err != nil && c.ctx.Err() != nil {
		c.mu.Lock()
		c.exitCode = -1
		c.mu.Unlock()
		return c.ctx.Err()
	}
	return err
}

func (c *command) Run() error {
	if err := c.Start(); err != nil {
		return err
	}
	return c.Wait()
}

func (c *command) ExitCode() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.exitCode
}

func (c *command) run(stdout io.WriteCloser, stderr io.WriteCloser) error {
	if err := c.ctx.Err(); err != nil {
		return err
	}

	c.mu.Lock()
	name := c.name
	args := append([]string{}, c.args...)
	dir := c.dir
	tty := c.tty
	c.mu.Unlock()

	if dir != "" {
		name = "sh"
		args = []string{"-lc", fmt.Sprintf("cd %s && %s", shellQuote(dir), joinCommand(c.name, c.args...))}
	}

	endpoint, err := url.Parse(c.sandbox.url)
	if err != nil {
		return fmt.Errorf("sprite sandbox: parse exec url: %w", err)
	}

	query := endpoint.Query()
	query.Add("cmd", name)
	for _, arg := range args {
		query.Add("cmd", arg)
	}
	if tty {
		query.Set("tty", "1")
	}
	endpoint.RawQuery = query.Encode()

	req, err := http.NewRequestWithContext(c.ctx, http.MethodPost, endpoint.String(), nil)
	if err != nil {
		return fmt.Errorf("sprite sandbox: build request: %w", err)
	}
	if c.sandbox.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.sandbox.apiKey)
	}

	res, err := c.sandbox.client.Do(req)
	if err != nil {
		return fmt.Errorf("sprite sandbox: execute request: %w", err)
	}
	defer res.Body.Close()

	contentType := strings.ToLower(res.Header.Get("Content-Type"))
	if strings.Contains(contentType, "application/json") {
		return c.handleJSON(res, stdout, stderr)
	}

	if res.StatusCode >= http.StatusBadRequest {
		body, _ := io.ReadAll(res.Body)
		c.setExitCode(1)
		_, _ = io.WriteString(stderr, strings.TrimSpace(string(body)))
		return &sandbox.ExitError{Code: 1}
	}

	_, err = io.Copy(stdout, res.Body)
	if err != nil {
		return fmt.Errorf("sprite sandbox: stream response: %w", err)
	}
	c.setExitCode(0)
	return nil
}

func (c *command) handleJSON(res *http.Response, stdout io.Writer, stderr io.Writer) error {
	body, err := io.ReadAll(res.Body)
	if err != nil {
		return fmt.Errorf("sprite sandbox: read response: %w", err)
	}
	if res.StatusCode >= http.StatusBadRequest {
		c.setExitCode(1)
		_, _ = io.WriteString(stderr, strings.TrimSpace(string(body)))
		return &sandbox.ExitError{Code: 1}
	}

	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		_, _ = stdout.Write(body)
		c.setExitCode(0)
		return nil
	}

	stdoutText := pickString(payload, "stdout", "output")
	stderrText := pickString(payload, "stderr", "error")
	exitCode := pickInt(payload, "exit_code", "exitCode", "status")

	if stdoutText != "" {
		_, _ = io.WriteString(stdout, stdoutText)
	}
	if stderrText != "" {
		_, _ = io.WriteString(stderr, stderrText)
	}

	c.setExitCode(exitCode)
	if exitCode != 0 {
		return &sandbox.ExitError{Code: exitCode, Stderr: stderrText}
	}
	return nil
}

func (c *command) setExitCode(code int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.exitCode = code
}

func pickString(payload map[string]any, keys ...string) string {
	for _, key := range keys {
		value, ok := payload[key]
		if !ok {
			continue
		}
		text, ok := value.(string)
		if ok {
			return text
		}
	}
	return ""
}

func pickInt(payload map[string]any, keys ...string) int {
	for _, key := range keys {
		value, ok := payload[key]
		if !ok {
			continue
		}
		switch v := value.(type) {
		case float64:
			return int(v)
		case int:
			return v
		case string:
			n, err := strconv.Atoi(v)
			if err == nil {
				return n
			}
		}
	}
	return 0
}

func shellQuote(input string) string {
	if input == "" {
		return "''"
	}
	return "'" + strings.ReplaceAll(input, "'", `'"'"'`) + "'"
}

func joinCommand(cmd string, args ...string) string {
	parts := make([]string, 0, len(args)+1)
	parts = append(parts, shellQuote(cmd))
	for _, arg := range args {
		parts = append(parts, shellQuote(arg))
	}
	return strings.Join(parts, " ")
}

type nopWriteCloser struct {
	io.Writer
}

func (n nopWriteCloser) Close() error {
	return nil
}
