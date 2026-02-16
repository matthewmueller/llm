package ssh

import (
	"context"
	"fmt"
	"io"
	"net"
	"os"
	"strconv"
	"strings"
	"sync"

	"github.com/matthewmueller/llm/sandbox"
	"github.com/matthewmueller/sshx"
	gossh "golang.org/x/crypto/ssh"
)

// Sandbox executes commands on a remote host over ssh.
type Sandbox struct {
	host         string
	port         int
	identityFile string
	extraArgs    []string
}

var _ sandbox.Sandbox = (*Sandbox)(nil)

// Option configures the ssh sandbox.
type Option func(*Sandbox)

// WithPort sets the ssh port.
func WithPort(port int) Option {
	return func(s *Sandbox) {
		s.port = port
	}
}

// WithIdentityFile sets the ssh key path.
func WithIdentityFile(path string) Option {
	return func(s *Sandbox) {
		s.identityFile = path
	}
}

// WithArgs appends raw ssh arguments.
func WithArgs(args ...string) Option {
	return func(s *Sandbox) {
		s.extraArgs = append(s.extraArgs, args...)
	}
}

// New creates a new ssh sandbox.
func New(host string, options ...Option) *Sandbox {
	s := &Sandbox{
		host: host,
		port: 22,
	}
	for _, option := range options {
		option(s)
	}
	return s
}

// CommandContext builds a command handle for remote execution.
func (s *Sandbox) CommandContext(ctx context.Context, cmd string, args ...string) sandbox.Cmd {
	return &command{
		ctx:      ctx,
		sandbox:  s,
		name:     cmd,
		args:     args,
		exitCode: -1,
	}
}

// Execute runs a command remotely.
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

	client  *gossh.Client
	session *gossh.Session
	closer  sync.Once

	exitCode int
}

var _ sandbox.Cmd = (*command)(nil)

func (c *command) SetDir(dir string) {
	c.dir = dir
}

func (c *command) SetTTY(tty bool) {
	c.tty = tty
}

func (c *command) StdinPipe() (io.WriteCloser, error) {
	if err := c.ensureSession(); err != nil {
		return nil, err
	}
	return c.session.StdinPipe()
}

func (c *command) StdoutPipe() (io.ReadCloser, error) {
	if err := c.ensureSession(); err != nil {
		return nil, err
	}
	reader, err := c.session.StdoutPipe()
	if err != nil {
		return nil, err
	}
	return io.NopCloser(reader), nil
}

func (c *command) StderrPipe() (io.ReadCloser, error) {
	if err := c.ensureSession(); err != nil {
		return nil, err
	}
	reader, err := c.session.StderrPipe()
	if err != nil {
		return nil, err
	}
	return io.NopCloser(reader), nil
}

func (c *command) Start() error {
	if err := c.ensureSession(); err != nil {
		return err
	}
	if err := c.session.Start(c.commandString()); err != nil {
		return err
	}
	go func() {
		<-c.ctx.Done()
		c.close()
	}()
	return nil
}

func (c *command) Wait() error {
	if err := c.ensureSession(); err != nil {
		return err
	}
	defer c.close()

	err := c.session.Wait()
	if err != nil {
		if exitErr, ok := err.(*gossh.ExitError); ok {
			c.exitCode = exitErr.ExitStatus()
			return err
		}
		if c.ctx.Err() != nil {
			c.exitCode = -1
			return c.ctx.Err()
		}
		c.exitCode = -1
		return err
	}

	c.exitCode = 0
	return nil
}

func (c *command) Run() error {
	if err := c.Start(); err != nil {
		return err
	}
	return c.Wait()
}

func (c *command) ExitCode() int {
	return c.exitCode
}

func (c *command) ensureSession() error {
	if c.session != nil {
		return nil
	}
	if err := c.ctx.Err(); err != nil {
		return err
	}
	if len(c.sandbox.extraArgs) > 0 {
		return fmt.Errorf("ssh sandbox: WithArgs is not supported with sshx")
	}

	user, host, err := sshx.Split(c.sandbox.host)
	if err != nil {
		return fmt.Errorf("ssh sandbox: parse host: %w", err)
	}
	host, err = overridePort(host, c.sandbox.port)
	if err != nil {
		return fmt.Errorf("ssh sandbox: parse host/port: %w", err)
	}

	signers, err := c.sandbox.signers()
	if err != nil {
		return err
	}

	config := sshx.Configure(user, host, signers...)
	client, err := sshx.DialConfig(host, config)
	if err != nil {
		return fmt.Errorf("ssh sandbox: dial: %w", err)
	}

	session, err := client.NewSession()
	if err != nil {
		client.Close()
		return fmt.Errorf("ssh sandbox: create session: %w", err)
	}

	if c.tty {
		if err := session.RequestPty("xterm", 80, 40, gossh.TerminalModes{}); err != nil {
			session.Close()
			client.Close()
			return fmt.Errorf("ssh sandbox: request pty: %w", err)
		}
	}

	c.client = client
	c.session = session
	return nil
}

func (c *command) close() {
	c.closer.Do(func() {
		if c.session != nil {
			_ = c.session.Close()
		}
		if c.client != nil {
			_ = c.client.Close()
		}
	})
}

func (c *command) commandString() string {
	command := joinCommand(c.name, c.args...)
	if c.dir == "" {
		return command
	}
	return fmt.Sprintf("cd %s && %s", shellQuote(c.dir), command)
}

func joinCommand(cmd string, args ...string) string {
	parts := make([]string, 0, len(args)+1)
	parts = append(parts, shellQuote(cmd))
	for _, arg := range args {
		parts = append(parts, shellQuote(arg))
	}
	return strings.Join(parts, " ")
}

func shellQuote(input string) string {
	if input == "" {
		return "''"
	}
	return "'" + strings.ReplaceAll(input, "'", `'"'"'`) + "'"
}

func overridePort(host string, port int) (string, error) {
	if port == 0 || port == 22 {
		return host, nil
	}
	name, _, err := net.SplitHostPort(host)
	if err != nil {
		return "", err
	}
	return net.JoinHostPort(name, strconv.Itoa(port)), nil
}

func (s *Sandbox) signers() ([]gossh.Signer, error) {
	if s.identityFile == "" {
		return nil, nil
	}
	keyData, err := os.ReadFile(s.identityFile)
	if err != nil {
		return nil, fmt.Errorf("ssh sandbox: read key %q: %w", s.identityFile, err)
	}
	signer, err := sshx.ParsePrivateKey(s.identityFile, keyData, func() ([]byte, error) {
		return nil, fmt.Errorf("ssh sandbox: passphrase-protected key %q is not supported", s.identityFile)
	})
	if err != nil {
		return nil, fmt.Errorf("ssh sandbox: parse key %q: %w", s.identityFile, err)
	}
	return []gossh.Signer{signer}, nil
}
