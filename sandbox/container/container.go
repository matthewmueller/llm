package container

import (
	"context"
	"fmt"
	"io"
	"os/exec"

	"github.com/matthewmueller/llm/sandbox"
)

// Sandbox executes commands inside a running container.
type Sandbox struct {
	runtime   string
	container string
	execArgs  []string
}

var _ sandbox.Sandbox = (*Sandbox)(nil)

// Option configures a container sandbox.
type Option func(*Sandbox)

// WithRuntime sets the container runtime binary (docker or podman).
func WithRuntime(runtime string) Option {
	return func(s *Sandbox) {
		s.runtime = runtime
	}
}

// WithExecArgs appends args to the runtime exec invocation.
func WithExecArgs(args ...string) Option {
	return func(s *Sandbox) {
		s.execArgs = append(s.execArgs, args...)
	}
}

// New creates a container sandbox for the target container name/id.
func New(container string, options ...Option) (*Sandbox, error) {
	s := &Sandbox{
		container: container,
	}
	for _, option := range options {
		option(s)
	}
	if s.runtime == "" {
		runtime, err := detectRuntime()
		if err != nil {
			return nil, err
		}
		s.runtime = runtime
	}
	return s, nil
}

// CommandContext builds a command handle for execution in the container.
func (s *Sandbox) CommandContext(ctx context.Context, cmd string, args ...string) sandbox.Cmd {
	return &command{
		ctx:       ctx,
		sandbox:   s,
		name:      cmd,
		args:      args,
		exitCode:  -1,
		innerArgs: append([]string{}, s.execArgs...),
	}
}

// Execute runs a command inside the configured container.
func (s *Sandbox) Execute(ctx context.Context, cmd string, args ...string) (sandbox.Result, error) {
	return sandbox.Execute(ctx, s, cmd, args...)
}

type command struct {
	ctx       context.Context
	sandbox   *Sandbox
	name      string
	args      []string
	dir       string
	tty       bool
	innerArgs []string
	cmd       *exec.Cmd
	exitCode  int
}

var _ sandbox.Cmd = (*command)(nil)

func (c *command) SetDir(dir string) {
	c.dir = dir
}

func (c *command) SetTTY(tty bool) {
	c.tty = tty
}

func (c *command) StdinPipe() (io.WriteCloser, error) {
	cmd := c.ensure()
	return cmd.StdinPipe()
}

func (c *command) StdoutPipe() (io.ReadCloser, error) {
	cmd := c.ensure()
	return cmd.StdoutPipe()
}

func (c *command) StderrPipe() (io.ReadCloser, error) {
	cmd := c.ensure()
	return cmd.StderrPipe()
}

func (c *command) Start() error {
	cmd := c.ensure()
	return cmd.Start()
}

func (c *command) Wait() error {
	cmd := c.ensure()
	err := cmd.Wait()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			c.exitCode = exitErr.ExitCode()
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

func (c *command) ensure() *exec.Cmd {
	if c.cmd != nil {
		return c.cmd
	}

	args := []string{"exec"}
	if c.tty {
		args = append(args, "-t")
	}
	if c.dir != "" {
		args = append(args, "-w", c.dir)
	}
	args = append(args, c.innerArgs...)
	args = append(args, c.sandbox.container, c.name)
	args = append(args, c.args...)

	c.cmd = exec.CommandContext(c.ctx, c.sandbox.runtime, args...)
	return c.cmd
}

func detectRuntime() (string, error) {
	if _, err := exec.LookPath("podman"); err == nil {
		return "podman", nil
	}
	if _, err := exec.LookPath("docker"); err == nil {
		return "docker", nil
	}
	return "", fmt.Errorf("container sandbox: unable to find podman or docker")
}
