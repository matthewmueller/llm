package local

import (
	"context"
	"io"
	"os/exec"

	"github.com/matthewmueller/llm/sandbox"
)

// Sandbox executes commands on the local machine.
type Sandbox struct {
	dir string
}

var _ sandbox.Sandbox = (*Sandbox)(nil)

// New creates a new local sandbox.
func New(dir string) *Sandbox {
	return &Sandbox{dir: dir}
}

// CommandContext builds a local command handle.
func (s *Sandbox) CommandContext(ctx context.Context, cmd string, args ...string) sandbox.Cmd {
	c := &command{
		ctx:      ctx,
		name:     cmd,
		args:     args,
		dir:      s.dir,
		exitCode: -1,
	}
	return c
}

// Execute runs a command locally and collects its output.
func (s *Sandbox) Execute(ctx context.Context, cmd string, args ...string) (sandbox.Result, error) {
	return sandbox.Execute(ctx, s, cmd, args...)
}

type command struct {
	ctx      context.Context
	name     string
	args     []string
	dir      string
	tty      bool
	cmd      *exec.Cmd
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
	cmd := exec.CommandContext(c.ctx, c.name, c.args...)
	if c.dir != "" {
		cmd.Dir = c.dir
	}
	if c.tty {
		// Kept for API compatibility with remote sandboxes.
	}
	c.cmd = cmd
	return cmd
}
