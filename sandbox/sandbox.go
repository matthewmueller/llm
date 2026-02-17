package sandbox

import (
	"context"
	"io"
)

type Executor interface {
	Run(ctx context.Context, cmd *Cmd) error
}

func New(executor Executor) *Exec {
	return &Exec{executor}
}

type Exec struct {
	exec Executor
}

func (e *Exec) Command(cmd string, args ...string) *Cmd {
	return e.CommandContext(context.Background(), cmd, args...)
}

func (e *Exec) CommandContext(ctx context.Context, cmd string, args ...string) *Cmd {
	return &Cmd{e.exec, ctx, cmd, args, "", nil, nil, nil, nil}
}

type Cmd struct {
	exec   Executor
	ctx    context.Context
	Path   string
	Args   []string
	Dir    string
	Env    []string
	Stdin  io.Reader
	Stdout io.Writer
	Stderr io.Writer
}

func (c *Cmd) Run() error {
	return c.exec.Run(c.ctx, c)
}
