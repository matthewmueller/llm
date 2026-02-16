package sandbox

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
)

// Result is the output of a command execution.
type Result struct {
	Stdout   string
	Stderr   string
	ExitCode int
}

// ExitError is returned when a command exits with a non-zero code.
type ExitError struct {
	Code   int
	Stderr string
}

func (e *ExitError) Error() string {
	if e.Stderr != "" {
		return fmt.Sprintf("sandbox: exit code %d: %s", e.Code, e.Stderr)
	}
	return fmt.Sprintf("sandbox: exit code %d", e.Code)
}

// Cmd is a command that can be started and streamed, similar to exec.Cmd.
type Cmd interface {
	SetDir(dir string)
	SetTTY(tty bool)
	StdinPipe() (io.WriteCloser, error)
	StdoutPipe() (io.ReadCloser, error)
	StderrPipe() (io.ReadCloser, error)
	Start() error
	Wait() error
	Run() error
	ExitCode() int
}

// Commander can construct command handles for execution.
type Commander interface {
	CommandContext(ctx context.Context, cmd string, args ...string) Cmd
}

// Sandbox executes commands in an isolated environment.
type Sandbox interface {
	Commander
	Execute(ctx context.Context, cmd string, args ...string) (Result, error)
}

// Execute runs a command and collects stdout/stderr/exit code.
func Execute(ctx context.Context, c Commander, cmd string, args ...string) (Result, error) {
	command := c.CommandContext(ctx, cmd, args...)
	return Collect(command)
}

// Collect runs a command and buffers stdout/stderr while preserving exit code.
func Collect(command Cmd) (Result, error) {
	stdout, err := command.StdoutPipe()
	if err != nil {
		return Result{}, fmt.Errorf("sandbox: stdout pipe: %w", err)
	}
	stderr, err := command.StderrPipe()
	if err != nil {
		return Result{}, fmt.Errorf("sandbox: stderr pipe: %w", err)
	}

	if err := command.Start(); err != nil {
		return Result{}, err
	}

	var stdoutBuf bytes.Buffer
	var stderrBuf bytes.Buffer

	var copyWG sync.WaitGroup
	copyWG.Add(2)
	go func() {
		defer copyWG.Done()
		_, _ = io.Copy(&stdoutBuf, stdout)
	}()
	go func() {
		defer copyWG.Done()
		_, _ = io.Copy(&stderrBuf, stderr)
	}()

	waitErr := command.Wait()
	copyWG.Wait()

	result := Result{
		Stdout:   stdoutBuf.String(),
		Stderr:   stderrBuf.String(),
		ExitCode: command.ExitCode(),
	}

	if waitErr != nil {
		if errors.Is(waitErr, context.Canceled) || errors.Is(waitErr, context.DeadlineExceeded) {
			return Result{}, waitErr
		}
		if result.ExitCode >= 0 {
			return result, nil
		}
		return Result{}, waitErr
	}

	return result, nil
}
