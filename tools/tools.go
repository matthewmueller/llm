// Package tools provides cross-provider built-in tools for LLM agents.
package tools

import (
	"bytes"
	"context"
	"net/http"
	"os/exec"
	"time"

	"github.com/matthewmueller/llm"
	"github.com/matthewmueller/llm/internal/ask"
	"github.com/matthewmueller/virt"
)

// Executor is an interface for executing shell commands.
type Executor interface {
	Execute(ctx context.Context, command string, workingDir string, timeout int) (stdout, stderr string, exitCode int, err error)
}

// All returns all built-in tools with the provided dependencies.
func All(a ask.Asker, fetcher *http.Client, fsys virt.FS, executor Executor) []llm.Tool {
	return []llm.Tool{
		Read(fsys),
		Write(fsys),
		Edit(fsys),
		Grep(fsys),
		Glob(fsys),
		ReadDir(fsys),
		Bash(executor),
		Ask(a),
		Fetch(fetcher),
	}
}

// DefaultExecutor implements the Executor interface using os/exec.
type DefaultExecutor struct{}

// Execute runs a shell command and returns the output.
func (e *DefaultExecutor) Execute(ctx context.Context, command string, workingDir string, timeout int) (stdout, stderr string, exitCode int, err error) {
	// Create context with timeout
	ctx, cancel := context.WithTimeout(ctx, time.Duration(timeout)*time.Second)
	defer cancel()

	// Create the command
	cmd := exec.CommandContext(ctx, "sh", "-c", command)
	if workingDir != "" {
		cmd.Dir = workingDir
	}

	// Capture stdout and stderr
	var stdoutBuf, stderrBuf bytes.Buffer
	cmd.Stdout = &stdoutBuf
	cmd.Stderr = &stderrBuf

	// Run the command
	err = cmd.Run()

	stdout = stdoutBuf.String()
	stderr = stderrBuf.String()

	// Get exit code
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			exitCode = exitErr.ExitCode()
			err = nil // Non-zero exit is not an error for us
		} else if ctx.Err() == context.DeadlineExceeded {
			err = context.DeadlineExceeded
			exitCode = -1
		}
	}

	return stdout, stderr, exitCode, err
}
