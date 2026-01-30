package tools

import (
	"context"
	"fmt"

	"github.com/matthewmueller/llm"
)

const (
	defaultBashTimeout = 60
	maxOutputSize      = 100 * 1024 // 100KB
)

// BashInput defines the input parameters for the Bash tool.
type BashInput struct {
	Command    string `json:"command" is:"required" description:"Shell command to execute"`
	Timeout    int    `json:"timeout" description:"Timeout in seconds (default: 60)"`
	WorkingDir string `json:"working_dir" description:"Working directory for command execution"`
}

// BashOutput defines the output of the Bash tool.
type BashOutput struct {
	Stdout   string `json:"stdout"`
	Stderr   string `json:"stderr"`
	ExitCode int    `json:"exit_code"`
}

// Bash creates a tool for executing shell commands.
func Bash(exec Executor) llm.Tool {
	return llm.Func("tool_bash",
		"Execute a shell command. Use this for running programs, installing packages, git commands, and other terminal tasks.",
		func(ctx context.Context, in BashInput) (BashOutput, error) {
			timeout := in.Timeout
			if timeout <= 0 {
				timeout = defaultBashTimeout
			}

			stdout, stderr, exitCode, err := exec.Execute(ctx, in.Command, in.WorkingDir, timeout)
			if err != nil {
				return BashOutput{}, fmt.Errorf("bash: execution failed: %w", err)
			}

			// Truncate output if too large
			if len(stdout) > maxOutputSize {
				stdout = stdout[:maxOutputSize] + "\n... [output truncated]"
			}
			if len(stderr) > maxOutputSize {
				stderr = stderr[:maxOutputSize] + "\n... [output truncated]"
			}

			return BashOutput{
				Stdout:   stdout,
				Stderr:   stderr,
				ExitCode: exitCode,
			}, nil
		},
	)
}
