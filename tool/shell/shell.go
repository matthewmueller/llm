package shell

import (
	"bytes"
	"context"
	"time"

	"github.com/matthewmueller/llm"
	"github.com/matthewmueller/llm/sandbox"
)

const defaultTimeout = 10_000 * time.Millisecond

const description = `Runs a shell command and returns a combined output of stdout and stderr.
- The arguments to ` + "`" + `shell` + "`" + ` will be passed to execvp(). Most terminal commands should be prefixed with ` + "`" + `sh -lc` + "`" + `.
- Always set the ` + "`" + `workdir` + "`" + ` param when using the shell function. By default the workdir is ` + "`" + `.` + "`" + `.
- Do not use ` + "`" + `cd` + "`" + ` unless absolutely necessary.
`

type In struct {
	Cmd       string   `json:"cmd" is:"required" description:"The name of the command to execute"`
	Args      []string `json:"args" is:"required" description:"The arguments to the command"`
	WorkDir   string   `json:"workdir" description:"The working directory to execute the command in"`
	TimeoutMs int      `json:"timeout_ms" description:"The timeout for the command in milliseconds"`
}

type Out struct {
	Output string `json:"output" description:"The combined output of stdout and stderr"`
}

func New(exec *sandbox.Exec) llm.Tool {
	return llm.Func("shell", description, func(ctx context.Context, in In) (*Out, error) {
		timeout := defaultTimeout
		if in.TimeoutMs > 0 {
			timeout = time.Duration(in.TimeoutMs) * time.Millisecond
		}
		ctx, cancel := context.WithTimeout(ctx, timeout)
		defer cancel()

		cmd := exec.CommandContext(ctx, in.Cmd, in.Args...)
		cmd.Dir = in.WorkDir

		out := new(bytes.Buffer)
		cmd.Stdout = out
		cmd.Stderr = out

		// Run the command
		if err := cmd.Run(); err != nil {
			return nil, err
		}

		// Return a combined output of stdout and stderr
		return &Out{
			Output: out.String(),
		}, nil
	})
}
