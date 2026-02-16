package local_test

import (
	"bufio"
	"context"
	"io"
	"testing"
	"time"

	"github.com/matryer/is"
	"github.com/matthewmueller/llm/sandbox/local"
)

func TestCommandContextStreaming(t *testing.T) {
	is := is.New(t)
	sb := local.New(t.TempDir())

	cmd := sb.CommandContext(context.Background(), "sh", "-lc", "printf 'hello\\n'; sleep 0.2; printf 'world\\n'")
	stdout, err := cmd.StdoutPipe()
	is.NoErr(err)

	is.NoErr(cmd.Start())

	reader := bufio.NewReader(stdout)
	firstLine := make(chan string, 1)
	go func() {
		line, _ := reader.ReadString('\n')
		firstLine <- line
	}()

	select {
	case line := <-firstLine:
		is.Equal(line, "hello\n")
	case <-time.After(1 * time.Second):
		t.Fatal("timed out waiting for first streamed line")
	}

	rest, err := io.ReadAll(reader)
	is.NoErr(err)
	is.Equal(string(rest), "world\n")

	is.NoErr(cmd.Wait())
	is.Equal(cmd.ExitCode(), 0)
}

func TestExecuteNonZeroExit(t *testing.T) {
	is := is.New(t)
	sb := local.New(t.TempDir())

	result, err := sb.Execute(context.Background(), "sh", "-lc", "echo 'nope' >&2; exit 42")
	is.NoErr(err)
	is.Equal(result.ExitCode, 42)
	is.Equal(result.Stderr, "nope\n")
}
