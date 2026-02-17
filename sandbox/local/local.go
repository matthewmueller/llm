package local

import (
	"context"
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/matthewmueller/llm/sandbox"
)

// New creates a new local sandbox
func New(root string) *sandbox.Exec {
	return sandbox.New(&Sandbox{root})
}

// Sandbox executes commands on the local machine.
type Sandbox struct {
	root string
}

var _ sandbox.Executor = (*Sandbox)(nil)

func (s *Sandbox) Run(ctx context.Context, c *sandbox.Cmd) error {
	rootDir, err := filepath.Abs(s.root)
	if err != nil {
		return fmt.Errorf("sandbox/local: resolving root dir: %w", err)
	}

	workDir, err := resolve(rootDir, c.Dir)
	if err != nil {
		return fmt.Errorf("sandbox/local: resolving working dir: %w", err)
	}

	isOutside, err := isOutsideRoot(rootDir, workDir)
	if err != nil {
		return fmt.Errorf("sandbox/local: unable to verify working dir: %w", err)
	} else if isOutside {
		// TODO: consider allowing this with permission
		return fmt.Errorf("sandbox/local: working dir %q is outside of root %q", c.Dir, s.root)
	}

	// Run the command
	cmd := exec.CommandContext(ctx, c.Path, c.Args...)
	cmd.Dir = workDir
	cmd.Stdin = c.Stdin
	cmd.Stdout = c.Stdout
	cmd.Stderr = c.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("sandbox/local: running command: %w", err)
	}

	return nil
}

func resolve(absDir string, dirs ...string) (string, error) {
	for _, dir := range dirs {
		if filepath.IsAbs(dir) {
			absDir = dir
			continue
		}
		absDir = filepath.Join(absDir, dir)
	}
	return absDir, nil
}

func isOutsideRoot(rootDir, workDir string) (bool, error) {
	rel, err := filepath.Rel(rootDir, workDir)
	if err != nil {
		return false, err
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return true, nil
	}
	return false, nil
}
