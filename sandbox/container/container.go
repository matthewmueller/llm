package container

import (
	"context"
	"fmt"
	"os/exec"
	"path"

	"github.com/matthewmueller/llm/sandbox"
)

func WithVolume(hostPath, containerPath string) Option {
	return func(c *Sandbox) {
		c.volumes = append(c.volumes, fmt.Sprintf("%s:%s", hostPath, containerPath))
	}
}

func WithWorkDir(workdir string) Option {
	workdir = path.Clean(workdir)
	return func(c *Sandbox) {
		c.workDir = workdir
	}
}

type Option func(*Sandbox)

// New creates a new local sandbox
func New(image string, options ...Option) *sandbox.Exec {
	box := &Sandbox{
		image,
		"/",
		nil,
	}
	for _, option := range options {
		option(box)
	}
	return sandbox.New(box)
}

// Sandbox executes commands on the local machine.
type Sandbox struct {
	image   string
	workDir string
	volumes []string
}

var _ sandbox.Executor = (*Sandbox)(nil)

func detectRuntime() (string, error) {
	if _, err := exec.LookPath("podman"); err == nil {
		return "podman", nil
	}
	if _, err := exec.LookPath("docker"); err == nil {
		return "docker", nil
	}
	return "", fmt.Errorf("container sandbox: unable to find podman or docker")
}

func resolve(rootDir string, dirs ...string) string {
	workDir := rootDir
	for _, dir := range dirs {
		if path.IsAbs(dir) {
			workDir = dir
			continue
		}
		workDir = path.Join(workDir, dir)
	}
	return workDir
}

func (s *Sandbox) Run(ctx context.Context, c *sandbox.Cmd) error {
	// docker or podman
	runtime, err := detectRuntime()
	if err != nil {
		return err
	}

	workDir := resolve(s.workDir, c.Dir)

	// Setup container arguments
	args := []string{"run", "--rm", "-i"}
	args = append(args, "-w", workDir)
	for _, volume := range s.volumes {
		args = append(args, "-v", volume)
	}
	args = append(args, s.image, c.Path)
	args = append(args, c.Args...)

	// Run the command inside a container
	cmd := exec.CommandContext(ctx, runtime, args...)
	cmd.Stdin = c.Stdin
	cmd.Stdout = c.Stdout
	cmd.Stderr = c.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("container sandbox: running command: %w", err)
	}

	return nil
}
