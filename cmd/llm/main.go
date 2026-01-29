package main

import (
	"context"
	"log/slog"
	"os"

	"github.com/matthewmueller/llm/internal/cli"
	"github.com/matthewmueller/logs"
)

func main() {
	ctx := context.Background()
	log := logs.Default()
	if err := run(ctx, log); err != nil {
		log.Error(err.Error())
		os.Exit(1)
	}
}

func run(ctx context.Context, log *slog.Logger) error {
	cli := cli.New(log)
	return cli.Parse(ctx, os.Args[1:]...)
}
