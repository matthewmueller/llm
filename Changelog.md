# 0.2.5 / 2026-02-18

- add support for getting usage and a /context summary in the llm cli

# 0.2.4 / 2026-02-18

- add `provider.Model` method to `llm.Provider` interface
- add model metadata for openai, anthropic and gemini models
- switch over to using `matthewmueller/prompt`
- add a `fetch` tool

# 0.2.3 / 2026-02-17

- clear out tools
- add container and local sandbox support
- add sandbox-enabled shell tool
- support tools with slices as arguments
- pass back tool calls for storing past tool call history in context

# 0.2.2 / 2026-02-11

- remove slog.Logger since we weren't using it

# 0.2.1 / 2026-02-10

- add struct tags

# 0.2.0 / 2026-02-10

- llm: feed tool errors back into the model to try and resolve (subject to `WithMaxTurns`)

# 0.1.4 / 2026-02-08

- llm: add `llm.Model(ctx, provider, model)`
- llm: add ability to filter models with `llm.Models(ctx, filteredProviders...)`
- llm: provide `model.Name` where provided
- cli: make passing the --provider optional

# 0.1.3 / 2026-02-07

- add a readme and license

# 0.1.2 / 2026-02-02

- add thought signatures for gemini 3
- enforce a provider to avoid needing to request the full list of models each time
- pass the model name directly through to the provider

# 0.1.1 2026-01-31

- add an install command

# 0.1.0 / 2026-01-31

- llm client working with tools and thinking across models

# 0.0.2 / 2026-01-30

- started on claude code provider
- add a bunch of tools and tests
- add thinking as an enum
- rename the tooling

# 0.0.1 / 2026-01-29

- initial commit
