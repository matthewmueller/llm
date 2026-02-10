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
