package env

import (
	env11 "github.com/caarlos0/env/v11"
)

// Env holds environment configuration for LLM providers
type Env struct {
	AnthropicKey string `env:"ANTHROPIC_API_KEY"`
	OpenAIKey    string `env:"OPENAI_API_KEY"`
	GeminiKey    string `env:"GEMINI_API_KEY"`
	OllamaHost   string `env:"OLLAMA_HOST" envDefault:"http://localhost:11434"`
	OllamaModel  string `env:"OLLAMA_MODEL"`
	ClaudeCode string `env:"CLAUDE_CODE"` // Claude Code CLI flags (e.g. "--permission-mode=plan --add-dir=/tmp")
}

// Load reads environment variables
func Load() (*Env, error) {
	env := new(Env)
	if err := env11.Parse(env); err != nil {
		return nil, err
	}
	return env, nil
}
