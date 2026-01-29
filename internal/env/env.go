package env

import (
	env11 "github.com/caarlos0/env/v11"
)

// Env holds environment configuration for LLM providers
type Env struct {
	AnthropicKey string `env:"ANTHROPIC_API_KEY"`
	OpenAIKey    string `env:"OPENAI_API_KEY"`
	GeminiKey    string `env:"GEMINI_API_KEY"`
	OllamaHost   string `env:"OLLAMA_HOST"`
}

// Load reads environment variables
func Load() (*Env, error) {
	env := new(Env)
	if err := env11.Parse(env); err != nil {
		return nil, err
	}
	return env, nil
}
