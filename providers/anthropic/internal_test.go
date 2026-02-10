package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/matryer/is"
)

func TestNormalizeToolArgumentsEmpty(t *testing.T) {
	is := is.New(t)
	args := normalizeToolArguments(nil)
	is.Equal(string(args), "{}")
}

func TestNormalizeToolArgumentsInvalid(t *testing.T) {
	is := is.New(t)
	args := normalizeToolArguments(json.RawMessage(`{"x":`))
	is.Equal(string(args), "{}")
}

func TestNormalizeToolArgumentsValid(t *testing.T) {
	is := is.New(t)
	args := normalizeToolArguments(json.RawMessage(` {"x":1} `))
	is.Equal(string(args), `{"x":1}`)
}
