package llm_test

import (
	"context"
	"testing"

	"github.com/matryer/is"
	"github.com/matthewmueller/llm"
)

func TestFuncSchemaSliceTypes(t *testing.T) {
	is := is.New(t)

	tool := llm.Func("slice_types", "tests slice schema generation", func(ctx context.Context, in struct {
		Strings []string         `json:"strings"`
		Ints    []int            `json:"ints"`
		Floats  []float64        `json:"floats"`
		Bools   []bool           `json:"bools"`
		Objects []map[string]any `json:"objects"`
		Nested  [][]int          `json:"nested"`
		Ptr     *[]string        `json:"ptr"`
	}) (string, error) {
		return "", nil
	})

	schema := tool.Schema()
	props := schema.Function.Parameters.Properties

	is.Equal(props["strings"].Type, "array")
	is.Equal(props["strings"].Items.Type, "string")

	is.Equal(props["ints"].Type, "array")
	is.Equal(props["ints"].Items.Type, "integer")

	is.Equal(props["floats"].Type, "array")
	is.Equal(props["floats"].Items.Type, "number")

	is.Equal(props["bools"].Type, "array")
	is.Equal(props["bools"].Items.Type, "boolean")

	is.Equal(props["objects"].Type, "array")
	is.Equal(props["objects"].Items.Type, "object")

	is.Equal(props["nested"].Type, "array")
	is.Equal(props["nested"].Items.Type, "array")
	is.Equal(props["nested"].Items.Items.Type, "integer")

	is.Equal(props["ptr"].Type, "array")
	is.Equal(props["ptr"].Items.Type, "string")
}
