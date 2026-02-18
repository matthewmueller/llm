package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
)

// Tool interface - high-level typed tool definition
type Tool interface {
	Schema() *ToolSchema
	Run(ctx context.Context, in json.RawMessage) (out []byte, err error)
}

// ToolCall represents a tool invocation from the model
type ToolCall struct {
	ID               string          `json:"id,omitzero"`
	Name             string          `json:"name,omitzero"`
	Arguments        json.RawMessage `json:"arguments,omitzero"`
	ThoughtSignature []byte          `json:"thought_signature,omitzero"`
}

// ToolSchema defines a tool's JSON schema specification
type ToolSchema struct {
	Type     string
	Function *ToolFunction
}

// ToolFunction defines the function details for a tool
type ToolFunction struct {
	Name        string
	Description string
	Parameters  *ToolFunctionParameters
}

// ToolFunctionParameters defines the parameters schema for a tool
type ToolFunctionParameters struct {
	Type       string
	Properties map[string]*ToolProperty
	Required   []string
}

// ToolProperty defines a single property in the tool schema
type ToolProperty struct {
	Type        string
	Description string
	Enum        []string
	Items       *ToolProperty
}

func toolSchemas(tools []Tool) []*ToolSchema {
	schemas := []*ToolSchema{}
	for _, t := range tools {
		schemas = append(schemas, t.Schema())
	}
	return schemas
}

// Func creates a typed tool with automatic JSON marshaling
func Func[In, Out any](name, description string, run func(ctx context.Context, in In) (Out, error)) Tool {
	return &typedFunc[In, Out]{
		name:        name,
		description: description,
		run:         run,
	}
}

// typedFunc wraps a typed function as a Tool
type typedFunc[In, Out any] struct {
	name        string
	description string
	run         func(ctx context.Context, in In) (Out, error)
}

func (t *typedFunc[In, Out]) Name() string        { return t.name }
func (t *typedFunc[In, Out]) Description() string { return t.description }

func (t *typedFunc[In, Out]) Schema() *ToolSchema {
	var in In
	return &ToolSchema{
		Type: "function",
		Function: &ToolFunction{
			Name:        t.name,
			Description: t.description,
			Parameters:  generateSchema(in),
		},
	}
}

func (t *typedFunc[In, Out]) Run(ctx context.Context, args json.RawMessage) ([]byte, error) {
	var in In
	if len(args) > 0 {
		if err := json.Unmarshal(args, &in); err != nil {
			return nil, fmt.Errorf("tool %s: unmarshaling input: %w", t.name, err)
		}
	}
	out, err := t.run(ctx, in)
	if err != nil {
		return nil, err
	}
	return json.Marshal(out)
}

// generateSchema creates ToolFunctionParameters from a struct type
// Supported struct tags:
//   - `json:"fieldname"` - JSON field name
//   - `description:"text"` - field description for the schema
//   - `enums:"a,b,c"` - allowed values (comma-separated)
//   - `is:"required"` - marks field as required (presence only, no value)
func generateSchema(v any) *ToolFunctionParameters {
	params := &ToolFunctionParameters{
		Type:       "object",
		Properties: make(map[string]*ToolProperty),
		Required:   []string{},
	}

	t := reflect.TypeOf(v)
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	if t.Kind() != reflect.Struct {
		return params
	}

	for i := range t.NumField() {
		field := t.Field(i)
		if !field.IsExported() {
			continue
		}

		// Get JSON field name
		name := field.Name
		if jsonTag := field.Tag.Get("json"); jsonTag != "" {
			parts := strings.Split(jsonTag, ",")
			if parts[0] != "" && parts[0] != "-" {
				name = parts[0]
			}
		}

		// Get description
		description := field.Tag.Get("description")

		// Get enums
		var enums []string
		if enumTag := field.Tag.Get("enums"); enumTag != "" {
			enums = strings.Split(enumTag, ",")
		}

		prop := schemaType(field.Type)
		prop.Description = description
		prop.Enum = enums
		params.Properties[name] = prop

		// Check if required
		if field.Tag.Get("is") == "required" {
			params.Required = append(params.Required, name)
		}
	}

	return params
}

func schemaType(t reflect.Type) *ToolProperty {
	for t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	prop := &ToolProperty{Type: "string"}
	switch t.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		prop.Type = "integer"
	case reflect.Float32, reflect.Float64:
		prop.Type = "number"
	case reflect.Bool:
		prop.Type = "boolean"
	case reflect.Slice, reflect.Array:
		prop.Type = "array"
		prop.Items = schemaType(t.Elem())
	case reflect.Struct, reflect.Map:
		prop.Type = "object"
	}

	return prop
}
