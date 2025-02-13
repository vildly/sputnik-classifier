// datacleaner/datacleaner.go
package datacleaner

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// contains checks whether a given string exists in a slice.
func contains(slice []string, item string) bool {
	for _, elem := range slice {
		if elem == item {
			return true
		}
	}
	return false
}

// CleanFiltered recursively cleans input data,
// trimming strings and for maps, retaining only allowed keys.
func CleanFiltered(input interface{}, allowedKeys []string) interface{} {
	switch value := input.(type) {
	case string:
		return strings.TrimSpace(value)
	case []interface{}:
		for i, elem := range value {
			value[i] = CleanFiltered(elem, allowedKeys)
		}
		return value
	case map[string]interface{}:
		newMap := make(map[string]interface{})
		for key, val := range value {
			if contains(allowedKeys, key) {
				newMap[key] = CleanFiltered(val, allowedKeys)
			}
		}
		return newMap
	default:
		return value
	}
}

// unmarshalJSON decodes raw JSON bytes into a generic interface.
// It only accepts a top-level JSON array; otherwise, an error is returned.
func unmarshalJSON(raw []byte) (interface{}, error) {
	var data interface{}
	if err := json.Unmarshal(raw, &data); err != nil {
		return nil, fmt.Errorf(
			"invalid JSON format: only a JSON array is allowed ([ {\"key\": \"value\"}, ... ]) | underlying error: %w",
			err,
		)
	}

	arr, ok := data.([]interface{})
	if !ok {
		return nil, errors.New(
			"invalid JSON format: only a JSON array is allowed ([ {\"key\": \"value\"}, ... ])",
		)
	}
	return arr, nil
}

// marshalJSON encodes data back into JSON bytes.
func marshalJSON(data interface{}) ([]byte, error) {
	return json.Marshal(data)
}

// CleanJSONFiltered accepts raw JSON bytes and a slice of allowed keys,
// cleans the data, and returns filtered JSON or an error.
// It returns an error if no keys are provided.
func CleanJSONFiltered(rawJSON []byte, allowedKeys []string) ([]byte, error) {
	if len(allowedKeys) == 0 {
		return nil, errors.New("error: keys parameter is required")
	}

	data, err := unmarshalJSON(rawJSON)
	if err != nil {
		return nil, err
	}

	cleanedData := CleanFiltered(data, allowedKeys)
	return marshalJSON(cleanedData)
}
