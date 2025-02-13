// datacleaner/datacleaner.go
// Package datacleaner provides functions to clean and filter JSON data.
package datacleaner

import (
	"encoding/json"
	"errors"
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

// CleanFiltered recursively cleans input data.
// It trims strings and, when encountering maps,
// only retains the keys specified in allowedKeys.
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

// unmarshalJSON is a helper to decode JSON into a generic interface.
func unmarshalJSON(raw []byte) (interface{}, error) {
	var data interface{}
	err := json.Unmarshal(raw, &data)
	if err != nil {
		return nil, err
	}
	return data, nil
}

// marshalJSON is a helper to encode data back into JSON.
func marshalJSON(data interface{}) ([]byte, error) {
	return json.Marshal(data)
}

// CleanJSONFiltered accepts raw JSON bytes and a slice of allowed keys.
// It returns filtered and cleaned JSON or an error if no keys are provided.
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

