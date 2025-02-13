package datacleaner

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// CleanFiltered recursively processes input data using the allowed keys.
// - For strings, it returns the trimmed string.
// - For arrays, it processes each element recursively.
// - For maps, it concatenates the cleaned values of allowed keys into one string.
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
		var tokens []string
		for _, key := range allowedKeys {
			if val, exists := value[key]; exists {
				cleaned := CleanFiltered(val, allowedKeys)
				// Convert the value to a string.
				token := fmt.Sprintf("%v", cleaned)
				if token != "" {
					tokens = append(tokens, token)
				}
			}
		}
		// Concatenate all token strings together.
		return strings.TrimSpace(strings.Join(tokens, " "))
	default:
		return value
	}
}

// unmarshalJSONArray decodes raw JSON bytes directly into a []interface{}.
// It returns an error if the provided raw JSON is not a valid array.
func unmarshalJSONArray(raw []byte) ([]interface{}, error) {
	var arr []interface{}
	if err := json.Unmarshal(raw, &arr); err != nil {
		return nil, fmt.Errorf("invalid JSON array: expected format [ {\"key\": \"value\"}, ... ] | underlying error: %w", err)
	}
	return arr, nil
}

// marshalJSON encodes the data back into JSON bytes.
func marshalJSON(data interface{}) ([]byte, error) {
	return json.Marshal(data)
}

// CleanJSONFiltered accepts raw JSON bytes and a slice of allowed keys.
// It processes each object in the top-level JSON array by concatenating values
// corresponding to the allowed keys into a single string. If no allowed keys are provided,
// it returns an error.
func CleanJSONFiltered(rawJSON []byte, allowedKeys []string) ([]byte, error) {
	if len(allowedKeys) == 0 {
		return nil, errors.New("error: keys parameter is required")
	}

	arr, err := unmarshalJSONArray(rawJSON)
	if err != nil {
		return nil, err
	}

	// Process each element of the array.
	cleanedData := CleanFiltered(arr, allowedKeys)
	return marshalJSON(cleanedData)
}
