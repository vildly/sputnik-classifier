// datacleaner.go
// Package datacleaner provides functions to clean input data from JSON.
// In addition to trimming whitespace, this version also keeps only specified keys.
package datacleaner

import (
	"encoding/json"
	"strings"
)

// contains checks whether a given string is in a slice.
func contains(slice []string, item string) bool {
	for _, elem := range slice {
		if elem == item {
			return true
		}
	}
	return false
}

// CleanFiltered recursively cleans an input data structure.
// It trims whitespace from strings and, when encountering a map,
// only keeps the keys specified in keysToKeep. If a slice is encountered,
// each element is cleaned recursively.
func CleanFiltered(input interface{}, keysToKeep []string) interface{} {
	switch v := input.(type) {
	case string:
		// Clean string: trim whitespace.
		return strings.TrimSpace(v)
	case []interface{}:
		// Recursively clean each element in a slice.
		for i, elem := range v {
			v[i] = CleanFiltered(elem, keysToKeep)
		}
		return v
	case map[string]interface{}:
		// Create a new map and copy only keys that are in keysToKeep.
		newMap := make(map[string]interface{})
		for key, val := range v {
			if contains(keysToKeep, key) {
				newMap[key] = CleanFiltered(val, keysToKeep)
			}
		}
		return newMap
	default:
		// For all other types, return the value as is.
		return v
	}
}

// CleanJSONFiltered accepts a raw JSON byte slice, cleans it using CleanFiltered,
// and returns the cleaned JSON. The keysToKeep parameter directs which keys to retain.
func CleanJSONFiltered(rawJSON []byte, keysToKeep []string) ([]byte, error) {
	// Unmarshal the JSON into a generic data structure.
	var data interface{}
	if err := json.Unmarshal(rawJSON, &data); err != nil {
		return nil, err
	}
	// Clean and filter the data recursively.
	cleanedData := CleanFiltered(data, keysToKeep)
	// Marshal the cleaned data back to JSON.
	return json.Marshal(cleanedData)
}

