// main.go
package main

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"

	"gopp/datacleaner"
)

// parseKeys splits a comma-separated string and returns a slice of trimmed keys.
func parseKeys(keysStr string) []string {
	if keysStr == "" {
		return nil
	}
	parts := strings.Split(keysStr, ",")
	var keys []string
	for _, part := range parts {
		trimmed := strings.TrimSpace(part)
		if trimmed != "" {
			keys = append(keys, trimmed)
		}
	}
	return keys
}

// readRequestBody reads the request body and returns its bytes.
func readRequestBody(r *http.Request) ([]byte, error) {
	defer r.Body.Close()
	return io.ReadAll(r.Body)
}

func dataCleanerHandler(w http.ResponseWriter, r *http.Request) {
	rawBody, err := readRequestBody(r)
	if err != nil {
		http.Error(w, "unable to read request", http.StatusBadRequest)
		return
	}

	// Extract and parse the "keys" query parameter.
	keysStr := r.URL.Query().Get("keys")
	keysToKeep := parseKeys(keysStr)

	// Clean and filter JSON using the datacleaner package.
	cleanedJSON, err := datacleaner.CleanJSONFiltered(rawBody, keysToKeep)
	if err != nil {
		http.Error(w, "error cleaning JSON data: "+err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	if _, err := w.Write(cleanedJSON); err != nil {
		http.Error(w, fmt.Sprintf("error writing response: %v", err), http.StatusInternalServerError)
		return
	}
}

func main() {
	http.HandleFunc("/clean", dataCleanerHandler)
	log.Println("Data Cleaner service running on port 8080...")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
