package main

import (
	"fmt"
	"io"
	"log"
	"net/http"

	"gopp/datacleaner"
)

func dataCleanerHandler(w http.ResponseWriter, r *http.Request) {
	// Close the request body when done.
	defer r.Body.Close()

	// Read the full request body.
	rawBody, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "unable to read request", http.StatusBadRequest)
		return
	}

	// Specify which keys you want to keep in the cleaned JSON.
	// Adjust this as necessary for your use case.
	keysToKeep := []string{"title", "description"}

	// Clean and filter the JSON using the new CleanJSONFiltered function.
	cleanedJSON, err := datacleaner.CleanJSONFiltered(rawBody, keysToKeep)
	if err != nil {
		http.Error(w, "error cleaning JSON data", http.StatusInternalServerError)
		return
	}

	// Set appropriate header and write the cleaned JSON as the response.
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
