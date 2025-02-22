/*
Package handlers provides the endpoints for the REST API. Each handler function
is responsible for handling a specific HTTP request. The handlers package also contains
the SetDataService function, which allows the main package to inject the data service into the handlers.
*/
package handlers

import (
	"encoding/json"
	"errors"
	"fmt"
	"github.com/gorilla/mux"
	"net/http"
	"rest_api_go/services"
)

var dataService *services.DataService
var ErrDataNotFound = errors.New("data not found")

// SetDataService injects the data service into the handlers.
func SetDataService(service *services.DataService) {
	dataService = service
}

// GetDataHandler retrieves all data documents.
func GetDataHandler(w http.ResponseWriter, r *http.Request) {
	dataItems, err := dataService.GetData()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(dataItems)
}

// GetDataByIDHandler retrieves a single data document by its ID.
func GetDataByIDHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	dataItem, err := dataService.GetDataByID(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(dataItem)
}

// UpdateDataHandler replaces an existing data document with the provided data.
func UpdateDataHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	var dataItem map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&dataItem); err != nil {
		http.Error(w, "Invalid request payload", http.StatusBadRequest)
		return
	}

	// Call the update method on the service layer.
	err := dataService.UpdateData(id, dataItem)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.WriteHeader(http.StatusNoContent) // 204 No Content
}

// PatchDataHandler updates specific fields in the data document.
func PatchDataHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	// Decode the request body into a map.
	var updates map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&updates); err != nil {
		http.Error(w, "Invalid request payload", http.StatusBadRequest)
		return
	}

	// Validate updates.
	if len(updates) == 0 {
		http.Error(w, "No fields to update", http.StatusBadRequest)
		return
	}

	// Call the patch method on the service layer.
	err := dataService.PatchData(id, updates)
	if err != nil {
		if errors.Is(err, ErrDataNotFound) {
			http.Error(w, "Data not found", http.StatusNotFound)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusNoContent) // 204 No Content
}

// CreateDataHandler creates a new data document.
func CreateDataHandler(w http.ResponseWriter, r *http.Request) {
	var dataItem map[string]interface{}

	// Decode the JSON request body into a map.
	if err := json.NewDecoder(r.Body).Decode(&dataItem); err != nil {
		http.Error(w, "Invalid request payload", http.StatusBadRequest)
		return
	}

	// Insert the new data document into the database.
	id, err := dataService.CreateData(dataItem)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Set the Location header for the newly created resource.
	w.Header().Set("Location", fmt.Sprintf("/data/%s", id))
	w.WriteHeader(http.StatusCreated) // 201 Created

	message := fmt.Sprintf("Data with id: %s has been created", id)
	w.Write([]byte(message))
}

// DeleteDataHandler deletes a data document.
func DeleteDataHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	if err := dataService.DeleteData(id); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.WriteHeader(http.StatusNoContent) // 204 No Content
}
