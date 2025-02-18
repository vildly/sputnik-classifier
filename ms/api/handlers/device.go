/*
Package handlers provides the endpoints for the REST API. Each handler function
is responsible for handling a specific HTTP request. The handlers package also contains
the SetDeviceService function, which allows the main package to inject the device service into the handlers.
*/
package handlers

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"rest_api_go/models"
	"rest_api_go/services"
	"strconv"

	"github.com/gorilla/mux"
)

var deviceService *services.DeviceService
var ErrDeviceNotFound = errors.New("device not found")

// Inject the device service into the handlers
func SetDeviceService(service *services.DeviceService) {
	deviceService = service
}

// GetDevicesHandler retrieves all devices
func GetDevicesHandler(w http.ResponseWriter, r *http.Request) {
	devices, err := deviceService.GetDevices()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	for i := range devices {
		devices[i].Links = []models.Link{
			{Rel: "self", Href: fmt.Sprintf("/devices/%d", devices[i].ID)},
			{Rel: "update", Href: fmt.Sprintf("/devices/%d", devices[i].ID)},
			{Rel: "delete", Href: fmt.Sprintf("/devices/%d", devices[i].ID)},
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(devices)
}

// GetDeviceHandler retrieves a device by its ID
func GetDeviceHandler(w http.ResponseWriter, r *http.Request) {

	vars := mux.Vars(r)
	id := vars["id"]

	device, err := deviceService.GetDevice(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	device.Links = []models.Link{
		{Rel: "self", Href: fmt.Sprintf("/devices/%d", device.ID)},
		{Rel: "update", Href: fmt.Sprintf("/devices/%d", device.ID)},
		{Rel: "delete", Href: fmt.Sprintf("/devices/%d", device.ID)},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(device)
}

// UpdateDeviceHandler updates a device
func UpdateDeviceHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	idStr := vars["id"]
	id, err := strconv.Atoi(idStr)
	if err != nil {
		http.Error(w, "Invalid device ID", http.StatusBadRequest)
		return
	}
	var dev models.Device

	if err := json.NewDecoder(r.Body).Decode(&dev); err != nil {
		http.Error(w, "Invalid request payload", http.StatusBadRequest)
		return
	}
	dev.ID = id
	err = deviceService.UpdateDevice(dev)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.WriteHeader(http.StatusNoContent) // 204 No Content
}

func PatchDeviceHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"] // id is already a string

	// Decode the request body into a map
	var updates map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&updates); err != nil {
		http.Error(w, "Invalid request payload", http.StatusBadRequest)
		return
	}

	// Validate updates
	if len(updates) == 0 {
		http.Error(w, "No fields to update", http.StatusBadRequest)
		return
	}

	// Call the update method on the service
	err := deviceService.PatchDevice(id, updates)
	if err != nil {
		if errors.Is(err, ErrDeviceNotFound) {
			http.Error(w, "Device not found", http.StatusNotFound)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusNoContent) // 204 No Content
}

// CreateDeviceHandler creates a new device
func CreateDeviceHandler(w http.ResponseWriter, r *http.Request) {
	var dev models.Device

	// Decode the JSON request body into a Device struct
	if err := json.NewDecoder(r.Body).Decode(&dev); err != nil {
		http.Error(w, "Invalid request payload", http.StatusBadRequest)
		return
	}

	// Insert the new device into the database
	id, err := deviceService.CreateDevice(dev)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Set the Location header for the newly created resource
	w.Header().Set("Location", fmt.Sprintf("/devices/%s", id))
	w.WriteHeader(http.StatusCreated) // 201 Created

	message := fmt.Sprintf("Device with id: %s has been created", id)
	w.Write([]byte(message))
}

func DeleteDeviceHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	// Declare err using the short assignment operator
	if err := deviceService.DeleteDevice(id); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.WriteHeader(http.StatusNoContent) // 204 No Content
}
