package handlers

import (
	"encoding/json"
	"net/http"
	"rest_api_go/auth"
	"rest_api_go/services"

	"github.com/gorilla/mux"
)

var userService *services.UserService

// SetUserService injects the user service into the handlers.
func SetUserService(service *services.UserService) {
	userService = service
}

// LoginHandler handles the login request.
//
// @Summary Login to get a new token
// @Description Authenticate user credentials and returns a JWT token if the credentials are valid.
// @Tags authentication
// @Accept application/x-www-form-urlencoded
// @Produce json
// @Param username formData string true "Username"
// @Param password formData string true "Password"
// @Success 200 {object} map[string]string "Token"
// @Failure 400 {object} map[string]string "Invalid credentials"
// @Failure 401 {object} map[string]string "Unauthorized"
// @Failure 500 {object} map[string]string "Internal server error"
// @Router /login [post]
func LoginHandler(w http.ResponseWriter, r *http.Request) {
	username := r.FormValue("username")
	password := r.FormValue("password")

	isValid, err := userService.Authenticate(username, password)
	if err != nil {
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}
	if !isValid {
		http.Error(w, "Invalid credentials", http.StatusUnauthorized)
		return
	}

	token, err := auth.GenerateJWT(username)
	if err != nil {
		http.Error(w, "Failed to generate token", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	resp := map[string]string{"token": token}
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		http.Error(w, "Failed to generate response", http.StatusInternalServerError)
		return
	}
}

// RegisterUserHandler handles user registration.
//
// @Summary Register a new user
// @Description Create a new user with a given username and password.
// @Tags users
// @Accept application/x-www-form-urlencoded
// @Produce json
// @Param username formData string true "Username"
// @Param password formData string true "Password"
// @Success 201 {object} map[string]string "Created"
// @Failure 500 {object} map[string]string "Internal server error"
// @Router /users [post]
func RegisterUserHandler(w http.ResponseWriter, r *http.Request) {
	username := r.FormValue("username")
	password := r.FormValue("password")

	created, err := userService.CreateUser(username, password)
	if err != nil {
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}
	if !created {
		http.Error(w, "Failed to create user", http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusCreated)
	resp := map[string]string{"message": "User created successfully"}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// RemoveUserHandler handles the deletion of a user.
//
// @Summary Delete a user
// @Description Delete an existing user by username.
// @Tags users
// @Produce json
// @Param username path string true "Username"
// @Success 204 {object} nil "No Content"
// @Failure 404 {object} map[string]string "User not found"
// @Failure 500 {object} map[string]string "Internal server error"
// @Router /users/{username} [delete]
func RemoveUserHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	username := vars["username"]

	success, err := userService.DeleteUser(username)
	if err != nil {
		http.Error(w, "Failed to delete user", http.StatusInternalServerError)
		return
	}
	if !success {
		http.Error(w, "User not found", http.StatusNotFound)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}
