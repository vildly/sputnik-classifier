/*
This package is responsible for validating the JWT token that is passed in the Authorization header of the request.
*/
package middleware

import (
	"net/http"
	"rest_api_go/auth"
	"strings"
)

func JWTMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authHeader := r.Header.Get("Authorization")
		if authHeader == "" {
			http.Error(w, "Authorization header is missing", http.StatusUnauthorized)
			return
		}

		tokenString := strings.TrimPrefix(authHeader, "Bearer ")
		_, err := auth.ValidateJWT(tokenString)
		if err != nil {
			http.Error(w, "Invalid token provided", http.StatusUnauthorized)
			return
		}

		next.ServeHTTP(w, r)
	})
}
