# Stage 1: Build the Go application
FROM golang:1.23-alpine AS builder

# Set the working directory inside the container
WORKDIR /app

# Copy go.mod and go.sum files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy the source code
COPY . .

# Build the Go application
RUN go build -o main .

# Stage 2: Create a smaller base image for running the app
FROM alpine:latest

# Set the working directory
WORKDIR /root/

# Copy the binary from the builder stage to this stage
COPY --from=builder /app/main .

# Install PostgreSQL client (for pg_isready)
RUN apk add --no-cache postgresql-client

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /entrypoint.sh

# Expose the application on port 80
EXPOSE 80

# Use the entrypoint script to start the app
ENTRYPOINT ["/entrypoint.sh"]

# Default command to run the executable
CMD ["./main"]
