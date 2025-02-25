# Define the base versions
# https://hub.docker.com/_/golang/tags
ARG GOLANG_BASE=1.24-alpine
# https://hub.docker.com/_/alpine/tags
ARG ALPINE_BASE=3.21.3

# BUILD STAGE
# #######################
FROM golang:$GOLANG_BASE AS builder

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

# FINAL STAGE
# #######################
# Run stage (final image)
FROM alpine:$ALPINE_BASE

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
