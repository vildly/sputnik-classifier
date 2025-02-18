package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	_ "go.mongodb.org/mongo-driver/mongo/readpref" // Import readpref for MongoDB
)

func initDB() *mongo.Client {
	// MongoDB connection details from environment variables

	mongoURI := os.Getenv("MONGODB_URI") // e.g., "mongodb://user:password@host:port/database"
	fmt.Println("MONGODB_URI: ", mongoURI)
	if mongoURI == "" {
		log.Fatal("MONGODB_URI environment variable is not set")
	}

	// Set client options
	clientOptions := options.Client().ApplyURI(mongoURI)

	// Connect to MongoDB
	client, err := mongo.Connect(context.Background(), clientOptions)
	if err != nil {
		log.Fatalf("Error connecting to MongoDB: %v", err)
	}

	// Check the connection (optional but recommended)
	err = client.Ping(context.Background(), nil)
	if err != nil {
		log.Fatalf("Error pinging MongoDB: %v", err)
	}

	fmt.Println("Connected to MongoDB")
	return client
}
