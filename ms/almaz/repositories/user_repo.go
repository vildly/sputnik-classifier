package repositories

import (
	"context"
	"crypto/sha256"
	"fmt"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	_ "go.mongodb.org/mongo-driver/mongo/readpref"
)

type UserRepository interface {
	ValidateUser(username, password string) (bool, error)
	CreateUser(username, password string) (bool, error)
	DeleteUser(username string) (bool, error)
}

type userRepo struct {
	DB         *mongo.Client
	database   string
	collection *mongo.Collection
}

func NewUserRepository(db *mongo.Client) UserRepository {
	database := "users"
	collectionName := "users"

	collection := db.Database(database).Collection(collectionName)
	return &userRepo{
		DB:         db,
		database:   database,
		collection: collection,
	}
}

func (repo *userRepo) ValidateUser(username, password string) (bool, error) {
	hashedPassword := hashPassword(password)

	filter := bson.M{"username": username, "password_hash": hashedPassword}
	count, err := repo.collection.CountDocuments(context.Background(), filter)
	if err != nil {
		return false, err
	}

	return count > 0, nil
}

func (repo *userRepo) CreateUser(username, password string) (bool, error) {
	hashedPassword := hashPassword(password)

	user := bson.M{"username": username, "password_hash": hashedPassword} // Use bson.M for MongoDB documents

	_, err := repo.collection.InsertOne(context.Background(), user)
	if err != nil {
		return false, err
	}
	return true, nil
}

func hashPassword(password string) string {
	return fmt.Sprintf("%x", sha256.Sum256([]byte(password)))
}

func (repo *userRepo) DeleteUser(username string) (bool, error) {
	filter := bson.M{"username": username}
	result, err := repo.collection.DeleteOne(context.Background(), filter)
	if err != nil {
		return false, err
	}

	return result.DeletedCount > 0, nil // Check if a document was actually deleted
}
