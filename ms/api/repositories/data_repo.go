/* Package repositories provides the implementation of the repository interfaces. */
package repositories

import (
	"context"
	"fmt"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
)

// DataRepository defines the methods that a data repository should implement.
type DataRepository interface {
	InsertData(data map[string]interface{}) (string, error)
	GetAllData() ([]map[string]interface{}, error)
	GetDataByID(id string) (map[string]interface{}, error)
	UpdateData(id string, data map[string]interface{}) error
	PatchData(id string, updates map[string]interface{}) error
	DeleteData(id string) error
}

// DataRepo is an implementation of DataRepository using MongoDB.
type DataRepo struct {
	DB         *mongo.Client
	database   string
	collection *mongo.Collection
}

// NewDataRepository returns a new instance of DataRepo.
// You can change the database and collection names if needed.
func NewDataRepository(db *mongo.Client) *DataRepo {
	database := "data"       // You might want to use a different database name here.
	collectionName := "data" // And a different collection if needed.

	collection := db.Database(database).Collection(collectionName)
	return &DataRepo{
		DB:         db,
		database:   database,
		collection: collection,
	}
}

// InsertData inserts the given JSON data into the collection and returns the generated ID.
func (repo *DataRepo) InsertData(data map[string]interface{}) (string, error) {
	result, err := repo.collection.InsertOne(context.Background(), data)
	if err != nil {
		return "", err
	}

	objID, ok := result.InsertedID.(primitive.ObjectID)
	if !ok {
		return "", fmt.Errorf("inserted ID is not an ObjectID")
	}
	return objID.Hex(), nil
}

// GetAllData returns all documents from the collection as a slice of maps.
func (repo *DataRepo) GetAllData() ([]map[string]interface{}, error) {
	var data []map[string]interface{}

	cursor, err := repo.collection.Find(context.Background(), bson.M{})
	if err != nil {
		return nil, err
	}
	defer cursor.Close(context.Background())

	if err := cursor.All(context.Background(), &data); err != nil {
		return nil, err
	}
	return data, nil
}

// GetDataByID finds a document by its ID.
func (repo *DataRepo) GetDataByID(id string) (map[string]interface{}, error) {
	objID, err := primitive.ObjectIDFromHex(id)
	if err != nil {
		return nil, fmt.Errorf("invalid data ID: %v", err)
	}
	filter := bson.M{"_id": objID}

	var result map[string]interface{}
	err = repo.collection.FindOne(context.Background(), filter).Decode(&result)
	if err != nil {
		if err == mongo.ErrNoDocuments {
			return nil, fmt.Errorf("data with ID %s not found", id)
		}
		return nil, err
	}
	return result, nil
}

// UpdateData replaces an existing document with the provided data.
// The provided data is assumed to contain the complete document (except for _id).
func (repo *DataRepo) UpdateData(id string, data map[string]interface{}) error {
	objID, err := primitive.ObjectIDFromHex(id)
	if err != nil {
		return fmt.Errorf("invalid data ID: %v", err)
	}

	// Ensure that _id is not modified.
	delete(data, "_id")

	filter := bson.M{"_id": objID}
	update := bson.M{"$set": data}

	_, err = repo.collection.UpdateOne(context.Background(), filter, update)
	return err
}

// PatchData updates specific fields of the document identified by id.
func (repo *DataRepo) PatchData(id string, updates map[string]interface{}) error {
	if len(updates) == 0 {
		return fmt.Errorf("no fields to update")
	}
	objID, err := primitive.ObjectIDFromHex(id)
	if err != nil {
		return fmt.Errorf("invalid data ID: %v", err)
	}
	filter := bson.M{"_id": objID}
	update := bson.M{"$set": updates}

	_, err = repo.collection.UpdateOne(context.Background(), filter, update)
	return err
}

// DeleteData removes the document identified by id from the collection.
func (repo *DataRepo) DeleteData(id string) error {
	objID, err := primitive.ObjectIDFromHex(id)
	if err != nil {
		return fmt.Errorf("invalid data ID: %v", err)
	}
	filter := bson.M{"_id": objID}
	result, err := repo.collection.DeleteOne(context.Background(), filter)
	if err != nil {
		return err
	}

	if result.DeletedCount == 0 {
		return fmt.Errorf("data with ID %s not found", id)
	}
	return nil
}
