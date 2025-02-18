/* Package repositories provides the implementation of the repositories interfaces. */
package repositories

import (
	"context"
	"fmt"
	"rest_api_go/models"
	"strconv"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	_ "go.mongodb.org/mongo-driver/mongo/readpref" // Import readpref for MongoDB
)

// DeviceRepository is an interface that defines the methods that a device repository should implement.
type DeviceRepository interface {
	GetAllDevices() ([]models.Device, error)
	GetDeviceByID(id string) (*models.Device, error)
	CreateDevice(dev models.Device) (string, error)
	UpdateDevice(dev models.Device) error
	PatchDevice(id string, updates map[string]interface{}) error
	DeleteDevice(id string) error
}

type DeviceRepo struct {
	DB         *mongo.Client
	database   string
	collection *mongo.Collection
}

func NewDeviceRepository(db *mongo.Client) *DeviceRepo { // Renamed function
	database := "devices"
	collectionName := "devices"

	collection := db.Database(database).Collection(collectionName)

	return &DeviceRepo{
		DB:         db,
		database:   database,
		collection: collection,
	}
}

func (repo *DeviceRepo) GetAllDevices() ([]models.Device, error) {
	var devices []models.Device

	cursor, err := repo.collection.Find(context.Background(), bson.M{})
	if err != nil {
		return nil, err
	}
	defer cursor.Close(context.Background())

	if err := cursor.All(context.Background(), &devices); err != nil {
		return nil, err
	}

	return devices, nil
}

func (repo *DeviceRepo) GetDeviceByID(id string) (*models.Device, error) {
	objID, err := primitive.ObjectIDFromHex(id)
	if err != nil {
		return nil, fmt.Errorf("invalid device ID: %v", err)
	}
	filter := bson.M{"_id": objID}

	var dev models.Device
	err = repo.collection.FindOne(context.Background(), filter).Decode(&dev)
	if err != nil {
		if err == mongo.ErrNoDocuments {
			return nil, fmt.Errorf("device with ID %s not found", id)
		}
		return nil, err
	}
	return &dev, nil
}

func (repo *DeviceRepo) CreateDevice(dev models.Device) (string, error) {
	result, err := repo.collection.InsertOne(context.Background(), dev)
	if err != nil {
		return "", err
	}

	// Handle ObjectID (most common case):
	objectID, ok := result.InsertedID.(primitive.ObjectID)
	if !ok {
		return "", fmt.Errorf("inserted ID is not an ObjectID")
	}
	// dev.ID = objectID.Hex()

	return objectID.Hex(), nil
}

func (repo *DeviceRepo) UpdateDevice(dev models.Device) error {

	objID, err := primitive.ObjectIDFromHex(strconv.Itoa(dev.ID))

	if err != nil {
		return fmt.Errorf("invalid device ID: %v", err)
	}
	filter := bson.M{"_id": objID}

	update := bson.M{"$set": dev}

	_, err = repo.collection.UpdateOne(context.Background(), filter, update)
	return err
}

func (repo *DeviceRepo) PatchDevice(id string, updates map[string]interface{}) error {
	if len(updates) == 0 {
		return fmt.Errorf("no fields to update")
	}
	objID, err := primitive.ObjectIDFromHex(id)
	if err != nil {
		return fmt.Errorf("invalid device ID: %v", err)
	}
	filter := bson.M{"_id": objID}

	update := bson.M{"$set": updates}

	_, err = repo.collection.UpdateOne(context.Background(), filter, update)
	return err
}

func (repo *DeviceRepo) DeleteDevice(id string) error {
	objID, err := primitive.ObjectIDFromHex(id)
	if err != nil {
		return fmt.Errorf("invalid device ID: %v", err)
	}
	filter := bson.M{"_id": objID}

	result, err := repo.collection.DeleteOne(context.Background(), filter)
	if err != nil {
		return err
	}

	if result.DeletedCount == 0 {
		return fmt.Errorf("device with ID %s not found", id)
	}

	return nil
}
