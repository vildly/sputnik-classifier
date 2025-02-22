package services

import (
	"rest_api_go/repositories"
)

// DataService provides methods to work with data using the underlying repository.
type DataService struct {
	repo repositories.DataRepository
}

// NewDataService creates a new data service with the given repository.
func NewDataService(repo repositories.DataRepository) *DataService {
	return &DataService{repo: repo}
}

// GetData retrieves all data documents.
func (service *DataService) GetData() ([]map[string]interface{}, error) {
	return service.repo.GetAllData()
}

// GetDataByID retrieves a single data document by its ID.
func (service *DataService) GetDataByID(id string) (map[string]interface{}, error) {
	return service.repo.GetDataByID(id)
}

// UpdateData updates an entire data document with the provided values.
func (service *DataService) UpdateData(id string, data map[string]interface{}) error {
	return service.repo.UpdateData(id, data)
}

// CreateData inserts a new data document and returns its generated ID.
func (service *DataService) CreateData(data map[string]interface{}) (string, error) {
	return service.repo.InsertData(data)
}

// DeleteData removes a data document identified by id.
func (service *DataService) DeleteData(id string) error {
	return service.repo.DeleteData(id)
}

// PatchData updates specific fields in a data document.
func (service *DataService) PatchData(id string, updates map[string]interface{}) error {
	return service.repo.PatchData(id, updates)
}
