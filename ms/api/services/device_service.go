package services

import (
	"rest_api_go/models"
	"rest_api_go/repositories"
)

type DeviceService struct {
	repo repositories.DeviceRepository
}

// NewDeviceService creates a new device service with the given repository.
func NewDeviceService(repo repositories.DeviceRepository) *DeviceService {
	return &DeviceService{repo: repo}
}

func (service *DeviceService) GetDevices() ([]*models.Device, error) {
	devices, err := service.repo.GetAllDevices()
	if err != nil {
		return nil, err
	}

	var devicePtrs []*models.Device
	for _, device := range devices {
		deviceCopy := device
		devicePtrs = append(devicePtrs, &deviceCopy)
	}
	return devicePtrs, nil
}

func (service *DeviceService) GetDevice(id string) (*models.Device, error) {
	return service.repo.GetDeviceByID(id)
}

func (service *DeviceService) UpdateDevice(dev models.Device) error {
	return service.repo.UpdateDevice(dev)
}

func (service *DeviceService) CreateDevice(dev models.Device) (string, error) {
	return service.repo.CreateDevice(dev)
}

func (service *DeviceService) DeleteDevice(id string) error {
	return service.repo.DeleteDevice(id)
}

func (service *DeviceService) PatchDevice(id string, updates map[string]interface{}) error {
	return service.repo.PatchDevice(id, updates)
}
