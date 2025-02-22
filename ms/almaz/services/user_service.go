package services

import (
	"rest_api_go/repositories"
)

type UserService struct {
	repo repositories.UserRepository
}

func NewUserService(repo repositories.UserRepository) *UserService {
	return &UserService{repo: repo}
}

func (service *UserService) Authenticate(username, password string) (bool, error) {
	return service.repo.ValidateUser(username, password)
}

func (service *UserService) CreateUser(username, password string) (bool, error) {
	return service.repo.CreateUser(username, password)
}

func (service *UserService) DeleteUser(username string) (bool, error) {
	return service.repo.DeleteUser(username)
}
