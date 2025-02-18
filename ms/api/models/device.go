// @Success 200 {array} models.Device
package models

type Link struct {
	Rel  string `json:"rel"`
	Href string `json:"href"`
}

type Device struct {
	ID      int    `json:"id"`
	Name    string `json:"name"`
	Version string `json:"version"`
	Links   []Link `json:"links,omitempty"`
}
