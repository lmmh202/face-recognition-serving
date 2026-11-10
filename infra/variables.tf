variable "location" {
  description = "Azure region; the Azure for Students policy limits the choices."
  type        = string
  default     = "koreacentral"
}

variable "resource_group_name" {
  description = "Resource group holding everything in this deployment."
  type        = string
  default     = "rg-face-recognition-serving"
}

variable "image_tag" {
  description = "Image tag in the registry."
  type        = string
}

variable "acr_name" {
  description = "Container registry name: globally unique, alphanumeric."
  type        = string
  default     = "crfacerecognitionserving"
}
