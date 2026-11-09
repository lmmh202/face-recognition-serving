output "acr_login_server" {
  description = "Registry host for image tags."
  value       = azurerm_container_registry.app.login_server
}
