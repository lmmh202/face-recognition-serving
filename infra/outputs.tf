output "acr_login_server" {
  description = "Registry host for image tags."
  value       = azurerm_container_registry.app.login_server
}

output "app_url" {
  description = "Public service address."
  value       = "https://${azurerm_container_app.app.ingress[0].fqdn}"
}
