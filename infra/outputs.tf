output "acr_login_server" {
  description = "Registry host for image tags."
  value       = azurerm_container_registry.app.login_server
}

output "environment_mode" {
  description = "Must be WorkloadProfiles; Express can't bind custom domains."
  value       = azapi_resource.environment.output.mode
}

output "domain_verification_id" {
  description = "Value for the asuid.<subdomain> TXT record."
  value       = azapi_resource.environment.output.verification_id
}

output "app_url" {
  description = "Public service address."
  value       = "https://${azurerm_container_app.app.ingress[0].fqdn}"
}
