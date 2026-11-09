resource "azurerm_container_registry" "app" {
  name                = var.acr_name
  resource_group_name = azurerm_resource_group.app.name
  location            = azurerm_resource_group.app.location
  sku                 = "Basic"
  admin_enabled       = false

  depends_on = [azurerm_resource_provider_registration.required]
}
