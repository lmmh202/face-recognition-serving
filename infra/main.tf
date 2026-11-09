resource "azurerm_resource_provider_registration" "required" {
  for_each = toset([
    "Microsoft.App",
    "Microsoft.ContainerRegistry",
    "Microsoft.Network",
    "Microsoft.OperationalInsights",
  ])

  name = each.value
}

resource "azurerm_resource_group" "app" {
  name     = var.resource_group_name
  location = var.location
}
