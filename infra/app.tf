resource "azurerm_log_analytics_workspace" "app" {
  name                = "log-face-recognition-serving"
  resource_group_name = azurerm_resource_group.app.name
  location            = azurerm_resource_group.app.location
  sku                 = "PerGB2018"
  retention_in_days   = 30
  daily_quota_gb      = 0.5 # caps ingestion cost; logs stop for the day past this

  depends_on = [azurerm_resource_provider_registration.required]
}

# A user-assigned identity can be granted AcrPull before the app exists,
# so the first revision can already pull its image.
resource "azurerm_user_assigned_identity" "app" {
  name                = "id-face-recognition-serving"
  resource_group_name = azurerm_resource_group.app.name
  location            = azurerm_resource_group.app.location

  depends_on = [azurerm_resource_provider_registration.required]
}

resource "azurerm_role_assignment" "acr_pull" {
  scope                = azurerm_container_registry.app.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.app.principal_id
  principal_type       = "ServicePrincipal"
}

# Azure creates a workload profiles environment with this Consumption profile.
# Plan management is charged only when a Dedicated profile exists, so none is added.
resource "azurerm_container_app_environment" "app" {
  name                       = "cae-face-recognition-serving"
  resource_group_name        = azurerm_resource_group.app.name
  location                   = azurerm_resource_group.app.location
  log_analytics_workspace_id = azurerm_log_analytics_workspace.app.id

  workload_profile {
    name                  = "Consumption"
    workload_profile_type = "Consumption"
  }
}

resource "azurerm_container_app" "app" {
  name                         = "ca-face-recognition-serving"
  resource_group_name          = azurerm_resource_group.app.name
  container_app_environment_id = azurerm_container_app_environment.app.id
  revision_mode                = "Single"
  workload_profile_name        = "Consumption"

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.app.id]
  }

  registry {
    server   = azurerm_container_registry.app.login_server
    identity = azurerm_user_assigned_identity.app.id
  }

  template {
    min_replicas = 0 # nothing is billed while no requests arrive
    max_replicas = 1

    container {
      name   = "api"
      image  = "${azurerm_container_registry.app.login_server}/face-recognition-serving:${var.image_tag}"
      cpu    = 1
      memory = "2Gi"

      # Startup loads TensorFlow and three models.
      startup_probe {
        transport               = "HTTP"
        port                    = 8000
        path                    = "/health"
        interval_seconds        = 5
        failure_count_threshold = 24
      }

      liveness_probe {
        transport = "HTTP"
        port      = 8000
        path      = "/health"
      }

      readiness_probe {
        transport = "HTTP"
        port      = 8000
        path      = "/health"
      }
    }
  }

  ingress {
    external_enabled           = true
    target_port                = 8000
    allow_insecure_connections = false

    # Public demo: max_replicas = 1 is what caps cost under unwanted traffic.
    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  depends_on = [azurerm_role_assignment.acr_pull]
}
