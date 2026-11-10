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

# Created with azapi to pin environmentMode: left unset, Azure made an Express
# environment, and Express doesn't support custom domains. Only the Consumption
# profile is used; plan management is charged only for Dedicated profiles.
resource "azapi_resource" "environment" {
  type      = "Microsoft.App/managedEnvironments@2026-07-01"
  name      = "cae-face-recognition-serving"
  parent_id = azurerm_resource_group.app.id
  location  = azurerm_resource_group.app.location

  # azapi's embedded schema predates environmentMode (added in 2026-07-01);
  # Azure still validates the request.
  schema_validation_enabled = false

  body = {
    properties = {
      environmentMode = "WorkloadProfiles"
      workloadProfiles = [
        {
          name                = "Consumption"
          workloadProfileType = "Consumption"
        }
      ]
      appLogsConfiguration = {
        destination = "log-analytics"
        logAnalyticsConfiguration = {
          customerId = azurerm_log_analytics_workspace.app.workspace_id
        }
      }
    }
  }

  # Write-only: Azure never returns the key, so keep it out of the drift check.
  sensitive_body = {
    properties = {
      appLogsConfiguration = {
        logAnalyticsConfiguration = {
          sharedKey = azurerm_log_analytics_workspace.app.primary_shared_key
        }
      }
    }
  }

  response_export_values = {
    mode            = "properties.environmentMode"
    default_domain  = "properties.defaultDomain"
    verification_id = "properties.customDomainConfiguration.customDomainVerificationId"
  }
}

resource "azurerm_container_app" "app" {
  name                         = "ca-face-recognition-serving"
  resource_group_name          = azurerm_resource_group.app.name
  container_app_environment_id = azapi_resource.environment.id
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
