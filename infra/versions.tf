terraform {
  required_version = "~> 1.17"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
    # azurerm can't set the environment mode yet; see app.tf.
    azapi = {
      source  = "Azure/azapi"
      version = "~> 2.13"
    }
  }
}

# The subscription comes from ARM_SUBSCRIPTION_ID. Providers are registered
# explicitly in main.tf so that plans show every change to the subscription.
provider "azurerm" {
  features {}
  resource_provider_registrations = "none"
}
