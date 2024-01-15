provider "google" {
  region = "europe-west2"
  project = "dtumlops-410010"
}

resource "google_storage_bucket" "b" {
  name = "mlops-dvc-435"
  location = "europe-west2"
}

resource "google_service_account" "dvc_service_account" {   
    account_id = "dvc-service-account"
    display_name = "DVC Service Account"
}

# assign admin role
resource "google_project_iam_member" "service_account_admin" {
  project = "dtumlops-410010"
  role    = "roles/iam.serviceAccountAdmin"
  member  = "serviceAccount:${google_service_account.dvc_service_account.email}"
}

resource "google_service_account_key" "dvc_service_account_key" {
    service_account_id = google_service_account.dvc_service_account.name
}

output "secret" {
    sensitive = true
    value = google_service_account_key.dvc_service_account_key.private_key
}
