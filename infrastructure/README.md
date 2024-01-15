# Terraform GCP Infrastructure

## GCP Setup
1. Install `gcloud`-CLI
2. Login to `gcloud`-CLI as described [here](https://dvc.org/doc/user-guide/data-management/remote-storage/google-cloud-storage)

## Parts
- `gcp_dvc_bucket.tf` - DVC Bucket Infrastructure

## Initialize project
`cd` to the `infrastructure` foldern and run the following command.
```bash
terraform init
```

## Deploy Infrastructure
First run the plan command, to check if everything will change as wished.

```bash
terraform plan
```

and next run the terraform apply command, to actually update the GCP infrastructure.

```bash
terraform apply
```

## Destroy Infrastructure
In the end, if you no more need the infrastructure it can be destroyed by
```bash
terraform destroy
```


## Resources:
- https://github.com/ktrnka/mlops_example_dvc