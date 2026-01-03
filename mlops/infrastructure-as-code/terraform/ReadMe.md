# Terraform

## Summary

Terraform is an infrastructure as code tool that enables declarative provisioning of cloud resources across multiple providers. For ML teams, Terraform manages compute instances, storage buckets, networking, Kubernetes clusters, and the cloud services that underpin training and serving infrastructure.

Key points to remember:

- Terraform uses HCL (HashiCorp Configuration Language) for declarative resource definitions
- State files track the mapping between configuration and real infrastructure
- Providers enable multi-cloud resource management with a consistent interface
- Modules encapsulate reusable infrastructure patterns
- Plan-apply workflow shows changes before execution

## Core Concepts

### Resources and Providers

Providers connect Terraform to cloud platforms:

```hcl
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}
```

Resources define infrastructure:

```hcl
resource "aws_s3_bucket" "ml_artifacts" {
  bucket = "my-ml-artifacts"

  tags = {
    Environment = "production"
    Team        = "ml-platform"
  }
}

resource "aws_s3_bucket_versioning" "ml_artifacts" {
  bucket = aws_s3_bucket.ml_artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}
```

### State Management

Terraform tracks infrastructure state:

```hcl
# Backend configuration for remote state
terraform {
  backend "s3" {
    bucket         = "my-terraform-state"
    key            = "ml-platform/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}
```

State enables:
- Tracking resource identifiers
- Detecting configuration drift
- Planning changes before applying
- Collaborative infrastructure management

### Variables and Outputs

Input variables parameterize configurations:

```hcl
variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "development"
}

variable "gpu_instance_types" {
  description = "Allowed GPU instance types"
  type        = list(string)
  default     = ["p3.2xlarge", "p3.8xlarge", "p4d.24xlarge"]
}

variable "training_config" {
  description = "Training infrastructure configuration"
  type = object({
    min_nodes = number
    max_nodes = number
    disk_size = number
  })
}
```

Outputs export values:

```hcl
output "bucket_arn" {
  description = "ARN of the ML artifacts bucket"
  value       = aws_s3_bucket.ml_artifacts.arn
}

output "eks_cluster_endpoint" {
  description = "Kubernetes API endpoint"
  value       = aws_eks_cluster.ml_cluster.endpoint
}
```

## ML Infrastructure Patterns

### Training Cluster

EKS cluster for training workloads:

```hcl
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = "ml-training-cluster"
  cluster_version = "1.28"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  eks_managed_node_groups = {
    cpu_nodes = {
      min_size     = 2
      max_size     = 10
      desired_size = 3

      instance_types = ["m5.2xlarge"]
      capacity_type  = "ON_DEMAND"

      labels = {
        workload = "general"
      }
    }

    gpu_nodes = {
      min_size     = 0
      max_size     = 8
      desired_size = 0

      instance_types = ["p3.2xlarge"]
      capacity_type  = "SPOT"

      labels = {
        workload = "training"
        gpu      = "true"
      }

      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }
  }
}
```

### Storage Infrastructure

S3 buckets with lifecycle policies:

```hcl
resource "aws_s3_bucket" "training_data" {
  bucket = "ml-training-data-${var.environment}"
}

resource "aws_s3_bucket_lifecycle_configuration" "training_data" {
  bucket = aws_s3_bucket.training_data.id

  rule {
    id     = "archive-old-checkpoints"
    status = "Enabled"

    filter {
      prefix = "checkpoints/"
    }

    transition {
      days          = 30
      storage_class = "GLACIER"
    }

    expiration {
      days = 365
    }
  }

  rule {
    id     = "cleanup-temp"
    status = "Enabled"

    filter {
      prefix = "temp/"
    }

    expiration {
      days = 7
    }
  }
}
```

### SageMaker Resources

SageMaker infrastructure:

```hcl
resource "aws_sagemaker_domain" "ml_domain" {
  domain_name = "ml-${var.environment}"
  auth_mode   = "IAM"
  vpc_id      = module.vpc.vpc_id
  subnet_ids  = module.vpc.private_subnets

  default_user_settings {
    execution_role = aws_iam_role.sagemaker_execution.arn

    jupyter_server_app_settings {
      default_resource_spec {
        instance_type = "ml.t3.medium"
      }
    }
  }
}

resource "aws_sagemaker_endpoint_configuration" "model_endpoint" {
  name = "fraud-model-config"

  production_variants {
    variant_name           = "primary"
    model_name             = aws_sagemaker_model.fraud_model.name
    initial_instance_count = 2
    instance_type          = "ml.m5.xlarge"
  }
}
```

### IAM for ML

Least-privilege IAM roles:

```hcl
resource "aws_iam_role" "training_role" {
  name = "ml-training-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "sagemaker.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "training_policy" {
  name = "training-policy"
  role = aws_iam_role.training_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.training_data.arn,
          "${aws_s3_bucket.training_data.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      }
    ]
  })
}
```

## Modules

### Creating Modules

Reusable ML infrastructure module:

```hcl
# modules/ml-training-env/main.tf

variable "environment" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "subnet_ids" {
  type = list(string)
}

resource "aws_s3_bucket" "artifacts" {
  bucket = "ml-artifacts-${var.environment}"
}

resource "aws_ecr_repository" "training" {
  name = "ml-training-${var.environment}"
}

output "artifacts_bucket" {
  value = aws_s3_bucket.artifacts.id
}

output "ecr_repository" {
  value = aws_ecr_repository.training.repository_url
}
```

Using the module:

```hcl
module "ml_dev" {
  source = "./modules/ml-training-env"

  environment = "development"
  vpc_id      = module.vpc.vpc_id
  subnet_ids  = module.vpc.private_subnets
}

module "ml_prod" {
  source = "./modules/ml-training-env"

  environment = "production"
  vpc_id      = module.vpc.vpc_id
  subnet_ids  = module.vpc.private_subnets
}
```

### Public Modules

Use community modules:

```hcl
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "ml-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["us-east-1a", "us-east-1b", "us-east-1c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = var.environment != "production"
}
```

## Workflow

### Plan and Apply

```bash
# Initialize working directory
terraform init

# Preview changes
terraform plan -out=tfplan

# Apply changes
terraform apply tfplan

# Destroy resources
terraform destroy
```

### Workspaces

Manage multiple environments:

```bash
# Create workspaces
terraform workspace new development
terraform workspace new production

# Switch workspace
terraform workspace select production

# List workspaces
terraform workspace list
```

Use workspace in configuration:

```hcl
locals {
  environment = terraform.workspace

  instance_count = {
    development = 1
    production  = 3
  }
}

resource "aws_instance" "training" {
  count         = local.instance_count[local.environment]
  instance_type = "p3.2xlarge"
}
```

## Best Practices

### State Management

- Use remote state with locking (S3 + DynamoDB)
- Enable state encryption
- Restrict state access with IAM
- Use separate state files per environment

### Code Organization

```
terraform/
  modules/
    ml-training-env/
    ml-serving-env/
  environments/
    development/
      main.tf
      variables.tf
      terraform.tfvars
    production/
      main.tf
      variables.tf
      terraform.tfvars
  global/
    iam/
    networking/
```

### Security

- Never commit secrets to version control
- Use AWS Secrets Manager or SSM Parameter Store
- Apply least-privilege IAM policies
- Enable resource tagging for cost tracking
- Use security groups with minimal access

### CI/CD Integration

```yaml
# GitHub Actions example
name: Terraform
on:
  push:
    branches: [main]
  pull_request:

jobs:
  terraform:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3

      - name: Terraform Init
        run: terraform init

      - name: Terraform Plan
        run: terraform plan -out=tfplan

      - name: Terraform Apply
        if: github.ref == 'refs/heads/main'
        run: terraform apply -auto-approve tfplan
```

## Common Pitfalls

### State Conflicts

Multiple engineers applying simultaneously causes conflicts. Use state locking and coordinate changes.

### Drift

Manual changes create drift between state and reality. Use `terraform plan` regularly to detect drift.

### Secrets in State

State files contain sensitive data. Encrypt state and restrict access.

### Large State Files

Monolithic configurations create large state files. Split into smaller, focused modules.

### Ignoring Costs

Terraform makes it easy to provision expensive resources. Use cost estimation tools and set up alerts.
