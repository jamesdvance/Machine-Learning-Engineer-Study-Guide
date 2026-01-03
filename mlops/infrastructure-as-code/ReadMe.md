# Infrastructure as Code for ML

## Summary

Infrastructure as Code (IaC) manages cloud infrastructure through declarative configuration files rather than manual provisioning. For ML teams, IaC enables reproducible training environments, consistent deployment infrastructure, and version-controlled infrastructure changes alongside model code.

Key points to remember:

- IaC brings software engineering practices (version control, code review, testing) to infrastructure
- Terraform is the most widely adopted multi-cloud IaC tool
- Pulumi uses real programming languages, appealing to Python-proficient ML teams
- CloudFormation provides native AWS integration without additional tooling
- Infrastructure should be as reproducible as model training code

## Why IaC for ML

ML infrastructure is complex and evolving:

Training infrastructure:
- GPU clusters that scale with demand
- Storage for datasets and model artifacts
- Networking for distributed training
- IAM roles with least-privilege access

Serving infrastructure:
- Kubernetes clusters for model deployment
- Load balancers and autoscaling
- Monitoring and logging
- Database connections for feature stores

Without IaC, teams face:
- Environment inconsistencies ("works on my cluster")
- Undocumented infrastructure changes
- Inability to reproduce training environments
- Difficulty scaling infrastructure
- Security and compliance gaps

## IaC Concepts

### Declarative vs Imperative

Declarative (Terraform, CloudFormation):
- Describe desired end state
- Tool determines how to reach that state
- Easier to reason about

Imperative (scripts, some Pulumi patterns):
- Describe steps to take
- More control over execution
- Harder to maintain

Most IaC tools favor declarative approaches, though Pulumi enables imperative patterns when needed.

### State Management

IaC tools track infrastructure state:

- What resources exist
- Resource identifiers
- Configuration values
- Dependencies between resources

State enables:
- Detecting drift from desired configuration
- Planning changes before applying
- Understanding resource relationships
- Enabling team collaboration

State storage options:
- Local files (development only)
- Remote backends (S3, GCS, Azure Blob)
- Managed services (Terraform Cloud, Pulumi Cloud)

### Idempotency

IaC operations should be idempotent:
- Running the same configuration multiple times produces the same result
- If infrastructure matches configuration, no changes occur
- Partial failures can be retried safely

Idempotency enables safe automation and recovery from failures.

## Tool Comparison

### Terraform

Industry standard for multi-cloud IaC:

Strengths:
- Largest ecosystem and community
- Works across all major clouds
- Mature and well-documented
- Strong module ecosystem

Considerations:
- HCL is a domain-specific language to learn
- State management requires careful handling
- Some resources have delayed provider support

Best for: Most teams, especially multi-cloud or cloud-agnostic needs.

### Pulumi

IaC with real programming languages:

Strengths:
- Python, TypeScript, Go support
- Full programming language features
- Better IDE support and testing
- Appeals to developer-focused teams

Considerations:
- Smaller community than Terraform
- Learning curve for IaC concepts
- Some providers less mature

Best for: Teams with strong programming skills who want infrastructure as code, literally.

### CloudFormation

AWS-native IaC:

Strengths:
- No additional tools needed
- Deep AWS service integration
- Built-in rollback
- AWS support included

Considerations:
- AWS-only
- YAML/JSON syntax verbose
- Slower feature updates than Terraform

Best for: AWS-only teams preferring native tooling.

### CDK (Cloud Development Kit)

AWS CDK and CDKTF offer programming language abstraction:

AWS CDK generates CloudFormation
CDKTF generates Terraform

Combines programming language benefits with underlying tool ecosystems.

## ML Infrastructure Patterns

### Reproducible Training Environments

Define complete training infrastructure:

```
training-infrastructure/
    vpc/              # Networking
    eks/              # Kubernetes cluster
    storage/          # S3 buckets, EBS volumes
    iam/              # Roles and policies
    sagemaker/        # SageMaker resources
```

Version control infrastructure alongside training code:

```
ml-project/
    src/              # Training code
    infrastructure/   # IaC definitions
    .github/          # CI/CD workflows
```

### Environment Parity

Maintain consistent infrastructure across environments:

```hcl
# Terraform workspaces or modules
module "ml_platform" {
  source = "./modules/ml-platform"

  environment    = terraform.workspace
  instance_type  = var.instance_types[terraform.workspace]
  min_nodes      = var.min_nodes[terraform.workspace]
}
```

Development, staging, and production use the same modules with different parameters.

### GPU Cluster Management

Define GPU node pools with IaC:

```yaml
# Kubernetes node pool definition
gpu_nodes:
  instance_types: ["p3.2xlarge", "p3.8xlarge"]
  min_size: 0  # Scale to zero when unused
  max_size: 16
  spot_enabled: true  # Cost optimization
  taints:
    - nvidia.com/gpu=true:NoSchedule
```

IaC enables consistent GPU cluster configuration across environments.

### Feature Store Infrastructure

Provision feature store dependencies:

- Redis/DynamoDB for online store
- S3/GCS for offline store
- Networking for low-latency access
- IAM for service-to-service auth

Infrastructure and feature store configuration evolve together.

## Workflow Integration

### CI/CD for Infrastructure

Automate infrastructure changes:

```yaml
# GitHub Actions example
name: Infrastructure

on:
  pull_request:
    paths:
      - 'infrastructure/**'
  push:
    branches: [main]

jobs:
  plan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Terraform Plan
        run: terraform plan -out=tfplan

  apply:
    needs: plan
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Terraform Apply
        run: terraform apply tfplan
```

Infrastructure changes follow the same review process as code.

### GitOps

Git as source of truth for infrastructure state:

1. Infrastructure defined in Git
2. Changes made through pull requests
3. Approved changes automatically applied
4. Drift detection and remediation

Tools like Argo CD and Flux enable GitOps for Kubernetes resources.

### Policy as Code

Enforce infrastructure policies:

```python
# Open Policy Agent example
deny[msg] {
    input.resource_type == "aws_s3_bucket"
    not input.versioning_enabled
    msg := "S3 buckets must have versioning enabled"
}

deny[msg] {
    input.resource_type == "aws_instance"
    input.instance_type == "p4d.24xlarge"
    input.environment != "production"
    msg := "Large GPU instances only allowed in production"
}
```

Policy enforcement prevents misconfigurations and cost overruns.

## Best Practices

### Modular Design

Create reusable modules:

```
modules/
    ml-training-env/
        main.tf
        variables.tf
        outputs.tf
    eks-cluster/
    sagemaker-domain/
```

Modules encapsulate complexity and ensure consistency.

### State Isolation

Separate state by:
- Environment (dev, staging, prod)
- Team or project
- Lifecycle (network vs application)

Isolation reduces blast radius and enables parallel work.

### Secrets Management

Never store secrets in IaC configurations:

```hcl
# Bad: hardcoded secret
resource "aws_db_instance" "db" {
  password = "mysecret123"  # Don't do this
}

# Good: reference secrets manager
data "aws_secretsmanager_secret_version" "db_password" {
  secret_id = "prod/db/password"
}

resource "aws_db_instance" "db" {
  password = data.aws_secretsmanager_secret_version.db_password.secret_string
}
```

### Documentation

Document infrastructure decisions:

```hcl
# GPU node pool configuration
#
# We use p3.2xlarge as default because:
# - Cost-effective for most training jobs
# - Available in all target regions
# - Sufficient memory for typical models
#
# For large model training, manually override to p3.8xlarge
# or p4d.24xlarge via Terraform variables.
```

### Cost Management

Tag resources for cost tracking:

```hcl
locals {
  common_tags = {
    Project     = "ml-platform"
    Environment = var.environment
    Team        = "ml-engineering"
    ManagedBy   = "terraform"
  }
}

resource "aws_instance" "training" {
  tags = merge(local.common_tags, {
    Name = "training-node"
  })
}
```

Tags enable cost allocation and unused resource identification.

## Common Pitfalls

### Manual Changes

Manual console changes create drift. Use IaC for all changes, even "quick fixes."

### Monolithic Configurations

Large single configurations are hard to manage. Split by concern and lifecycle.

### Ignoring Costs

IaC makes provisioning easy; it also makes over-provisioning easy. Monitor costs and right-size resources.

### Poor State Management

State corruption or loss causes major issues. Use remote backends with versioning and locking.

### Insufficient Testing

Test infrastructure changes in lower environments before production. Use tools like Terratest or Pulumi testing frameworks.

## Choosing a Tool

| Factor | Terraform | Pulumi | CloudFormation |
|--------|-----------|--------|----------------|
| Multi-cloud | Excellent | Good | AWS only |
| Language | HCL | Python, TS, Go | YAML/JSON |
| Community | Largest | Growing | AWS-focused |
| Learning curve | Moderate | Lower for devs | Moderate |
| AWS integration | Good | Good | Best |
| State management | Flexible | Flexible | Managed |

For most ML teams:
- Start with Terraform for broadest applicability
- Consider Pulumi if team is Python-centric
- Use CloudFormation if AWS-only and prefer native tooling
