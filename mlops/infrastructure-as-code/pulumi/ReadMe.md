# Pulumi

## Summary

Pulumi is an infrastructure as code platform that enables defining cloud infrastructure using general-purpose programming languages like Python, TypeScript, Go, and C#. For ML teams comfortable with Python, Pulumi offers a natural way to manage infrastructure alongside ML code, leveraging familiar programming constructs and IDE support.

Key points to remember:

- Infrastructure is defined in real programming languages, not DSLs
- Python support makes it accessible to ML engineers
- Full programming language features: loops, conditionals, functions, classes
- Same state management concepts as Terraform
- Strong typing and IDE autocomplete improve developer experience

## Core Concepts

### Programs and Projects

A Pulumi program is code that defines infrastructure:

```python
# __main__.py
import pulumi
import pulumi_aws as aws

# Create an S3 bucket
bucket = aws.s3.Bucket("ml-artifacts",
    bucket="my-ml-artifacts",
    versioning=aws.s3.BucketVersioningArgs(
        enabled=True
    ),
    tags={
        "Environment": "production",
        "Team": "ml-platform"
    }
)

# Export the bucket name
pulumi.export("bucket_name", bucket.id)
pulumi.export("bucket_arn", bucket.arn)
```

Project configuration in Pulumi.yaml:

```yaml
name: ml-infrastructure
runtime: python
description: ML platform infrastructure
```

### Resources and Components

Create resources with full Python expressiveness:

```python
import pulumi
import pulumi_aws as aws

# Use Python for dynamic configuration
environments = ["dev", "staging", "prod"]
buckets = {}

for env in environments:
    bucket = aws.s3.Bucket(f"artifacts-{env}",
        bucket=f"ml-artifacts-{env}",
        tags={"Environment": env}
    )
    buckets[env] = bucket

# Conditional logic
if pulumi.get_stack() == "production":
    # Add extra configuration for production
    aws.s3.BucketPolicy("prod-policy",
        bucket=buckets["prod"].id,
        policy=production_policy_json
    )
```

### Stacks

Stacks represent environment instances:

```bash
# Create stacks
pulumi stack init development
pulumi stack init production

# Switch stacks
pulumi stack select production

# Configure stack-specific values
pulumi config set aws:region us-east-1
pulumi config set --secret dbPassword mySecret123
```

Access configuration in code:

```python
import pulumi

config = pulumi.Config()

# Stack name
stack = pulumi.get_stack()

# Configuration values
region = config.require("aws:region")
db_password = config.require_secret("dbPassword")
instance_count = config.get_int("instanceCount") or 3
```

## ML Infrastructure Patterns

### EKS Cluster for Training

```python
import pulumi
import pulumi_aws as aws
import pulumi_eks as eks

# VPC setup
vpc = aws.ec2.Vpc("ml-vpc",
    cidr_block="10.0.0.0/16",
    enable_dns_hostnames=True,
    tags={"Name": "ml-vpc"}
)

# Create subnets programmatically
azs = ["us-east-1a", "us-east-1b", "us-east-1c"]
private_subnets = []
public_subnets = []

for i, az in enumerate(azs):
    private_subnet = aws.ec2.Subnet(f"private-{az}",
        vpc_id=vpc.id,
        cidr_block=f"10.0.{i+1}.0/24",
        availability_zone=az,
        tags={"Name": f"private-{az}"}
    )
    private_subnets.append(private_subnet)

    public_subnet = aws.ec2.Subnet(f"public-{az}",
        vpc_id=vpc.id,
        cidr_block=f"10.0.{i+101}.0/24",
        availability_zone=az,
        map_public_ip_on_launch=True,
        tags={"Name": f"public-{az}"}
    )
    public_subnets.append(public_subnet)

# EKS cluster
cluster = eks.Cluster("ml-cluster",
    vpc_id=vpc.id,
    subnet_ids=[s.id for s in private_subnets],
    instance_type="m5.2xlarge",
    desired_capacity=3,
    min_size=2,
    max_size=10,
    node_associate_public_ip_address=False
)

# GPU node group
gpu_nodes = eks.ManagedNodeGroup("gpu-nodes",
    cluster=cluster,
    instance_types=["p3.2xlarge"],
    scaling_config=aws.eks.NodeGroupScalingConfigArgs(
        desired_size=0,
        min_size=0,
        max_size=8
    ),
    labels={"gpu": "true"},
    taints=[{
        "key": "nvidia.com/gpu",
        "value": "true",
        "effect": "NO_SCHEDULE"
    }]
)

pulumi.export("kubeconfig", cluster.kubeconfig)
```

### SageMaker Setup

```python
import pulumi
import pulumi_aws as aws

# Execution role
execution_role = aws.iam.Role("sagemaker-execution",
    assume_role_policy="""{
        "Version": "2012-10-17",
        "Statement": [{
            "Action": "sts:AssumeRole",
            "Effect": "Allow",
            "Principal": {
                "Service": "sagemaker.amazonaws.com"
            }
        }]
    }"""
)

# Attach policies
aws.iam.RolePolicyAttachment("sagemaker-full-access",
    role=execution_role.name,
    policy_arn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
)

# SageMaker Domain
domain = aws.sagemaker.Domain("ml-domain",
    domain_name=f"ml-{pulumi.get_stack()}",
    auth_mode="IAM",
    vpc_id=vpc.id,
    subnet_ids=[s.id for s in private_subnets],
    default_user_settings=aws.sagemaker.DomainDefaultUserSettingsArgs(
        execution_role=execution_role.arn
    )
)
```

### Modular Components

Create reusable Python classes:

```python
import pulumi
from pulumi import ComponentResource, ResourceOptions
import pulumi_aws as aws

class MLTrainingEnvironment(ComponentResource):
    def __init__(self, name: str, environment: str, opts=None):
        super().__init__("custom:ml:TrainingEnvironment", name, None, opts)

        # S3 bucket for artifacts
        self.artifacts_bucket = aws.s3.Bucket(
            f"{name}-artifacts",
            bucket=f"ml-artifacts-{environment}",
            versioning=aws.s3.BucketVersioningArgs(enabled=True),
            opts=ResourceOptions(parent=self)
        )

        # ECR repository
        self.ecr_repo = aws.ecr.Repository(
            f"{name}-repo",
            name=f"ml-training-{environment}",
            image_scanning_configuration=aws.ecr.RepositoryImageScanningConfigurationArgs(
                scan_on_push=True
            ),
            opts=ResourceOptions(parent=self)
        )

        # IAM role
        self.training_role = aws.iam.Role(
            f"{name}-role",
            assume_role_policy=self._get_assume_role_policy(),
            opts=ResourceOptions(parent=self)
        )

        self.register_outputs({
            "bucket_name": self.artifacts_bucket.id,
            "ecr_url": self.ecr_repo.repository_url,
            "role_arn": self.training_role.arn
        })

    def _get_assume_role_policy(self):
        return """{
            "Version": "2012-10-17",
            "Statement": [{
                "Action": "sts:AssumeRole",
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"}
            }]
        }"""


# Usage
dev_env = MLTrainingEnvironment("dev", environment="development")
prod_env = MLTrainingEnvironment("prod", environment="production")

pulumi.export("dev_bucket", dev_env.artifacts_bucket.id)
pulumi.export("prod_bucket", prod_env.artifacts_bucket.id)
```

## Programming Language Benefits

### Loops and Conditionals

```python
# Dynamic resource creation
instance_types = ["p3.2xlarge", "p3.8xlarge", "p4d.24xlarge"]

for i, instance_type in enumerate(instance_types):
    aws.ec2.LaunchTemplate(f"gpu-template-{i}",
        instance_type=instance_type,
        image_id=gpu_ami,
        tags={"InstanceType": instance_type}
    )

# Conditional configuration
stack = pulumi.get_stack()

if stack == "production":
    replicas = 3
    instance_type = "p3.8xlarge"
else:
    replicas = 1
    instance_type = "p3.2xlarge"
```

### Functions and Classes

```python
def create_iam_role(name: str, service: str) -> aws.iam.Role:
    """Create an IAM role for a specific service."""
    return aws.iam.Role(name,
        assume_role_policy=f"""{{
            "Version": "2012-10-17",
            "Statement": [{{
                "Action": "sts:AssumeRole",
                "Effect": "Allow",
                "Principal": {{"Service": "{service}.amazonaws.com"}}
            }}]
        }}"""
    )

sagemaker_role = create_iam_role("sagemaker", "sagemaker")
lambda_role = create_iam_role("lambda", "lambda")
```

### Type Hints and IDE Support

```python
from typing import List, Optional
import pulumi
import pulumi_aws as aws

def create_ml_bucket(
    name: str,
    environment: str,
    lifecycle_days: int = 30
) -> aws.s3.Bucket:
    """Create an S3 bucket for ML artifacts with lifecycle rules."""
    bucket = aws.s3.Bucket(f"ml-{name}",
        bucket=f"ml-{name}-{environment}",
        versioning=aws.s3.BucketVersioningArgs(enabled=True),
        lifecycle_rules=[aws.s3.BucketLifecycleRuleArgs(
            enabled=True,
            prefix="temp/",
            expiration=aws.s3.BucketLifecycleRuleExpirationArgs(
                days=lifecycle_days
            )
        )]
    )
    return bucket
```

## State and Backends

### Pulumi Cloud (Default)

```bash
# Login to Pulumi Cloud
pulumi login

# View state in web console
pulumi stack
```

### Self-Managed State

```bash
# Use S3 backend
pulumi login s3://my-pulumi-state

# Use local filesystem
pulumi login --local
```

## Comparison with Terraform

### When to Choose Pulumi

- Team is Python-first
- Complex conditional logic needed
- Want IDE autocomplete and type checking
- Prefer testing infrastructure with pytest
- Need to share code with ML applications

### When to Choose Terraform

- Team has Terraform experience
- Simpler infrastructure needs
- Want largest provider ecosystem
- Prefer declarative-only approach
- Need widest community support

### Key Differences

| Aspect | Pulumi | Terraform |
|--------|--------|-----------|
| Language | Python, TypeScript, Go | HCL |
| Logic | Full programming | Limited DSL |
| Testing | pytest, unittest | Terratest |
| IDE support | Full autocomplete | Limited |
| Learning curve | Lower for devs | Terraform-specific |

## Best Practices

### Project Structure

```
infrastructure/
    __main__.py
    config.py
    components/
        __init__.py
        ml_environment.py
        networking.py
    Pulumi.yaml
    Pulumi.dev.yaml
    Pulumi.prod.yaml
    requirements.txt
```

### Testing

```python
# test_infrastructure.py
import pulumi
import pytest

class MockResource:
    def __init__(self, name, inputs):
        self.name = name
        self.inputs = inputs

@pytest.fixture
def stack():
    pulumi.runtime.set_mocks(MyMocks())
    return pulumi.automation.create_or_select_stack(
        stack_name="test",
        project_name="test",
        program=lambda: None
    )

def test_bucket_versioning_enabled():
    # Test that buckets have versioning enabled
    pass
```

### Secrets Management

```python
import pulumi

config = pulumi.Config()

# Secrets are encrypted in state
db_password = config.require_secret("dbPassword")

# Output secrets are also encrypted
pulumi.export("connection_string",
    pulumi.Output.secret(f"postgres://user:{db_password}@host/db")
)
```

## Common Pitfalls

### Circular Dependencies

Python makes it easy to create circular resource dependencies. Use `depends_on` explicitly when needed.

### State Drift

Same as Terraform: manual changes cause drift. Run `pulumi preview` regularly.

### Large Programs

Very large programs can be slow. Split into multiple stacks or use Automation API.

### Python Environment

Dependencies must be managed carefully. Use virtual environments and pin versions.
