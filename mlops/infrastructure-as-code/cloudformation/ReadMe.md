# AWS CloudFormation

## Summary

AWS CloudFormation is AWS's native infrastructure as code service for provisioning and managing AWS resources. It uses YAML or JSON templates to define infrastructure, with deep integration into AWS services. For teams fully committed to AWS, CloudFormation offers seamless service integration and no additional tooling costs.

Key points to remember:

- Native AWS service with no additional tools required
- Deep integration with AWS services including SAM for serverless
- Templates can be YAML or JSON; YAML is more readable
- Stacks manage related resources with built-in rollback
- Drift detection identifies manual changes

## Core Concepts

### Templates

Templates define AWS resources:

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: ML Training Infrastructure

Parameters:
  Environment:
    Type: String
    AllowedValues:
      - development
      - staging
      - production
    Default: development

  InstanceType:
    Type: String
    Default: p3.2xlarge

Resources:
  MLArtifactsBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub 'ml-artifacts-${Environment}'
      VersioningConfiguration:
        Status: Enabled
      Tags:
        - Key: Environment
          Value: !Ref Environment

  TrainingRole:
    Type: AWS::IAM::Role
    Properties:
      RoleName: !Sub 'ml-training-role-${Environment}'
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: sagemaker.amazonaws.com
            Action: sts:AssumeRole

Outputs:
  BucketArn:
    Description: ARN of ML artifacts bucket
    Value: !GetAtt MLArtifactsBucket.Arn
    Export:
      Name: !Sub '${Environment}-MLArtifactsBucketArn'
```

### Stacks

Stacks are instances of templates:

```bash
# Create stack
aws cloudformation create-stack \
    --stack-name ml-infrastructure-dev \
    --template-body file://template.yaml \
    --parameters ParameterKey=Environment,ParameterValue=development \
    --capabilities CAPABILITY_NAMED_IAM

# Update stack
aws cloudformation update-stack \
    --stack-name ml-infrastructure-dev \
    --template-body file://template.yaml

# Delete stack
aws cloudformation delete-stack \
    --stack-name ml-infrastructure-dev
```

### Intrinsic Functions

CloudFormation provides built-in functions:

```yaml
Resources:
  Bucket:
    Type: AWS::S3::Bucket
    Properties:
      # Substitute variables
      BucketName: !Sub 'ml-${AWS::Region}-${Environment}'

      # Reference other resources
      LoggingConfiguration:
        DestinationBucketName: !Ref LoggingBucket

      # Conditional logic
      VersioningConfiguration:
        Status: !If [IsProd, Enabled, Suspended]

      # Join strings
      Tags:
        - Key: FullName
          Value: !Join ['-', [!Ref Environment, artifacts, bucket]]

Conditions:
  IsProd: !Equals [!Ref Environment, production]
```

## ML Infrastructure Templates

### SageMaker Training Environment

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: SageMaker Training Environment

Parameters:
  Environment:
    Type: String
  VpcId:
    Type: AWS::EC2::VPC::Id
  SubnetIds:
    Type: List<AWS::EC2::Subnet::Id>

Resources:
  # Execution Role
  SageMakerExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: sagemaker.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
      Policies:
        - PolicyName: S3Access
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - s3:GetObject
                  - s3:PutObject
                  - s3:ListBucket
                Resource:
                  - !GetAtt TrainingDataBucket.Arn
                  - !Sub '${TrainingDataBucket.Arn}/*'

  # Training Data Bucket
  TrainingDataBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub 'ml-training-data-${Environment}-${AWS::AccountId}'
      VersioningConfiguration:
        Status: Enabled
      PublicAccessBlockConfiguration:
        BlockPublicAcls: true
        BlockPublicPolicy: true
        IgnorePublicAcls: true
        RestrictPublicBuckets: true

  # SageMaker Domain
  SageMakerDomain:
    Type: AWS::SageMaker::Domain
    Properties:
      DomainName: !Sub 'ml-domain-${Environment}'
      AuthMode: IAM
      VpcId: !Ref VpcId
      SubnetIds: !Ref SubnetIds
      DefaultUserSettings:
        ExecutionRole: !GetAtt SageMakerExecutionRole.Arn

Outputs:
  DomainId:
    Value: !Ref SageMakerDomain
  ExecutionRoleArn:
    Value: !GetAtt SageMakerExecutionRole.Arn
```

### EKS Cluster for Training

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: EKS Cluster for ML Training

Parameters:
  ClusterName:
    Type: String
    Default: ml-training-cluster
  VpcId:
    Type: AWS::EC2::VPC::Id
  SubnetIds:
    Type: List<AWS::EC2::Subnet::Id>

Resources:
  EKSClusterRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: eks.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/AmazonEKSClusterPolicy

  EKSCluster:
    Type: AWS::EKS::Cluster
    Properties:
      Name: !Ref ClusterName
      Version: '1.28'
      RoleArn: !GetAtt EKSClusterRole.Arn
      ResourcesVpcConfig:
        SubnetIds: !Ref SubnetIds
        EndpointPrivateAccess: true
        EndpointPublicAccess: true

  NodeGroupRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: ec2.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy
        - arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
        - arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy

  GPUNodeGroup:
    Type: AWS::EKS::Nodegroup
    Properties:
      ClusterName: !Ref EKSCluster
      NodegroupName: gpu-nodes
      NodeRole: !GetAtt NodeGroupRole.Arn
      Subnets: !Ref SubnetIds
      InstanceTypes:
        - p3.2xlarge
      ScalingConfig:
        MinSize: 0
        MaxSize: 8
        DesiredSize: 0
      Labels:
        gpu: 'true'
      Taints:
        - Key: nvidia.com/gpu
          Value: 'true'
          Effect: NO_SCHEDULE

Outputs:
  ClusterEndpoint:
    Value: !GetAtt EKSCluster.Endpoint
  ClusterArn:
    Value: !GetAtt EKSCluster.Arn
```

## Nested Stacks and Cross-Stack References

### Nested Stacks

Break large templates into manageable pieces:

```yaml
# parent-template.yaml
Resources:
  NetworkStack:
    Type: AWS::CloudFormation::Stack
    Properties:
      TemplateURL: https://s3.amazonaws.com/templates/network.yaml
      Parameters:
        Environment: !Ref Environment

  StorageStack:
    Type: AWS::CloudFormation::Stack
    Properties:
      TemplateURL: https://s3.amazonaws.com/templates/storage.yaml
      Parameters:
        Environment: !Ref Environment
        VpcId: !GetAtt NetworkStack.Outputs.VpcId

  ComputeStack:
    Type: AWS::CloudFormation::Stack
    Properties:
      TemplateURL: https://s3.amazonaws.com/templates/compute.yaml
      Parameters:
        VpcId: !GetAtt NetworkStack.Outputs.VpcId
        BucketArn: !GetAtt StorageStack.Outputs.BucketArn
```

### Cross-Stack References

Share resources between stacks:

```yaml
# network-stack.yaml
Outputs:
  VpcId:
    Value: !Ref VPC
    Export:
      Name: !Sub '${Environment}-VpcId'

# compute-stack.yaml
Resources:
  Instance:
    Type: AWS::EC2::Instance
    Properties:
      SubnetId: !ImportValue
        !Sub '${Environment}-SubnetId'
```

## Change Sets

Preview changes before applying:

```bash
# Create change set
aws cloudformation create-change-set \
    --stack-name ml-infrastructure \
    --template-body file://template.yaml \
    --change-set-name update-instance-type

# Review changes
aws cloudformation describe-change-set \
    --stack-name ml-infrastructure \
    --change-set-name update-instance-type

# Execute change set
aws cloudformation execute-change-set \
    --stack-name ml-infrastructure \
    --change-set-name update-instance-type
```

## AWS SAM for Serverless ML

AWS SAM simplifies serverless applications:

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: Serverless ML Inference

Globals:
  Function:
    Timeout: 30
    MemorySize: 1024
    Runtime: python3.11

Resources:
  InferenceFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: inference/
      Handler: app.handler
      Events:
        Inference:
          Type: Api
          Properties:
            Path: /predict
            Method: post
      Environment:
        Variables:
          MODEL_BUCKET: !Ref ModelBucket
      Policies:
        - S3ReadPolicy:
            BucketName: !Ref ModelBucket

  ModelBucket:
    Type: AWS::S3::Bucket

Outputs:
  InferenceApi:
    Value: !Sub 'https://${ServerlessRestApi}.execute-api.${AWS::Region}.amazonaws.com/Prod/predict'
```

Deploy with SAM CLI:

```bash
sam build
sam deploy --guided
```

## Best Practices

### Template Organization

```
cloudformation/
    templates/
        network.yaml
        storage.yaml
        compute.yaml
        ml-platform.yaml
    parameters/
        dev.json
        prod.json
    scripts/
        deploy.sh
        validate.sh
```

### Parameter Files

```json
// parameters/prod.json
[
    {
        "ParameterKey": "Environment",
        "ParameterValue": "production"
    },
    {
        "ParameterKey": "InstanceType",
        "ParameterValue": "p3.8xlarge"
    }
]
```

```bash
aws cloudformation create-stack \
    --stack-name ml-prod \
    --template-body file://templates/ml-platform.yaml \
    --parameters file://parameters/prod.json
```

### Validation

```bash
# Validate template syntax
aws cloudformation validate-template \
    --template-body file://template.yaml

# Use cfn-lint for additional checks
cfn-lint template.yaml
```

## Comparison with Terraform

### CloudFormation Advantages

- Native AWS integration
- No additional tools to install
- Built-in rollback on failures
- AWS support included
- Free to use

### Terraform Advantages

- Multi-cloud support
- Larger community
- Better state management options
- More flexible syntax
- Faster provider updates

### When to Choose CloudFormation

- AWS-only infrastructure
- Want native AWS support
- Using AWS SAM for serverless
- Prefer no additional tooling

## Common Pitfalls

### Circular Dependencies

Resources referencing each other create deployment failures. Use `DependsOn` to break cycles.

### Rollback Failures

Failed rollbacks leave stacks in broken states. Enable termination protection and use change sets.

### Resource Limits

CloudFormation has limits (500 resources per stack). Use nested stacks for large deployments.

### Slow Updates

Some resources take long to update. Design for minimal disruption and use UpdatePolicy.

### Drift

Manual changes cause drift. Use drift detection regularly:

```bash
aws cloudformation detect-stack-drift --stack-name my-stack
aws cloudformation describe-stack-resource-drifts --stack-name my-stack
```
