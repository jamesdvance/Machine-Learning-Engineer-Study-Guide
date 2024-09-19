# ML Ops On SageMaker
AWS Skillshare

## AWS Service Catalog
* Enables all the features in Sagemaker, off the bat

## Model Registry
* Save and catalog models with their metadata 
* Integrate with Sagemaker Pipelines
* Various versions per model group
* Can save metadata, then compare and decide which goes into production
* Can access in SM Studio UI, or programmatically
* Can be shared cross-account
* Can view ROC curve, confusion matrix, etc
* Integrates with Sagemaker Pipelines

## ML Lineage Tracking
* Logs every step of ML Workflow
* Data processing, feature engineering, etc
* Automatically creates a connected graph of lineage entity metadata tracking your workflow
* Tracks training jobs, processing jobs, transform jobs

### Key Advantages
* Retrieve all datasets that went into the creation of the model
* Retrieve all jobs that went into the creation of the an endpoint
* Retrieve all models that use a data set
* Retrieve all endpoints that use a model
* Retrieve which endpoints are derived from a certain dataset
* Retraieve relationships between entities

### Concepts
* Linage Graph - connected graph of ML workflow E2E
* Artifacts - URI addressable object or data
* Actions - an action taken such as computation, transformation, or job. Its inputs and outputs are artifacts
* Contexts - Provides a method to logically group entities
* Associations - directed edge in the lineage graph
* Experiment entities - Trials, Experiments, Trial Components

### API
* Is accessible via python api
* Can go forward or backwards from an artifact
* ModelArtifact, DatasetArtifact, EndpointContext are wrappers over the LineageQuery api
* Case study - get all datasets associated with a model. 

## Sagemaker Projects
* Can do Source Control, ML Pipelines, Model Registry, CICD Deployment
* Can be triggered w/ Amazon EventBridge
	* Could be triggered by some performance degredation threshold for example

### Sagemaker Project Templates 
* Have templates in the AWS Service Catalog that have for example
 	* 3rd party source control
 	* 3rd party CICD
 	* AWS cloud formation 
 * Can also create own templates within AWS Service Catalog and deploy within AWS Sagemaker Studio

### Sagemaker Project
* Can setup so that Data Scientists work only in notebooks in Sagemaker studio, and kick off deployment steps 
* Can include manual evaluation before deployment

### Sagemaker Pipelines
* Can Prepare Data, Train model, Evaluate and Test Model, and Register model
* Can be triggered upon git commit (?)
* Model pipelines can be deployed

## Sample Architecture
* 

### Additional Resources
* MLOps With SageMaker WhitePaper
*