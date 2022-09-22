# ML Ops Tactics 

## [Operationalizing Machine Learning: An Interview Study](https://arxiv.org/pdf/2209.09125.pdf)

### Three Keys To Success ("Three V's")
* Velocity - prototyping and iterating on ideas quickly is crucial
	* high velocity experimentation development environments
	* debugging environments to test hypotheses quickly
* Validation - test changes, cut bad ideas, and check for bugs as early and often as possible
	* proactively monitoring pipeline
* Versioning - store and manage multiple versions of models and data for querying, debugging and rollbacks
	* buggy models in production can be rolled back to retrained, simpler, or historical models

### Experimentation
* Since stage-to-stage deployments can take weeks or months, its good to prune bad ideas quickly
* A test environment can be used to validate models in a sandbox
* Smaller code changes are preferable as they lead to easier code review, fewer merge conflicts, and easier validation
* Changes in large orgs were primarily made in config instead of code
	* Random seeds, package versions, data versions

### Model Evaluation
* Data changes, business & product goals change rapidly
* Dynamic validation datasets help uncover bugs more rapidly
* When errors were critical to manage, like in self-driving cars, the system needed to not just fix bugs but change the data to learn from the bugs
* Systematically bucketing data by its errors in production can help to uncover groups / attributes that are prone to errors and bugs
* Another way to handle errors in production is to automatically put errors into a dataset to be used in offline training as the validation set
* To coordinate multiple improvements to models happening at once, and ensure validation is consistent between them, one co used a 'merge queue'
	1. Creates temporary branches w/ special prefix. Purpose of temp branch is to validate PR changes
	2. Group changes in PR into a merge group - the base branch + all branches before and including this one
	* Saves the coordination and timing required with merging multiple branches into the main branch 
	* Allows single testing suite run on the merge group altogether to find bad merges quicker
* A standardized deployment system based on a standard evaluation built trust with the business in high-stakes ML situations
* Evaluation should be done with business/product metrics not (largely) ML-specific metrics like MAP
* The first task of every ML project should be to figure out the most important metric
* Certain metrics could target crucial customers or groups, and could require certain thresholds to be successful 

### Model Deployment
* Having multiple stages w/evaluation at each stage helps to catch failures and more quickly iterate
* A stage might be in production, but just on a small % of traffic
* Shadow Stage - predictions are generated live, but not shown to users
	* Both the prod model and the candidate model's data is saved to the lake and they can be evaluated against each other
	* Also helpful to validate to stakeholders that a change is worth putting into production
	* Shadow mode is impossible for recommender systems, since the feedback loop is required for validation

### Production Processes
* Retraining is often done in such a cadence to maximize operational efficiency, not necessarily scientific rigor
* A layer of heuristics can help filter results based on domain knowledge / observations 
	* Generally, creating filters is more effective than trying to re-train the model to filter a certain result
* Data validation is crucial for features, and predictions
	* Hard constraints - bounds on values
	* Completeness - fraction of non-nulls
	* Schema checks - expected cols and types in those  cols
* Old versions of models can be kept as a fallback
* Some orgs encourage using fewer, but larger neural networks that share embeddings and might have multiple outputs
	* The idea is they reduce intermediate dependencies and eliminate bugs

### On-Call Processes
* On-call rotations often last 1-2 weeks. At the end of a shift, an engineer creates an incident report for each bug or for major issues and how they were fixed
* Often, there is a central queue of production ML bugs that every engineer adds tickets to and processes tickets from
* Often it is larger than what engineers could process, so priorities are added
* SLO's can help provide clarity on when a pipeline is broken

### Pain Points / Anti-Patterns
* Mismatch between development and production environments
	* tension betweene velocity and validating easy. Its hard to exactly match to production and will slow down experimentation
* Code review and coding best practices are often skipped, because they hinder velocity

### Data Leakage
* E.g. embedding model is used on same split of data as downstream models
* E.g. class imbalance meant not enough labeled data for subpopulation at training, but was more prevalent at serving
* E.g. feedback delay (time between getting labels) was ignored during training

### Jupyter notebooks
* Sometimes, they're used in production, making it easier to debug and fix problems
* Can modularize code by putting each component of a pipeline into a notebook
* Othertimes, even training is done outside of notebooks, by policy

### Alerts
* Alert fatigue is a consistent problem. Filtering is needed to still respond to alerts but not miss important ones 
* If a data validation flag triggers a fallback to an older model by default, it creates a tension with versioning, as models are consistently be rolled back, often for false alerts
* Some companies use a model to try to classify useful alerts
* Data alerts are further compounded by label lag (feedback delay), such that performance drops can't be measured in a timely way
	* Recommenders don't have this issue
* Natural vs unnatural data drift
	* Natural - slow expected change over time. Can be easily fixed with frequent retrains
		* However, it can cause problems with hand-curated features, such as bucketized features
	* Unnatural - upstream bug, unusual customer input, freak event (Covid). 
* Bugs are often unpredictable and one-off, but their symptons (such as immediate degredation in prediction vs offline) are predictable

### Debugging
* Adding observability, to pipelines along with raw data access and visualization tools reduces friction for debugging
* There's no perfect debugging system, especially because ML bugs frequently don't trigger failed tests but minor degredation in performance

### Experimentation Methods
* High velocity / parallel experiments can lead to disorganized results and a versioning mess
* Focusing on the best experiment, then building on its success yielded the highest productivity

### MLOps Layers & Tools
* Run layer - record of an execution of an ML or data pipeline. 
	* Often managed by data catalogs, model registries, training dashboards
	* E.g. W&B, MLFlow, Hive Metastore, AWS Glue
* Pipeline layer - specifies the dependencies between artifacts and details of the corresponding computations
	* E.g. Papermill, Airflow, Tensorflow Extended, Sagemaker
	* Pipelines change less frequently than runs, and have more components
* Component layer - a component is an individual node of computation in a pipeline
	* E.g. Python, Spark, Pytorch, Tensorflow
	* Large organizations may have a library of common components
* Infrastructure Layer - (cloud) storage and compute
	* E.g. Docker, AWS, GCP
* Config-based development helps reduce complexity of managing pipelines
* Most changes limited to Run layer, such as changing hyper parameters
* Feature stores help provide versioning, which is key in debugging