# MLOps: Overview, Definition, Architecture
https://arxiv.org/abs/2205.02302

## DevOps
* collab & knowledge sharing
* continuous integration
* deployment automation
* monitoring and logging

## MLOps Principles
1. CI/CD Automation
2. Workflow orchestration
4. Versioning
5. Collaboration
6. Continuous ML training and evaluation
7. ML metadata tracking/logging
8. Continuous monitoring
9. Feedback loops

## Technical Componenets
* CI/CD
* Code repo
* Workflow orchestration - Airflow, Sagemaker Pipelines
* Feature Store - offilne and online
* Model training infrastructure
* Model registry
* ML Metadata stores
* Model Serving Component - REST API or Kubeflow for serving, Spark for batch
* Monitoring Component

## Architecture and Workflow
* Ideal state is continuous retraining once a *concept* drift threshold is reached

## Open Challenges

### Organizational
* Model-driven machine learning towards product-oriented
* Data-centric vs model driven
* MLOPs training and education lacking

## Discussion Points
* The end-to-end process of collaboation starts with the business stakeholder telling a Solution Architect who creates a system? Have never encountered such a clean start
	* usually a data science project starts for proof of concept and then evolves into a productionized model
* What does does the data engineer do that is out of scope of the data scientist?
	* IMO "refining" of raw data *generated* by the ML model can lead to leakage and training/inference performance divergence
* Are they describing a best case? what is the best case for ml workflows?
* How can workflows stay flexible to iteration? 
* How much does real-time data change the equation? What about continuous training? 
* How much does model lineage and tracking disrupt the flow of experimentation?
* Two new paradigms - data-centric and product-oriented. I feel the iteration and expirmentation creates a tension between these two. Thoughts?
* Specialized tools exist (such as Weights & Biases) that might be better at one thing than an integrated system like Sagemaker. When to use?
* Concept drift is mentioned as a trigger for automated retrainnig. When to use vs performance metrics?
