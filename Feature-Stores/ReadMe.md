# Feature Stores

## Feathr (LinkedIn). MLOps.community podcast. 9/2/22

### Lessons Learned working on Feathr
* Feature stores and feature management isn't a solved problem
* Having clean concepts is important to serve users
* A feature store is a device to help ML Engineers think more simply about their deployment system
* So need named concepts that resonate with users. Getting those right early on is very helpful
* Ask users upfront what they want from this system
* Entity features are the important use case in feature stores
* The idea of decoupling the artifact / entity of a feature from just code is the innovation of feature stores
	* And can be referenced in experimentation, offline, and online environments 

### Feathr vs other feature stores
* Is still ambiguity of what feature store means. Not just a database

### Real-Time Features
* Real-time features are still a work in progress for most
* Speed to experiment with real-time features is crucial for ML engineers
* Real-time feature matter most in areas like advertising
***

## [AWS Sagemaker-Feature Store Introduction](https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-featurestore/feature_store_introduction.html)

### Feature Groups
* Used to Sagemaker Feature STore to store metadata for all data stored in the feature store. 
* A feature group is a logical grouping of feature
* Consists of feature definitions, a record identifier name and configs for its online and offline store
* To define in Python, can create object using name of the feature group and the Sagemaker session
* After definition, an actual Feature Group can be created with the Create method, with these parameters
	* s3_uri
	* record_identifier_name
	* event_time_feature_name
	* role_arn
	* enable_online_store 

### Offline vs Online
* Offline - historical data in s3
* Online - available via GetRecord API for low latency
* Each Feature Group contains an OnlineStoreConfig and OfflineStoreConfig 
