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
***

## [Eugene Yan - Feature Stores](https://eugeneyan.com/writing/feature-stores/)

### Heirarchy of Feature Store Needs
From foundational to nice to have
1. Access - access to feature information & data, transparency, lineage
	* transparency - users can view logic and code in creating the features
	* lineage - trace upstream sources of the features
2. Serving - availability in production at high throughput / low latency
	* NOSQl for low laetncy
	* ability to sync from offline to online feature stores
	* ability to perform real-time feature transformations within the feature store itself
	* don't want to re-implement pipelines that data scientists have created while expirimenting
3. Integrity - minimize train-serve skew, point-in-time correct data, monitoring
	* minimizes train-serve skew
	* point-in-time correctness. Makes sure historical features used offline don't have data leaks
	* point-in-time is made trickier when features update intraday
4. Convenience - easy to use, intuitive API's, interactivity
	* Easy and quick to use, offline and online. 
5. Autopilot - automated backfilling of alerts, feature selection
	* backfilling features - when a definition changes, updating a feature's value in all historical data sets 
		* performed via a UI in Airflow
	* automated alerts on feature distributions
	* feature selection via providing correlations with a given label

### Feature Store Tools
* Feast - feature store that acts as interface between data scientists, data engs, and ml engs. 
	* inspired to solve data re-creation pain-points (similar / same features always re-created)
	* provide unified SDK's across Python, Java and Go for retrieving features from offline and online stores

### Feature Store Implemtations
* Palette - feature store created by Uber. Enables reuse and sharing of features in a single location.
	* Various teams could contribute features to and from Palette
	* Syncs using offline store with Hive 
	* Feature not available in Cassandra are generated online via Flink before being saved to Cassandra
	* Syncing goes both ways. New features added to Hive are synced to Cassandra; real time features added to Cassandra are ETL'd to Hive
* [Monzo bank](https://nlathia.github.io/2020/12/Building-a-feature-store.html) - synced from analytics store SQL to Cassandra (NoSQL)
	* tags get added to SQL queries 
	* A feature store service checks the schema of the feature tables 
	* A cron job checks for updated feature tables and syncs them from SQL (BigQuery) to NoSQL (Cassandra) via a disk staging area
* [Doordash Gigascale feature store](https://doordash.engineering/2020/11/19/building-a-gigascale-ml-feature-store-with-redis/)
	* stores billions of records to persistant scaleable storage
	* scales to millions of QPS (10M)
	* most features refreshed daily via fast batch writes. Real-time features are updated uniformly throughout the day
	* Uses Redis after a long comparison [process](https://doordash.engineering/2020/11/19/building-a-gigascale-ml-feature-store-with-redis/)
* [Alibaba Basic Feature Server](https://102.alibaba.com/detail?id=183) 
	* computes statistical features on user interactions in real time for Alibaba's real-time recsys

### [Distributed Time Travel](https://netflixtechblog.com/distributed-time-travel-for-feature-generation-389cccdd3907?gi=7f62ff23602)
* Helps solve the difficulty of creating point-in-time accurate features to simulate production when offline
	* Doing so incorrectly leads to data leaks
* To do so, they take snapshots of offline and online data context sets
* Stratify sample context sets over time, to generate a well-distributed dataset to train and validate on
* Sampling done by Spark, storing snapshots in S3
* Data scientists can request features at a given point in time
* Time travel features can be requested with
	* context - where when and how model is used, with time being the most critical
	* items - to score/rank
	* labels - for supervised learning
	* feature encoder - how to combine context and item data to create features such as feature-crosses

### Train-Serve Skew
* One solution is to use Apache Beam as the data processing pipeline
	* Unifies batch and streaming pipelines into one
	* Ingests batch and streaming into both offline and online stores
	* Serving pipelines don't need to be re-written
* Netflix uses shared feature encoders between offline and online feature generation tasks

### Monitoring
* Could show distribution of features, correlations between features and labels, and clustering analysis (used by Airbnb's Zipline)
* Uber uses Data Quality Monitor, which provides users with daily data quality scores based on 
	* metrics over features
	* multi-dimensional time series over those metrics
	* PCA over that time series
	* time series on the top components. 
	* Compare one-step ahead forecast to live calculation, flag anomaly if outside of range