# AWS Glue

## Components
* Data Catalog - holds metadata and structure of data
* Database - used to create or access the database for the sources and targets
* Crawler - Used to retrieve data from the source using built-in or custom classifiers
* Job - business logic that carriers out an ETL task. Internally, Glue is Apache Spark with Python or Scala
* Trigger - Starts the ETL job on demand or at specific time
* Development endpoint - endpoint where ETL job script can be tested, developed or debugged

## Glue Attributes
* Serverless, so cost-effective
* Fast. Based on Python Scala (which it provides)
* Can output data to many different places (RDS, Redshift, S3)

## Development
* A Python SDK can be used to write scripts. The code runs on top of Spark
* Also a GUI