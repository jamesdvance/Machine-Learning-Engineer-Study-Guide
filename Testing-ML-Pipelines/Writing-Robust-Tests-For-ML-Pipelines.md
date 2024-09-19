
# [Writing Robust Tests for Machine Learning Pipelines](https://eugeneyan.com/writing/testing-pipelines/)
Eugene Yan, 2022
***

## Types of Tests
* Unit - individual methods or classes
* Integration - integration points in pipeline
* Functional - ensure business requirements & logic are met
* End-To-End - test pipeline output integrates correctly with the site and correct rec's are made
* Acceptance - visual inspection & QA of live system
* Performance - load testing and assessing latency and throughput
* Smoke - handcrafted URL's to trigger QA of common items / situations
* A/B - evaluate long-term performance of system 

Additionally, ML pipelines often had additional tests on data quality model evaluation, and model behavioral checks that run in production with each new model or batch
* Data Quality - [Great Expectations](https://greatexpectations.io/), [Deequ](https://github.com/awslabs/deequ) by AWS
* Model Evaluation - e.g. sklearn metrics

### Integration Tests
Integration Tests are useful for ensuring an entire pipeline works as expected
<br> 
In reality this can mean stretching out tests across multiple repos and infra
