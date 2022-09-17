 # Testing

## [PyTest Documentation](https://docs.pytest.org/en/6.2.x/contents.html)
***
### Basics
Simple quickstart
1. Define tests in a directory or subdirectory that have `test_*.py` or `_test.py` in the name
2. Call 'pytest' from the command line from the root directory or test directory

### Testing Exceptions
* Can create assertions for exceptions using pytest.raises(*SomeException*)

### Syntax
* Uses simple assert syntax, not the various methods like self.assertEqual in the unittest library

### Fixture
In software testing, a text fixture sets up a process by initializing it, and satisfying preconditions
* Provide a fixed baseline for repeatable and stable results
* pytests's fixtures provide improvement over unittest's setup method
* defined using the @pytest.fixture decorator
* pytest also has many built-in fixtures

### Pytest Watch
* Runs 

***

## [Writing Robust Tests for Machine Learning Pipelines](https://eugeneyan.com/writing/testing-pipelines/)
Eugene Yan, 2022
***

### Types of Tests
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

#### Integration Tests
Integration Tests are useful for ensuring an entire pipeline works as expected
<br> 
In reality this can mean stretching out tests across multiple repos and infra


***

## [Pandas Testing Documentation](https://pandas.pydata.org/docs/reference/general_utility_functions.html)
***

***