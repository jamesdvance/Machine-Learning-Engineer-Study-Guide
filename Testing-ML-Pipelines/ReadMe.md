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
* pytest also has many built-in fixtures, such as `monkeypatch` (replaces values and behaviors)
* Don't use fixtures for tests that requier slight variations in the data
* Can be made modular and importable in any test by defining fixtures in `conftest.py`

### Marks
* markers enable users to define categories for tests sos they don't all have to be run at once
* Tests can be marked with many categories
* At runtime, marks can be excluded using a parameter
* Common categories are by subsystem or dependencies. For example, a mark could be `database_access`
* built-in marks: 
    * `skip` - skips a test always
    * `skipif` - skips if the expression passed to it is true
    * `xfail` - 'expected fail'. aka if this tests fails, that's expected so the whole suite can still pass
    * `parametrize` - creates multiple variants of a test with different values as arguments

### Configuration
* Uses pytest.ini for config

### Parameterization
* Fixtures reduce code by extracting common dependencies but aren't as useful when you have several tests w/different inputs and outputs
* Instead, can parameterize a test definition, and pytest will create variants of the tests for you with different parameters you specify

### Pytest Watch
* Automatically runs the pytest tests when changes to the codebase are detected
* Notifies when problems are found

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