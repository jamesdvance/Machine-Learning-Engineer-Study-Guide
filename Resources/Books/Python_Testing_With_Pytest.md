# Python Testing With Pytest
Brian Okken

## Chapter 1: Getting Started
* Pytest shows which line failed, the error message, and the surrounding code in a *traceback*
* -vv parameter to command line gives verbose results

## Chapter 2: Writing Test Functions

### assert
* Use `assert` for almost all assertions
* Will have verbose error explanations

### assertion helpers
* an *assertion helper* is used to wrap up a complicated assertion check
* The assertion helper function can be called at the end of a test function instead of a plain assert statement
* The Cards data class 
* `pytest.fail()` can be used to fail a test given a condition and write a specific message
* This makes `pytest.fail()` the right tool for assertion helpers
`__traceback-hide__ = True` can be used inside of an assertion 
	* makes it so failing tests don't use this function in the traceback
	* more clarity since the assertion helper already outputs a helpful message

### Testing For Expected Exceptions
* Sometimes we want to test that code raises an exception when its supposed to
* Can use `pytest.raises()` 
* Can be used inside of a with statement like `with pytest.raises(TypeError):`
* Entire assertion phrases can check things like correct error message, ect after `with` statement
* `match` parameter takes a regular expression and matches it with an exception message
* Can test against type `ExceptionInfo` objects with a custom variable called `exc_info`

### Structuring Test Functions
* Make sure asserts are kept at the end of test functions
	* Pattern is called both "Given-When-Then" and "Arrange-Act-Assert"
 	* Tests that focus on more than one behavior are hard to debug

### Grouping Tests With Classes
* pytest allows grouping tests with classes
* just using functions is sufficient for many projects(!)
* then, could run `test*` all methods in a given class by calling class from file like `python test_classes.py::TestEquality`
* inheritance can be utilized if necessary, but can often lead to confusion
* class functionality is mostly useful for grouping similar tests

### Running A Subset of Tests
* Test classes can be used to run a subset of tests
* Can also run subset of tests like this
	* run single test method `pytest test_module.py::Test-Class::test_method`
	* All tests in a class `pytest test_module.py::Test-Class`
	* All tests in a module `pytest test_module.py`
	* All tests in a directory `pytest test_dir`
	* All tests matching a name pattern `pytest -k pattern`
	* Tests by marker
* `-k` argument takes an expression and tells pytest to run all tests that contain a substring match
	* can be given an explicit and, not, or, () logic in the form of an expression
	* Is especially helpufl in debugging problems that might span certain test characteristics

## Chapter 3: Pytest Fixtures
* Fixtures run before (and sometimes after) test functions
* Fixture names are defined as parameters in the test function 
* The code can have different purposes, such as getting a dataset, or getting system into a known state.
* Fixtures can get data ready for multiple tests
* The `@pytest.fixture` decorator tells pytest to run it before running the test
* The fixture can get called in a test function 
* Name the test that uses the fixture like `test_some_fixture_name` for the fixture `some_fixture_name`
* Generally used as 'getting ready for' or 'cleaning up after' tests
* An error within a fixture gets treated like an Error instead of a Fail that would happen inside a test func

### Fixtures for Setup and Teardown
* Example use case would be initializiing a db connection and closing it after tests are run
* `yield` keywork can be used so that setup happens first (db initialized) then the object is available to all functions who call it (`yield`) 
	* then `db.close()` can run after completion for teardown
* `--setup-show` command-line flag can be used to show the order of tests & fixtures
	* helps identify order setup, test and teardown run

### Specifying Fixture Scope
* scope is passes as a parameter to the fixture decorator
* Default 'scope' of a fixture is 'function' scope. That means it will run before each test that needs it
* Other scopes:
	* class - run once per class
	* module - fixture is run once per module
	* package - run once per 'package' or test directory
	* session - once per session. All tests share one setup and teardown call
* In order to use session and package scopes, fixtures need to be in the conftest.py file
* A function fixture can depend on a class or module fixture, but a module fixture can't depend on a function fixture

### Sharing Fixtures through conftest.py
* To be shared among multiple test files, put fixtures in a conftest.py file in the same directory or some parent directory
* You never have to import conftest.py anywhere because its automatically read by the cli

### Finding Where Fixtures Are Defined
* `pytest --fixtures -v` can be used to show where all fixtuers are
* Fixtures found in conftest.py will be at the bottom of the list
* `fixtures-per-test` parameter is used to show where are fixtures used by a particular test

### Using Multiple Fixture Levels
* Multiple stages (i.e. session-scoped first, then function-scoped for a given test)

### Using Multiple Fixtures Per Test Or Fixture
* functions and fixtures can contain multiple fixtures

### Deciding Fixture Scope Dynamically
* Can be decided dynamically, by putting in a function name parameter for scope passed into the fixture 
* That function can return the string corresponding to the scope, depending on some logic
* The logic can be based on CLI using `config.getoption("--scope_func", None)`

### Using autouse for Fixtures That Always Get Used
* `autouse=True` gets a fixture to run all the time
* Example use case could be to add test times after each test
* Shouldn't be the norm

### Renaming Fixtures
* can rename a fixture by passing a name parameter to the decorator

## Chapter 4: BuiltIn Fixtures
See all with `pytest --fixtures`
* tmp_path, tmp_path_factory - create temporary dirs on the fly
	* temp dirs still exist after test runs
	* pytest cleans up all but the most recent during each run
	* are numbered by run #
* capsys - capturing output (stdout, stderr)
	* normally, pytest captures all the output (and doesn't print it)
		* to change add cli param, -s or --capture=no 
		* can use capsys.disabled() for same effect
	* similar built in fixtures
		* capfd - captures file descripters (usually same as stdout and stderr)
		* capsysbinary - captures bytes instead of text
		* capfdbinary - file descipters binary
		* caplog - captures output writting with logging package
* monkeypatch - changing the environment or application code
	* lightweight mocking
	* dynamic modification of class or module during runtime
	
## Chapter 5: Parametrization
Prametrized testing is a way to send multiple sets of data through
the same test and have pytest report if any of the sets failed
* @pytest.mark.parametrize()

### Parametrizing Test Functions
* First argument - list of names of parameters. Can be a list of strings, or comma-seperated string
* Second argument - list of test cases. Each list element is a tuple or list that has one elem for each argument
* These are passed into `@pytest.mark.parametrize` decorator

### Parametrizing Fixtures
* Define a fixture function and pass a params argument (list) into the `@pytest.fixture decorator`
* Can just iterate over each list element (as a tuple) if more than one arg needed

### Parametrizing with pytest_generate_tests
* `pytest_generate_tests` is a hook function 
* takes `metafunc` as an input
* Calls metafunc.parametrize then takes same params as @pytest.mark.parametrize

## Chapter 6: Markers
* Markers are tags or labels for tests
* Some are built-in like @pytest.mark.slow 
* Can use them to mark tests by type like @pytest.mark.smoke
* Types of built-in marks:
	* skip
	* skipif
	* filterwarnings
	* xfail
	* parametrize
	* usefixtures
* can apply file-level marks by definine `pytestmark` variable as a mark object
* can also apply marks to entire classes
* can also mark specific parametrizations of a parametrized test (using *marks* parameter)

### Calling from CLI
* Can use *and*, *or*, *not*, and *()* when calling markers from the CLI
* Can use `--strict-markers` CLI param to cause error if marker referenced doesn't exist

### Markers and Fixtures
* Can use markers to input parameters into fixtures when called 
* The fixture calls request.node.get_closest_marker(*marker_name*)
* `request` is a parameter into the fixture
* `request.node` is pytest's representation of a test

## Chapter 7: Strategy
* Python code can be imported and tested fine-grained
* Other language code is tested like a black box

### Features to Test

*Recent* - new features / areas of code
*Core* - unique parts of software, essential functions to product
*Risk* - areas that pose more risk, such as areas crucial to customers but not frequently updated
*Problematic* - often break
*Expertise* - understand by subset of people

