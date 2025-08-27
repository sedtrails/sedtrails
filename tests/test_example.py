"""
This is an example of how to use the pytest framework to test your code.
For more examples and information, see the pytest documentation: https://docs.pytest.org/
"""

import pytest


##########################
# Code to be tested
##########################
def func(x):
                                """
                                This function increments the input by 1
                                """
                                return x + 1


##########################
# Fixtures
##########################
# Fitures are values that are passed to test functions
# Fixtures are defined using the pytest.fixture decorator


# This is a global fixture, available to all tests in this file
@pytest.fixture
def input_value():
    """
    Define a global fixture available to all tests in this file
    """
    return 39


##########################
# Tests can be grouped into classes
# This is not required, but can be useful for organising
# tests that are related to the same function/class/file
##########################


class TestFunc:
    """Test the func function"""
    def test_func_pass(self, input_value):
        """
        Test that the function returns the expected value
        """
        assert func(input_value) == 40  # use asserts to check the outputs

    def test_func_fail(self, input_value):
        """
        Test that the function does not return an incorrect value.
        This is not a good test, but it is here to show how to write a failing test.
        """
        assert func(input_value) != 200

    def test_func_exception(self):
        """
        Test that the function raises an exception when given a string
        """
        with pytest.raises(TypeError):
            func("hello")


##########################
# Running the tests
##########################
# Run the tests by calling `pytest` from the command line
# pytest will automatically find and run all tests in the current directory
# You can also run `pytest -v` flag for more verbose output

# To runs specific tests, you can specify the test file, class, and function:
# pytest tests/test_example.py::TestFunc::test_func_pass
