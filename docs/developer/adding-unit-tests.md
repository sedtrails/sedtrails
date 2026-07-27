# Adding Unit Tests

All new plugins and features should include documentation of the new features and unit tests to ensure that everything is working as expected. This document explains how to add unit tests for the ``physics_lib.py`` functions, and has been prepared with input from Copilot (Claude Sonnet 4).

## Test Structure

The test file `test_physics_lib.py` is organized into test classes, with each class focusing on testing a specific function or group of related functions:

- `TestComputeGrainProperties` - Tests for `compute_grain_properties()`
- `TestComputeShearVelocity` - Tests for `compute_shear_velocity()`  
- `TestComputeShields` - Tests for `compute_shields()`
- `TestComputeBedLoadVelocity` - Tests for `compute_bed_load_velocity()`
- And more...

## Example: Adding a New Test

Here's how to add a test for a specific known input/output case:

```python
def test_my_function_known_case(self):
    """Test my_function with known input/output values."""
    # Given specific input parameters
    input_param1 = 1.0
    input_param2 = 2.0
    
    # When calling the function
    result = my_function(input_param1, input_param2)
    
    # Then expect a specific output (with tolerance for floating point)
    expected_output = 3.14159
    assert_allclose(result, expected_output, rtol=0.01)  # 1% tolerance
```

## Test Categories

### 1. Known Value Tests
Test with specific input values where you know the expected output:
```python
def test_settling_velocity_known_value(self, grain_parameters):
    """Test settling velocity calculation with known input/output values."""
    result = compute_grain_properties(**grain_parameters)
    expected_settling_velocity = 0.0202  # Known result for these params
    assert_allclose(result['settling_velocity'], expected_settling_velocity, rtol=0.01)
```

### 2. Formula Verification Tests
Verify that the implementation matches the mathematical formula:
```python
def test_shear_velocity_formula(self):
    """Test that shear velocity formula u* = sqrt(τ/ρ) is correctly implemented."""
    bed_shear_stress = np.array([1.0])
    water_density = 1000.0
    
    result = compute_shear_velocity(bed_shear_stress, water_density)
    expected = np.sqrt(bed_shear_stress[0] / water_density)  # Manual calculation
    
    assert_allclose(result[0], expected)
```

### 3. Edge Case Tests
Test boundary conditions and special cases:
```python
def test_zero_input(self):
    """Test function behavior with zero input."""
    result = my_function(0.0)
    assert result == 0.0

def test_negative_input(self):
    """Test function behavior with negative input."""
    result = my_function(-1.0)
    # Verify expected behavior (e.g., absolute value, error, etc.)
```

### 4. Range/Sanity Tests
Test that outputs are in reasonable physical ranges:
```python
def test_settling_velocity_range(self):
    """Test that settling velocity is in reasonable range for medium sand."""
    result = compute_grain_properties(**params)
    # For 200 micron sand, settling velocity should be around 0.02 m/s
    assert 0.01 <= result['settling_velocity'] <= 0.05
```

## Running Tests

Run specific test class:
```bash
python -m pytest tests/transport_converter/test_physics_lib.py::TestComputeGrainProperties -v
```

Run specific test method:
```bash
python -m pytest tests/transport_converter/test_physics_lib.py::TestComputeGrainProperties::test_settling_velocity_known_value -v
```

Run all physics lib tests:
```bash
python -m pytest tests/transport_converter/test_physics_lib.py -v
```

## Integration Tests and Diagnostic Plots

 Some converter tests are marked as integration tests and rely on sample datasets.
 Some of these datasets are not committed to the repository by default; see
 ``sample-data/readme.md`` for download instructions before running (for example,
 Delft3D4 NetCDF conversion expects a Delft3D-4 map NetCDF file).

```bash
python -m pytest -m integration tests/transport_converter/plugins/test_d3d4_netcdf.py
```

The Delft3D4 converter test can optionally emit diagnostic plots for visual
inspection. Plotting is opt-in and controlled via environment variables:

- ``SEDTRAILS_PLOT_DIR``: directory where plots are written. If not set, no
    plots are produced.
- ``SEDTRAILS_PLOT_TIME_INDEX``: optional time index to plot (default: 0).

Example:

```bash
set SEDTRAILS_PLOT_DIR=tests_output/plots
set SEDTRAILS_PLOT_TIME_INDEX=10
python -m pytest -m integration tests/transport_converter/plugins/test_d3d4_netcdf.py
```

## Adding Your Own Known Value Tests

1. **Collect your known input/output data**: Gather the specific parameter values and expected results you want to test.

2. **Create a test method**: Add a new test method to the appropriate test class:
   ```python
   def test_my_known_case(self):
       """Test with my specific known values."""
       # Your input parameters
       param1 = your_value1
       param2 = your_value2
       
       # Call the function
       result = the_function(param1, param2)
       
       # Check against known output
       expected = your_expected_result
       assert_allclose(result, expected, rtol=0.01)  # Adjust tolerance as needed
   ```

3. **Add fixtures for common parameters**: If you have multiple tests using similar parameters, create a fixture:
   ```python
   @pytest.fixture
   def my_test_params(self):
       return {
           'param1': value1,
           'param2': value2,
           # ... more parameters
       }
   
   def test_with_fixture(self, my_test_params):
       result = my_function(**my_test_params)
       # ... assertions
   ```

## Tips for Good Tests

1. **Use descriptive names**: Test method names should clearly describe what is being tested
2. **Include docstrings**: Explain what the test is checking
3. **Use appropriate tolerances**: For floating point comparisons, use `assert_allclose` with reasonable tolerances
4. **Test edge cases**: Include tests for zero values, negative values, very large/small values
5. **Group related tests**: Use test classes to organize tests for the same function
6. **Comment your expected values**: Explain where known values come from (literature, manual calculation, etc.)

## Example: Complete Test for New Function

If you had a new function `compute_drag_coefficient(reynolds_number)`, here's a complete test class:

```python
class TestComputeDragCoefficient:
    """Test the compute_drag_coefficient function."""
    
    def test_known_reynolds_case(self):
        """Test with known Reynolds number from literature."""
        reynolds = 1000.0
        result = compute_drag_coefficient(reynolds)
        expected = 0.44  # Known value from fluid mechanics literature
        assert_allclose(result, expected, rtol=0.05)  # 5% tolerance
    
    def test_low_reynolds_limit(self):
        """Test Stokes flow limit (Re << 1)."""
        reynolds = 0.1
        result = compute_drag_coefficient(reynolds)
        # Should approach 24/Re for low Reynolds numbers
        expected = 24.0 / reynolds
        assert_allclose(result, expected, rtol=0.1)
    
    def test_array_input(self):
        """Test with array of Reynolds numbers."""
        reynolds_array = np.array([100.0, 1000.0, 10000.0])
        result = compute_drag_coefficient(reynolds_array)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == reynolds_array.shape
        assert np.all(result > 0)  # Drag coefficient should be positive
```