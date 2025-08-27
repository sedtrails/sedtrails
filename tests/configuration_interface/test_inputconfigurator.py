"""
Unit tests for the YAMLConfigValidator class in the sedtrails.configuration_interface.validator module.
"""

import tempfile
import pytest
import yaml
import json
from sedtrails.configuration_interface.validator import YAMLConfigValidator
from sedtrails.exceptions import YamlValidationError, YamlOutputError


@pytest.fixture
def validator():
    # You can use any valid schema file path, but for testing _apply_defaults, schema_content is enough
    dummy_schema = {}
    dummy_schema_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yml')
    yaml.dump(dummy_schema, dummy_schema_file)
    dummy_schema_file.close()

    DummyValidator = YAMLConfigValidator()

    return DummyValidator


class TestYAMLConfigValidator:
    """
    Test suite for validating YAML configuration files using YAMLConfigValidator.
    """

    def test_apply_defaults_object(self, validator):
        schema = {
            'type': 'object',
            'properties': {
                'a': {'type': 'string', 'default': 'foo'},
                'b': {'type': 'number', 'default': 42},
            },
        }
        config = {}
        result = validator._apply_defaults(schema, config)
        assert result == {'a': 'foo', 'b': 42}

    def test_apply_defaults_partial_object(self, validator):
        schema = {
            'type': 'object',
            'properties': {
                'a': {'type': 'string', 'default': 'foo'},
                'b': {'type': 'number', 'default': 42},
            },
        }
        config = {'a': 'bar'}
        result = validator._apply_defaults(schema, config)
        assert result == {'a': 'bar', 'b': 42}

    def test_apply_defaults_nested_object(self, validator):
        schema = {
            'type': 'object',
            'properties': {'outer': {'type': 'object', 'properties': {'inner': {'type': 'string', 'default': 'baz'}}}},
        }
        config = {'outer': {}}
        result = validator._apply_defaults(schema, config)
        assert result == {'outer': {'inner': 'baz'}}

    def test_apply_defaults_array(self, validator):
        schema = {'type': 'array', 'items': {'type': 'object', 'properties': {'x': {'type': 'integer', 'default': 1}}}}
        config = [{'x': 2}, {}]
        result = validator._apply_defaults(schema, config)
        assert result == [{'x': 2}, {'x': 1}]

    # -----------------------------
    # Tests for validate_yaml
    # -----------------------------
    def test_validate_yaml_success(self, tmp_path):
        """
        Test YAML file validation when a YAML file is created successfully
        """

        # Create a temporary YAML configuration file containing only "path"
        config_data = {
            'general': {'input_model': {'format': 'fm_netcdf', 'reference_date': '2023-01-01'}},
        }
        config_file = tmp_path / 'valid_config.yml'
        config_file.write_text(yaml.dump(config_data))

        # Instantiate the validator with the schema file.
        validator = YAMLConfigValidator()
        result = validator.validate_yaml(str(config_file))

        assert result['general']['input_model']['format'] == 'fm_netcdf'
        assert result['general']['input_model']['reference_date'] == '2023-01-01'  # Default applied

    def test_validate_yaml_validation_error(self, tmp_path):
        """
        Test YAML file validation error is generated
        """
        # Schema requires "path", so if it's missing, validation should error.
        schema = {
            '$schema': 'https://json-schema.org/draft/2020-12/schema#',
            'type': 'object',
            'properties': {'path': {'type': 'string'}},
            'required': ['path'],
        }
        # Create a temporary schema file.
        schema_file = tmp_path / 'schema.json'
        schema_file.write_text(json.dumps(schema))

        # Create a YAML file missing the required "path" property.
        config_data = {'name': 'some_name'}
        config_file = tmp_path / 'config_invalid.yml'
        config_file.write_text(yaml.dump(config_data))
        validator_instance = YAMLConfigValidator()
        with pytest.raises(YamlValidationError, match='YAML config validation error'):
            validator_instance.validate_yaml(str(config_file))

    # -----------------------------
    # Tests for export_schema
    # -----------------------------
    def test_export_schema_return_yaml_string(self, validator):
        """
        Test export_schema method returns schema as YAML string when no output file is specified
        """
        # Call export_schema without output_file parameter
        result = validator.export_schema()

        # Check that result is a string
        assert isinstance(result, str)

        # Check that the string is valid YAML by parsing it
        parsed_yaml = yaml.safe_load(result)
        assert isinstance(parsed_yaml, dict)

        # Check that the parsed YAML matches the schema content
        assert parsed_yaml == validator.schema_content

    def test_export_schema_write_to_file(self, tmp_path, validator):
        """
        Test export_schema method writes schema to file when output_file is specified
        """
        # Define the path for the temporary YAML file
        output_file = tmp_path / 'schema_output.yml'

        # Call export_schema with output_file parameter
        result = validator.export_schema(str(output_file))

        # Check that the method returns None when writing to file
        assert result is None

        # Check if the YAML file has been created
        assert output_file.exists()

        # Read the file content and verify it's correct
        with open(output_file, 'r') as f:
            file_content = f.read()

        # Parse the file content as YAML
        parsed_yaml = yaml.safe_load(file_content)

        # Check that the file content matches the schema content
        assert parsed_yaml == validator.schema_content

    def test_export_schema_file_write_error(self, validator, tmp_path):
        """
        Test export_schema method raises YamlOutputError when file cannot be written
        """
        # Create a directory path that doesn't exist and can't be created
        invalid_output_file = tmp_path / 'nonexistent_dir' / 'subdir' / 'schema.yml'

        # Ensure the parent directory doesn't exist
        assert not invalid_output_file.parent.exists()

        # Mock the file opening to raise an exception
        import unittest.mock

        with unittest.mock.patch('builtins.open', side_effect=PermissionError('Permission denied')):
            with pytest.raises(YamlOutputError, match='Error writing schema to file'):
                validator.export_schema(str(invalid_output_file))

    def test_export_schema_file_content_matches_string_output(self, tmp_path, validator):
        """
        Test that export_schema produces the same content whether returning string or writing to file
        """
        # Get schema as string
        schema_string = validator.export_schema()

        # Write schema to file
        output_file = tmp_path / 'schema_compare.yml'
        validator.export_schema(str(output_file))

        # Read file content
        with open(output_file, 'r') as f:
            file_content = f.read()

        # Both should be identical
        assert schema_string == file_content

    # -----------------------------
    # Tests for export_schema_to_yaml: file creation
    # -----------------------------
    def test_export_schema_yaml_file_creation(self, tmp_path, validator):
        """
        Test if export_schema creates a YAML file when output_file is provided
        """
        # Define the path for the temporary YAML file
        output_file = tmp_path / 'schema_output.yml'

        # Write the data to the YAML file
        validator.export_schema(str(output_file))

        # Check if the YAML file has been created
        assert output_file.exists()

        # Verify the file contains valid YAML content
        with open(output_file, 'r') as f:
            content = f.read()

        # Should be able to parse as YAML without error
        parsed_content = yaml.safe_load(content)
        assert isinstance(parsed_content, dict)
