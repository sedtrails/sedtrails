"""
Unit tests for the YAMLConfigValidator class in the sedtrails.configuration_interface.validator module.
"""

import tempfile
import pytest
import yaml
import json
from sedtrails.application_interfaces.validator import YAMLConfigValidator
from sedtrails.exceptions import YamlValidationError, YamlOutputError


@pytest.fixture
def validator():
    """Provide a YAMLConfigValidator instance for tests."""
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
        """Apply object defaults when all optional keys are missing."""
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
        """Preserve provided values while filling missing object defaults."""
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
        """Apply defaults recursively in nested object properties."""
        schema = {
            'type': 'object',
            'properties': {'outer': {'type': 'object', 'properties': {'inner': {'type': 'string', 'default': 'baz'}}}},
        }
        config = {'outer': {}}
        result = validator._apply_defaults(schema, config)
        assert result == {'outer': {'inner': 'baz'}}

    def test_apply_defaults_missing_referenced_object(self, validator):
        """Apply defaults inside a missing object property defined by a reference."""
        schema = {
            'type': 'object',
            '$defs': {
                'input_model_type': {
                    'type': 'object',
                    'properties': {
                        'format': {'type': 'string', 'default': 'fm_netcdf'},
                        'reference_date': {'type': 'string', 'default': '1970-01-01'},
                    },
                }
            },
            'properties': {
                'input_model': {'$ref': '#/$defs/input_model_type'},
            },
        }
        config = {}

        result = validator._apply_defaults(schema, config)

        assert result == {
            'input_model': {
                'format': 'fm_netcdf',
                'reference_date': '1970-01-01',
            }
        }

    def test_apply_defaults_does_not_create_missing_referenced_choice(self, validator):
        """Do not materialize unselected referenced objects in choice maps."""
        schema = {
            'type': 'object',
            '$defs': {
                'method': {
                    'type': 'object',
                    'properties': {
                        'flow_field_name': {
                            'type': 'array',
                            'default': ['depth_avg_flow_velocity'],
                        },
                    },
                }
            },
            'properties': {
                'tracer_methods': {
                    'type': 'object',
                    'properties': {
                        'selected': {'$ref': '#/$defs/method'},
                        'unselected': {'$ref': '#/$defs/method'},
                    },
                    'maxProperties': 1,
                }
            },
        }
        config = {'tracer_methods': {'selected': {}}}

        result = validator._apply_defaults(schema, config)

        assert result == {
            'tracer_methods': {
                'selected': {
                    'flow_field_name': ['depth_avg_flow_velocity'],
                }
            }
        }

    def test_apply_defaults_array(self, validator):
        """Apply item defaults for each element in an array of objects."""
        schema = {'type': 'array', 'items': {'type': 'object', 'properties': {'x': {'type': 'integer', 'default': 1}}}}
        config = [{'x': 2}, {}]
        result = validator._apply_defaults(schema, config)
        assert result == [{'x': 2}, {'x': 1}]

    def test_apply_defaults_missing_referenced_array(self, validator):
        """Apply defaults for missing array properties defined by references."""
        schema = {
            'type': 'object',
            '$defs': {
                'tags': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'default': ['active'],
                },
                'tag': {
                    'type': 'string',
                    'default': 'active',
                },
                'names': {
                    'type': 'array',
                    'items': {'$ref': '#/$defs/tag'},
                },
            },
            'properties': {
                'tags': {'$ref': '#/$defs/tags'},
                'names': {'$ref': '#/$defs/names'},
            },
        }
        config = {}

        result = validator._apply_defaults(schema, config)

        assert result == {'tags': ['active'], 'names': []}

    def test_create_config_template_includes_referenced_object_defaults(self, tmp_path):
        """Include defaults from referenced object schemas in generated templates."""
        output_file = tmp_path / 'sedtrails-template.yml'
        validator = YAMLConfigValidator()

        validator.create_config_template(str(output_file))

        template = yaml.safe_load(output_file.read_text(encoding='utf-8'))
        assert template['general']['input_model']['format'] == 'fm_netcdf'
        assert template['general']['input_model']['reference_date'] == '1970-01-01'
        assert template['general']['input_model']['morfac'] == 1

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
            'inputs': {'data': 'dummy.nc', 'repeat_eulerian_fields': True},
        }
        config_file = tmp_path / 'valid_config.yml'
        config_file.write_text(yaml.dump(config_data))

        # Instantiate the validator with the schema file.
        validator = YAMLConfigValidator()
        result = validator.validate_yaml(str(config_file))

        assert result['general']['input_model']['format'] == 'fm_netcdf'
        assert result['general']['input_model']['reference_date'] == '2023-01-01'  # Default applied
        assert result['inputs']['repeat_eulerian_fields'] is True

    def test_validate_yaml_applies_repeat_eulerian_fields_default(self, tmp_path):
        """The Eulerian forcing loop option defaults to disabled."""

        config_data = {
            'general': {'input_model': {'format': 'fm_netcdf', 'reference_date': '2023-01-01'}},
            'inputs': {'data': 'dummy.nc'},
        }
        config_file = tmp_path / 'valid_config.yml'
        config_file.write_text(yaml.dump(config_data))

        validator = YAMLConfigValidator()
        result = validator.validate_yaml(str(config_file))

        assert result['inputs']['repeat_eulerian_fields'] is False

    def test_validate_yaml_accepts_report_domain_exit_updates(self, tmp_path):
        """General config accepts optional per-timestep domain-exit update logging."""

        config_data = {
            'general': {
                'input_model': {'format': 'fm_netcdf', 'reference_date': '2023-01-01'},
                'report_domain_exit_updates': True,
            },
            'inputs': {'data': 'dummy.nc'},
        }
        config_file = tmp_path / 'valid_config.yml'
        config_file.write_text(yaml.dump(config_data))

        validator = YAMLConfigValidator()
        result = validator.validate_yaml(str(config_file))

        assert result['general']['report_domain_exit_updates'] is True

    def test_validate_yaml_accepts_output_sync_interval(self, tmp_path):
        """Output config accepts an optional NetCDF flush cadence."""

        config_data = {
            'general': {'input_model': {'format': 'fm_netcdf', 'reference_date': '2023-01-01'}},
            'inputs': {'data': 'dummy.nc'},
            'outputs': {
                'save_interval': '30M',
                'sync_interval': '2H',
                'store_tracks': True,
            },
        }
        config_file = tmp_path / 'valid_config.yml'
        config_file.write_text(yaml.dump(config_data))

        validator = YAMLConfigValidator()
        result = validator.validate_yaml(str(config_file))

        assert result['outputs']['sync_interval'] == '2H'

    def test_validate_yaml_accepts_inner_boundary_pol_files(self, tmp_path):
        """Domain config accepts one or more island/cutout polygon files."""

        config_data = {
            'general': {'input_model': {'format': 'fm_netcdf', 'reference_date': '2023-01-01'}},
            'inputs': {'data': 'dummy.nc'},
            'domain': {
                'pol_file': 'outer.pol',
                'inner_boundary_pol_files': ['island_01.pol', 'island_02.pol'],
            },
        }
        config_file = tmp_path / 'valid_config.yml'
        config_file.write_text(yaml.dump(config_data))

        validator = YAMLConfigValidator()
        result = validator.validate_yaml(str(config_file))

        assert result['domain']['inner_boundary_pol_files'] == ['island_01.pol', 'island_02.pol']

    def test_validate_yaml_rejects_non_string_inner_boundary_pol_files(self, tmp_path):
        """Inner boundary file entries must be paths encoded as strings."""

        config_data = {
            'general': {'input_model': {'format': 'fm_netcdf', 'reference_date': '2023-01-01'}},
            'inputs': {'data': 'dummy.nc'},
            'domain': {
                'pol_file': 'outer.pol',
                'inner_boundary_pol_files': [123],
            },
        }
        config_file = tmp_path / 'invalid_config.yml'
        config_file.write_text(yaml.dump(config_data))

        validator = YAMLConfigValidator()
        with pytest.raises(YamlValidationError, match='YAML config validation error'):
            validator.validate_yaml(str(config_file))

    def test_validate_yaml_accepts_boundary_class_pol_files(self, tmp_path):
        """Domain config accepts open and land boundary override polygon files."""

        config_data = {
            'general': {'input_model': {'format': 'fm_netcdf', 'reference_date': '2023-01-01'}},
            'inputs': {'data': 'dummy.nc'},
            'domain': {
                'pol_file': 'outer.pol',
                'boundary_class_pol_files': {
                    'open': ['offshore_01.pol', 'offshore_02.pol'],
                    'land': ['coastline.pol'],
                },
            },
        }
        config_file = tmp_path / 'valid_config.yml'
        config_file.write_text(yaml.dump(config_data))

        validator = YAMLConfigValidator()
        result = validator.validate_yaml(str(config_file))

        assert result['domain']['boundary_class_pol_files']['open'] == ['offshore_01.pol', 'offshore_02.pol']
        assert result['domain']['boundary_class_pol_files']['land'] == ['coastline.pol']

    def test_validate_yaml_rejects_boundary_class_pol_files_without_domain_extent(self, tmp_path):
        """Boundary class polygons modify a domain but do not define one."""

        config_data = {
            'general': {'input_model': {'format': 'fm_netcdf', 'reference_date': '2023-01-01'}},
            'inputs': {'data': 'dummy.nc'},
            'domain': {
                'boundary_class_pol_files': {
                    'open': ['offshore.pol'],
                },
            },
        }
        config_file = tmp_path / 'invalid_config.yml'
        config_file.write_text(yaml.dump(config_data))

        validator = YAMLConfigValidator()
        with pytest.raises(YamlValidationError, match='YAML config validation error'):
            validator.validate_yaml(str(config_file))

    def test_validate_yaml_rejects_non_string_boundary_class_pol_files(self, tmp_path):
        """Boundary override file entries must be paths encoded as strings."""

        config_data = {
            'general': {'input_model': {'format': 'fm_netcdf', 'reference_date': '2023-01-01'}},
            'inputs': {'data': 'dummy.nc'},
            'domain': {
                'pol_file': 'outer.pol',
                'boundary_class_pol_files': {
                    'open': [123],
                },
            },
        }
        config_file = tmp_path / 'invalid_config.yml'
        config_file.write_text(yaml.dump(config_data))

        validator = YAMLConfigValidator()
        with pytest.raises(YamlValidationError, match='YAML config validation error'):
            validator.validate_yaml(str(config_file))

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
