import json
from copy import deepcopy
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, Optional

import jsonschema
import yaml

from sedtrails.exceptions import YamlOutputError, YamlParsingError, YamlValidationError

# Schema file names (will be resolved using importlib.resources or pkg_resources)
ROOT_SCHEMA = 'main.schema.json'
REF_SCHEMAS = [
    # Add sub-schemas here
    'population.schema.json',
    'visualization.schema.json',
]


class SedtrailsYamlLoader(yaml.SafeLoader):
    """
    Custom YAML loader that avoids converting datetime strings to datetime objects.
    A custom YAML loader is necessary because default loader always converts datetime strings to datetime objects
    We want to keep datetime as strings, for convenience conversions are handled internally
    by the SedTRAILS configuration interface"""

    pass


# Add a constructor that treats timestamps as strings instead of datetime objects
def construct_timestamp_as_string(loader, node):
    """
    Construct timestamp nodes as strings instead of datetime objects.

    Parameters
    ----------
    loader : object
        YAML loader instance.
    node : object
        YAML scalar node to construct.

    Returns
    -------
    str
        Scalar timestamp text constructed from the YAML node.
    """
    return loader.construct_scalar(node)


SedtrailsYamlLoader.add_constructor('tag:yaml.org,2002:timestamp', construct_timestamp_as_string)


class YAMLConfigValidator:
    """
    A class to load, validate a YAML configuration file based on a JSON Schema.

    Attributes
    ----------
    config : Dict[str, Any]
        The validated configuration data as a nested dictionary.
    """

    def __init__(self) -> None:
        """
        Initialize a validator to validate SedTRAILS configuration files written in YAML.

        """

        self.config: Dict[str, Any] = {}
        self.raw_config: Dict[str, Any] = {}
        self._applied_defaults: bool = False
        self.__registry = None
        self.__validator = self._validator()

    def _load_schema_from_file(self, schema_file: str) -> Dict[str, Any]:
        """
        Loads the JSON schema from a package resource file.

        Parameters
        ----------
        schema_file : str
            The name of the JSON schema file (e.g., 'main.schema.json').

        Returns
        -------
        Dict[str, Any]
            The loaded schema as a dictionary.

        Raises
        ------
        json.JSONDecodeError
            If the schema file is not a valid JSON file.
        FileNotFoundError
            If the schema file cannot be found in the package resources.
        """
        try:
            # Use importlib.resources to access schema files in the package
            config_files = files('sedtrails') / 'config'
            schema_file_path = config_files / schema_file

            # Read the schema file content
            with schema_file_path.open('r', encoding='utf-8') as f:
                schema_data: Any = json.load(f)

        except json.JSONDecodeError as err:
            raise err
        except FileNotFoundError as err:
            raise FileNotFoundError(f'Schema file {schema_file} not found in package resources') from err
        except Exception as err:
            raise RuntimeError(f'Error loading schema file {schema_file}: {err}') from err
        else:
            return schema_data

    def _apply_defaults(self, schema_content: Dict[str, Any], config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively applies default values from the schema to the data dictionary.
        If a default is specified as a dictionary with "$ref" and transformation keys,
        it is computed accordingly.

        Parameters
        ----------
        schema_content : dict
            The JSON schema (or subschema).
        config_data : dict
            The configuration data to which defaults should be applied.

        Returns
        -------
        dict
            The configuration data with defaults applied.

        Raises
        ------
        ValueError
            If the default value cannot be resolved.
        """
        return self._apply_defaults_with_resolver(schema_content, config_data, self.__validator)

    def _apply_defaults_with_resolver(
        self,
        schema_content: Dict[str, Any],
        config_data: Dict[str, Any],
        validator: jsonschema.Draft202012Validator,
        schema_root: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Internal method that applies defaults with access to the schema resolver.
        """
        if schema_root is None:
            schema_root = schema_content

        schema_type = schema_content.get('type')

        # Handle $ref references
        if '$ref' in schema_content:
            resolved_schema = self._resolve_schema_reference(schema_content, schema_root)
            if resolved_schema is not schema_content:
                resolved_root = self._schema_root_for_reference(schema_content, resolved_schema, schema_root)
                return self._apply_defaults_with_resolver(resolved_schema, config_data, validator, resolved_root)
            return config_data

        if schema_type == 'object' and isinstance(config_data, dict):
            properties = schema_content.get('properties', {})

            # Handle allOf, anyOf, oneOf schemas
            for conditional_key in ['allOf', 'anyOf', 'oneOf']:
                if conditional_key in schema_content:
                    for sub_schema in schema_content[conditional_key]:
                        if conditional_key == 'allOf' or self._schema_matches(sub_schema, config_data, validator):
                            config_data = self._apply_defaults_with_resolver(
                                sub_schema, config_data, validator, schema_root
                            )

            for key, prop_schema in properties.items():
                resolved_prop_schema = self._resolve_schema_reference(prop_schema, schema_root)
                resolved_prop_root = self._schema_root_for_reference(prop_schema, resolved_prop_schema, schema_root)
                if key not in config_data:
                    can_create_missing_container = self._can_create_missing_container(schema_content)
                    # Create missing property with default value
                    if 'default' in prop_schema:
                        config_data[key] = self._deep_copy_default(prop_schema['default'])
                    elif 'default' in resolved_prop_schema and (
                        resolved_prop_schema.get('type') not in {'object', 'array'} or can_create_missing_container
                    ):
                        config_data[key] = self._deep_copy_default(resolved_prop_schema['default'])
                    elif can_create_missing_container and resolved_prop_schema.get('type') == 'object':
                        # Create empty object and apply defaults recursively
                        config_data[key] = {}
                        config_data[key] = self._apply_defaults_with_resolver(
                            resolved_prop_schema, config_data[key], validator, resolved_prop_root
                        )
                    elif (
                        can_create_missing_container
                        and resolved_prop_schema.get('type') == 'array'
                        and 'items' in resolved_prop_schema
                        and self._array_items_define_default(resolved_prop_schema, resolved_prop_root)
                    ):
                        # Handle arrays with default items
                        config_data[key] = []
                else:
                    # Property exists, apply defaults recursively if it's an object or array
                    if resolved_prop_schema.get('type') == 'object' and isinstance(config_data[key], dict):
                        config_data[key] = self._apply_defaults_with_resolver(
                            resolved_prop_schema, config_data[key], validator, resolved_prop_root
                        )
                    elif resolved_prop_schema.get('type') == 'array' and isinstance(config_data[key], list):
                        raw_item_schema = resolved_prop_schema.get('items', {})
                        item_schema = self._resolve_schema_reference(raw_item_schema, resolved_prop_root)
                        item_schema_root = self._schema_root_for_reference(
                            raw_item_schema, item_schema, resolved_prop_root
                        )
                        for i, item in enumerate(config_data[key]):
                            if isinstance(item, dict) and item_schema.get('type') == 'object':
                                config_data[key][i] = self._apply_defaults_with_resolver(
                                    item_schema, item, validator, item_schema_root
                                )

        elif schema_type == 'array' and isinstance(config_data, list):
            raw_item_schema = schema_content.get('items', {})
            item_schema = self._resolve_schema_reference(raw_item_schema, schema_root)
            item_schema_root = self._schema_root_for_reference(raw_item_schema, item_schema, schema_root)
            for i, item in enumerate(config_data):
                if isinstance(item, dict) and item_schema.get('type') == 'object':
                    config_data[i] = self._apply_defaults_with_resolver(item_schema, item, validator, item_schema_root)

        return config_data

    @staticmethod
    def _can_create_missing_container(schema_content: Dict[str, Any]) -> bool:
        """Return whether missing object or array properties should be materialized."""
        choice_keywords = {'anyOf', 'oneOf', 'maxProperties'}
        return not any(keyword in schema_content for keyword in choice_keywords)

    def _array_items_define_default(self, schema_content: Dict[str, Any], schema_root: Dict[str, Any]) -> bool:
        """Return whether an array schema's items define a default."""
        raw_item_schema = schema_content.get('items', {})
        item_schema = self._resolve_schema_reference(raw_item_schema, schema_root)
        return 'default' in item_schema

    def _resolve_schema_reference(
        self, schema_content: Dict[str, Any], schema_root: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Resolve a schema ``$ref`` if present."""
        if '$ref' not in schema_content:
            return schema_content

        ref = schema_content['$ref']
        if ref.startswith('#/') and schema_root is not None:
            try:
                resolved_schema = schema_root
                for part in ref[2:].split('/'):
                    resolved_schema = resolved_schema[part]
            except (KeyError, TypeError):
                print(f'Warning: Could not resolve $ref {ref}')
                return schema_content
            return resolved_schema

        try:
            resolver = self.__registry.resolver()
            resolved = resolver.lookup(ref)
        except Exception as e:
            print(f'Warning: Could not resolve $ref {ref}: {e}')
            return schema_content

        return resolved.contents

    @staticmethod
    def _schema_root_for_reference(
        schema_content: Dict[str, Any], resolved_schema: Dict[str, Any], schema_root: Dict[str, Any]
    ) -> Dict[str, Any]:
        ref = schema_content.get('$ref')
        if ref and not ref.startswith('#/'):
            return resolved_schema
        return schema_root

    def _schema_matches(
        self, schema: Dict[str, Any], data: Dict[str, Any], validator: jsonschema.Draft202012Validator
    ) -> bool:
        """
        Check if data matches a schema (used for anyOf/oneOf conditions).
        """
        try:
            # Create a temporary validator for this schema
            temp_validator = jsonschema.Draft202012Validator(schema, registry=self.__registry)
            temp_validator.validate(data)
            return True
        except jsonschema.ValidationError:
            return False

    def _deep_copy_default(self, value: Any) -> Any:
        """
        Deep copy default values to avoid reference issues.
        """
        import copy

        return copy.deepcopy(value)

    def _get_root_schema_content(self) -> Dict[str, Any]:
        """
        Get the root schema content from the validator.
        """
        return self.__validator.schema

    @property
    def schema_content(self) -> Dict[str, Any]:
        """
        Property to access the root schema content.

        Returns
        -------
        Dict[str, Any]
            The schema content value.
        """
        return self._get_root_schema_content()

    def _validator(self) -> jsonschema.Draft202012Validator:
        """
        Creates a JSON schema validator using the SedTRAILS schema.

        Returns
        -------
        jsonschema.Draft202012Validator
            A validator instance that can be used to validate YAML configurations
            against the SedTRAILS JSON schema.

        """

        from referencing import Registry, Resource
        from referencing.jsonschema import DRAFT202012

        registry = Registry()

        root_schema_content = self._load_schema_from_file(ROOT_SCHEMA)

        for schema_file in REF_SCHEMAS:
            schema_content = self._load_schema_from_file(schema_file)
            resource = Resource(contents=schema_content, specification=DRAFT202012)
            uri = f'urn:sedtrails:config:{Path(schema_file).name}'
            registry = registry.with_resource(uri=uri, resource=resource)

        self.__registry = registry

        validator_class = jsonschema.Draft202012Validator
        validator = validator_class(schema=root_schema_content, registry=registry)
        try:
            validator.check_schema(root_schema_content)
        except jsonschema.SchemaError as e:
            raise ValueError(f'Schema validation error: {e.message}') from e

        return validator

    def validate_yaml(self, yml_filepath: str) -> Dict[str, Any]:
        """
        Loads the YAML file, validates it against the SedTRAILS JSON schema,
        applies default values (including processing default folder directives), and
        returns the resulting configuration as a nested dictionary.

        Parameters
        ----------
        yml_filepath : str
            The path to the YAML configuration file to validate.

        Returns
        -------
        dict
            The validated and default-populated configuration as a nested dictionary.

        Raises
        ------
        YamlParsingError
            If the YAML file cannot be read
        YamlValidationError
            If the YAML configuration is invalid.
        ValueError
            If there is an error applying default values from the schema.
        """

        try:
            with open(yml_filepath, 'r') as f:
                yaml_data: Dict[str, Any] = yaml.load(f, Loader=SedtrailsYamlLoader)
        except Exception as e:
            raise YamlParsingError(f'Error reading YAML file: {e}') from e
        self.raw_config = deepcopy(yaml_data)

        # Validate the YAML data against the schema
        try:
            # jsonschema.validate(instance=yaml_data, schema=self.schema_content, resolver=new_resolver)
            self.__validator.validate(yaml_data)
        except jsonschema.ValidationError as e:
            raise YamlValidationError(f'YAML config validation error: {e.message}') from e

        # apply default values from the schema
        if self._applied_defaults:  # skip applying defaults if already done
            return self.config

        # Apply default values from the schema
        try:
            config_with_defaults = self._apply_defaults(self.schema_content, deepcopy(yaml_data))
            self.config = config_with_defaults
            self._applied_defaults = True
        except Exception as e:
            raise ValueError(f'Error applying defaults: {e}') from e

        return self.config

    def export_schema(self, output_file: Optional[str] = None) -> str | None:
        """
        Exports the JSON schema as a valid YAML string.

        Parameters
        ----------
        output_file : str, optional
            If provided, the YAML schema is written to this file.

        Returns
        -------
        str
            The schema as a YAML formatted string, if no output file is specified.
        If an output file is specified, the schema is written to that file.

        Raises
        ------
        YamlOutputError
            If there is an error while writing the schema to the file.
        """
        schema_yaml: str = yaml.dump(self.schema_content, sort_keys=False)
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(schema_yaml)
            except Exception as e:
                raise YamlOutputError(f'Error writing schema to file: {e}') from e
        else:
            return schema_yaml

    def export_config(self, output_file: str = './example.sedtrails.yaml') -> None:
        """
        Exports the validated configuration as a valid YAML string.

        Parameters
        ----------
        output_file : str
            path to file where the YAML configuration is written.

        Returns
        -------
        The configuration is written to that file.

        Raises
        ------
        YamlOutputError
            If there is an error while writing the configuration to the file.
        """

        if not self._applied_defaults:
            self._apply_defaults(self.schema_content, self.config)
        print(f'Applied defaults to config: {self.config}')

        config_yaml: str = yaml.dump(self.config, sort_keys=False)

        print(f'config yaml: {config_yaml}')
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(config_yaml)
            except Exception as e:
                raise YamlOutputError(f'Error writing config to file: {e}') from e

    def create_config_template(self, output_file: str = './sedtrails.template.yaml') -> None:
        """
        Creates a configuration template with all schema defaults applied.

        Parameters
        ----------
        output_file : str
            Path to file where the YAML configuration template is written.

        Raises
        ------
        YamlOutputError
            If there is an error while writing the configuration to the file.
        """

        try:
            # Create empty config and apply all schema defaults
            # empty_config = {}
            self._apply_defaults(self.schema_content, self.config)

            # Create output directory if needed
            from pathlib import Path

            output_path = Path(output_file)

            # Convert to YAML
            config_yaml = yaml.dump(
                self.config, default_flow_style=False, sort_keys=False, indent=2, allow_unicode=True
            )

            # Write to file with header
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('# SedTRAILS Configuration Template\n')
                f.write('# Generated using defaults for configuration properties\n\n')
                f.write(config_yaml)

            print(f'Configuration template exported to: {output_path.absolute()}')

        except Exception as e:
            raise YamlOutputError(f'Error writing config template to file: {e}') from e


if __name__ == '__main__':
    validator = YAMLConfigValidator()

    data = validator.validate_yaml('examples/sedtrails-example.yaml')

    print(f'Validated data: {data}')
