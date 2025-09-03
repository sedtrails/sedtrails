:py:mod:`sedtrails.configuration_interface.validator`
=====================================================

.. py:module:: sedtrails.configuration_interface.validator

.. autodoc2-docstring:: sedtrails.configuration_interface.validator
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`YAMLConfigValidator <sedtrails.configuration_interface.validator.YAMLConfigValidator>`
     - .. autodoc2-docstring:: sedtrails.configuration_interface.validator.YAMLConfigValidator
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`ROOT_SCHEMA <sedtrails.configuration_interface.validator.ROOT_SCHEMA>`
     - .. autodoc2-docstring:: sedtrails.configuration_interface.validator.ROOT_SCHEMA
          :summary:
   * - :py:obj:`REF_SCHEMAS <sedtrails.configuration_interface.validator.REF_SCHEMAS>`
     - .. autodoc2-docstring:: sedtrails.configuration_interface.validator.REF_SCHEMAS
          :summary:

API
~~~

.. py:data:: ROOT_SCHEMA
   :canonical: sedtrails.configuration_interface.validator.ROOT_SCHEMA
   :value: 'main.schema.json'

   .. autodoc2-docstring:: sedtrails.configuration_interface.validator.ROOT_SCHEMA

.. py:data:: REF_SCHEMAS
   :canonical: sedtrails.configuration_interface.validator.REF_SCHEMAS
   :value: ['population.schema.json', 'visualization.schema.json']

   .. autodoc2-docstring:: sedtrails.configuration_interface.validator.REF_SCHEMAS

.. py:class:: YAMLConfigValidator()
   :canonical: sedtrails.configuration_interface.validator.YAMLConfigValidator

   .. autodoc2-docstring:: sedtrails.configuration_interface.validator.YAMLConfigValidator

   .. rubric:: Initialization

   .. autodoc2-docstring:: sedtrails.configuration_interface.validator.YAMLConfigValidator.__init__

   .. py:method:: _load_schema_from_file(schema_file: str) -> typing.Dict[str, typing.Any]
      :canonical: sedtrails.configuration_interface.validator.YAMLConfigValidator._load_schema_from_file

      .. autodoc2-docstring:: sedtrails.configuration_interface.validator.YAMLConfigValidator._load_schema_from_file

   .. py:method:: _apply_defaults(schema_content: typing.Dict[str, typing.Any], config_data: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]
      :canonical: sedtrails.configuration_interface.validator.YAMLConfigValidator._apply_defaults

      .. autodoc2-docstring:: sedtrails.configuration_interface.validator.YAMLConfigValidator._apply_defaults

   .. py:method:: _apply_defaults_with_resolver(schema_content: typing.Dict[str, typing.Any], config_data: typing.Dict[str, typing.Any], validator: jsonschema.Draft202012Validator) -> typing.Dict[str, typing.Any]
      :canonical: sedtrails.configuration_interface.validator.YAMLConfigValidator._apply_defaults_with_resolver

      .. autodoc2-docstring:: sedtrails.configuration_interface.validator.YAMLConfigValidator._apply_defaults_with_resolver

   .. py:method:: _schema_matches(schema: typing.Dict[str, typing.Any], data: typing.Dict[str, typing.Any], validator: jsonschema.Draft202012Validator) -> bool
      :canonical: sedtrails.configuration_interface.validator.YAMLConfigValidator._schema_matches

      .. autodoc2-docstring:: sedtrails.configuration_interface.validator.YAMLConfigValidator._schema_matches

   .. py:method:: _deep_copy_default(value: typing.Any) -> typing.Any
      :canonical: sedtrails.configuration_interface.validator.YAMLConfigValidator._deep_copy_default

      .. autodoc2-docstring:: sedtrails.configuration_interface.validator.YAMLConfigValidator._deep_copy_default

   .. py:method:: _get_root_schema_content() -> typing.Dict[str, typing.Any]
      :canonical: sedtrails.configuration_interface.validator.YAMLConfigValidator._get_root_schema_content

      .. autodoc2-docstring:: sedtrails.configuration_interface.validator.YAMLConfigValidator._get_root_schema_content

   .. py:property:: schema_content
      :canonical: sedtrails.configuration_interface.validator.YAMLConfigValidator.schema_content
      :type: typing.Dict[str, typing.Any]

      .. autodoc2-docstring:: sedtrails.configuration_interface.validator.YAMLConfigValidator.schema_content

   .. py:method:: _validator() -> jsonschema.Draft202012Validator
      :canonical: sedtrails.configuration_interface.validator.YAMLConfigValidator._validator

      .. autodoc2-docstring:: sedtrails.configuration_interface.validator.YAMLConfigValidator._validator

   .. py:method:: validate_yaml(yml_filepath: str) -> typing.Dict[str, typing.Any]
      :canonical: sedtrails.configuration_interface.validator.YAMLConfigValidator.validate_yaml

      .. autodoc2-docstring:: sedtrails.configuration_interface.validator.YAMLConfigValidator.validate_yaml

   .. py:method:: export_schema(output_file: typing.Optional[str] = None) -> str | None
      :canonical: sedtrails.configuration_interface.validator.YAMLConfigValidator.export_schema

      .. autodoc2-docstring:: sedtrails.configuration_interface.validator.YAMLConfigValidator.export_schema

   .. py:method:: export_config(output_file: str = './example.sedtrails.yaml') -> None
      :canonical: sedtrails.configuration_interface.validator.YAMLConfigValidator.export_config

      .. autodoc2-docstring:: sedtrails.configuration_interface.validator.YAMLConfigValidator.export_config

   .. py:method:: create_config_template(output_file: str = './sedtrails.template.yaml') -> None
      :canonical: sedtrails.configuration_interface.validator.YAMLConfigValidator.create_config_template

      .. autodoc2-docstring:: sedtrails.configuration_interface.validator.YAMLConfigValidator.create_config_template
