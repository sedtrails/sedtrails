:py:mod:`sedtrails.configuration_interface.cli.main`
====================================================

.. py:module:: sedtrails.configuration_interface.cli.main

.. autodoc2-docstring:: sedtrails.configuration_interface.cli.main
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`version_callback <sedtrails.configuration_interface.cli.main.version_callback>`
     - .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.version_callback
          :summary:
   * - :py:obj:`main <sedtrails.configuration_interface.cli.main.main>`
     - .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.main
          :summary:
   * - :py:obj:`load_config <sedtrails.configuration_interface.cli.main.load_config>`
     - .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.load_config
          :summary:
   * - :py:obj:`create_config_template <sedtrails.configuration_interface.cli.main.create_config_template>`
     - .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.create_config_template
          :summary:
   * - :py:obj:`run_simulation <sedtrails.configuration_interface.cli.main.run_simulation>`
     - .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.run_simulation
          :summary:
   * - :py:obj:`analyze <sedtrails.configuration_interface.cli.main.analyze>`
     - .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.analyze
          :summary:
   * - :py:obj:`network_analysis <sedtrails.configuration_interface.cli.main.network_analysis>`
     - .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.network_analysis
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`app <sedtrails.configuration_interface.cli.main.app>`
     - .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.app
          :summary:
   * - :py:obj:`config_app <sedtrails.configuration_interface.cli.main.config_app>`
     - .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.config_app
          :summary:

API
~~~

.. py:function:: version_callback(value: bool)
   :canonical: sedtrails.configuration_interface.cli.main.version_callback

   .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.version_callback

.. py:data:: app
   :canonical: sedtrails.configuration_interface.cli.main.app
   :value: 'Typer(...)'

   .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.app

.. py:data:: config_app
   :canonical: sedtrails.configuration_interface.cli.main.config_app
   :value: 'Typer(...)'

   .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.config_app

.. py:function:: main(version: bool = typer.Option(False, '--version', '-v', callback=version_callback, is_eager=True, help='Show version and exit.'))
   :canonical: sedtrails.configuration_interface.cli.main.main

   .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.main

.. py:function:: load_config(config_file: str = typer.Option('sedtrails.yml', '--config', '-c', help='Path to the SedTRAILS configuration file.')) -> dict
   :canonical: sedtrails.configuration_interface.cli.main.load_config

   .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.load_config

.. py:function:: create_config_template(output_file: str = typer.Option('./sedtrails-template.yml', '--output', '-o', help='Path to the output configuration template file.'))
   :canonical: sedtrails.configuration_interface.cli.main.create_config_template

   .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.create_config_template

.. py:function:: run_simulation(config_file: str = typer.Option('sedtrails.yml', '--config', '-c', help='Path to the SedTRAILS configuration file.'), output_file: str = typer.Option('sedtrails.nc', '--output', '-o', help='Path to the output SedTRAILS netCDF file.'))
   :canonical: sedtrails.configuration_interface.cli.main.run_simulation

   .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.run_simulation

.. py:function:: analyze(input_file: pathlib.Path = typer.Option('sedtrails.nc', '--input', '-i', help='Input SedTRAILS netCDF file containing particle tracks.'), output_file: pathlib.Path = typer.Option('analysis.nc', '--output', '-o', help='Output SedTRAILS netCDF file containing statistical and connectivity results.'))
   :canonical: sedtrails.configuration_interface.cli.main.analyze

   .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.analyze

.. py:function:: network_analysis(input_file: pathlib.Path = typer.Option('sedtrails.nc', '--input', '-i', help='Input netCDF file containing particle tracking results.'), output_file: pathlib.Path = typer.Option('analysis.nc', '--output', '-o', help='Path to the output SedTRAILS netCDF file containing statistical and connectivity results.'))
   :canonical: sedtrails.configuration_interface.cli.main.network_analysis

   .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.network_analysis
