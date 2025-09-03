:py:mod:`sedtrails.simulation`
==============================

.. py:module:: sedtrails.simulation

.. autodoc2-docstring:: sedtrails.simulation
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Simulation <sedtrails.simulation.Simulation>`
     - .. autodoc2-docstring:: sedtrails.simulation.Simulation
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`setup_global_exception_logging <sedtrails.simulation.setup_global_exception_logging>`
     - .. autodoc2-docstring:: sedtrails.simulation.setup_global_exception_logging
          :summary:

API
~~~

.. py:function:: setup_global_exception_logging(logger_manager)
   :canonical: sedtrails.simulation.setup_global_exception_logging

   .. autodoc2-docstring:: sedtrails.simulation.setup_global_exception_logging

.. py:class:: Simulation(config_file: str)
   :canonical: sedtrails.simulation.Simulation

   .. autodoc2-docstring:: sedtrails.simulation.Simulation

   .. rubric:: Initialization

   .. autodoc2-docstring:: sedtrails.simulation.Simulation.__init__

   .. py:method:: _create_dashboard()
      :canonical: sedtrails.simulation.Simulation._create_dashboard

      .. autodoc2-docstring:: sedtrails.simulation.Simulation._create_dashboard

   .. py:method:: _get_format_config()
      :canonical: sedtrails.simulation.Simulation._get_format_config

      .. autodoc2-docstring:: sedtrails.simulation.Simulation._get_format_config

   .. py:method:: _get_output_dir()
      :canonical: sedtrails.simulation.Simulation._get_output_dir

      .. autodoc2-docstring:: sedtrails.simulation.Simulation._get_output_dir

   .. py:method:: _get_physics_config()
      :canonical: sedtrails.simulation.Simulation._get_physics_config

      .. autodoc2-docstring:: sedtrails.simulation.Simulation._get_physics_config

   .. py:property:: config
      :canonical: sedtrails.simulation.Simulation.config

      .. autodoc2-docstring:: sedtrails.simulation.Simulation.config

   .. py:property:: population_config
      :canonical: sedtrails.simulation.Simulation.population_config

      .. autodoc2-docstring:: sedtrails.simulation.Simulation.population_config

   .. py:property:: start_time
      :canonical: sedtrails.simulation.Simulation.start_time

      .. autodoc2-docstring:: sedtrails.simulation.Simulation.start_time

   .. py:property:: flow_field
      :canonical: sedtrails.simulation.Simulation.flow_field
      :type: sedtrails.transport_converter.format_converter.SedtrailsData

      .. autodoc2-docstring:: sedtrails.simulation.Simulation.flow_field

   .. py:method:: validate_config() -> bool
      :canonical: sedtrails.simulation.Simulation.validate_config

      .. autodoc2-docstring:: sedtrails.simulation.Simulation.validate_config

   .. py:method:: get_parameter(key: str) -> typing.Any
      :canonical: sedtrails.simulation.Simulation.get_parameter

      .. autodoc2-docstring:: sedtrails.simulation.Simulation.get_parameter

   .. py:method:: run()
      :canonical: sedtrails.simulation.Simulation.run

      .. autodoc2-docstring:: sedtrails.simulation.Simulation.run
