:py:mod:`sedtrails.transport_converter.sedtrails_data`
======================================================

.. py:module:: sedtrails.transport_converter.sedtrails_data

.. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_data
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`SedtrailsData <sedtrails.transport_converter.sedtrails_data.SedtrailsData>`
     - .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_data.SedtrailsData
          :summary:

API
~~~

.. py:class:: SedtrailsData
   :canonical: sedtrails.transport_converter.sedtrails_data.SedtrailsData

   .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_data.SedtrailsData

   .. py:attribute:: times
      :canonical: sedtrails.transport_converter.sedtrails_data.SedtrailsData.times
      :type: numpy.ndarray
      :value: None

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_data.SedtrailsData.times

   .. py:attribute:: reference_date
      :canonical: sedtrails.transport_converter.sedtrails_data.SedtrailsData.reference_date
      :type: numpy.datetime64
      :value: None

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_data.SedtrailsData.reference_date

   .. py:attribute:: x
      :canonical: sedtrails.transport_converter.sedtrails_data.SedtrailsData.x
      :type: numpy.ndarray
      :value: None

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_data.SedtrailsData.x

   .. py:attribute:: y
      :canonical: sedtrails.transport_converter.sedtrails_data.SedtrailsData.y
      :type: numpy.ndarray
      :value: None

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_data.SedtrailsData.y

   .. py:attribute:: bed_level
      :canonical: sedtrails.transport_converter.sedtrails_data.SedtrailsData.bed_level
      :type: numpy.ndarray
      :value: None

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_data.SedtrailsData.bed_level

   .. py:attribute:: depth_avg_flow_velocity
      :canonical: sedtrails.transport_converter.sedtrails_data.SedtrailsData.depth_avg_flow_velocity
      :type: typing.Dict[str, numpy.ndarray]
      :value: None

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_data.SedtrailsData.depth_avg_flow_velocity

   .. py:attribute:: fractions
      :canonical: sedtrails.transport_converter.sedtrails_data.SedtrailsData.fractions
      :type: int
      :value: None

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_data.SedtrailsData.fractions

   .. py:attribute:: bed_load_transport
      :canonical: sedtrails.transport_converter.sedtrails_data.SedtrailsData.bed_load_transport
      :type: typing.Dict[str, numpy.ndarray]
      :value: None

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_data.SedtrailsData.bed_load_transport

   .. py:attribute:: suspended_transport
      :canonical: sedtrails.transport_converter.sedtrails_data.SedtrailsData.suspended_transport
      :type: typing.Dict[str, numpy.ndarray]
      :value: None

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_data.SedtrailsData.suspended_transport

   .. py:attribute:: water_depth
      :canonical: sedtrails.transport_converter.sedtrails_data.SedtrailsData.water_depth
      :type: numpy.ndarray
      :value: None

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_data.SedtrailsData.water_depth

   .. py:attribute:: mean_bed_shear_stress
      :canonical: sedtrails.transport_converter.sedtrails_data.SedtrailsData.mean_bed_shear_stress
      :type: numpy.ndarray
      :value: None

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_data.SedtrailsData.mean_bed_shear_stress

   .. py:attribute:: max_bed_shear_stress
      :canonical: sedtrails.transport_converter.sedtrails_data.SedtrailsData.max_bed_shear_stress
      :type: numpy.ndarray
      :value: None

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_data.SedtrailsData.max_bed_shear_stress

   .. py:attribute:: sediment_concentration
      :canonical: sedtrails.transport_converter.sedtrails_data.SedtrailsData.sediment_concentration
      :type: numpy.ndarray
      :value: None

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_data.SedtrailsData.sediment_concentration

   .. py:attribute:: nonlinear_wave_velocity
      :canonical: sedtrails.transport_converter.sedtrails_data.SedtrailsData.nonlinear_wave_velocity
      :type: typing.Dict[str, numpy.ndarray]
      :value: None

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_data.SedtrailsData.nonlinear_wave_velocity

   .. py:attribute:: metadata
      :canonical: sedtrails.transport_converter.sedtrails_data.SedtrailsData.metadata
      :type: sedtrails.transport_converter.sedtrails_metadata.SedtrailsMetadata
      :value: None

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_data.SedtrailsData.metadata

   .. py:method:: __post_init__()
      :canonical: sedtrails.transport_converter.sedtrails_data.SedtrailsData.__post_init__

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_data.SedtrailsData.__post_init__

   .. py:method:: _calculate_timestep()
      :canonical: sedtrails.transport_converter.sedtrails_data.SedtrailsData._calculate_timestep

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_data.SedtrailsData._calculate_timestep

   .. py:method:: _compute_grid_metadata()
      :canonical: sedtrails.transport_converter.sedtrails_data.SedtrailsData._compute_grid_metadata

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_data.SedtrailsData._compute_grid_metadata

   .. py:method:: _validate_metadata()
      :canonical: sedtrails.transport_converter.sedtrails_data.SedtrailsData._validate_metadata

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_data.SedtrailsData._validate_metadata

   .. py:method:: add_physics_field(name: str, data)
      :canonical: sedtrails.transport_converter.sedtrails_data.SedtrailsData.add_physics_field

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_data.SedtrailsData.add_physics_field

   .. py:method:: has_physics_field(name: str) -> bool
      :canonical: sedtrails.transport_converter.sedtrails_data.SedtrailsData.has_physics_field

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_data.SedtrailsData.has_physics_field

   .. py:method:: get_physics_fields() -> list
      :canonical: sedtrails.transport_converter.sedtrails_data.SedtrailsData.get_physics_fields

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_data.SedtrailsData.get_physics_fields

   .. py:method:: has_physics_data() -> bool
      :canonical: sedtrails.transport_converter.sedtrails_data.SedtrailsData.has_physics_data

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_data.SedtrailsData.has_physics_data

   .. py:method:: __getitem__(time_index: int) -> typing.Dict
      :canonical: sedtrails.transport_converter.sedtrails_data.SedtrailsData.__getitem__

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_data.SedtrailsData.__getitem__
