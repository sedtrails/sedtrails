:py:mod:`sedtrails.transport_converter.physics_converter`
=========================================================

.. py:module:: sedtrails.transport_converter.physics_converter

.. autodoc2-docstring:: sedtrails.transport_converter.physics_converter
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`PhysicsConfig <sedtrails.transport_converter.physics_converter.PhysicsConfig>`
     - .. autodoc2-docstring:: sedtrails.transport_converter.physics_converter.PhysicsConfig
          :summary:
   * - :py:obj:`PhysicsConverter <sedtrails.transport_converter.physics_converter.PhysicsConverter>`
     - .. autodoc2-docstring:: sedtrails.transport_converter.physics_converter.PhysicsConverter
          :summary:

API
~~~

.. py:class:: PhysicsConfig
   :canonical: sedtrails.transport_converter.physics_converter.PhysicsConfig

   .. autodoc2-docstring:: sedtrails.transport_converter.physics_converter.PhysicsConfig

   .. py:attribute:: tracer_method
      :canonical: sedtrails.transport_converter.physics_converter.PhysicsConfig.tracer_method
      :type: str
      :value: 'van_westen'

      .. autodoc2-docstring:: sedtrails.transport_converter.physics_converter.PhysicsConfig.tracer_method

   .. py:attribute:: gravity
      :canonical: sedtrails.transport_converter.physics_converter.PhysicsConfig.gravity
      :type: float
      :value: 9.81

      .. autodoc2-docstring:: sedtrails.transport_converter.physics_converter.PhysicsConfig.gravity

   .. py:attribute:: von_karman_constant
      :canonical: sedtrails.transport_converter.physics_converter.PhysicsConfig.von_karman_constant
      :type: float
      :value: 0.4

      .. autodoc2-docstring:: sedtrails.transport_converter.physics_converter.PhysicsConfig.von_karman_constant

   .. py:attribute:: kinematic_viscosity
      :canonical: sedtrails.transport_converter.physics_converter.PhysicsConfig.kinematic_viscosity
      :type: float
      :value: 1.36e-06

      .. autodoc2-docstring:: sedtrails.transport_converter.physics_converter.PhysicsConfig.kinematic_viscosity

   .. py:attribute:: water_density
      :canonical: sedtrails.transport_converter.physics_converter.PhysicsConfig.water_density
      :type: float
      :value: 1027.0

      .. autodoc2-docstring:: sedtrails.transport_converter.physics_converter.PhysicsConfig.water_density

   .. py:attribute:: particle_density
      :canonical: sedtrails.transport_converter.physics_converter.PhysicsConfig.particle_density
      :type: float
      :value: 2650.0

      .. autodoc2-docstring:: sedtrails.transport_converter.physics_converter.PhysicsConfig.particle_density

   .. py:attribute:: porosity
      :canonical: sedtrails.transport_converter.physics_converter.PhysicsConfig.porosity
      :type: float
      :value: 0.4

      .. autodoc2-docstring:: sedtrails.transport_converter.physics_converter.PhysicsConfig.porosity

   .. py:attribute:: grain_diameter
      :canonical: sedtrails.transport_converter.physics_converter.PhysicsConfig.grain_diameter
      :type: float
      :value: 0.00025

      .. autodoc2-docstring:: sedtrails.transport_converter.physics_converter.PhysicsConfig.grain_diameter

   .. py:attribute:: morfac
      :canonical: sedtrails.transport_converter.physics_converter.PhysicsConfig.morfac
      :type: float
      :value: 1.0

      .. autodoc2-docstring:: sedtrails.transport_converter.physics_converter.PhysicsConfig.morfac

.. py:class:: PhysicsConverter(config: typing.Optional[sedtrails.transport_converter.physics_converter.PhysicsConfig] = None)
   :canonical: sedtrails.transport_converter.physics_converter.PhysicsConverter

   .. autodoc2-docstring:: sedtrails.transport_converter.physics_converter.PhysicsConverter

   .. rubric:: Initialization

   .. autodoc2-docstring:: sedtrails.transport_converter.physics_converter.PhysicsConverter.__init__

   .. py:property:: grain_properties
      :canonical: sedtrails.transport_converter.physics_converter.PhysicsConverter.grain_properties

      .. autodoc2-docstring:: sedtrails.transport_converter.physics_converter.PhysicsConverter.grain_properties

   .. py:method:: _calculate_grain_properties() -> None
      :canonical: sedtrails.transport_converter.physics_converter.PhysicsConverter._calculate_grain_properties

      .. autodoc2-docstring:: sedtrails.transport_converter.physics_converter.PhysicsConverter._calculate_grain_properties

   .. py:property:: physics_plugin
      :canonical: sedtrails.transport_converter.physics_converter.PhysicsConverter.physics_plugin

      .. autodoc2-docstring:: sedtrails.transport_converter.physics_converter.PhysicsConverter.physics_plugin

   .. py:method:: convert_physics(sedtrails_data, transport_probability_method: str = None) -> None
      :canonical: sedtrails.transport_converter.physics_converter.PhysicsConverter.convert_physics

      .. autodoc2-docstring:: sedtrails.transport_converter.physics_converter.PhysicsConverter.convert_physics
