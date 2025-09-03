:py:mod:`sedtrails.transport_converter.physics_lib`
===================================================

.. py:module:: sedtrails.transport_converter.physics_lib

.. autodoc2-docstring:: sedtrails.transport_converter.physics_lib
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`SuspendedVelocityMethod <sedtrails.transport_converter.physics_lib.SuspendedVelocityMethod>`
     - .. autodoc2-docstring:: sedtrails.transport_converter.physics_lib.SuspendedVelocityMethod
          :summary:
   * - :py:obj:`MixingLayerMethod <sedtrails.transport_converter.physics_lib.MixingLayerMethod>`
     - .. autodoc2-docstring:: sedtrails.transport_converter.physics_lib.MixingLayerMethod
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`compute_shear_velocity <sedtrails.transport_converter.physics_lib.compute_shear_velocity>`
     - .. autodoc2-docstring:: sedtrails.transport_converter.physics_lib.compute_shear_velocity
          :summary:
   * - :py:obj:`compute_shields <sedtrails.transport_converter.physics_lib.compute_shields>`
     - .. autodoc2-docstring:: sedtrails.transport_converter.physics_lib.compute_shields
          :summary:
   * - :py:obj:`compute_bed_load_velocity <sedtrails.transport_converter.physics_lib.compute_bed_load_velocity>`
     - .. autodoc2-docstring:: sedtrails.transport_converter.physics_lib.compute_bed_load_velocity
          :summary:
   * - :py:obj:`compute_transport_layer_thickness <sedtrails.transport_converter.physics_lib.compute_transport_layer_thickness>`
     - .. autodoc2-docstring:: sedtrails.transport_converter.physics_lib.compute_transport_layer_thickness
          :summary:
   * - :py:obj:`compute_suspended_velocity <sedtrails.transport_converter.physics_lib.compute_suspended_velocity>`
     - .. autodoc2-docstring:: sedtrails.transport_converter.physics_lib.compute_suspended_velocity
          :summary:
   * - :py:obj:`compute_directions_from_magnitude <sedtrails.transport_converter.physics_lib.compute_directions_from_magnitude>`
     - .. autodoc2-docstring:: sedtrails.transport_converter.physics_lib.compute_directions_from_magnitude
          :summary:
   * - :py:obj:`compute_mixing_layer_thickness <sedtrails.transport_converter.physics_lib.compute_mixing_layer_thickness>`
     - .. autodoc2-docstring:: sedtrails.transport_converter.physics_lib.compute_mixing_layer_thickness
          :summary:
   * - :py:obj:`compute_grain_properties <sedtrails.transport_converter.physics_lib.compute_grain_properties>`
     - .. autodoc2-docstring:: sedtrails.transport_converter.physics_lib.compute_grain_properties
          :summary:

API
~~~

.. py:class:: SuspendedVelocityMethod(*args, **kwds)
   :canonical: sedtrails.transport_converter.physics_lib.SuspendedVelocityMethod

   Bases: :py:obj:`enum.Enum`

   .. autodoc2-docstring:: sedtrails.transport_converter.physics_lib.SuspendedVelocityMethod

   .. rubric:: Initialization

   .. autodoc2-docstring:: sedtrails.transport_converter.physics_lib.SuspendedVelocityMethod.__init__

   .. py:attribute:: VAN_WESTEN_2025
      :canonical: sedtrails.transport_converter.physics_lib.SuspendedVelocityMethod.VAN_WESTEN_2025
      :value: 'van_westen_2025'

      .. autodoc2-docstring:: sedtrails.transport_converter.physics_lib.SuspendedVelocityMethod.VAN_WESTEN_2025

   .. py:attribute:: SOULSBY_2011
      :canonical: sedtrails.transport_converter.physics_lib.SuspendedVelocityMethod.SOULSBY_2011
      :value: 'soulsby_2011'

      .. autodoc2-docstring:: sedtrails.transport_converter.physics_lib.SuspendedVelocityMethod.SOULSBY_2011

.. py:class:: MixingLayerMethod(*args, **kwds)
   :canonical: sedtrails.transport_converter.physics_lib.MixingLayerMethod

   Bases: :py:obj:`enum.Enum`

   .. autodoc2-docstring:: sedtrails.transport_converter.physics_lib.MixingLayerMethod

   .. rubric:: Initialization

   .. autodoc2-docstring:: sedtrails.transport_converter.physics_lib.MixingLayerMethod.__init__

   .. py:attribute:: BERTIN_2008
      :canonical: sedtrails.transport_converter.physics_lib.MixingLayerMethod.BERTIN_2008
      :value: 'bertin_2008'

      .. autodoc2-docstring:: sedtrails.transport_converter.physics_lib.MixingLayerMethod.BERTIN_2008

   .. py:attribute:: HARRIS_WIBERG
      :canonical: sedtrails.transport_converter.physics_lib.MixingLayerMethod.HARRIS_WIBERG
      :value: 'harris_wiberg'

      .. autodoc2-docstring:: sedtrails.transport_converter.physics_lib.MixingLayerMethod.HARRIS_WIBERG

.. py:function:: compute_shear_velocity(bed_shear_stress: numpy.ndarray, water_density: float) -> numpy.ndarray
   :canonical: sedtrails.transport_converter.physics_lib.compute_shear_velocity

   .. autodoc2-docstring:: sedtrails.transport_converter.physics_lib.compute_shear_velocity

.. py:function:: compute_shields(bed_shear_stress: numpy.ndarray, gravity: float, sediment_density: float, water_density: float, grain_diameter: float) -> numpy.ndarray
   :canonical: sedtrails.transport_converter.physics_lib.compute_shields

   .. autodoc2-docstring:: sedtrails.transport_converter.physics_lib.compute_shields

.. py:function:: compute_bed_load_velocity(shields_number: numpy.ndarray, critical_shields: float, mean_shear_velocity: numpy.ndarray) -> numpy.ndarray
   :canonical: sedtrails.transport_converter.physics_lib.compute_bed_load_velocity

   .. autodoc2-docstring:: sedtrails.transport_converter.physics_lib.compute_bed_load_velocity

.. py:function:: compute_transport_layer_thickness(transport_magnitude: numpy.ndarray, velocity_magnitude: numpy.ndarray, sediment_density: float, porosity: float) -> numpy.ndarray
   :canonical: sedtrails.transport_converter.physics_lib.compute_transport_layer_thickness

   .. autodoc2-docstring:: sedtrails.transport_converter.physics_lib.compute_transport_layer_thickness

.. py:function:: compute_suspended_velocity(flow_velocity_magnitude: numpy.ndarray, bed_load_velocity: numpy.ndarray, settling_velocity: float, von_karman_constant: float, max_shear_velocity: numpy.ndarray, shields_number: numpy.ndarray, critical_shields: float, method: sedtrails.transport_converter.physics_lib.SuspendedVelocityMethod = SuspendedVelocityMethod.SOULSBY_2011) -> numpy.ndarray
   :canonical: sedtrails.transport_converter.physics_lib.compute_suspended_velocity

   .. autodoc2-docstring:: sedtrails.transport_converter.physics_lib.compute_suspended_velocity

.. py:function:: compute_directions_from_magnitude(velocity_magnitude: numpy.ndarray, transport_x: numpy.ndarray, transport_y: numpy.ndarray, transport_magnitude: numpy.ndarray) -> typing.Tuple[numpy.ndarray, numpy.ndarray]
   :canonical: sedtrails.transport_converter.physics_lib.compute_directions_from_magnitude

   .. autodoc2-docstring:: sedtrails.transport_converter.physics_lib.compute_directions_from_magnitude

.. py:function:: compute_mixing_layer_thickness(max_bed_shear_stress: numpy.ndarray, critical_shear_stress: float, method: sedtrails.transport_converter.physics_lib.MixingLayerMethod = MixingLayerMethod.BERTIN_2008) -> numpy.ndarray
   :canonical: sedtrails.transport_converter.physics_lib.compute_mixing_layer_thickness

   .. autodoc2-docstring:: sedtrails.transport_converter.physics_lib.compute_mixing_layer_thickness

.. py:function:: compute_grain_properties(grain_diameter: float, gravity: float, sediment_density: float, water_density: float, kinematic_viscosity: float) -> dict[str, float]
   :canonical: sedtrails.transport_converter.physics_lib.compute_grain_properties

   .. autodoc2-docstring:: sedtrails.transport_converter.physics_lib.compute_grain_properties
