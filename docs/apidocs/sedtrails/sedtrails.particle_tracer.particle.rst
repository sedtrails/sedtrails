:py:mod:`sedtrails.particle_tracer.particle`
============================================

.. py:module:: sedtrails.particle_tracer.particle

.. autodoc2-docstring:: sedtrails.particle_tracer.particle
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Particle <sedtrails.particle_tracer.particle.Particle>`
     - .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Particle
          :summary:
   * - :py:obj:`PhysicalProperties <sedtrails.particle_tracer.particle.PhysicalProperties>`
     - .. autodoc2-docstring:: sedtrails.particle_tracer.particle.PhysicalProperties
          :summary:
   * - :py:obj:`Sand <sedtrails.particle_tracer.particle.Sand>`
     - .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Sand
          :summary:
   * - :py:obj:`Mud <sedtrails.particle_tracer.particle.Mud>`
     - .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Mud
          :summary:
   * - :py:obj:`Passive <sedtrails.particle_tracer.particle.Passive>`
     - .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Passive
          :summary:
   * - :py:obj:`InterpolatedValue <sedtrails.particle_tracer.particle.InterpolatedValue>`
     - .. autodoc2-docstring:: sedtrails.particle_tracer.particle.InterpolatedValue
          :summary:

API
~~~

.. py:class:: Particle
   :canonical: sedtrails.particle_tracer.particle.Particle

   Bases: :py:obj:`abc.ABC`

   .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Particle

   .. py:attribute:: _id_counter
      :canonical: sedtrails.particle_tracer.particle.Particle._id_counter
      :type: typing.ClassVar[int]
      :value: 0

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Particle._id_counter

   .. py:attribute:: id
      :canonical: sedtrails.particle_tracer.particle.Particle.id
      :type: int
      :value: 'field(...)'

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Particle.id

   .. py:attribute:: _x
      :canonical: sedtrails.particle_tracer.particle.Particle._x
      :type: float
      :value: 'field(...)'

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Particle._x

   .. py:attribute:: _y
      :canonical: sedtrails.particle_tracer.particle.Particle._y
      :type: float
      :value: 'field(...)'

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Particle._y

   .. py:attribute:: _release_time
      :canonical: sedtrails.particle_tracer.particle.Particle._release_time
      :type: str
      :value: 'field(...)'

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Particle._release_time

   .. py:attribute:: _burial_depth
      :canonical: sedtrails.particle_tracer.particle.Particle._burial_depth
      :type: float
      :value: 'field(...)'

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Particle._burial_depth

   .. py:attribute:: _is_mobile
      :canonical: sedtrails.particle_tracer.particle.Particle._is_mobile
      :type: bool
      :value: 'field(...)'

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Particle._is_mobile

   .. py:attribute:: name
      :canonical: sedtrails.particle_tracer.particle.Particle.name
      :type: typing.Optional[str]
      :value: 'field(...)'

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Particle.name

   .. py:attribute:: trace
      :canonical: sedtrails.particle_tracer.particle.Particle.trace
      :type: typing.Dict
      :value: 'field(...)'

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Particle.trace

   .. py:method:: __post_init__()
      :canonical: sedtrails.particle_tracer.particle.Particle.__post_init__

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Particle.__post_init__

   .. py:method:: add_position(position: typing.Tuple) -> None
      :canonical: sedtrails.particle_tracer.particle.Particle.add_position

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Particle.add_position

   .. py:property:: x
      :canonical: sedtrails.particle_tracer.particle.Particle.x
      :type: float

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Particle.x

   .. py:property:: burial_depth
      :canonical: sedtrails.particle_tracer.particle.Particle.burial_depth
      :type: float

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Particle.burial_depth

   .. py:property:: y
      :canonical: sedtrails.particle_tracer.particle.Particle.y
      :type: float

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Particle.y

   .. py:property:: release_time
      :canonical: sedtrails.particle_tracer.particle.Particle.release_time
      :type: str

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Particle.release_time

   .. py:property:: is_mobile
      :canonical: sedtrails.particle_tracer.particle.Particle.is_mobile
      :type: bool

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Particle.is_mobile

   .. py:method:: particle_velocity() -> float
      :canonical: sedtrails.particle_tracer.particle.Particle.particle_velocity
      :abstractmethod:

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Particle.particle_velocity

.. py:class:: PhysicalProperties
   :canonical: sedtrails.particle_tracer.particle.PhysicalProperties

   .. autodoc2-docstring:: sedtrails.particle_tracer.particle.PhysicalProperties

   .. py:attribute:: density
      :canonical: sedtrails.particle_tracer.particle.PhysicalProperties.density
      :type: float
      :value: None

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.PhysicalProperties.density

   .. py:attribute:: diameter
      :canonical: sedtrails.particle_tracer.particle.PhysicalProperties.diameter
      :type: float
      :value: None

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.PhysicalProperties.diameter

   .. py:method:: __post_init__()
      :canonical: sedtrails.particle_tracer.particle.PhysicalProperties.__post_init__

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.PhysicalProperties.__post_init__

.. py:class:: Sand
   :canonical: sedtrails.particle_tracer.particle.Sand

   Bases: :py:obj:`sedtrails.particle_tracer.particle.Particle`

   .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Sand

   .. py:attribute:: physical_properties
      :canonical: sedtrails.particle_tracer.particle.Sand.physical_properties
      :type: sedtrails.particle_tracer.particle.PhysicalProperties
      :value: 'field(...)'

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Sand.physical_properties

   .. py:method:: __post_init__()
      :canonical: sedtrails.particle_tracer.particle.Sand.__post_init__

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Sand.__post_init__

   .. py:method:: particle_velocity() -> float
      :canonical: sedtrails.particle_tracer.particle.Sand.particle_velocity

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Sand.particle_velocity

.. py:class:: Mud
   :canonical: sedtrails.particle_tracer.particle.Mud

   Bases: :py:obj:`sedtrails.particle_tracer.particle.Particle`

   .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Mud

   .. py:attribute:: physical_properties
      :canonical: sedtrails.particle_tracer.particle.Mud.physical_properties
      :type: sedtrails.particle_tracer.particle.PhysicalProperties
      :value: 'field(...)'

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Mud.physical_properties

   .. py:method:: __post_init__()
      :canonical: sedtrails.particle_tracer.particle.Mud.__post_init__

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Mud.__post_init__

   .. py:method:: particle_velocity() -> float
      :canonical: sedtrails.particle_tracer.particle.Mud.particle_velocity

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Mud.particle_velocity

.. py:class:: Passive
   :canonical: sedtrails.particle_tracer.particle.Passive

   Bases: :py:obj:`sedtrails.particle_tracer.particle.Particle`

   .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Passive

   .. py:attribute:: physical_properties
      :canonical: sedtrails.particle_tracer.particle.Passive.physical_properties
      :type: sedtrails.particle_tracer.particle.PhysicalProperties
      :value: 'field(...)'

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Passive.physical_properties

   .. py:method:: __post_init__()
      :canonical: sedtrails.particle_tracer.particle.Passive.__post_init__

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Passive.__post_init__

   .. py:method:: particle_velocity() -> float
      :canonical: sedtrails.particle_tracer.particle.Passive.particle_velocity

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.Passive.particle_velocity

.. py:class:: InterpolatedValue
   :canonical: sedtrails.particle_tracer.particle.InterpolatedValue

   .. autodoc2-docstring:: sedtrails.particle_tracer.particle.InterpolatedValue

   .. py:attribute:: bed_level
      :canonical: sedtrails.particle_tracer.particle.InterpolatedValue.bed_level
      :type: float
      :value: None

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.InterpolatedValue.bed_level

   .. py:attribute:: bed_load_sediment
      :canonical: sedtrails.particle_tracer.particle.InterpolatedValue.bed_load_sediment
      :type: numpy.ndarray
      :value: None

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.InterpolatedValue.bed_load_sediment

   .. py:attribute:: suspended_sediment
      :canonical: sedtrails.particle_tracer.particle.InterpolatedValue.suspended_sediment
      :type: float
      :value: None

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.InterpolatedValue.suspended_sediment

   .. py:attribute:: sediment_concentration
      :canonical: sedtrails.particle_tracer.particle.InterpolatedValue.sediment_concentration
      :type: float
      :value: None

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.InterpolatedValue.sediment_concentration

   .. py:attribute:: water_depth
      :canonical: sedtrails.particle_tracer.particle.InterpolatedValue.water_depth
      :type: float
      :value: None

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.InterpolatedValue.water_depth

   .. py:attribute:: mean_bed_shear_stress
      :canonical: sedtrails.particle_tracer.particle.InterpolatedValue.mean_bed_shear_stress
      :type: float
      :value: None

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.InterpolatedValue.mean_bed_shear_stress

   .. py:attribute:: max_bed_shear_stress
      :canonical: sedtrails.particle_tracer.particle.InterpolatedValue.max_bed_shear_stress
      :type: float
      :value: None

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.InterpolatedValue.max_bed_shear_stress

   .. py:attribute:: wave_velocity
      :canonical: sedtrails.particle_tracer.particle.InterpolatedValue.wave_velocity
      :type: numpy.ndarray
      :value: None

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.InterpolatedValue.wave_velocity

   .. py:attribute:: depth_avg_flow_velocity
      :canonical: sedtrails.particle_tracer.particle.InterpolatedValue.depth_avg_flow_velocity
      :type: float
      :value: None

      .. autodoc2-docstring:: sedtrails.particle_tracer.particle.InterpolatedValue.depth_avg_flow_velocity
