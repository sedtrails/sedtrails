:py:mod:`sedtrails.particle_tracer.timer`
=========================================

.. py:module:: sedtrails.particle_tracer.timer

.. autodoc2-docstring:: sedtrails.particle_tracer.timer
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Duration <sedtrails.particle_tracer.timer.Duration>`
     - .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Duration
          :summary:
   * - :py:obj:`Time <sedtrails.particle_tracer.timer.Time>`
     - .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Time
          :summary:
   * - :py:obj:`Timer <sedtrails.particle_tracer.timer.Timer>`
     - .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Timer
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`convert_duration_string_to_seconds <sedtrails.particle_tracer.timer.convert_duration_string_to_seconds>`
     - .. autodoc2-docstring:: sedtrails.particle_tracer.timer.convert_duration_string_to_seconds
          :summary:
   * - :py:obj:`convert_datetime_string_to_datetime64 <sedtrails.particle_tracer.timer.convert_datetime_string_to_datetime64>`
     - .. autodoc2-docstring:: sedtrails.particle_tracer.timer.convert_datetime_string_to_datetime64
          :summary:

API
~~~

.. py:function:: convert_duration_string_to_seconds(duration_str: str) -> int
   :canonical: sedtrails.particle_tracer.timer.convert_duration_string_to_seconds

   .. autodoc2-docstring:: sedtrails.particle_tracer.timer.convert_duration_string_to_seconds

.. py:function:: convert_datetime_string_to_datetime64(datetime_str: str) -> numpy.datetime64
   :canonical: sedtrails.particle_tracer.timer.convert_datetime_string_to_datetime64

   .. autodoc2-docstring:: sedtrails.particle_tracer.timer.convert_datetime_string_to_datetime64

.. py:class:: Duration(duration: str)
   :canonical: sedtrails.particle_tracer.timer.Duration

   .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Duration

   .. rubric:: Initialization

   .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Duration.__init__

   .. py:property:: string
      :canonical: sedtrails.particle_tracer.timer.Duration.string
      :type: str

      .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Duration.string

   .. py:property:: seconds
      :canonical: sedtrails.particle_tracer.timer.Duration.seconds
      :type: int

      .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Duration.seconds

   .. py:property:: deltatime
      :canonical: sedtrails.particle_tracer.timer.Duration.deltatime
      :type: numpy.timedelta64

      .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Duration.deltatime

.. py:class:: Time
   :canonical: sedtrails.particle_tracer.timer.Time

   .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Time

   .. py:attribute:: _start
      :canonical: sedtrails.particle_tracer.timer.Time._start
      :type: str
      :value: None

      .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Time._start

   .. py:attribute:: time_step
      :canonical: sedtrails.particle_tracer.timer.Time.time_step
      :type: sedtrails.particle_tracer.timer.Duration
      :value: 'field(...)'

      .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Time.time_step

   .. py:attribute:: duration
      :canonical: sedtrails.particle_tracer.timer.Time.duration
      :type: sedtrails.particle_tracer.timer.Duration
      :value: 'field(...)'

      .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Time.duration

   .. py:attribute:: read_input_timestep
      :canonical: sedtrails.particle_tracer.timer.Time.read_input_timestep
      :type: sedtrails.particle_tracer.timer.Duration
      :value: 'field(...)'

      .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Time.read_input_timestep

   .. py:attribute:: reference_date
      :canonical: sedtrails.particle_tracer.timer.Time.reference_date
      :type: str
      :value: 'field(...)'

      .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Time.reference_date

   .. py:attribute:: cfl_condition
      :canonical: sedtrails.particle_tracer.timer.Time.cfl_condition
      :type: float
      :value: 'field(...)'

      .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Time.cfl_condition

   .. py:attribute:: _start_time_np
      :canonical: sedtrails.particle_tracer.timer.Time._start_time_np
      :type: numpy.datetime64
      :value: 'field(...)'

      .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Time._start_time_np

   .. py:method:: __post_init__()
      :canonical: sedtrails.particle_tracer.timer.Time.__post_init__

      .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Time.__post_init__

   .. py:property:: start
      :canonical: sedtrails.particle_tracer.timer.Time.start
      :type: int

      .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Time.start

   .. py:property:: end
      :canonical: sedtrails.particle_tracer.timer.Time.end
      :type: int

      .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Time.end

.. py:class:: Timer
   :canonical: sedtrails.particle_tracer.timer.Timer

   .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Timer

   .. py:attribute:: simulation_time
      :canonical: sedtrails.particle_tracer.timer.Timer.simulation_time
      :type: sedtrails.particle_tracer.timer.Time
      :value: None

      .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Timer.simulation_time

   .. py:attribute:: _current
      :canonical: sedtrails.particle_tracer.timer.Timer._current
      :type: int | float
      :value: 'field(...)'

      .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Timer._current

   .. py:attribute:: _current_timestep
      :canonical: sedtrails.particle_tracer.timer.Timer._current_timestep
      :type: int | float
      :value: 'field(...)'

      .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Timer._current_timestep

   .. py:attribute:: cfl_condition
      :canonical: sedtrails.particle_tracer.timer.Timer.cfl_condition
      :type: float
      :value: 'field(...)'

      .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Timer.cfl_condition

   .. py:attribute:: stop
      :canonical: sedtrails.particle_tracer.timer.Timer.stop
      :type: bool
      :value: False

      .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Timer.stop

   .. py:method:: __post_init__()
      :canonical: sedtrails.particle_tracer.timer.Timer.__post_init__

      .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Timer.__post_init__

   .. py:property:: current
      :canonical: sedtrails.particle_tracer.timer.Timer.current
      :type: int | float

      .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Timer.current

   .. py:property:: current_timestep
      :canonical: sedtrails.particle_tracer.timer.Timer.current_timestep
      :type: int | float

      .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Timer.current_timestep

   .. py:property:: next
      :canonical: sedtrails.particle_tracer.timer.Timer.next
      :type: int | float

      .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Timer.next

   .. py:method:: set_timestep(timestep: float) -> None
      :canonical: sedtrails.particle_tracer.timer.Timer.set_timestep

      .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Timer.set_timestep

   .. py:method:: advance() -> None
      :canonical: sedtrails.particle_tracer.timer.Timer.advance

      .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Timer.advance

   .. py:method:: compute_cfl_timestep(flow_data_list: list, sedtrails_data) -> None
      :canonical: sedtrails.particle_tracer.timer.Timer.compute_cfl_timestep

      .. autodoc2-docstring:: sedtrails.particle_tracer.timer.Timer.compute_cfl_timestep
