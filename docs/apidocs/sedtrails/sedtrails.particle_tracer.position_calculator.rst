:py:mod:`sedtrails.particle_tracer.position_calculator`
=======================================================

.. py:module:: sedtrails.particle_tracer.position_calculator

.. autodoc2-docstring:: sedtrails.particle_tracer.position_calculator
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`ParticlePositionCalculator <sedtrails.particle_tracer.position_calculator.ParticlePositionCalculator>`
     - .. autodoc2-docstring:: sedtrails.particle_tracer.position_calculator.ParticlePositionCalculator
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`_update_particles_rk4 <sedtrails.particle_tracer.position_calculator._update_particles_rk4>`
     - .. autodoc2-docstring:: sedtrails.particle_tracer.position_calculator._update_particles_rk4
          :summary:
   * - :py:obj:`_update_particles_rk4_parallel <sedtrails.particle_tracer.position_calculator._update_particles_rk4_parallel>`
     - .. autodoc2-docstring:: sedtrails.particle_tracer.position_calculator._update_particles_rk4_parallel
          :summary:
   * - :py:obj:`_interpolate_field <sedtrails.particle_tracer.position_calculator._interpolate_field>`
     - .. autodoc2-docstring:: sedtrails.particle_tracer.position_calculator._interpolate_field
          :summary:

API
~~~

.. py:function:: _update_particles_rk4(x0: numpy.typing.NDArray, y0: numpy.typing.NDArray, grid_u: numpy.typing.NDArray, grid_v: numpy.typing.NDArray, grid_x: numpy.typing.NDArray, grid_y: numpy.typing.NDArray, triangles: numpy.typing.NDArray, dt: numpy.float32, igeo: int, geofac: numpy.float32) -> tuple[numpy.typing.NDArray, numpy.typing.NDArray]
   :canonical: sedtrails.particle_tracer.position_calculator._update_particles_rk4

   .. autodoc2-docstring:: sedtrails.particle_tracer.position_calculator._update_particles_rk4

.. py:function:: _update_particles_rk4_parallel(x0: numpy.typing.NDArray, y0: numpy.typing.NDArray, grid_u: numpy.typing.NDArray, grid_v: numpy.typing.NDArray, grid_x: numpy.typing.NDArray, grid_y: numpy.typing.NDArray, triangles: numpy.typing.NDArray, dt: numpy.float32, igeo: int, geofac: numpy.float32) -> tuple[numpy.typing.NDArray, numpy.typing.NDArray]
   :canonical: sedtrails.particle_tracer.position_calculator._update_particles_rk4_parallel

   .. autodoc2-docstring:: sedtrails.particle_tracer.position_calculator._update_particles_rk4_parallel

.. py:function:: _interpolate_field(field: numpy.typing.NDArray, x_points: numpy.typing.NDArray, y_points: numpy.typing.NDArray, grid_x: numpy.typing.NDArray, grid_y: numpy.typing.NDArray, triangles: numpy.typing.NDArray) -> numpy.typing.NDArray
   :canonical: sedtrails.particle_tracer.position_calculator._interpolate_field

   .. autodoc2-docstring:: sedtrails.particle_tracer.position_calculator._interpolate_field

.. py:class:: ParticlePositionCalculator(grid_x: numpy.typing.NDArray, grid_y: numpy.typing.NDArray, grid_u: numpy.typing.NDArray, grid_v: numpy.typing.NDArray, triangles: typing.Optional[numpy.typing.NDArray] = None, igeo: int = 0)
   :canonical: sedtrails.particle_tracer.position_calculator.ParticlePositionCalculator

   .. autodoc2-docstring:: sedtrails.particle_tracer.position_calculator.ParticlePositionCalculator

   .. rubric:: Initialization

   .. autodoc2-docstring:: sedtrails.particle_tracer.position_calculator.ParticlePositionCalculator.__init__

   .. py:method:: interpolate_field(field: numpy.typing.NDArray, x_pts: numpy.typing.NDArray, y_pts: numpy.typing.NDArray) -> numpy.typing.NDArray
      :canonical: sedtrails.particle_tracer.position_calculator.ParticlePositionCalculator.interpolate_field

      .. autodoc2-docstring:: sedtrails.particle_tracer.position_calculator.ParticlePositionCalculator.interpolate_field

   .. py:method:: update_particles(x0: numpy.typing.NDArray, y0: numpy.typing.NDArray, dt: numpy.float32, parallel: bool = False, num_workers: typing.Optional[int] = None) -> typing.Tuple[numpy.typing.NDArray, numpy.typing.NDArray]
      :canonical: sedtrails.particle_tracer.position_calculator.ParticlePositionCalculator.update_particles

      .. autodoc2-docstring:: sedtrails.particle_tracer.position_calculator.ParticlePositionCalculator.update_particles
