:py:mod:`sedtrails.particle_tracer.position_calculator_numba`
=============================================================

.. py:module:: sedtrails.particle_tracer.position_calculator_numba

.. autodoc2-docstring:: sedtrails.particle_tracer.position_calculator_numba
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`create_numba_particle_calculator <sedtrails.particle_tracer.position_calculator_numba.create_numba_particle_calculator>`
     - .. autodoc2-docstring:: sedtrails.particle_tracer.position_calculator_numba.create_numba_particle_calculator
          :summary:
   * - :py:obj:`find_triangle <sedtrails.particle_tracer.position_calculator_numba.find_triangle>`
     - .. autodoc2-docstring:: sedtrails.particle_tracer.position_calculator_numba.find_triangle
          :summary:
   * - :py:obj:`interpolate_field <sedtrails.particle_tracer.position_calculator_numba.interpolate_field>`
     - .. autodoc2-docstring:: sedtrails.particle_tracer.position_calculator_numba.interpolate_field
          :summary:
   * - :py:obj:`update_particles_rk4 <sedtrails.particle_tracer.position_calculator_numba.update_particles_rk4>`
     - .. autodoc2-docstring:: sedtrails.particle_tracer.position_calculator_numba.update_particles_rk4
          :summary:
   * - :py:obj:`update_particles_rk4_parallel <sedtrails.particle_tracer.position_calculator_numba.update_particles_rk4_parallel>`
     - .. autodoc2-docstring:: sedtrails.particle_tracer.position_calculator_numba.update_particles_rk4_parallel
          :summary:

API
~~~

.. py:function:: create_numba_particle_calculator(grid_x, grid_y, triangles=None)
   :canonical: sedtrails.particle_tracer.position_calculator_numba.create_numba_particle_calculator

   .. autodoc2-docstring:: sedtrails.particle_tracer.position_calculator_numba.create_numba_particle_calculator

.. py:function:: find_triangle(x, y, grid_x, grid_y, triangles)
   :canonical: sedtrails.particle_tracer.position_calculator_numba.find_triangle

   .. autodoc2-docstring:: sedtrails.particle_tracer.position_calculator_numba.find_triangle

.. py:function:: interpolate_field(field, x_points, y_points, grid_x, grid_y, triangles)
   :canonical: sedtrails.particle_tracer.position_calculator_numba.interpolate_field

   .. autodoc2-docstring:: sedtrails.particle_tracer.position_calculator_numba.interpolate_field

.. py:function:: update_particles_rk4(x0, y0, grid_u, grid_v, grid_x, grid_y, triangles, dt, igeo=0)
   :canonical: sedtrails.particle_tracer.position_calculator_numba.update_particles_rk4

   .. autodoc2-docstring:: sedtrails.particle_tracer.position_calculator_numba.update_particles_rk4

.. py:function:: update_particles_rk4_parallel(x0, y0, grid_u, grid_v, grid_x, grid_y, triangles, dt, igeo=0)
   :canonical: sedtrails.particle_tracer.position_calculator_numba.update_particles_rk4_parallel

   .. autodoc2-docstring:: sedtrails.particle_tracer.position_calculator_numba.update_particles_rk4_parallel
