:py:mod:`sedtrails.particle_tracer.state_checker`
=================================================

.. py:module:: sedtrails.particle_tracer.state_checker

.. autodoc2-docstring:: sedtrails.particle_tracer.state_checker
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Status <sedtrails.particle_tracer.state_checker.Status>`
     - .. autodoc2-docstring:: sedtrails.particle_tracer.state_checker.Status
          :summary:
   * - :py:obj:`StateChecker <sedtrails.particle_tracer.state_checker.StateChecker>`
     - .. autodoc2-docstring:: sedtrails.particle_tracer.state_checker.StateChecker
          :summary:

API
~~~

.. py:class:: Status(*args, **kwds)
   :canonical: sedtrails.particle_tracer.state_checker.Status

   Bases: :py:obj:`enum.Enum`

   .. autodoc2-docstring:: sedtrails.particle_tracer.state_checker.Status

   .. rubric:: Initialization

   .. autodoc2-docstring:: sedtrails.particle_tracer.state_checker.Status.__init__

   .. py:attribute:: ACTIVE
      :canonical: sedtrails.particle_tracer.state_checker.Status.ACTIVE
      :value: 'active'

      .. autodoc2-docstring:: sedtrails.particle_tracer.state_checker.Status.ACTIVE

   .. py:attribute:: DEAD
      :canonical: sedtrails.particle_tracer.state_checker.Status.DEAD
      :value: 'dead'

      .. autodoc2-docstring:: sedtrails.particle_tracer.state_checker.Status.DEAD

   .. py:attribute:: ALIVE
      :canonical: sedtrails.particle_tracer.state_checker.Status.ALIVE
      :value: 'alive'

      .. autodoc2-docstring:: sedtrails.particle_tracer.state_checker.Status.ALIVE

   .. py:attribute:: STUCK
      :canonical: sedtrails.particle_tracer.state_checker.Status.STUCK
      :value: 'stuck'

      .. autodoc2-docstring:: sedtrails.particle_tracer.state_checker.Status.STUCK

.. py:class:: StateChecker(**config)
   :canonical: sedtrails.particle_tracer.state_checker.StateChecker

   .. autodoc2-docstring:: sedtrails.particle_tracer.state_checker.StateChecker

   .. rubric:: Initialization

   .. autodoc2-docstring:: sedtrails.particle_tracer.state_checker.StateChecker.__init__

   .. py:method:: check_state(particle: sedtrails.particle_tracer.particle.Particle, flow_field) -> str
      :canonical: sedtrails.particle_tracer.state_checker.StateChecker.check_state

      .. autodoc2-docstring:: sedtrails.particle_tracer.state_checker.StateChecker.check_state
