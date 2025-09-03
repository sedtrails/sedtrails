:py:mod:`sedtrails.particle_tracer.diffusion_library`
=====================================================

.. py:module:: sedtrails.particle_tracer.diffusion_library

.. autodoc2-docstring:: sedtrails.particle_tracer.diffusion_library
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`DiffusionStrategy <sedtrails.particle_tracer.diffusion_library.DiffusionStrategy>`
     - .. autodoc2-docstring:: sedtrails.particle_tracer.diffusion_library.DiffusionStrategy
          :summary:
   * - :py:obj:`GradientDiffusion <sedtrails.particle_tracer.diffusion_library.GradientDiffusion>`
     - .. autodoc2-docstring:: sedtrails.particle_tracer.diffusion_library.GradientDiffusion
          :summary:
   * - :py:obj:`RandomDiffusion <sedtrails.particle_tracer.diffusion_library.RandomDiffusion>`
     - .. autodoc2-docstring:: sedtrails.particle_tracer.diffusion_library.RandomDiffusion
          :summary:
   * - :py:obj:`DiffusionCalculator <sedtrails.particle_tracer.diffusion_library.DiffusionCalculator>`
     - .. autodoc2-docstring:: sedtrails.particle_tracer.diffusion_library.DiffusionCalculator
          :summary:

API
~~~

.. py:class:: DiffusionStrategy
   :canonical: sedtrails.particle_tracer.diffusion_library.DiffusionStrategy

   Bases: :py:obj:`abc.ABC`

   .. autodoc2-docstring:: sedtrails.particle_tracer.diffusion_library.DiffusionStrategy

   .. py:method:: calculate(dt: float, x: numpy.ndarray, y: numpy.ndarray, u: numpy.ndarray, v: numpy.ndarray, nu: float) -> typing.Tuple[numpy.ndarray, numpy.ndarray]
      :canonical: sedtrails.particle_tracer.diffusion_library.DiffusionStrategy.calculate
      :abstractmethod:

      .. autodoc2-docstring:: sedtrails.particle_tracer.diffusion_library.DiffusionStrategy.calculate

.. py:class:: GradientDiffusion
   :canonical: sedtrails.particle_tracer.diffusion_library.GradientDiffusion

   Bases: :py:obj:`sedtrails.particle_tracer.diffusion_library.DiffusionStrategy`

   .. autodoc2-docstring:: sedtrails.particle_tracer.diffusion_library.GradientDiffusion

   .. py:method:: calculate(dt: float, x: numpy.ndarray, y: numpy.ndarray, u: numpy.ndarray, v: numpy.ndarray, nu: float) -> typing.Tuple[numpy.ndarray, numpy.ndarray]
      :canonical: sedtrails.particle_tracer.diffusion_library.GradientDiffusion.calculate

      .. autodoc2-docstring:: sedtrails.particle_tracer.diffusion_library.GradientDiffusion.calculate

.. py:class:: RandomDiffusion
   :canonical: sedtrails.particle_tracer.diffusion_library.RandomDiffusion

   Bases: :py:obj:`sedtrails.particle_tracer.diffusion_library.DiffusionStrategy`

   .. autodoc2-docstring:: sedtrails.particle_tracer.diffusion_library.RandomDiffusion

   .. py:method:: calculate(dt: float, x: numpy.ndarray, y: numpy.ndarray, u: numpy.ndarray, v: numpy.ndarray, nu: float) -> typing.Tuple[float, float]
      :canonical: sedtrails.particle_tracer.diffusion_library.RandomDiffusion.calculate

.. py:class:: DiffusionCalculator(strategy: sedtrails.particle_tracer.diffusion_library.DiffusionStrategy)
   :canonical: sedtrails.particle_tracer.diffusion_library.DiffusionCalculator

   .. autodoc2-docstring:: sedtrails.particle_tracer.diffusion_library.DiffusionCalculator

   .. rubric:: Initialization

   .. autodoc2-docstring:: sedtrails.particle_tracer.diffusion_library.DiffusionCalculator.__init__

   .. py:property:: strategy
      :canonical: sedtrails.particle_tracer.diffusion_library.DiffusionCalculator.strategy
      :type: sedtrails.particle_tracer.diffusion_library.DiffusionStrategy

      .. autodoc2-docstring:: sedtrails.particle_tracer.diffusion_library.DiffusionCalculator.strategy

   .. py:method:: calc_diffusion(x: float, y: float, u: numpy.ndarray, v: numpy.ndarray, nu: float, dt: float) -> typing.Tuple[numpy.ndarray, numpy.ndarray]
      :canonical: sedtrails.particle_tracer.diffusion_library.DiffusionCalculator.calc_diffusion

      .. autodoc2-docstring:: sedtrails.particle_tracer.diffusion_library.DiffusionCalculator.calc_diffusion
