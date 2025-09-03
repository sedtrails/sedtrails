:py:mod:`sedtrails.particle_tracer.data_retriever`
==================================================

.. py:module:: sedtrails.particle_tracer.data_retriever

.. autodoc2-docstring:: sedtrails.particle_tracer.data_retriever
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`FieldDataRetriever <sedtrails.particle_tracer.data_retriever.FieldDataRetriever>`
     - .. autodoc2-docstring:: sedtrails.particle_tracer.data_retriever.FieldDataRetriever
          :summary:

API
~~~

.. py:class:: FieldDataRetriever(sedtrails_data: sedtrails.transport_converter.format_converter.SedtrailsData, fraction_index: int = 0)
   :canonical: sedtrails.particle_tracer.data_retriever.FieldDataRetriever

   .. autodoc2-docstring:: sedtrails.particle_tracer.data_retriever.FieldDataRetriever

   .. rubric:: Initialization

   .. autodoc2-docstring:: sedtrails.particle_tracer.data_retriever.FieldDataRetriever.__init__

   .. py:attribute:: MIN_WEIGHT
      :canonical: sedtrails.particle_tracer.data_retriever.FieldDataRetriever.MIN_WEIGHT
      :value: 0.0

      .. autodoc2-docstring:: sedtrails.particle_tracer.data_retriever.FieldDataRetriever.MIN_WEIGHT

   .. py:attribute:: MAX_WEIGHT
      :canonical: sedtrails.particle_tracer.data_retriever.FieldDataRetriever.MAX_WEIGHT
      :value: 1.0

      .. autodoc2-docstring:: sedtrails.particle_tracer.data_retriever.FieldDataRetriever.MAX_WEIGHT

   .. py:method:: get_interpolation_indices(target_time: float) -> typing.Tuple[int, int, float]
      :canonical: sedtrails.particle_tracer.data_retriever.FieldDataRetriever.get_interpolation_indices

      .. autodoc2-docstring:: sedtrails.particle_tracer.data_retriever.FieldDataRetriever.get_interpolation_indices

   .. py:method:: _interpolate_linearly(lower_value: numpy.ndarray, upper_value: numpy.ndarray, weight: float) -> numpy.ndarray
      :canonical: sedtrails.particle_tracer.data_retriever.FieldDataRetriever._interpolate_linearly

      .. autodoc2-docstring:: sedtrails.particle_tracer.data_retriever.FieldDataRetriever._interpolate_linearly

   .. py:method:: _extract_fraction(field_data)
      :canonical: sedtrails.particle_tracer.data_retriever.FieldDataRetriever._extract_fraction

      .. autodoc2-docstring:: sedtrails.particle_tracer.data_retriever.FieldDataRetriever._extract_fraction

   .. py:method:: get_flow_field(time: float, flow_field_name: str) -> typing.Dict[str, numpy.ndarray]
      :canonical: sedtrails.particle_tracer.data_retriever.FieldDataRetriever.get_flow_field

      .. autodoc2-docstring:: sedtrails.particle_tracer.data_retriever.FieldDataRetriever.get_flow_field

   .. py:method:: get_scalar_field(time: float, scalar_field_name: str) -> typing.Dict[str, numpy.ndarray]
      :canonical: sedtrails.particle_tracer.data_retriever.FieldDataRetriever.get_scalar_field

      .. autodoc2-docstring:: sedtrails.particle_tracer.data_retriever.FieldDataRetriever.get_scalar_field
