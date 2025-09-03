:py:mod:`sedtrails.transport_converter.format_converter`
========================================================

.. py:module:: sedtrails.transport_converter.format_converter

.. autodoc2-docstring:: sedtrails.transport_converter.format_converter
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`FormatConverter <sedtrails.transport_converter.format_converter.FormatConverter>`
     - .. autodoc2-docstring:: sedtrails.transport_converter.format_converter.FormatConverter
          :summary:

API
~~~

.. py:class:: FormatConverter(config: typing.Dict)
   :canonical: sedtrails.transport_converter.format_converter.FormatConverter

   .. autodoc2-docstring:: sedtrails.transport_converter.format_converter.FormatConverter

   .. rubric:: Initialization

   .. autodoc2-docstring:: sedtrails.transport_converter.format_converter.FormatConverter.__init__

   .. py:method:: __post_init__()
      :canonical: sedtrails.transport_converter.format_converter.FormatConverter.__post_init__

      .. autodoc2-docstring:: sedtrails.transport_converter.format_converter.FormatConverter.__post_init__

   .. py:property:: input_file
      :canonical: sedtrails.transport_converter.format_converter.FormatConverter.input_file

      .. autodoc2-docstring:: sedtrails.transport_converter.format_converter.FormatConverter.input_file

   .. py:property:: input_format
      :canonical: sedtrails.transport_converter.format_converter.FormatConverter.input_format
      :type: str | None

      .. autodoc2-docstring:: sedtrails.transport_converter.format_converter.FormatConverter.input_format

   .. py:property:: reference_date
      :canonical: sedtrails.transport_converter.format_converter.FormatConverter.reference_date
      :type: str

      .. autodoc2-docstring:: sedtrails.transport_converter.format_converter.FormatConverter.reference_date

   .. py:property:: morfac
      :canonical: sedtrails.transport_converter.format_converter.FormatConverter.morfac
      :type: float

      .. autodoc2-docstring:: sedtrails.transport_converter.format_converter.FormatConverter.morfac

   .. py:property:: format_plugin
      :canonical: sedtrails.transport_converter.format_converter.FormatConverter.format_plugin

      .. autodoc2-docstring:: sedtrails.transport_converter.format_converter.FormatConverter.format_plugin

   .. py:method:: convert_to_sedtrails(current_time=None, reading_interval=None) -> sedtrails.transport_converter.sedtrails_data.SedtrailsData
      :canonical: sedtrails.transport_converter.format_converter.FormatConverter.convert_to_sedtrails

      .. autodoc2-docstring:: sedtrails.transport_converter.format_converter.FormatConverter.convert_to_sedtrails
