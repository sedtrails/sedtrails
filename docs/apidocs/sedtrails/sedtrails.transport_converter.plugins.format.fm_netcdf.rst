:py:mod:`sedtrails.transport_converter.plugins.format.fm_netcdf`
================================================================

.. py:module:: sedtrails.transport_converter.plugins.format.fm_netcdf

.. autodoc2-docstring:: sedtrails.transport_converter.plugins.format.fm_netcdf
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`FormatPlugin <sedtrails.transport_converter.plugins.format.fm_netcdf.FormatPlugin>`
     - .. autodoc2-docstring:: sedtrails.transport_converter.plugins.format.fm_netcdf.FormatPlugin
          :summary:

API
~~~

.. py:class:: FormatPlugin(input_file: str, morfac: float = 1.0)
   :canonical: sedtrails.transport_converter.plugins.format.fm_netcdf.FormatPlugin

   Bases: :py:obj:`sedtrails.transport_converter.plugins.BaseFormatPlugin`

   .. autodoc2-docstring:: sedtrails.transport_converter.plugins.format.fm_netcdf.FormatPlugin

   .. rubric:: Initialization

   .. autodoc2-docstring:: sedtrails.transport_converter.plugins.format.fm_netcdf.FormatPlugin.__init__

   .. py:method:: __post_init__()
      :canonical: sedtrails.transport_converter.plugins.format.fm_netcdf.FormatPlugin.__post_init__

      .. autodoc2-docstring:: sedtrails.transport_converter.plugins.format.fm_netcdf.FormatPlugin.__post_init__

   .. py:property:: variables
      :canonical: sedtrails.transport_converter.plugins.format.fm_netcdf.FormatPlugin.variables
      :type: typing.List[str]

      .. autodoc2-docstring:: sedtrails.transport_converter.plugins.format.fm_netcdf.FormatPlugin.variables

   .. py:method:: _decompress_time(time_info: typing.Dict) -> typing.Dict
      :canonical: sedtrails.transport_converter.plugins.format.fm_netcdf.FormatPlugin._decompress_time

      .. autodoc2-docstring:: sedtrails.transport_converter.plugins.format.fm_netcdf.FormatPlugin._decompress_time

   .. py:method:: convert(current_time=None, reading_interval=None) -> sedtrails.transport_converter.sedtrails_data.SedtrailsData
      :canonical: sedtrails.transport_converter.plugins.format.fm_netcdf.FormatPlugin.convert

      .. autodoc2-docstring:: sedtrails.transport_converter.plugins.format.fm_netcdf.FormatPlugin.convert

   .. py:method:: _calculate_time_slice(current_time, reading_interval, time_info)
      :canonical: sedtrails.transport_converter.plugins.format.fm_netcdf.FormatPlugin._calculate_time_slice

      .. autodoc2-docstring:: sedtrails.transport_converter.plugins.format.fm_netcdf.FormatPlugin._calculate_time_slice

   .. py:method:: load() -> typing.Any
      :canonical: sedtrails.transport_converter.plugins.format.fm_netcdf.FormatPlugin.load

      .. autodoc2-docstring:: sedtrails.transport_converter.plugins.format.fm_netcdf.FormatPlugin.load

   .. py:method:: _slice_time_info(time_info: typing.Dict, time_slice: slice) -> typing.Dict
      :canonical: sedtrails.transport_converter.plugins.format.fm_netcdf.FormatPlugin._slice_time_info

      .. autodoc2-docstring:: sedtrails.transport_converter.plugins.format.fm_netcdf.FormatPlugin._slice_time_info

   .. py:method:: _get_time_info(input_data: typing.Union[xugrid.UgridDataset, xarray.Dataset], reference_date: numpy.datetime64) -> typing.Dict
      :canonical: sedtrails.transport_converter.plugins.format.fm_netcdf.FormatPlugin._get_time_info

      .. autodoc2-docstring:: sedtrails.transport_converter.plugins.format.fm_netcdf.FormatPlugin._get_time_info

   .. py:method:: _map_dfm_variables(time_info, time_start_idx: typing.Optional[int] = None, time_end_idx: typing.Optional[int] = None) -> typing.Dict
      :canonical: sedtrails.transport_converter.plugins.format.fm_netcdf.FormatPlugin._map_dfm_variables

      .. autodoc2-docstring:: sedtrails.transport_converter.plugins.format.fm_netcdf.FormatPlugin._map_dfm_variables
