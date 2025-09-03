:py:mod:`sedtrails.data_manager.simulation_buffer`
==================================================

.. py:module:: sedtrails.data_manager.simulation_buffer

.. autodoc2-docstring:: sedtrails.data_manager.simulation_buffer
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`SimulationDataBuffer <sedtrails.data_manager.simulation_buffer.SimulationDataBuffer>`
     - .. autodoc2-docstring:: sedtrails.data_manager.simulation_buffer.SimulationDataBuffer
          :summary:

API
~~~

.. py:class:: SimulationDataBuffer()
   :canonical: sedtrails.data_manager.simulation_buffer.SimulationDataBuffer

   .. autodoc2-docstring:: sedtrails.data_manager.simulation_buffer.SimulationDataBuffer

   .. rubric:: Initialization

   .. autodoc2-docstring:: sedtrails.data_manager.simulation_buffer.SimulationDataBuffer.__init__

   .. py:method:: add(particle_id, time, x, y)
      :canonical: sedtrails.data_manager.simulation_buffer.SimulationDataBuffer.add

      .. autodoc2-docstring:: sedtrails.data_manager.simulation_buffer.SimulationDataBuffer.add

   .. py:method:: clear()
      :canonical: sedtrails.data_manager.simulation_buffer.SimulationDataBuffer.clear

      .. autodoc2-docstring:: sedtrails.data_manager.simulation_buffer.SimulationDataBuffer.clear

   .. py:method:: get_data()
      :canonical: sedtrails.data_manager.simulation_buffer.SimulationDataBuffer.get_data

      .. autodoc2-docstring:: sedtrails.data_manager.simulation_buffer.SimulationDataBuffer.get_data

   .. py:method:: to_ugrid_dataset(node_x, node_y, face_node_connectivity, fill_value=-1)
      :canonical: sedtrails.data_manager.simulation_buffer.SimulationDataBuffer.to_ugrid_dataset

      .. autodoc2-docstring:: sedtrails.data_manager.simulation_buffer.SimulationDataBuffer.to_ugrid_dataset

   .. py:method:: write_to_disk(node_x, node_y, face_node_connectivity, fill_value, writer, filename)
      :canonical: sedtrails.data_manager.simulation_buffer.SimulationDataBuffer.write_to_disk

      .. autodoc2-docstring:: sedtrails.data_manager.simulation_buffer.SimulationDataBuffer.write_to_disk

   .. py:method:: merge_output_files(output_dir, merged_filename='merged_output.nc')
      :canonical: sedtrails.data_manager.simulation_buffer.SimulationDataBuffer.merge_output_files
      :staticmethod:

      .. autodoc2-docstring:: sedtrails.data_manager.simulation_buffer.SimulationDataBuffer.merge_output_files
