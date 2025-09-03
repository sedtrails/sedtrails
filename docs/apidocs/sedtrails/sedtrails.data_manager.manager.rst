:py:mod:`sedtrails.data_manager.manager`
========================================

.. py:module:: sedtrails.data_manager.manager

.. autodoc2-docstring:: sedtrails.data_manager.manager
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`DataManager <sedtrails.data_manager.manager.DataManager>`
     - .. autodoc2-docstring:: sedtrails.data_manager.manager.DataManager
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`NODE_X <sedtrails.data_manager.manager.NODE_X>`
     - .. autodoc2-docstring:: sedtrails.data_manager.manager.NODE_X
          :summary:
   * - :py:obj:`NODE_Y <sedtrails.data_manager.manager.NODE_Y>`
     - .. autodoc2-docstring:: sedtrails.data_manager.manager.NODE_Y
          :summary:
   * - :py:obj:`FACE_NODE_CONNECTIVITY <sedtrails.data_manager.manager.FACE_NODE_CONNECTIVITY>`
     - .. autodoc2-docstring:: sedtrails.data_manager.manager.FACE_NODE_CONNECTIVITY
          :summary:
   * - :py:obj:`FILL_VALUE <sedtrails.data_manager.manager.FILL_VALUE>`
     - .. autodoc2-docstring:: sedtrails.data_manager.manager.FILL_VALUE
          :summary:

API
~~~

.. py:data:: NODE_X
   :canonical: sedtrails.data_manager.manager.NODE_X
   :value: 'array(...)'

   .. autodoc2-docstring:: sedtrails.data_manager.manager.NODE_X

.. py:data:: NODE_Y
   :canonical: sedtrails.data_manager.manager.NODE_Y
   :value: 'array(...)'

   .. autodoc2-docstring:: sedtrails.data_manager.manager.NODE_Y

.. py:data:: FACE_NODE_CONNECTIVITY
   :canonical: sedtrails.data_manager.manager.FACE_NODE_CONNECTIVITY
   :value: 'array(...)'

   .. autodoc2-docstring:: sedtrails.data_manager.manager.FACE_NODE_CONNECTIVITY

.. py:data:: FILL_VALUE
   :canonical: sedtrails.data_manager.manager.FILL_VALUE
   :value: None

   .. autodoc2-docstring:: sedtrails.data_manager.manager.FILL_VALUE

.. py:class:: DataManager(output_dir: str, max_bytes=512 * 1024 * 1024)
   :canonical: sedtrails.data_manager.manager.DataManager

   .. autodoc2-docstring:: sedtrails.data_manager.manager.DataManager

   .. rubric:: Initialization

   .. autodoc2-docstring:: sedtrails.data_manager.manager.DataManager.__init__

   .. py:method:: _cleanup_chunk_files()
      :canonical: sedtrails.data_manager.manager.DataManager._cleanup_chunk_files

      .. autodoc2-docstring:: sedtrails.data_manager.manager.DataManager._cleanup_chunk_files

   .. py:method:: set_mesh(node_x=NODE_X, node_y=NODE_Y, face_node_connectivity=FACE_NODE_CONNECTIVITY, fill_value=FILL_VALUE)
      :canonical: sedtrails.data_manager.manager.DataManager.set_mesh

      .. autodoc2-docstring:: sedtrails.data_manager.manager.DataManager.set_mesh

   .. py:method:: add_data(particle_id, time, x, y)
      :canonical: sedtrails.data_manager.manager.DataManager.add_data

      .. autodoc2-docstring:: sedtrails.data_manager.manager.DataManager.add_data

   .. py:method:: write(filename=None)
      :canonical: sedtrails.data_manager.manager.DataManager.write

      .. autodoc2-docstring:: sedtrails.data_manager.manager.DataManager.write

   .. py:method:: merge(merged_filename='merged_output.nc')
      :canonical: sedtrails.data_manager.manager.DataManager.merge

      .. autodoc2-docstring:: sedtrails.data_manager.manager.DataManager.merge

   .. py:method:: dump(merge=True, merged_filename='final_output.nc', cleanup_chunks=True)
      :canonical: sedtrails.data_manager.manager.DataManager.dump

      .. autodoc2-docstring:: sedtrails.data_manager.manager.DataManager.dump
