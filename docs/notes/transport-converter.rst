Transport Converter
===================

This is an overview of the SedTRAILS transport converter, which is designed to convert sediment transport data into a format suitable for analysis and visualization in the SedTRAILS framework. The transport converter is implemented as a Python package, and the source code is available in the `sedtrails.transport_converter` module. The package includes various plugins for different sediment transport models and physics, allowing users to convert data from these models into the SedTRAILS format.



Format Converter
----------------

.. automodule:: sedtrails.transport_converter.format_converter
                    :members:

Physics Library
---------------

.. automodule:: sedtrails.transport_converter.physics_lib
                    :members:

Physics Converters
-------------------
.. automodule:: sedtrails.transport_converter.physics_converter
                    :members:

Format Plug-ins
---------------

Delft3D-4 (trim-*.dat)
^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: sedtrails.transport_converter.plugins.format.delft3d4_trim
                    :members:

D-Flow FM (netcdf)
^^^^^^^^^^^^^^^^^^
.. automodule:: sedtrails.transport_converter.plugins.format.fm_netcdf
                    :members:


Physics Plug-ins
------------------

Van Westen et al. (2025)
^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: sedtrails.transport_converter.plugins.physics.van_westen
                    :members:

Soulsby et al. (2011)
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: sedtrails.transport_converter.plugins.physics.soulsby
                    :members:
