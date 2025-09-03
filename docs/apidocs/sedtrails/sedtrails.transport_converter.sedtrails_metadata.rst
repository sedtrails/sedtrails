:py:mod:`sedtrails.transport_converter.sedtrails_metadata`
==========================================================

.. py:module:: sedtrails.transport_converter.sedtrails_metadata

.. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_metadata
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`SedtrailsMetadata <sedtrails.transport_converter.sedtrails_metadata.SedtrailsMetadata>`
     - .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_metadata.SedtrailsMetadata
          :summary:

API
~~~

.. py:class:: SedtrailsMetadata
   :canonical: sedtrails.transport_converter.sedtrails_metadata.SedtrailsMetadata

   .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_metadata.SedtrailsMetadata

   .. py:attribute:: flowfield_domain
      :canonical: sedtrails.transport_converter.sedtrails_metadata.SedtrailsMetadata.flowfield_domain
      :type: typing.Dict[str, float]
      :value: None

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_metadata.SedtrailsMetadata.flowfield_domain

   .. py:attribute:: REQUIRED_DOMAIN_KEYS
      :canonical: sedtrails.transport_converter.sedtrails_metadata.SedtrailsMetadata.REQUIRED_DOMAIN_KEYS
      :value: None

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_metadata.SedtrailsMetadata.REQUIRED_DOMAIN_KEYS

   .. py:attribute:: RESERVED_KEYS
      :canonical: sedtrails.transport_converter.sedtrails_metadata.SedtrailsMetadata.RESERVED_KEYS
      :value: None

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_metadata.SedtrailsMetadata.RESERVED_KEYS

   .. py:method:: __post_init__()
      :canonical: sedtrails.transport_converter.sedtrails_metadata.SedtrailsMetadata.__post_init__

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_metadata.SedtrailsMetadata.__post_init__

   .. py:method:: __setattr__(name: str, value: typing.Any)
      :canonical: sedtrails.transport_converter.sedtrails_metadata.SedtrailsMetadata.__setattr__

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_metadata.SedtrailsMetadata.__setattr__

   .. py:method:: add(key: str, value: typing.Any)
      :canonical: sedtrails.transport_converter.sedtrails_metadata.SedtrailsMetadata.add

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_metadata.SedtrailsMetadata.add

   .. py:method:: update(metadata_dict: typing.Mapping[str, typing.Any])
      :canonical: sedtrails.transport_converter.sedtrails_metadata.SedtrailsMetadata.update

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_metadata.SedtrailsMetadata.update

   .. py:method:: get(key: str, default=None) -> typing.Any
      :canonical: sedtrails.transport_converter.sedtrails_metadata.SedtrailsMetadata.get

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_metadata.SedtrailsMetadata.get

   .. py:method:: to_dict() -> typing.Dict[str, typing.Any]
      :canonical: sedtrails.transport_converter.sedtrails_metadata.SedtrailsMetadata.to_dict

      .. autodoc2-docstring:: sedtrails.transport_converter.sedtrails_metadata.SedtrailsMetadata.to_dict
