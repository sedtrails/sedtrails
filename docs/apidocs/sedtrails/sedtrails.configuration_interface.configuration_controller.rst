:py:mod:`sedtrails.configuration_interface.configuration_controller`
====================================================================

.. py:module:: sedtrails.configuration_interface.configuration_controller

.. autodoc2-docstring:: sedtrails.configuration_interface.configuration_controller
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Controller <sedtrails.configuration_interface.configuration_controller.Controller>`
     - .. autodoc2-docstring:: sedtrails.configuration_interface.configuration_controller.Controller
          :summary:
   * - :py:obj:`ConfigurationController <sedtrails.configuration_interface.configuration_controller.ConfigurationController>`
     - .. autodoc2-docstring:: sedtrails.configuration_interface.configuration_controller.ConfigurationController
          :summary:

API
~~~

.. py:class:: Controller
   :canonical: sedtrails.configuration_interface.configuration_controller.Controller

   Bases: :py:obj:`abc.ABC`

   .. autodoc2-docstring:: sedtrails.configuration_interface.configuration_controller.Controller

   .. py:method:: load_config(config_file: str) -> None
      :canonical: sedtrails.configuration_interface.configuration_controller.Controller.load_config
      :abstractmethod:

      .. autodoc2-docstring:: sedtrails.configuration_interface.configuration_controller.Controller.load_config

   .. py:method:: get_config() -> typing.Dict[str, typing.Any]
      :canonical: sedtrails.configuration_interface.configuration_controller.Controller.get_config
      :abstractmethod:

      .. autodoc2-docstring:: sedtrails.configuration_interface.configuration_controller.Controller.get_config

   .. py:method:: get(keys: str, default=None) -> typing.Any
      :canonical: sedtrails.configuration_interface.configuration_controller.Controller.get
      :abstractmethod:

      .. autodoc2-docstring:: sedtrails.configuration_interface.configuration_controller.Controller.get

.. py:class:: ConfigurationController(config_file: str, logger: typing.Optional[sedtrails.logger.logger.LoggerManager] = None)
   :canonical: sedtrails.configuration_interface.configuration_controller.ConfigurationController

   Bases: :py:obj:`sedtrails.configuration_interface.configuration_controller.Controller`

   .. autodoc2-docstring:: sedtrails.configuration_interface.configuration_controller.ConfigurationController

   .. rubric:: Initialization

   .. autodoc2-docstring:: sedtrails.configuration_interface.configuration_controller.ConfigurationController.__init__

   .. py:method:: load_config(config_file: str) -> None
      :canonical: sedtrails.configuration_interface.configuration_controller.ConfigurationController.load_config

      .. autodoc2-docstring:: sedtrails.configuration_interface.configuration_controller.ConfigurationController.load_config

   .. py:method:: log_after_load_config() -> None
      :canonical: sedtrails.configuration_interface.configuration_controller.ConfigurationController.log_after_load_config

      .. autodoc2-docstring:: sedtrails.configuration_interface.configuration_controller.ConfigurationController.log_after_load_config

   .. py:method:: get_config() -> dict
      :canonical: sedtrails.configuration_interface.configuration_controller.ConfigurationController.get_config

      .. autodoc2-docstring:: sedtrails.configuration_interface.configuration_controller.ConfigurationController.get_config

   .. py:method:: get(keys: str, default=None) -> typing.Any
      :canonical: sedtrails.configuration_interface.configuration_controller.ConfigurationController.get

      .. autodoc2-docstring:: sedtrails.configuration_interface.configuration_controller.ConfigurationController.get
