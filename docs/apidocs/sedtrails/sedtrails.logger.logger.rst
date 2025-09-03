:py:mod:`sedtrails.logger.logger`
=================================

.. py:module:: sedtrails.logger.logger

.. autodoc2-docstring:: sedtrails.logger.logger
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`LoggerManager <sedtrails.logger.logger.LoggerManager>`
     - .. autodoc2-docstring:: sedtrails.logger.logger.LoggerManager
          :summary:

API
~~~

.. py:class:: LoggerManager(log_dir: str = 'logs')
   :canonical: sedtrails.logger.logger.LoggerManager

   .. autodoc2-docstring:: sedtrails.logger.logger.LoggerManager

   .. rubric:: Initialization

   .. autodoc2-docstring:: sedtrails.logger.logger.LoggerManager.__init__

   .. py:method:: setup_logger()
      :canonical: sedtrails.logger.logger.LoggerManager.setup_logger

      .. autodoc2-docstring:: sedtrails.logger.logger.LoggerManager.setup_logger

   .. py:method:: log_simulation_state(state: dict, level=logging.INFO) -> None
      :canonical: sedtrails.logger.logger.LoggerManager.log_simulation_state

      .. autodoc2-docstring:: sedtrails.logger.logger.LoggerManager.log_simulation_state

   .. py:method:: log_exception(e: Exception, context: str = None) -> None
      :canonical: sedtrails.logger.logger.LoggerManager.log_exception

      .. autodoc2-docstring:: sedtrails.logger.logger.LoggerManager.log_exception

   .. py:method:: _format_seeding_strategy(strategy)
      :canonical: sedtrails.logger.logger.LoggerManager._format_seeding_strategy

      .. autodoc2-docstring:: sedtrails.logger.logger.LoggerManager._format_seeding_strategy
