:py:mod:`sedtrails.pathway_visualizer.simulation_dashboard`
===========================================================

.. py:module:: sedtrails.pathway_visualizer.simulation_dashboard

.. autodoc2-docstring:: sedtrails.pathway_visualizer.simulation_dashboard
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`SimulationDashboard <sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard>`
     - .. autodoc2-docstring:: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard
          :summary:

API
~~~

.. py:class:: SimulationDashboard(reference_date: str = '1970-01-01')
   :canonical: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard

   .. autodoc2-docstring:: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard

   .. rubric:: Initialization

   .. autodoc2-docstring:: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard.__init__

   .. py:method:: initialize_dashboard(figsize: typing.Tuple[float, float] = (16, 10)) -> None
      :canonical: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard.initialize_dashboard

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard.initialize_dashboard

   .. py:method:: _show_and_raise_window()
      :canonical: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard._show_and_raise_window

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard._show_and_raise_window

   .. py:method:: _on_close(event)
      :canonical: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard._on_close

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard._on_close

   .. py:method:: keep_window_open()
      :canonical: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard.keep_window_open

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard.keep_window_open

   .. py:method:: _setup_time_series_plots() -> None
      :canonical: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard._setup_time_series_plots

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard._setup_time_series_plots

   .. py:method:: _setup_progress_bar() -> None
      :canonical: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard._setup_progress_bar

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard._setup_progress_bar

   .. py:method:: _set_titles() -> None
      :canonical: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard._set_titles

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard._set_titles

   .. py:method:: update(flow_field: typing.Dict[str, numpy.ndarray], bathymetry: numpy.ndarray, particles: typing.Dict[str, numpy.ndarray], current_time: float, timestep: float, plot_interval: float, simulation_start_time: float = 0, simulation_end_time: float | None = None) -> None
      :canonical: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard.update

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard.update

   .. py:method:: _store_particle_data(particles: typing.Dict[str, numpy.ndarray], current_time: float, timestep: float, flow_field: typing.Dict[str, numpy.ndarray]) -> None
      :canonical: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard._store_particle_data

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard._store_particle_data

   .. py:method:: _update_flowfield_plot(flow_field: typing.Dict[str, numpy.ndarray], bathymetry: numpy.ndarray) -> None
      :canonical: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard._update_flowfield_plot

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard._update_flowfield_plot

   .. py:method:: _update_bathymetry_plot(flow_field: typing.Dict[str, numpy.ndarray], bathymetry: numpy.ndarray, particles: typing.Dict[str, numpy.ndarray]) -> None
      :canonical: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard._update_bathymetry_plot

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard._update_bathymetry_plot

   .. py:method:: _update_time_series_plots() -> None
      :canonical: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard._update_time_series_plots

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard._update_time_series_plots

   .. py:method:: _update_progress_bar(current_time: float, start_time: float, end_time: float) -> None
      :canonical: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard._update_progress_bar

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard._update_progress_bar

   .. py:method:: _setup_window_position() -> None
      :canonical: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard._setup_window_position

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard._setup_window_position

   .. py:method:: close() -> None
      :canonical: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard.close

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard.close

   .. py:method:: save(save_path: str) -> None
      :canonical: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard.save

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.simulation_dashboard.SimulationDashboard.save
