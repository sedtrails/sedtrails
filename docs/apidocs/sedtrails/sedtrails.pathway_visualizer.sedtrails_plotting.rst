:py:mod:`sedtrails.pathway_visualizer.sedtrails_plotting`
=========================================================

.. py:module:: sedtrails.pathway_visualizer.sedtrails_plotting

.. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`TrajectoryArrays <sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays>`
     - .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays
          :summary:
   * - :py:obj:`InteractivePolygonTool <sedtrails.pathway_visualizer.sedtrails_plotting.InteractivePolygonTool>`
     - .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.InteractivePolygonTool
          :summary:
   * - :py:obj:`ParticleStats <sedtrails.pathway_visualizer.sedtrails_plotting.ParticleStats>`
     - .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.ParticleStats
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`_flatten_valid <sedtrails.pathway_visualizer.sedtrails_plotting._flatten_valid>`
     - .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting._flatten_valid
          :summary:
   * - :py:obj:`plot_trajectories_by_age <sedtrails.pathway_visualizer.sedtrails_plotting.plot_trajectories_by_age>`
     - .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.plot_trajectories_by_age
          :summary:
   * - :py:obj:`_rotate_points <sedtrails.pathway_visualizer.sedtrails_plotting._rotate_points>`
     - .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting._rotate_points
          :summary:
   * - :py:obj:`plot_trajectories_by_baseline <sedtrails.pathway_visualizer.sedtrails_plotting.plot_trajectories_by_baseline>`
     - .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.plot_trajectories_by_baseline
          :summary:
   * - :py:obj:`animate_particles <sedtrails.pathway_visualizer.sedtrails_plotting.animate_particles>`
     - .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.animate_particles
          :summary:
   * - :py:obj:`_points_in_poly <sedtrails.pathway_visualizer.sedtrails_plotting._points_in_poly>`
     - .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting._points_in_poly
          :summary:
   * - :py:obj:`particles_originating_in_polygon <sedtrails.pathway_visualizer.sedtrails_plotting.particles_originating_in_polygon>`
     - .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.particles_originating_in_polygon
          :summary:
   * - :py:obj:`particles_passing_through_polygon <sedtrails.pathway_visualizer.sedtrails_plotting.particles_passing_through_polygon>`
     - .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.particles_passing_through_polygon
          :summary:
   * - :py:obj:`particles_between_two_polygons <sedtrails.pathway_visualizer.sedtrails_plotting.particles_between_two_polygons>`
     - .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.particles_between_two_polygons
          :summary:
   * - :py:obj:`particles_include_exclude <sedtrails.pathway_visualizer.sedtrails_plotting.particles_include_exclude>`
     - .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.particles_include_exclude
          :summary:
   * - :py:obj:`compute_particle_stats <sedtrails.pathway_visualizer.sedtrails_plotting.compute_particle_stats>`
     - .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.compute_particle_stats
          :summary:
   * - :py:obj:`stats_to_csv <sedtrails.pathway_visualizer.sedtrails_plotting.stats_to_csv>`
     - .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.stats_to_csv
          :summary:
   * - :py:obj:`quick_explorer <sedtrails.pathway_visualizer.sedtrails_plotting.quick_explorer>`
     - .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.quick_explorer
          :summary:
   * - :py:obj:`plot_density_heatmap <sedtrails.pathway_visualizer.sedtrails_plotting.plot_density_heatmap>`
     - .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.plot_density_heatmap
          :summary:
   * - :py:obj:`load_from_xarray <sedtrails.pathway_visualizer.sedtrails_plotting.load_from_xarray>`
     - .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.load_from_xarray
          :summary:

API
~~~

.. py:class:: TrajectoryArrays
   :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays

   .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays

   .. py:attribute:: time
      :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays.time
      :type: numpy.ndarray
      :value: None

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays.time

   .. py:attribute:: x
      :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays.x
      :type: numpy.ndarray
      :value: None

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays.x

   .. py:attribute:: y
      :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays.y
      :type: numpy.ndarray
      :value: None

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays.y

   .. py:attribute:: status_alive
      :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays.status_alive
      :type: typing.Optional[numpy.ndarray]
      :value: None

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays.status_alive

   .. py:attribute:: status_domain
      :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays.status_domain
      :type: typing.Optional[numpy.ndarray]
      :value: None

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays.status_domain

   .. py:attribute:: status_released
      :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays.status_released
      :type: typing.Optional[numpy.ndarray]
      :value: None

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays.status_released

   .. py:attribute:: status_mobile
      :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays.status_mobile
      :type: typing.Optional[numpy.ndarray]
      :value: None

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays.status_mobile

   .. py:attribute:: population_id
      :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays.population_id
      :type: typing.Optional[numpy.ndarray]
      :value: None

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays.population_id

   .. py:attribute:: trajectory_id
      :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays.trajectory_id
      :type: typing.Optional[typing.Sequence[str]]
      :value: None

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays.trajectory_id

   .. py:method:: valid_mask() -> numpy.ndarray
      :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays.valid_mask

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays.valid_mask

   .. py:method:: release_time() -> numpy.ndarray
      :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays.release_time

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays.release_time

   .. py:method:: age() -> numpy.ndarray
      :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays.age

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays.age

.. py:function:: _flatten_valid(x: numpy.ndarray, y: numpy.ndarray, c: numpy.ndarray, mask: typing.Optional[numpy.ndarray] = None) -> typing.Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
   :canonical: sedtrails.pathway_visualizer.sedtrails_plotting._flatten_valid

   .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting._flatten_valid

.. py:function:: plot_trajectories_by_age(tr: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays, units_scale: float = 1.0, point_size: float = 8.0, cmap: str = 'viridis', first_stable_index: int = 0, show_start: bool = True, show_end: bool = True, ax: typing.Optional[matplotlib.pyplot.Axes] = None) -> matplotlib.pyplot.Axes
   :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.plot_trajectories_by_age

   .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.plot_trajectories_by_age

.. py:function:: _rotate_points(x: numpy.ndarray, y: numpy.ndarray, rotation_deg: float, origin_xy: typing.Tuple[float, float]) -> typing.Tuple[numpy.ndarray, numpy.ndarray]
   :canonical: sedtrails.pathway_visualizer.sedtrails_plotting._rotate_points

   .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting._rotate_points

.. py:function:: plot_trajectories_by_baseline(tr: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays, rotation_deg: float = 0.0, first_stable_index: int = 0, origin_xy: typing.Optional[typing.Tuple[float, float]] = None, units_scale: float = 1.0, point_size: float = 8.0, cmap: str = 'viridis', ax: typing.Optional[matplotlib.pyplot.Axes] = None) -> matplotlib.pyplot.Axes
   :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.plot_trajectories_by_baseline

   .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.plot_trajectories_by_baseline

.. py:function:: animate_particles(tr: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays, rotation_deg: float = 0.0, first_stable_index: int = 0, origin_xy: typing.Optional[typing.Tuple[float, float]] = None, color_mode: str = 'baseline', units_scale: float = 1.0, point_size: float = 12.0, interval_ms: int = 80, t_indices: typing.Optional[typing.Sequence[int]] = None, save_path: typing.Optional[str] = None, dpi: int = 150)
   :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.animate_particles

   .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.animate_particles

.. py:function:: _points_in_poly(x: numpy.ndarray, y: numpy.ndarray, poly_xy: numpy.ndarray) -> numpy.ndarray
   :canonical: sedtrails.pathway_visualizer.sedtrails_plotting._points_in_poly

   .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting._points_in_poly

.. py:function:: particles_originating_in_polygon(tr: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays, poly_xy: numpy.ndarray, first_stable_index: int = 0) -> numpy.ndarray
   :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.particles_originating_in_polygon

   .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.particles_originating_in_polygon

.. py:function:: particles_passing_through_polygon(tr: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays, poly_xy: numpy.ndarray, first_stable_index: int = 0) -> numpy.ndarray
   :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.particles_passing_through_polygon

   .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.particles_passing_through_polygon

.. py:function:: particles_between_two_polygons(tr: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays, poly_a: numpy.ndarray, poly_b: numpy.ndarray, order: typing.Optional[str] = None, first_stable_index: int = 0) -> numpy.ndarray
   :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.particles_between_two_polygons

   .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.particles_between_two_polygons

.. py:function:: particles_include_exclude(tr: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays, include_polys: typing.List[numpy.ndarray], exclude_polys: typing.Optional[typing.List[numpy.ndarray]] = None) -> numpy.ndarray
   :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.particles_include_exclude

   .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.particles_include_exclude

.. py:class:: InteractivePolygonTool(ax: matplotlib.pyplot.Axes, on_done)
   :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.InteractivePolygonTool

   .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.InteractivePolygonTool

   .. rubric:: Initialization

   .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.InteractivePolygonTool.__init__

   .. py:method:: _onselect(verts)
      :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.InteractivePolygonTool._onselect

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.InteractivePolygonTool._onselect

   .. py:method:: disconnect()
      :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.InteractivePolygonTool.disconnect

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.InteractivePolygonTool.disconnect

   .. py:property:: polygon
      :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.InteractivePolygonTool.polygon
      :type: typing.Optional[numpy.ndarray]

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.InteractivePolygonTool.polygon

.. py:class:: ParticleStats
   :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.ParticleStats

   .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.ParticleStats

   .. py:attribute:: srcx
      :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.ParticleStats.srcx
      :type: float
      :value: None

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.ParticleStats.srcx

   .. py:attribute:: srcy
      :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.ParticleStats.srcy
      :type: float
      :value: None

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.ParticleStats.srcy

   .. py:attribute:: srct
      :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.ParticleStats.srct
      :type: float
      :value: None

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.ParticleStats.srct

   .. py:attribute:: netDispX
      :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.ParticleStats.netDispX
      :type: float
      :value: None

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.ParticleStats.netDispX

   .. py:attribute:: netDispY
      :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.ParticleStats.netDispY
      :type: float
      :value: None

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.ParticleStats.netDispY

   .. py:attribute:: netDispMag
      :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.ParticleStats.netDispMag
      :type: float
      :value: None

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.ParticleStats.netDispMag

   .. py:attribute:: duration
      :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.ParticleStats.duration
      :type: float
      :value: None

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.ParticleStats.duration

   .. py:attribute:: uLRV
      :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.ParticleStats.uLRV
      :type: float
      :value: None

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.ParticleStats.uLRV

   .. py:attribute:: vLRV
      :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.ParticleStats.vLRV
      :type: float
      :value: None

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.ParticleStats.vLRV

   .. py:attribute:: grossDisp
      :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.ParticleStats.grossDisp
      :type: float
      :value: None

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.ParticleStats.grossDisp

   .. py:attribute:: ratio
      :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.ParticleStats.ratio
      :type: float
      :value: None

      .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.ParticleStats.ratio

.. py:function:: compute_particle_stats(tr: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays, first_stable_index: int = 0) -> typing.List[sedtrails.pathway_visualizer.sedtrails_plotting.ParticleStats]
   :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.compute_particle_stats

   .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.compute_particle_stats

.. py:function:: stats_to_csv(stats: typing.List[sedtrails.pathway_visualizer.sedtrails_plotting.ParticleStats], path: str) -> None
   :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.stats_to_csv

   .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.stats_to_csv

.. py:function:: quick_explorer(tr: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays, rotation_deg: float = 0.0, first_stable_index: int = 0, origin_xy: typing.Optional[typing.Tuple[float, float]] = None, units_scale: float = 1.0)
   :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.quick_explorer

   .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.quick_explorer

.. py:function:: plot_density_heatmap(tr: sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays, bins: int = 200, units_scale: float = 1.0, first_stable_index: int = 0, ax: typing.Optional[matplotlib.pyplot.Axes] = None) -> matplotlib.pyplot.Axes
   :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.plot_density_heatmap

   .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.plot_density_heatmap

.. py:function:: load_from_xarray(ds) -> sedtrails.pathway_visualizer.sedtrails_plotting.TrajectoryArrays
   :canonical: sedtrails.pathway_visualizer.sedtrails_plotting.load_from_xarray

   .. autodoc2-docstring:: sedtrails.pathway_visualizer.sedtrails_plotting.load_from_xarray
