"""Regression test: burial depth must stay exactly zero on a static bed.

Guards the invariant documented in ``ParticlePopulation.update_burial_depth``:
``bed_level_previous`` always holds the bed level at the particle's *current*
position, so the burial increment is a purely temporal bed change (zero for a
static bed). The invariant requires ``update_bed_level_change_after_movement``
to run after EVERY ``update_position`` — i.e. inside the per-flow-field loop
of the simulation manager, because the vanwesten tracer moves particles once
per flow field (bed load + suspended).

History: the invariant was established in bdf7a99 (2026-06-02) and silently
regressed in merge 90d0b5a (2026-06-12), where an upstream re-indent of the
flow-field loop left the re-sample call outside the loop without any of its
lines changing. The result: the spatial bed gradient along each bed-load
displacement was booked as a temporal bed change, and the ``max(burial, 0)``
floor rectified that noise into a one-way burial ratchet. In a year-long
morphostatic Westerschelde run this spuriously buried and immobilised ~99% of
all particles.

The test runs a real end-to-end simulation on a synthetic forcing with a
STATIC sloped bed and steady eastward transport: particles must move, and
burial depth must remain exactly zero.
"""

import logging
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytest

from sedtrails.simulation_orchestrator.simulation_manager import Simulation

REFERENCE_DATE = '2020-01-01 00:00:00'


def _write_forcing(path: Path) -> None:
    """Synthetic SedTRAILS forcing: static bed sloping up in x, steady flow."""
    nx = ny = 60
    xs = np.linspace(0.0, 10_000.0, nx)
    ys = np.linspace(0.0, 10_000.0, ny)
    xg, yg = np.meshgrid(xs, ys, indexing='xy')
    x = xg.ravel()
    y = yg.ravel()
    n_nodes = x.size
    times = np.array([0.0, 43_200.0, 86_400.0])  # 3 steps, 12 h apart

    bedlevel = -10.0 + 1.0e-3 * x  # static slope: 1 m per km, up-slope eastward

    with nc.Dataset(path, 'w', format='NETCDF4') as ds:
        ds.createDimension('time', len(times))
        ds.createDimension('nNodes', n_nodes)
        ds.createDimension('nSedTot', 1)

        v = ds.createVariable('time', 'f8', ('time',))
        v.units = f'seconds since {REFERENCE_DATE}'
        v.calendar = 'standard'
        v[:] = times

        for name, values in (('net_xcc', x), ('net_ycc', y)):
            v = ds.createVariable(name, 'f8', ('nNodes',))
            v.units = 'm'
            v[:] = values

        def _steady_2d(name, value_per_node):
            var = ds.createVariable(name, 'f8', ('time', 'nNodes'))
            var[:] = np.tile(value_per_node, (len(times), 1))

        def _steady_3d(name, value_per_node):
            var = ds.createVariable(name, 'f8', ('time', 'nSedTot', 'nNodes'))
            var[:] = np.tile(value_per_node, (len(times), 1, 1))

        _steady_2d('bedlevel', bedlevel)
        _steady_2d('waterdepth', np.full(n_nodes, 5.0))
        _steady_2d('sea_water_x_velocity', np.full(n_nodes, 0.5))
        _steady_2d('sea_water_y_velocity', np.zeros(n_nodes))
        _steady_2d('mean_bss_magnitude', np.full(n_nodes, 1.0))
        _steady_2d('max_bss_magnitude', np.full(n_nodes, 2.0))  # >> tau_cr: mobile everywhere
        _steady_3d('bedload_x_comp', np.full(n_nodes, 0.1))
        _steady_3d('bedload_y_comp', np.zeros(n_nodes))
        _steady_3d('susload_x_comp', np.full(n_nodes, 0.05))
        _steady_3d('susload_y_comp', np.zeros(n_nodes))
        _steady_3d('suspended_sed_conc', np.full(n_nodes, 0.05))


def _write_config(path: Path, forcing: Path, output_dir: Path) -> None:
    path.write_text(
        f"""general:
  input_model:
    format: fm_netcdf
    reference_date: 2020-01-01
    morfac: 1
inputs:
  data: {forcing.as_posix()}
  read_interval: 2D
time:
  start: 2020-01-01 00:00:00
  timestep: 60S
  duration: 1H
  cfl_condition: 0.9
particles:
  populations:
    - name: static_bed_sand
      particle_type: sand
      characteristics:
        grain_size: 0.00018
        density: 2650.0
      tracer_methods:
        vanwesten:
          flow_field_name:
            - bed_load_velocity
            - suspended_velocity
          suspended_velocity_method: soulsby_2011
      transport_probability: reduced_velocity
      seeding:
        burial_depth:
          constant: 0
        release_start: 2020-01-01 00:00:00
        quantity: 1
        strategy:
          point:
            locations:
              - "3000,5000"
              - "4000,4000"
              - "5000,5000"
              - "6000,6000"
outputs:
  directory: {output_dir.as_posix()}
  store_tracks: true
  save_interval: 600S
visualization:
  dashboard:
    enable: false
    update_interval: 1H
""",
        encoding='utf-8',
    )


@pytest.fixture
def _preserve_logging_state():
    """Running a full Simulation reconfigures the global 'sedtrails' logger;
    restore it afterwards so log-capture tests elsewhere keep working."""
    loggers = [logging.getLogger(), logging.getLogger('sedtrails')]
    saved = [(lg, lg.level, lg.propagate, list(lg.handlers)) for lg in loggers]
    yield
    for lg, level, propagate, handlers in saved:
        for handler in list(lg.handlers):
            if handler not in handlers:
                handler.close()
                lg.removeHandler(handler)
        lg.level = level
        lg.propagate = propagate


@pytest.mark.integration
def test_burial_depth_stays_zero_on_static_bed(tmp_path, _preserve_logging_state):
    forcing = tmp_path / 'static_bed_forcing.nc'
    config = tmp_path / 'static_bed.yaml'
    output_dir = tmp_path / 'output'
    _write_forcing(forcing)
    _write_config(config, forcing, output_dir)

    Simulation(str(config), enable_dashboard=False).run()

    results = output_dir / 'sedtrails_results.nc'
    assert results.exists(), 'simulation did not produce sedtrails_results.nc'

    with nc.Dataset(results) as ds:
        x = np.asarray(ds.variables['x'][:], dtype=float)
        burial = np.asarray(ds.variables['burial_depth'][:], dtype=float)

    # The particles must actually have moved (the bug only manifests during
    # transport over a bed gradient) ...
    assert np.nanmax(np.abs(x[-1, :] - x[0, :])) > 1.0, (
        'particles did not move; the burial assertion below would be vacuous'
    )

    # ... and the bed is static, so burial depth must remain exactly zero.
    # Any positive value means spatial bed differences leaked into the
    # burial bookkeeping (regression of bdf7a99, see module docstring).
    assert np.nanmax(burial) <= 1e-12, (
        f'burial_depth reached {np.nanmax(burial):.6f} m on a static bed'
    )
