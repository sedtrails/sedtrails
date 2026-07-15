"""
NetCDF Writer
=============

Writes NetCDF files produced by the SedTrails Particle Tracer System.

"""

import logging
import netCDF4 as nc4
import numpy as np
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

DEFAULT_PARTICLE_CHUNK = 65_536
DEFAULT_SYNC_INTERVAL = 10

_COORDINATE_DTYPES = {
    'float32': 'f4',
    'f4': 'f4',
    np.dtype('float32'): 'f4',
    'float64': 'f8',
    'f8': 'f8',
    np.dtype('float64'): 'f8',
}
_STATUS_DTYPES = {
    'uint8': ('u1', np.uint8(255)),
    'u1': ('u1', np.uint8(255)),
    np.dtype('uint8'): ('u1', np.uint8(255)),
    'int32': ('i4', np.int32(-1)),
    'i4': ('i4', np.int32(-1)),
    np.dtype('int32'): ('i4', np.int32(-1)),
}
_STATUS_DEFAULTS = {
    'status_alive': 1,
    'status_buried': 0,
    'status_domain': 1,
    'status_transported': 0,
    'status_released': 1,
    'status_mobile': 0,
    'status_beached': 0,
    'status_left_domain': 0,
    'status_suspended': 0,
    'status_deposited': 0,
    'status_available_for_entrainment': 0,
    'status_entrained_now': 0,
    'status_deposited_now': 0,
}

_CORE_FLOAT_FIELDS = ('x', 'y', 'z', 'z_p', 'z_burial', 'burial_depth', 'mixing_depth')
_Q3D_MIN_FLOAT_FIELDS = (
    'q3d_first_substep_z_p',
    'centroid_particle_velocity_x',
    'centroid_particle_velocity_y',
    'centroid_particle_velocity',
    'first_substep_modified_centroid_particle_velocity_x',
    'first_substep_modified_centroid_particle_velocity_y',
    'first_substep_modified_centroid_particle_velocity',
    'first_substep_horizontal_particle_velocity_x',
    'first_substep_horizontal_particle_velocity_y',
    'first_substep_horizontal_particle_velocity',
    'first_substep_vertical_particle_velocity',
    'q3d_entrainment_probability',
    'q3d_entrainment_height_above_bed',
)
_Q3D_FULL_FLOAT_FIELDS = (
    'first_substep_horizontal_diffusion_velocity_x',
    'first_substep_horizontal_diffusion_velocity_y',
    'first_substep_horizontal_diffusion_velocity',
    'first_substep_vertical_advection_velocity',
    'first_substep_vertical_diffusion_velocity',
    'first_substep_horizontal_diffusion_coefficient',
    'first_substep_vertical_diffusion_coefficient',
    'first_substep_bed_level',
    'first_substep_water_depth',
    'first_substep_skin_roughness_height',
    'first_substep_max_shear_velocity',
    'first_substep_profile_roughness_height',
    'first_substep_total_transport_centroid_elevation',
    'first_substep_q3d_velocity_deficit_coefficient',
    'first_substep_q3d_vertical_velocity_gradient',
    'first_substep_settling_velocity',
    'first_substep_depth_avg_flow_velocity_magnitude',
    'first_substep_rouse_number',
    'turbulent_shields_number',
    'critical_shields_number',
)
_Q3D_INT_FIELDS = (
    'q3d_vertical_update_scheme_code',
    'q3d_motion_substeps',
)


class NetCDFWriter:
    """
    Streaming NetCDF writer for SedTRAILS particle trajectory output.

    The writer keeps one NetCDF4 handle open and writes one saved trajectory
    slot at a time. It deliberately avoids building an in-memory xarray Dataset
    for the full particle/time cube.

    Attributes
    ----------
    output_dir : pathlib.Path
        The directory where output files (NetCDF, images, etc.) are stored.

    """

    def __init__(self, output_dir):
        output_dir = Path(output_dir)
        # If the output directory already exists, we add a timestamp to avoid overwriting
        # if output_dir.exists():
        #     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        #     output_dir = output_dir.parent / f'{output_dir.name}_{timestamp}'
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._streaming_path = None
        self._write_count = 0
        self._sync_interval = DEFAULT_SYNC_INTERVAL
        self._reopen_interval = None

    def _validate_filename(self, filename):
        """
        Validates the file name to ensure it is a NetCDF file.
        """
        if not filename.endswith('.nc'):
            raise ValueError('Output file must have a .nc extension.')

    @staticmethod
    def _normalize_coordinate_dtype(dtype) -> str:
        try:
            return _COORDINATE_DTYPES[dtype]
        except KeyError as exc:
            valid = ', '.join(('float32', 'float64'))
            raise ValueError(f'Unsupported coordinate dtype {dtype!r}. Use one of: {valid}.') from exc

    @staticmethod
    def _normalize_status_dtype(dtype) -> tuple[str, np.integer]:
        try:
            return _STATUS_DTYPES[dtype]
        except KeyError as exc:
            valid = ', '.join(('uint8', 'int32'))
            raise ValueError(f'Unsupported status dtype {dtype!r}. Use one of: {valid}.') from exc

    @staticmethod
    def _compression_kwargs(enabled: bool, compression_level: int, shuffle: bool) -> dict:
        if not enabled or compression_level == 0:
            return {}
        return {
            'zlib': True,
            'complevel': int(compression_level),
            'shuffle': bool(shuffle),
        }

    @staticmethod
    def _time_particle_chunks(n_slots: int, n_particles: int, time_chunk: int, particle_chunk: int) -> tuple[int, int]:
        return (
            max(1, min(int(time_chunk), max(1, int(n_slots)))),
            max(1, min(int(particle_chunk), max(1, int(n_particles)))),
        )

    @staticmethod
    def _write_name(var, index: int, value: str, name_strlen: int) -> None:
        var[index, :] = np.array(list(value[:name_strlen].ljust(name_strlen)), dtype='S1')

    @classmethod
    def _create_static_metadata(
        cls,
        ds,
        n_particles: int,
        n_populations: int,
        n_flowfields: int,
        populations: list,
        flow_field_names: list,
        name_strlen: int,
    ) -> None:
        """Create and write metadata that does not vary with output time."""
        ds.createVariable('population_name', 'S1', ('n_populations', 'name_strlen'))
        ds.createVariable('population_particle_type', 'i4', ('n_populations',))
        ds.createVariable('population_start_idx', 'i8', ('n_populations',))
        ds.createVariable('population_count', 'i8', ('n_populations',))
        ds.createVariable('population_repr_volume', 'f8', ('n_populations',))
        ds.createVariable('trajectory_id', 'i8', ('n_particles',))
        ds.createVariable('population_id', 'i4', ('n_particles',))
        ds.createVariable('flowfield_name', 'S1', ('n_flowfields', 'name_strlen'))

        ds['trajectory_id'][:] = np.arange(n_particles, dtype=np.int64)

        particle_offset = 0
        for pop_idx, population in enumerate(populations):
            pop_name = getattr(population, 'name', f'population_{pop_idx}')
            cls._write_name(ds['population_name'], pop_idx, str(pop_name), name_strlen)
            ds['population_particle_type'][pop_idx] = int(getattr(population, 'particle_type', 0))
            ds['population_start_idx'][pop_idx] = particle_offset
            n_part = len(population.particles['x'])
            ds['population_count'][pop_idx] = n_part
            repr_vol = getattr(population, 'repr_volume', np.nan)
            ds['population_repr_volume'][pop_idx] = float(repr_vol) if repr_vol is not None else np.nan
            ds['population_id'][particle_offset:particle_offset + n_part] = pop_idx
            particle_offset += n_part

        for ff_idx in range(n_flowfields):
            ff_name = flow_field_names[ff_idx] if flow_field_names and ff_idx < len(flow_field_names) else ''
            cls._write_name(ds['flowfield_name'], ff_idx, str(ff_name), name_strlen)

    @staticmethod
    def _particle_field(particles: dict, name: str, default):
        if name in particles:
            return particles[name]
        return default

    @staticmethod
    def _q3d_float_fields(level: str) -> tuple[str, ...]:
        normalized = str(level or 'minimal').strip().lower().replace('-', '_')
        if normalized not in {'minimal', 'full'}:
            raise ValueError("q3d_diagnostics must be either 'minimal' or 'full'.")
        if normalized == 'full':
            return _Q3D_MIN_FLOAT_FIELDS + _Q3D_FULL_FLOAT_FIELDS
        return _Q3D_MIN_FLOAT_FIELDS

    def open_output(
        self,
        filename: str,
        n_slots: int,
        N_particles: int,
        N_populations: int,
        N_flowfields: int,
        populations: list,
        flow_field_names: list,
        name_strlen: int = 24,
        coordinate_dtype: str = 'float32',
        status_dtype: str = 'uint8',
        compression: bool = True,
        compression_level: int = 1,
        shuffle: bool = True,
        time_chunk: int = 1,
        particle_chunk: int = DEFAULT_PARTICLE_CHUNK,
        sync_interval: int | None = DEFAULT_SYNC_INTERVAL,
        reopen_interval: int | None = None,
        q3d_diagnostics: str = 'minimal',
    ):
        """
        Open a streaming output file with pre-allocated dimensions.

        Creates the file, defines all dimensions and variables, writes static
        population and flow-field metadata, and returns the open handle.
        The file is kept open throughout the simulation; call ``close_output()``
        when done (ideally inside a ``try/finally`` block).

        Parameters
        ----------
        filename : str
            Name of the NetCDF file to create (must end with ``.nc``).
        n_slots : int
            Number of time slots to pre-allocate (one per ``save_interval``
            boundary plus one for the initial state).
        N_particles : int
            Total number of particles across all populations.
        N_populations : int
            Number of particle populations.
        N_flowfields : int
            Number of flow-field tracers.
        populations : list
            Population objects; used to write static metadata (name, type, count).
        flow_field_names : list
            Names of the flow fields; written as static metadata.
        name_strlen : int, optional
            Maximum character length for string variables (default 24).
        coordinate_dtype : {'float32', 'float64'}, optional
            Floating point dtype for large trajectory variables. Defaults to
            ``float32`` to reduce output volume.
        status_dtype : {'uint8', 'int32'}, optional
            Integer dtype for status flags. Defaults to ``uint8``.
        compression : bool, optional
            Whether to compress large chunked variables.
        compression_level : int, optional
            NetCDF/HDF5 zlib compression level, 0-9. Level 1 is intentionally
            cheap and usually gives most of the size win for particle tracks.
        shuffle : bool, optional
            Whether to enable the HDF5 shuffle filter when compression is on.
        time_chunk, particle_chunk : int, optional
            Chunk shape for time-major trajectory variables.
        sync_interval : int or None, optional
            Flush every N successful writes. Use 0/None to sync only on close.
        reopen_interval : int or None, optional
            Proactively close and reopen the output file every N writes. This is
            useful on unstable network filesystems but is disabled by default.
        Returns
        -------
        netCDF4.Dataset
            Open file handle for use with ``record_output()`` and ``close_output()``.
        """
        self._validate_filename(filename)
        output_path = self.output_dir / filename

        coordinate_dtype = self._normalize_coordinate_dtype(coordinate_dtype)
        status_dtype, status_fill = self._normalize_status_dtype(status_dtype)
        compression_level = int(compression_level)
        if compression_level < 0 or compression_level > 9:
            raise ValueError('compression_level must be between 0 and 9.')
        time_particle_chunks = self._time_particle_chunks(n_slots, N_particles, time_chunk, particle_chunk)
        compression_kwargs = self._compression_kwargs(compression, compression_level, shuffle)
        q3d_float_fields = self._q3d_float_fields(q3d_diagnostics)

        ds = nc4.Dataset(str(output_path), 'w', format='NETCDF4')

        # Time-major layout writes one full particle slice per saved time. This
        # matches the simulation access pattern and keeps chunk writes contiguous.
        ds.createDimension('n_particles', N_particles)
        ds.createDimension('n_populations', N_populations)
        ds.createDimension('n_timesteps', n_slots)
        ds.createDimension('n_flowfields', N_flowfields)
        ds.createDimension('name_strlen', name_strlen)

        # Global attributes
        ds.title = 'SedTRAILS Particle Simulation Results'
        ds.institution = 'SedTRAILS Particle Tracer System'
        ds.created_on = datetime.now().isoformat()
        ds.sedtrails_output_schema = 'trajectory_v2'
        ds.trajectory_layout = 'time_particle'
        ds.coordinate_dtype = coordinate_dtype
        ds.status_dtype = status_dtype
        ds.estimated_output_slots = int(n_slots)
        ds.written_slots = 0
        ds.sync_interval = 0 if sync_interval is None else int(sync_interval)
        ds.reopen_interval = 0 if reopen_interval is None else int(reopen_interval)
        ds.q3d_diagnostics = str(q3d_diagnostics)

        self._create_static_metadata(
            ds,
            N_particles,
            N_populations,
            N_flowfields,
            populations,
            flow_field_names,
            name_strlen,
        )

        # Time-varying trajectory variables. ``time`` is one value per saved
        # output slot; all particles share the same simulation clock.
        ds.createVariable(
            'time',
            'f8',
            ('n_timesteps',),
            fill_value=np.nan,
            chunksizes=(time_particle_chunks[0],),
            **compression_kwargs,
        )
        for var_name in _CORE_FLOAT_FIELDS + q3d_float_fields:
            ds.createVariable(
                var_name,
                coordinate_dtype,
                ('n_timesteps', 'n_particles'),
                fill_value=np.nan,
                chunksizes=time_particle_chunks,
                **compression_kwargs,
            )
        for var_name in _Q3D_INT_FIELDS:
            ds.createVariable(
                var_name,
                'i4',
                ('n_timesteps', 'n_particles'),
                fill_value=np.int32(-1),
                chunksizes=time_particle_chunks,
                **compression_kwargs,
            )
        for var_name in _STATUS_DEFAULTS:
            ds.createVariable(
                var_name,
                status_dtype,
                ('n_timesteps', 'n_particles'),
                fill_value=status_fill,
                chunksizes=time_particle_chunks,
                **compression_kwargs,
            )

        # Store path so record_output can reopen on network/HDF errors
        self._streaming_path = str(output_path)
        self._write_count = 0
        self._sync_interval = 0 if sync_interval is None else int(sync_interval)
        self._reopen_interval = None if reopen_interval is None else int(reopen_interval)

        ds.sync()
        return ds

    @classmethod
    def _write_slot(cls, h, populations: list, slot_idx: int, current_time: float) -> None:
        """Write one save-interval slot to an open netCDF4 handle."""
        particle_offset = 0
        h['time'][slot_idx] = current_time
        for population in populations:
            particles = population.particles
            num_particles = len(population.particles['x'])
            sl = slice(particle_offset, particle_offset + num_particles)

            for field_name in _CORE_FLOAT_FIELDS:
                default = 0.0 if field_name == 'z' else np.nan
                h[field_name][slot_idx, sl] = cls._particle_field(particles, field_name, default)
            for field_name in _Q3D_MIN_FLOAT_FIELDS + _Q3D_FULL_FLOAT_FIELDS:
                if field_name in h.variables:
                    h[field_name][slot_idx, sl] = cls._particle_field(particles, field_name, np.nan)
            for field_name in _Q3D_INT_FIELDS:
                h[field_name][slot_idx, sl] = cls._particle_field(particles, field_name, np.int32(-1))
            for status_name, default in _STATUS_DEFAULTS.items():
                h[status_name][slot_idx, sl] = cls._particle_field(particles, status_name, default)

            particle_offset += num_particles

        h.written_slots = max(int(getattr(h, 'written_slots', 0)), int(slot_idx) + 1)

    def _reopen_handle(self, nc_handle) -> 'nc4.Dataset':
        """Close a broken/stale handle and reopen the file in read-write mode."""
        path = getattr(self, '_streaming_path', None) or nc_handle.filepath()
        try:
            nc_handle.close()
        except Exception:
            pass
        return nc4.Dataset(str(path), 'r+', format='NETCDF4')

    def record_output(
        self, nc_handle, populations: list, slot_idx: int, current_time: float
    ) -> 'nc4.Dataset':
        """
        Write current particle state to one time slot in the streaming output file.

        On network (SMB/NFS) drives the HDF5 file descriptor can go stale after
        a reconnect or server-side idle timeout.  Two defences are available:

        1. **Proactive reopen** can refresh the OS file descriptor before it
           can go stale.
        2. **Reactive reopen**: on any HDF/IO ``RuntimeError`` or ``OSError`` the
           handle is closed, the file is reopened in ``'r+'`` mode, and the write
           is retried once.

        Parameters
        ----------
        nc_handle : netCDF4.Dataset
            Open handle returned by ``open_output()`` or a previous call to this
            method.  May be silently replaced by a fresh handle on reopen.
        populations : list
            Current population objects whose particle state will be written.
        slot_idx : int
            Zero-based index of the time slot to write into.
        current_time : float
            Simulation time (seconds) to store in the ``time`` variable.

        Returns
        -------
        netCDF4.Dataset
            The (possibly refreshed) file handle.  **Callers must reassign**:
            ``nc_handle = writer.record_output(nc_handle, ...)``
        """
        # Proactive reopen prevents stale FD on long-running network-drive writes.
        self._write_count = getattr(self, '_write_count', 0) + 1
        reopen_interval = getattr(self, '_reopen_interval', None)
        if reopen_interval and reopen_interval > 0 and self._write_count % reopen_interval == 0:
            logger.debug('Proactive netCDF handle reopen at slot %d (write #%d)',
                         slot_idx, self._write_count)
            nc_handle = self._reopen_handle(nc_handle)

        # Attempt write; on HDF/IO error reopen and retry once
        try:
            self._write_slot(nc_handle, populations, slot_idx, current_time)
        except (RuntimeError, OSError) as exc:
            err_str = str(exc)
            if not any(kw in err_str for kw in ('HDF', 'NetCDF', 'errno', 'I/O')):
                raise
            logger.warning(
                'HDF/network error writing slot %d: %r; reopening output file and retrying',
                slot_idx, exc,
            )
            nc_handle = self._reopen_handle(nc_handle)
            self._write_slot(nc_handle, populations, slot_idx, current_time)
            logger.info('Retry succeeded for slot %d', slot_idx)

        sync_interval = getattr(self, '_sync_interval', DEFAULT_SYNC_INTERVAL)
        if sync_interval and sync_interval > 0 and self._write_count % sync_interval == 0:
            nc_handle.sync()

        return nc_handle

    def _write_particle_snapshot(
        self,
        filename: str,
        populations: list,
        current_time: float,
        *,
        title: str,
        file_kind: str,
        output_schema: str,
        layout: str,
        reference_date: str | None = None,
        time_units: str | None = None,
        name_strlen: int = 24,
        coordinate_dtype: str = 'float32',
        status_dtype: str = 'uint8',
        compression: bool = True,
        compression_level: int = 1,
        shuffle: bool = True,
        particle_chunk: int = DEFAULT_PARTICLE_CHUNK,
        q3d_diagnostics: str = 'minimal',
    ) -> Path:
        """Write a compact one-snapshot particle-state NetCDF file."""
        self._validate_filename(filename)
        output_path = self.output_dir / filename
        tmp_path = output_path.with_name(f'.{output_path.name}.tmp')

        coordinate_dtype = self._normalize_coordinate_dtype(coordinate_dtype)
        status_dtype, status_fill = self._normalize_status_dtype(status_dtype)
        n_particles = sum(len(population.particles['x']) for population in populations)
        n_populations = len(populations)
        particle_chunk = max(1, min(int(particle_chunk), max(1, int(n_particles))))
        compression_kwargs = self._compression_kwargs(compression, int(compression_level), shuffle)
        q3d_float_fields = self._q3d_float_fields(q3d_diagnostics)

        if tmp_path.exists():
            tmp_path.unlink()

        ds = nc4.Dataset(str(tmp_path), 'w', format='NETCDF4')
        try:
            ds.createDimension('n_particles', n_particles)
            ds.createDimension('n_populations', n_populations)
            ds.createDimension('n_flowfields', 1)
            ds.createDimension('name_strlen', name_strlen)

            ds.title = title
            ds.institution = 'SedTRAILS Particle Tracer System'
            ds.created_on = datetime.now().isoformat()
            ds.sedtrails_file_kind = file_kind
            ds.sedtrails_output_schema = output_schema
            ds.trajectory_layout = layout
            ds.q3d_diagnostics = str(q3d_diagnostics)
            if reference_date is not None:
                ds.reference_date = str(reference_date)
            if time_units is not None:
                ds.time_units = str(time_units)

            self._create_static_metadata(
                ds,
                n_particles,
                n_populations,
                1,
                populations,
                [],
                name_strlen,
            )

            ds.createVariable('time', 'f8', (), fill_value=np.nan)
            ds['time'][...] = float(current_time)

            for var_name in _CORE_FLOAT_FIELDS + q3d_float_fields:
                ds.createVariable(
                    var_name,
                    coordinate_dtype,
                    ('n_particles',),
                    fill_value=np.nan,
                    chunksizes=(particle_chunk,),
                    **compression_kwargs,
                )
            for var_name in _Q3D_INT_FIELDS:
                ds.createVariable(
                    var_name,
                    'i4',
                    ('n_particles',),
                    fill_value=np.int32(-1),
                    chunksizes=(particle_chunk,),
                    **compression_kwargs,
                )
            for var_name in _STATUS_DEFAULTS:
                ds.createVariable(
                    var_name,
                    status_dtype,
                    ('n_particles',),
                    fill_value=status_fill,
                    chunksizes=(particle_chunk,),
                    **compression_kwargs,
                )

            particle_offset = 0
            for population in populations:
                particles = population.particles
                n_part = len(particles['x'])
                sl = slice(particle_offset, particle_offset + n_part)
                for field_name in _CORE_FLOAT_FIELDS:
                    default = 0.0 if field_name == 'z' else np.nan
                    ds[field_name][sl] = self._particle_field(particles, field_name, default)
                for field_name in q3d_float_fields:
                    ds[field_name][sl] = self._particle_field(particles, field_name, np.nan)
                for field_name in _Q3D_INT_FIELDS:
                    ds[field_name][sl] = self._particle_field(particles, field_name, np.int32(-1))
                for status_name, default in _STATUS_DEFAULTS.items():
                    ds[status_name][sl] = self._particle_field(particles, status_name, default)
                particle_offset += n_part

            ds.sync()
        finally:
            ds.close()

        tmp_path.replace(output_path)
        return output_path

    def write_checkpoint(
        self,
        filename: str,
        populations: list,
        current_time: float,
        *,
        reference_date: str | None = None,
        time_units: str | None = None,
        name_strlen: int = 24,
        coordinate_dtype: str = 'float32',
        status_dtype: str = 'uint8',
        compression: bool = True,
        compression_level: int = 1,
        shuffle: bool = True,
        particle_chunk: int = DEFAULT_PARTICLE_CHUNK,
        q3d_diagnostics: str = 'minimal',
    ) -> Path:
        """
        Write a compact restart checkpoint containing only the current state.

        Parameters
        ----------
        filename : str
            Name of the file to write.
        populations : list
            Particle populations to process.
        current_time : float
            Current simulation time in seconds.
        reference_date : str | None
            Reference date for converting model times.
        time_units : str | None
            NetCDF time units string.
        name_strlen : int
            Maximum stored population-name length.
        coordinate_dtype : str
            NumPy dtype used for coordinate variables.
        status_dtype : str
            NumPy dtype used for status variables.
        compression : bool
            Whether NetCDF variables are compressed.
        compression_level : int
            Compression level for NetCDF variables.
        shuffle : bool
            Whether the NetCDF shuffle filter is enabled.
        particle_chunk : int
            Particle chunk size for NetCDF variables.

        Returns
        -------
        Path
            Path to the checkpoint file.
        """
        return self._write_particle_snapshot(
            filename,
            populations,
            current_time,
            title='SedTRAILS Particle Simulation Checkpoint',
            file_kind='checkpoint',
            output_schema='checkpoint_v1',
            layout='checkpoint',
            reference_date=reference_date,
            time_units=time_units,
            name_strlen=name_strlen,
            coordinate_dtype=coordinate_dtype,
            status_dtype=status_dtype,
            compression=compression,
            compression_level=compression_level,
            shuffle=shuffle,
            particle_chunk=particle_chunk,
            q3d_diagnostics=q3d_diagnostics,
        )

    def write_end_positions(
        self,
        filename: str,
        populations: list,
        current_time: float,
        *,
        reference_date: str | None = None,
        time_units: str | None = None,
        name_strlen: int = 24,
        coordinate_dtype: str = 'float32',
        status_dtype: str = 'uint8',
        compression: bool = True,
        compression_level: int = 1,
        shuffle: bool = True,
        particle_chunk: int = DEFAULT_PARTICLE_CHUNK,
        q3d_diagnostics: str = 'minimal',
    ) -> Path:
        """
        Write compact end-position results containing one state per particle.

        Parameters
        ----------
        filename : str
            Name of the result file to write.
        populations : list
            Particle populations to process.
        current_time : float
            Final simulation time in seconds.
        reference_date : str | None
            Reference date for converting model times.
        time_units : str | None
            NetCDF time units string.
        name_strlen : int
            Maximum stored population-name length.
        coordinate_dtype : str
            NumPy dtype used for coordinate variables.
        status_dtype : str
            NumPy dtype used for status variables.
        compression : bool
            Whether NetCDF variables are compressed.
        compression_level : int
            Compression level for NetCDF variables.
        shuffle : bool
            Whether the NetCDF shuffle filter is enabled.
        particle_chunk : int
            Particle chunk size for NetCDF variables.

        Returns
        -------
        Path
            Path to the end-position result file.
        """
        return self._write_particle_snapshot(
            filename,
            populations,
            current_time,
            title='SedTRAILS Particle Simulation End Positions',
            file_kind='end_positions',
            output_schema='end_positions_v1',
            layout='end_positions',
            reference_date=reference_date,
            time_units=time_units,
            name_strlen=name_strlen,
            coordinate_dtype=coordinate_dtype,
            status_dtype=status_dtype,
            compression=compression,
            compression_level=compression_level,
            shuffle=shuffle,
            particle_chunk=particle_chunk,
            q3d_diagnostics=q3d_diagnostics,
        )

    def close_output(self, nc_handle) -> Path:
        """
        Close the streaming output file and return its path.

        Parameters
        ----------
        nc_handle : netCDF4.Dataset
            Open handle returned by ``open_output()``.

        Returns
        -------
        pathlib.Path
            Absolute path to the closed NetCDF file.
        """
        path = Path(nc_handle.filepath())
        nc_handle.sync()
        nc_handle.close()
        return path


if __name__ == '__main__':
    pass
