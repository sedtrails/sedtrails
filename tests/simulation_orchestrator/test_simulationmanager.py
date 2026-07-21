"""
Unit tests for the Simulation class.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from sedtrails.application_interfaces.configuration_controller import ConfigurationController
from sedtrails.application_interfaces.validator import YAMLConfigValidator
from sedtrails.exceptions.exceptions import ConfigurationError
from sedtrails.particle_tracer.timer import Duration, Time
from sedtrails.simulation_orchestrator import parallel_worker as _pw
from sedtrails.simulation_orchestrator import simulation_manager
from sedtrails.simulation_orchestrator.runtime_plan import validate_population_runtime_configurations
from sedtrails.simulation_orchestrator.simulation_manager import Simulation


class _FakePopulation:
    """Minimal population double for permanent-burial integration tests."""

    def __init__(self, remove_permanently_buried):
        """Store the configured removal flag and captured exposure fields."""
        self.population_config = SimpleNamespace(remove_permanently_buried=remove_permanently_buried)
        self.exposure_fields = []

    def remove_permanently_buried_particles(self, max_exposure_depth):
        """Capture the exposure field passed by the simulation helper."""
        self.exposure_fields.append(np.asarray(max_exposure_depth))
        return 0


class _ExposurePlugin:
    """Format-plugin double returning deterministic maximum exposure inputs."""

    def __init__(self, max_erosion=None, max_bss=None):
        """Store exposure inputs and track calls."""
        self.max_erosion = np.asarray(max_erosion if max_erosion is not None else [0.5, 1.0])
        self.max_bss = np.asarray(max_bss if max_bss is not None else [5.0, 2.0])
        self.calls = 0

    def get_max_exposure_depth_fields(self):
        """Return maximum erosion and bed shear stress arrays."""
        self.calls += 1
        return self.max_erosion, self.max_bss


def _runtime_plan(population, critical_shear_stress=1.0, bertin_coefficient=0.08):
    """Build a minimal runtime plan with converter config and grain properties."""
    grain_properties = {}
    if critical_shear_stress is not None:
        grain_properties['critical_shear_stress'] = critical_shear_stress
    converter = SimpleNamespace(
        grain_properties=grain_properties,
        config=SimpleNamespace(bertin_coefficient=bertin_coefficient),
    )
    return SimpleNamespace(population=population, tracer=SimpleNamespace(converter=converter))


def _simulation_with_plugin(plugin):
    """Create an uninitialized Simulation with only helper dependencies set."""
    manager = object.__new__(Simulation)
    manager._profile_enabled = False
    manager.format_converter = SimpleNamespace(format_plugin=plugin)
    return manager


def _minimal_config():
    """Build a minimal valid configuration for controller-backed simulation tests."""
    return {
        'general': {'input_model': {'format': 'fm_netcdf', 'reference_date': '1970-01-01'}},
        'inputs': {'data': 'dummy.nc'},
        'particles': {
            'populations': [
                {
                    'name': 'sand',
                    'particle_type': 'sand',
                    'characteristics': {
                        'density': 2650.0,
                        'grain_size': 0.00025,
                    },
                    'tracer_methods': {
                        'vanwesten': {
                            'flow_field_name': ['bed_load_velocity'],
                        },
                    },
                    'seeding': {
                        'burial_depth': {'constant': 0.0},
                        'release_start': '2016-09-21 19:30:00',
                        'quantity': 1,
                        'strategy': {
                            'point': {
                                'locations': ['0.0,0.0'],
                            },
                        },
                    },
                }
            ]
        },
    }


def _simulation_with_config_file(tmp_path, config):
    """Create an uninitialized Simulation using the real configuration controller."""
    config_file = tmp_path / 'sedtrails.yml'
    config_file.write_text(yaml.dump(config), encoding='utf-8')
    manager = object.__new__(Simulation)
    manager._controller = ConfigurationController(str(config_file))
    return manager


class _Controller:
    """Minimal controller double backed by a key-value mapping."""

    def __init__(self, values=None):
        """Store values returned by ``get``."""
        self.values = values or {}

    def get(self, key, default=None):
        """Return the configured value or the provided default."""
        return self.values.get(key, default)


class _CheckpointWriter:
    """Writer double that captures checkpoint write calls."""

    def __init__(self):
        """Initialize the captured call list."""
        self.calls = []

    def write_checkpoint(self, *args, **kwargs):
        """Capture checkpoint arguments."""
        self.calls.append((args, kwargs))


class TestSimulationManagerParallelConfig:
    """Tests for local parallel configuration parsing."""

    def test_parallel_worker_count_uses_config_before_slurm(self, monkeypatch):
        """Explicit config should take precedence over scheduler environment."""
        monkeypatch.setenv('SLURM_CPUS_PER_TASK', '16')
        manager = object.__new__(Simulation)
        manager._controller = _Controller({'compute.n_workers': 4})

        assert manager._parallel_worker_count() == 4

    def test_parallel_worker_count_uses_slurm_fallback(self, monkeypatch):
        """Scheduler CPU allocation should be used when config is unset."""
        monkeypatch.setenv('SLURM_CPUS_PER_TASK', '8')
        manager = object.__new__(Simulation)
        manager._controller = _Controller()

        assert manager._parallel_worker_count() == 8

    @pytest.mark.parametrize(
        ('value', 'message'),
        [
            ('0', 'SLURM_CPUS_PER_TASK'),
            ('many', 'SLURM_CPUS_PER_TASK'),
        ],
    )
    def test_parallel_worker_count_rejects_invalid_slurm_values(self, monkeypatch, value, message):
        """Invalid scheduler worker counts should not silently fall back to serial."""
        monkeypatch.setenv('SLURM_CPUS_PER_TASK', value)
        manager = object.__new__(Simulation)
        manager._controller = _Controller()

        with pytest.raises(ConfigurationError, match=message):
            manager._parallel_worker_count()

    @pytest.mark.parametrize(
        ('task_id', 'n_tasks', 'message'),
        [
            ('0', '0', 'SEDTRAILS_N_TASKS'),
            ('2', '2', 'SEDTRAILS_TASK_ID'),
            ('left', '2', 'SEDTRAILS_TASK_ID'),
        ],
    )
    def test_parallel_task_context_rejects_invalid_environment(self, monkeypatch, task_id, n_tasks, message):
        """Worker task IDs should be validated before output paths are resolved."""
        monkeypatch.setenv('SEDTRAILS_TASK_ID', task_id)
        monkeypatch.setenv('SEDTRAILS_N_TASKS', n_tasks)

        with pytest.raises(ConfigurationError, match=message):
            Simulation._parallel_task_context_from_env()

    def test_run_parallel_clears_preload_when_pool_creation_fails(self, monkeypatch, tmp_path):
        """A failed worker pool should not keep the parent input dataset referenced."""
        import gc
        import multiprocessing

        class InputData:
            nbytes = 1024

            def load(self):
                return None

        class Plugin:
            def __init__(self):
                self.input_data = InputData()

            def load(self):
                return None

        class FailingForkContext:
            def Pool(self, processes):
                raise RuntimeError('pool creation failed')

        manager = object.__new__(Simulation)
        manager._controller = _Controller(
            {
                'compute.preload_input': True,
                'inputs.read_interval': '1H',
                'time.duration': '1H',
            }
        )
        manager._config_file = 'config.yml'
        manager.logger = SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None)
        plugin = Plugin()
        manager.format_converter = SimpleNamespace(format_plugin=plugin)

        monkeypatch.setattr(manager, '_get_output_dir', lambda: tmp_path)
        monkeypatch.setattr(multiprocessing, 'get_context', lambda method: FailingForkContext())
        monkeypatch.setattr(_pw, '_warn_if_oom_risk', lambda *args, **kwargs: None)
        monkeypatch.setattr(_pw, '_merge_outputs', lambda *args, **kwargs: pytest.fail('merge should not run'))
        monkeypatch.setattr(gc, 'collect', lambda: 0)

        with pytest.raises(RuntimeError, match='pool creation failed'):
            manager._run_parallel(2)

        assert _pw._PRELOADED_INPUT_DATA is None
        assert plugin.input_data is None

    def test_run_parallel_defaults_to_no_preload(self, monkeypatch, tmp_path):
        """Parallel runs should not load the full input dataset unless requested."""
        import gc
        import multiprocessing

        class Plugin:
            input_data = object()

            def load(self):
                pytest.fail('default parallel execution should not preload input')

        class FailingForkContext:
            def Pool(self, processes):
                raise RuntimeError('pool creation failed')

        manager = object.__new__(Simulation)
        manager._controller = _Controller(
            {
                'inputs.read_interval': '1H',
                'time.duration': '1H',
            }
        )
        manager._config_file = 'config.yml'
        manager.logger = SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None)
        manager.format_converter = SimpleNamespace(format_plugin=Plugin())

        monkeypatch.setattr(manager, '_get_output_dir', lambda: tmp_path)
        monkeypatch.setattr(multiprocessing, 'get_context', lambda method: FailingForkContext())
        monkeypatch.setattr(_pw, '_merge_outputs', lambda *args, **kwargs: pytest.fail('merge should not run'))
        monkeypatch.setattr(gc, 'collect', lambda: 0)

        with pytest.raises(RuntimeError, match='pool creation failed'):
            manager._run_parallel(2)

        assert _pw._PRELOADED_INPUT_DATA is None

    def test_create_dashboard_disables_parallel_dashboard(self, monkeypatch):
        """Parallel parent simulations should not create GUI dashboards."""
        import sedtrails.simulation_orchestrator.simulation_manager as manager_module

        manager = object.__new__(Simulation)
        manager._enable_dashboard_override = None
        manager._controller = _Controller(
            {
                'visualization.dashboard.enable': True,
                'compute.n_workers': 2,
            }
        )
        warning_calls = []
        manager.logger = SimpleNamespace(warning=lambda *args, **kwargs: warning_calls.append(args))

        monkeypatch.setattr(
            manager_module,
            'SimulationDashboard',
            lambda *args, **kwargs: pytest.fail('parallel runs should not create dashboards'),
        )

        assert manager._create_dashboard() is None
        assert warning_calls


class TestSimulationManagerPreflight:
    """Tests for configuration validation before seeding work starts."""

    def test_run_impl_rejects_passive_burial_depth_before_seeding(self):
        """Invalid passive burial depth should fail before time or seeding dependencies are accessed."""
        manager = object.__new__(Simulation)
        manager._config_is_read = True
        manager._controller = _Controller(
            {
                'particles.populations': [
                    {
                        'particle_type': 'passive',
                        'tracer_methods': {'passive_tracer': {}},
                        'seeding': {'burial_depth': {'constant': 0.0}},
                    }
                ]
            }
        )

        with pytest.raises(ConfigurationError, match='seeding.burial_depth'):
            manager._run_impl()

    @pytest.mark.parametrize('example_name', ['config.example_sfincs.yaml', 'sedtrails-example-passive.yaml'])
    def test_passive_examples_pass_runtime_preflight(self, example_name):
        """Committed passive examples must satisfy their runtime-only constraints."""
        example_file = Path(__file__).parents[2] / 'examples' / example_name
        config = YAMLConfigValidator().validate_yaml(str(example_file))

        validate_population_runtime_configurations(config['particles']['populations'])

    def test_plan_retrievers_pass_population_and_global_fraction_selection(self, monkeypatch):
        """Each plan build should receive its population config and shared input-model defaults."""
        calls = []

        def fake_build_plan_sedtrails_data(
            sedtrails_data,
            tracer_plan,
            population_config=None,
            default_fraction_index=0,
            default_fraction_name=None,
        ):
            calls.append(
                {
                    'sedtrails_data': sedtrails_data,
                    'tracer_plan': tracer_plan,
                    'population_config': population_config,
                    'default_fraction_index': default_fraction_index,
                    'default_fraction_name': default_fraction_name,
                }
            )
            return object()

        monkeypatch.setattr(simulation_manager, 'build_plan_sedtrails_data', fake_build_plan_sedtrails_data)
        manager = object.__new__(Simulation)
        manager._controller = _Controller(
            {
                'general.input_model': {
                    'sediment_fraction_index': 2,
                    'sediment_fraction_name': 'global_silt',
                }
            }
        )
        sedtrails_data = object()
        runtime_plans = (
            SimpleNamespace(
                population_index=3,
                population_config={'sediment_fraction_index': 1},
                tracer=object(),
            ),
            SimpleNamespace(
                population_index=7,
                population_config={'sediment_fraction_name': 'local_sand'},
                tracer=object(),
            ),
            SimpleNamespace(population_index=9, population_config={}, tracer=object()),
        )

        retrievers = manager._build_plan_retrievers(sedtrails_data, runtime_plans)

        assert set(retrievers) == {3, 7, 9}
        assert [call['population_config'] for call in calls] == [
            {'sediment_fraction_index': 1},
            {'sediment_fraction_name': 'local_sand'},
            {},
        ]
        assert all(call['sedtrails_data'] is sedtrails_data for call in calls)
        assert [call['default_fraction_index'] for call in calls] == [2, 2, 2]
        assert [call['default_fraction_name'] for call in calls] == ['global_silt'] * 3


class TestSimulationManagerTimeConfig:
    """Tests for simulation-time construction from configuration."""

    def test_simulation_time_uses_input_model_reference_date(self):
        """SFINCS-style date-only reference dates should align timer seconds with SedTRAILS times."""

        class Controller:
            values = {
                'time.start': '2024-05-01 00:00:00',
                'time.duration': '5M',
                'time.timestep': '60S',
                'inputs.read_interval': '30M',
                'general.input_model.reference_date': '2024-05-01',
            }

            def get(self, key, default=None):
                return self.values.get(key, default)

        manager = object.__new__(Simulation)
        manager._controller = Controller()

        simulation_time = manager._create_simulation_time()

        assert simulation_time.reference_date == '2024-05-01'
        assert simulation_time.start == 0

    @pytest.mark.parametrize('current_time', [0.0, 6011.0, 7200.0])
    def test_loaded_chunk_is_reused_through_final_interpolation_interval(self, current_time):
        """A loaded chunk remains valid until current time moves beyond its last timestamp."""

        class SedtrailsData:
            times = np.array([0.0, 1200.0, 2400.0, 3600.0, 4800.0, 6000.0, 7200.0])

        assert not Simulation._needs_sedtrails_reload(SedtrailsData(), current_time)

    @pytest.mark.parametrize('current_time', [-1.0, 7200.1])
    def test_loaded_chunk_reloads_outside_coverage(self, current_time):
        """Reload only when current time is outside the loaded chunk."""

        class SedtrailsData:
            times = np.array([0.0, 1200.0, 2400.0, 3600.0, 4800.0, 6000.0, 7200.0])

        assert Simulation._needs_sedtrails_reload(SedtrailsData(), current_time)

    def test_input_exhaustion_detects_time_after_loaded_data(self):
        """If reload still leaves current time beyond the loaded data, input is exhausted."""

        class SedtrailsData:
            times = np.array([4800.0, 6000.0, 7200.0])

        assert Simulation._is_after_loaded_sedtrails_data(SedtrailsData(), 7200.1)
        assert not Simulation._is_after_loaded_sedtrails_data(SedtrailsData(), 7200.0)

    def test_simulation_window_must_reach_first_input_time(self):
        """A simulation that ends before input forcing starts should fail clearly."""

        class SimulationTime:
            start = 0.0
            end = 3600.0
            reference_date = '2000-01-01'

        class SedtrailsData:
            times = np.array([4600.0, 8200.0])

        with pytest.raises(ConfigurationError, match='ends before the first available input field timestamp'):
            Simulation._validate_simulation_time_matches_input(SimulationTime(), SedtrailsData())

    def test_simulation_window_must_not_start_before_first_input_time(self):
        """A simulation may not spend CFL steps before forcing data exists."""

        class SimulationTime:
            start = 0.0
            end = 7200.0
            reference_date = '2000-01-01'

        class SedtrailsData:
            times = np.array([4600.0, 8200.0])

        with pytest.raises(ConfigurationError, match='starts before the first available input field timestamp'):
            Simulation._validate_simulation_time_matches_input(SimulationTime(), SedtrailsData())

    def test_simulation_window_may_end_at_first_input_time(self):
        """The first forcing timestamp is a valid simulation endpoint."""

        class SimulationTime:
            start = 4600.0
            end = 4600.0
            reference_date = '2000-01-01'

        class SedtrailsData:
            times = np.array([4600.0, 8200.0])

        Simulation._validate_simulation_time_matches_input(SimulationTime(), SedtrailsData())

    def test_exhausted_input_suppresses_further_reload_attempts(self):
        """After input exhaustion, later timesteps should reuse the last loaded fields."""

        class SedtrailsData:
            times = np.array([4800.0, 6000.0, 7200.0])

        assert Simulation._should_attempt_sedtrails_reload(SedtrailsData(), 7200.1, input_data_exhausted=False)
        assert not Simulation._should_attempt_sedtrails_reload(SedtrailsData(), 7200.1, input_data_exhausted=True)

    def test_simulation_start_must_be_within_forcing_window(self):
        """A run must start on forcing data; repeat mapping is only allowed after that."""

        class SimulationTime:
            start = 0.0
            reference_date = '2000-01-01'
            _start = '2000-01-01 00:00:00'

        with pytest.raises(ConfigurationError, match='time.start must be within the input forcing window'):
            Simulation._validate_simulation_start_matches_input(SimulationTime(), (3600.0, 7200.0))

    def test_simulation_start_at_forcing_edges_is_valid(self):
        """The first and last forcing timestamps are valid simulation starts."""

        class SimulationTime:
            reference_date = '2000-01-01'
            _start = '2000-01-01 01:00:00'

            def __init__(self, start):
                self.start = start

        Simulation._validate_simulation_start_matches_input(SimulationTime(3600.0), (3600.0, 7200.0))
        Simulation._validate_simulation_start_matches_input(SimulationTime(7200.0), (3600.0, 7200.0))

    def test_eulerian_time_is_unchanged_when_repeat_disabled(self):
        """Without looping, field lookups use the simulation clock."""

        mapped_time = Simulation._map_eulerian_field_time(
            current_time_seconds=25.0,
            repeat_eulerian_fields=False,
            input_time_bounds=(0.0, 10.0),
        )

        assert mapped_time == 25.0

    def test_eulerian_time_before_input_start_is_not_mapped_forward(self):
        """Before-start lookups remain invalid so validation/reload failures stay clear."""

        mapped_time = Simulation._map_eulerian_field_time(
            current_time_seconds=50.0,
            repeat_eulerian_fields=True,
            input_time_bounds=(100.0, 200.0),
        )

        assert mapped_time == 50.0

    @pytest.mark.parametrize(
        'current_time,expected_time',
        [
            (5.0, 5.0),
            (10.0, 10.0),
            (12.5, 2.5),
            (25.0, 5.0),
        ],
    )
    def test_eulerian_time_repeats_from_input_start(self, current_time, expected_time):
        """When looping is enabled, times after forcing end wrap to the first input timestamp."""

        mapped_time = Simulation._map_eulerian_field_time(
            current_time_seconds=current_time,
            repeat_eulerian_fields=True,
            input_time_bounds=(0.0, 10.0),
        )

        assert mapped_time == expected_time

    def test_eulerian_time_repeats_with_nonzero_input_start(self):
        """Looping preserves input-series offsets when the forcing does not start at zero seconds."""

        mapped_time = Simulation._map_eulerian_field_time(
            current_time_seconds=125.0,
            repeat_eulerian_fields=True,
            input_time_bounds=(100.0, 110.0),
        )

        assert mapped_time == 105.0

    def test_eulerian_time_repeat_uses_morfac_decompressed_bounds(self):
        """Looping should use the already-decompressed forcing span reported by the converter."""

        mapped_time = Simulation._map_eulerian_field_time(
            current_time_seconds=250.0,
            repeat_eulerian_fields=True,
            input_time_bounds=(100.0, 220.0),
        )

        assert mapped_time == 130.0

    def test_output_save_interval_uses_outputs_config(self):
        """Trajectory output cadence should be read from outputs.save_interval."""

        class Controller:
            def get(self, key, default=None):
                if key == 'outputs.save_interval':
                    return '30M'
                return default

        manager = object.__new__(Simulation)
        manager._controller = Controller()

        assert manager._output_save_interval_seconds() == 1800

    def test_output_save_interval_defaults_to_one_hour(self):
        """When unset, saved trajectory samples should default to one-hour spacing."""
        manager = object.__new__(Simulation)
        manager._controller = type('Controller', (), {'get': lambda self, key, default=None: default})()

        assert manager._output_save_interval_seconds() == 3600

    def test_output_save_interval_rejects_zero_duration(self):
        """A zero save interval would make output scheduling ambiguous."""
        manager = object.__new__(Simulation)
        manager._controller = type('Controller', (), {'get': lambda self, key, default=None: '0S'})()

        with pytest.raises(ConfigurationError, match='outputs.save_interval'):
            manager._output_save_interval_seconds()

    def test_output_sync_interval_defaults_to_save_interval(self):
        """NetCDF flush cadence should default to the trajectory save cadence."""
        manager = object.__new__(Simulation)
        manager._controller = type('Controller', (), {'get': lambda self, key, default=None: default})()

        assert manager._output_sync_interval_seconds(save_interval_seconds=1800) == 1800

    def test_output_sync_interval_uses_outputs_config(self):
        """Configured sync interval should be converted to seconds."""

        class Controller:
            def get(self, key, default=None):
                if key == 'outputs.sync_interval':
                    return '2H'
                return default

        manager = object.__new__(Simulation)
        manager._controller = Controller()

        assert manager._output_sync_interval_seconds(save_interval_seconds=1800) == 7200
        assert Simulation._sync_every_n_writes(save_interval_seconds=1800, sync_interval_seconds=7200) == 4

    def test_output_sync_interval_rejects_zero_duration(self):
        """A zero sync interval would make streaming flush cadence ambiguous."""
        manager = object.__new__(Simulation)
        manager._controller = type('Controller', (), {'get': lambda self, key, default=None: '0S'})()

        with pytest.raises(ConfigurationError, match='outputs.sync_interval'):
            manager._output_sync_interval_seconds(save_interval_seconds=1800)

    @pytest.mark.parametrize(
        'duration,save_interval,expected_count',
        [
            ('3H', 3600, 4),
            ('30M', 3600, 2),
            ('2H30M', 3600, 4),
        ],
    )
    def test_estimate_output_timesteps_counts_initial_scheduled_and_final(
        self, duration, save_interval, expected_count
    ):
        """Output allocation follows save cadence, not CFL integration cadence."""
        simulation_time = Time(
            _start='2000-01-01 00:00:00',
            duration=Duration(duration),
            reference_date='2000-01-01',
        )

        assert Simulation._estimate_output_timesteps(simulation_time, save_interval) == expected_count

    def test_next_scheduled_output_time_caps_at_simulation_end(self):
        """The final output target should be the simulation end, not a time after it."""

        class SimulationTime:
            start = 0.0
            end = 9000.0

        assert Simulation._next_scheduled_output_time(SimulationTime(), 3600, 1) == 3600.0
        assert Simulation._next_scheduled_output_time(SimulationTime(), 3600, 3) == 9000.0

    def test_limit_timestep_to_output_schedule_hits_save_boundary(self):
        """A CFL step that crosses an output boundary should land exactly on it."""
        limited = Simulation._limit_timestep_to_output_schedule(
            current_time=3598.0,
            current_timestep=10.0,
            next_output_time=3600.0,
        )

        assert limited == 2.0

    def test_limit_timestep_to_output_schedule_keeps_short_cfl_step(self):
        """CFL steps shorter than the remaining save interval should not be changed."""
        limited = Simulation._limit_timestep_to_output_schedule(
            current_time=10.0,
            current_timestep=2.0,
            next_output_time=3600.0,
        )

        assert limited == 2.0

    @pytest.mark.parametrize(
        'sample_time,next_output_time,end_time,expected',
        [
            (3599.0, 3600.0, 7200.0, False),
            (3600.0, 3600.0, 7200.0, True),
            (7200.0, 10800.0, 7200.0, True),
        ],
    )
    def test_output_sample_due_on_interval_or_final_time(self, sample_time, next_output_time, end_time, expected):
        """Samples should be saved only on configured boundaries or at final time."""
        assert Simulation._is_output_sample_due(sample_time, next_output_time, end_time) is expected

    @pytest.mark.parametrize(
        'method_name,transport_probability_method,expected',
        [
            ('vanwesten', 'stochastic_transport', True),
            ('vanwesten', 'reduced_velocity', True),
            ('vanwesten', 'no_probability', True),
            ('soulsby', 'no_probability', True),
            ('passive_tracer', 'no_probability', True),
            ('soulsby', 'reduced_velocity', False),
            ('passive_tracer', 'stochastic_transport', False),
        ],
    )
    def test_should_update_bed_level_after_movement_policy(
        self,
        method_name,
        transport_probability_method,
        expected,
    ):
        """Post-move bed-level updates should follow tracer and transport policy rules."""
        tracer_plan = SimpleNamespace(
            method_name=method_name,
            transport_probability_method=transport_probability_method,
        )

        assert Simulation._should_update_bed_level_after_movement(tracer_plan) is expected

    def test_initialize_population_output_status_supplies_required_fields(self):
        """The seeded initial sample should have status fields before the first physics update."""

        class Population:
            particles = {
                'x': np.array([1.0, 2.0]),
                'y': np.array([3.0, 4.0]),
                'release_time': np.array([0.0, 10.0]),
                'burial_depth': np.array([0.0, 0.0]),
            }
            _particle_simplices = np.array([5, -1])

        population = Population()

        Simulation._initialize_population_output_status([population], current_time=5.0)

        expected_keys = {
            'status_alive',
            'status_buried',
            'status_domain',
            'status_transported',
            'status_released',
            'status_mobile',
        }
        assert expected_keys.issubset(population.particles)
        np.testing.assert_array_equal(population.particles['status_domain'], np.array([True, False]))
        np.testing.assert_array_equal(population.particles['status_released'], np.array([True, False]))
        np.testing.assert_array_equal(population.particles['status_transported'], np.array([False, False]))
        np.testing.assert_array_equal(population.particles['status_mobile'], np.array([False, False]))


class TestSimulationManagerNetCDFOutputOptions:
    """Tests for NetCDF output and checkpoint option plumbing."""

    def test_output_netcdf_options_use_large_track_defaults(self):
        """Default NetCDF writer options should match the simulation policy."""
        manager = object.__new__(Simulation)
        manager._controller = _Controller()

        assert manager._output_netcdf_options() == {
            'coordinate_dtype': 'float32',
            'status_dtype': 'uint8',
            'compression': 'auto',
            'compression_auto_threshold_mb': 1024,
            'compression_level': 1,
            'shuffle': True,
            'time_chunk': 1,
            'particle_chunk': 65_536,
            'sync_interval': 10,
            'reopen_interval': None,
        }

    def test_output_netcdf_options_preserve_configured_values(self):
        """Configured NetCDF values should be forwarded without losing zero/null settings."""
        netcdf_config = {
            'coordinate_dtype': 'float64',
            'status_dtype': 'int32',
            'compression': False,
            'compression_auto_threshold_mb': 256,
            'compression_level': 0,
            'shuffle': False,
            'time_chunk': 8,
            'particle_chunk': 128,
            'sync_interval': 0,
            'reopen_interval': 25,
        }
        manager = object.__new__(Simulation)
        manager._controller = _Controller({'outputs.netcdf': netcdf_config})

        assert manager._output_netcdf_options() == netcdf_config

    def test_output_netcdf_options_use_legacy_sync_after_defaults(self, tmp_path):
        """A defaulted nested sync interval should not mask legacy duration config."""
        config = _minimal_config()
        config['outputs'] = {
            'save_interval': '30M',
            'sync_interval': '2H',
            'store_tracks': True,
            'netcdf': {},
        }
        manager = _simulation_with_config_file(tmp_path, config)

        assert manager._output_netcdf_options()['sync_interval'] == 4

    def test_output_netcdf_options_nested_sync_overrides_legacy_sync(self, tmp_path):
        """Explicit nested NetCDF sync configuration should take precedence."""
        config = _minimal_config()
        config['outputs'] = {
            'save_interval': '30M',
            'sync_interval': '2H',
            'store_tracks': True,
            'netcdf': {'sync_interval': 3},
        }
        manager = _simulation_with_config_file(tmp_path, config)

        assert manager._output_netcdf_options()['sync_interval'] == 3

    def test_estimate_netcdf_payload_bytes_uses_particle_slots_and_dtypes(self):
        """Payload estimates should track the particle-slot fields written by NetCDFWriter."""
        estimated = Simulation._estimate_netcdf_payload_bytes(
            n_particles=100,
            n_output_slots=10,
            coordinate_dtype='float32',
            status_dtype='uint8',
        )

        assert estimated == 100 * 10 * ((5 * 4) + (Simulation._OUTPUT_STATUS_FIELD_COUNT * 1))

    @pytest.mark.parametrize(
        ('n_particles', 'n_slots', 'threshold_mb', 'expected'),
        [
            (1_000, 2, 1, False),
            (1_000_000, 100, 1, True),
            (1_000_000, 1, 1024, False),
        ],
    )
    def test_resolve_output_netcdf_options_auto_compression(
        self, n_particles, n_slots, threshold_mb, expected
    ):
        """Auto compression should resolve from estimated output payload size."""
        raw_options = {
            'coordinate_dtype': 'float32',
            'status_dtype': 'uint8',
            'compression': 'auto',
            'compression_auto_threshold_mb': threshold_mb,
            'compression_level': 1,
            'shuffle': True,
            'time_chunk': 1,
            'particle_chunk': 65_536,
            'sync_interval': 10,
            'reopen_interval': None,
        }

        resolved = Simulation._resolve_output_netcdf_options(raw_options, n_particles, n_slots)

        assert resolved['compression'] is expected
        assert 'compression_auto_threshold_mb' not in resolved

    @pytest.mark.parametrize('compression', [True, False])
    def test_resolve_output_netcdf_options_preserves_explicit_compression(self, compression):
        """Explicit compression booleans should bypass auto-size decisions."""
        raw_options = {
            'coordinate_dtype': 'float32',
            'status_dtype': 'uint8',
            'compression': compression,
            'compression_auto_threshold_mb': 0,
            'compression_level': 1,
            'shuffle': True,
            'time_chunk': 1,
            'particle_chunk': 65_536,
            'sync_interval': 10,
            'reopen_interval': None,
        }

        resolved = Simulation._resolve_output_netcdf_options(raw_options, 0, 1)

        assert resolved['compression'] is compression
        assert 'compression_auto_threshold_mb' not in resolved

    def test_resolved_output_writer_options_sizes_tracks_and_snapshots_separately(self):
        """Full-track auto compression should not force snapshot compression."""
        manager = object.__new__(Simulation)
        manager._controller = _Controller()

        netcdf_options, snapshot_options, checkpoint_options = manager._resolved_output_writer_options(
            total_particles=1_000_000,
            n_output_slots=100,
            store_tracks=True,
        )

        assert netcdf_options['compression'] is True
        assert snapshot_options['compression'] is False
        assert checkpoint_options['writer_kwargs']['compression'] is False
        assert 'compression_auto_threshold_mb' not in netcdf_options
        assert 'compression_auto_threshold_mb' not in snapshot_options
        assert 'compression_auto_threshold_mb' not in checkpoint_options['writer_kwargs']

    def test_resolved_output_writer_options_sizes_end_positions_as_one_snapshot(self):
        """End-position output should resolve auto compression using one saved state."""
        manager = object.__new__(Simulation)
        manager._controller = _Controller()

        netcdf_options, snapshot_options, checkpoint_options = manager._resolved_output_writer_options(
            total_particles=1_000_000,
            n_output_slots=0,
            store_tracks=False,
        )

        assert netcdf_options['compression'] is False
        assert snapshot_options['compression'] is False
        assert checkpoint_options['writer_kwargs']['compression'] is False

    def test_output_checkpoint_options_use_defaults(self):
        """Default checkpoint policy should reuse checkpoint-safe writer defaults."""
        manager = object.__new__(Simulation)
        manager._controller = _Controller()

        assert manager._output_checkpoint_options() == {
            'enabled': True,
            'interval': 0,
            'writer_kwargs': {
                'coordinate_dtype': 'float32',
                'status_dtype': 'uint8',
                'compression': False,
                'compression_level': 1,
                'shuffle': True,
                'particle_chunk': 65_536,
            },
        }

    def test_output_checkpoint_options_forward_shared_writer_kwargs(self):
        """Checkpoint options should include policy fields and checkpoint-safe writer kwargs."""
        manager = object.__new__(Simulation)
        manager._controller = _Controller(
            {
                'outputs.netcdf': {
                    'coordinate_dtype': 'float64',
                    'status_dtype': 'int32',
                    'compression': False,
                    'compression_auto_threshold_mb': 256,
                    'compression_level': 0,
                    'shuffle': False,
                    'time_chunk': 4,
                    'particle_chunk': 512,
                    'sync_interval': 0,
                    'reopen_interval': 10,
                    'checkpoint': False,
                    'checkpoint_interval': 3,
                }
            }
        )

        checkpoint_options = manager._output_checkpoint_options()

        assert checkpoint_options == {
            'enabled': False,
            'interval': 3,
            'writer_kwargs': {
                'coordinate_dtype': 'float64',
                'status_dtype': 'int32',
                'compression': False,
                'compression_level': 0,
                'shuffle': False,
                'particle_chunk': 512,
            },
        }

    @pytest.mark.parametrize(
        ('checkpoint_options', 'saved_slots', 'final', 'expected_calls'),
        [
            ({'enabled': False, 'interval': 1, 'writer_kwargs': {}}, 1, True, 0),
            ({'enabled': True, 'interval': 0, 'writer_kwargs': {}}, 3, False, 0),
            ({'enabled': True, 'interval': 3, 'writer_kwargs': {}}, 2, False, 0),
            ({'enabled': True, 'interval': 3, 'writer_kwargs': {}}, 6, False, 1),
            ({'enabled': True, 'interval': 0, 'writer_kwargs': {}}, 7, True, 1),
        ],
    )
    def test_maybe_write_checkpoint_obeys_policy(
        self, checkpoint_options, saved_slots, final, expected_calls
    ):
        """Checkpoint writes should follow enabled, interval, and final-save policy."""
        writer = _CheckpointWriter()
        manager = object.__new__(Simulation)
        manager.data_manager = SimpleNamespace(writer=writer)
        simulation_time = SimpleNamespace(reference_date='2000-01-01')

        manager._maybe_write_checkpoint(
            populations=['population'],
            current_time=123.0,
            simulation_time=simulation_time,
            saved_slots=saved_slots,
            checkpoint_options=checkpoint_options,
            final=final,
        )

        assert len(writer.calls) == expected_calls
        if expected_calls:
            args, kwargs = writer.calls[0]
            assert args == ('sedtrails_checkpoint.nc', ['population'], 123.0)
            assert kwargs['reference_date'] == '2000-01-01'
            assert kwargs['time_units'] == 'seconds since 2000-01-01'

    def test_maybe_write_checkpoint_forwards_writer_kwargs(self):
        """Checkpoint writer kwargs should be passed through to the NetCDF writer."""
        writer = _CheckpointWriter()
        manager = object.__new__(Simulation)
        manager.data_manager = SimpleNamespace(writer=writer)

        manager._maybe_write_checkpoint(
            populations=[],
            current_time=0,
            simulation_time=SimpleNamespace(reference_date='1970-01-01'),
            saved_slots=1,
            checkpoint_options={
                'enabled': True,
                'interval': 1,
                'writer_kwargs': {
                    'coordinate_dtype': 'float64',
                    'status_dtype': 'int32',
                    'compression': False,
                    'compression_level': 0,
                    'shuffle': False,
                    'particle_chunk': 256,
                },
            },
        )

        _, kwargs = writer.calls[0]
        assert kwargs['coordinate_dtype'] == 'float64'
        assert kwargs['status_dtype'] == 'int32'
        assert kwargs['compression'] is False
        assert kwargs['compression_level'] == 0
        assert kwargs['shuffle'] is False
        assert kwargs['particle_chunk'] == 256

    @pytest.mark.parametrize(
        ('controller_values', 'expected'),
        [
            ({}, True),
            ({'outputs.store_tracks': True}, True),
            ({'outputs.store_tracks': False}, False),
            ({'outputs.store_end_positions': True}, False),
            ({'outputs.store_tracks': True, 'outputs.store_end_positions': True}, False),
        ],
    )
    def test_output_store_tracks_selects_trajectory_or_end_position_mode(
        self, controller_values, expected
    ):
        """End-position output should disable full trajectory streaming."""
        manager = object.__new__(Simulation)
        manager._controller = _Controller(controller_values)

        assert manager._output_store_tracks() is expected


class TestSimulationPermanentBurialIntegration:
    """Tests for simulation-level permanent-burial removal wiring."""

    def test_no_flagged_populations_skip_exposure_scan(self):
        """The expensive exposure scan should only run when removal is enabled."""
        population = _FakePopulation(remove_permanently_buried=False)
        plugin = _ExposurePlugin()
        manager = _simulation_with_plugin(plugin)

        manager._remove_permanently_buried_populations([population], [_runtime_plan(population)])

        assert plugin.calls == 0
        assert population.exposure_fields == []

    def test_flagged_population_receives_erosion_plus_custom_bertin_mixing(self):
        """Flagged populations should receive max erosion plus max mixing depth."""
        flagged = _FakePopulation(remove_permanently_buried=True)
        unflagged = _FakePopulation(remove_permanently_buried=False)
        plugin = _ExposurePlugin(max_erosion=[0.5, 1.0], max_bss=[5.0, 2.0])
        manager = _simulation_with_plugin(plugin)
        runtime_plans = [
            _runtime_plan(flagged, critical_shear_stress=1.0, bertin_coefficient=0.08),
            _runtime_plan(unflagged, critical_shear_stress=1.0, bertin_coefficient=0.08),
        ]

        manager._remove_permanently_buried_populations([flagged, unflagged], runtime_plans)

        assert plugin.calls == 1
        assert len(flagged.exposure_fields) == 1
        np.testing.assert_allclose(flagged.exposure_fields[0], [0.66, 1.08])
        assert unflagged.exposure_fields == []

    def test_missing_exposure_plugin_support_raises(self):
        """Enabled removal should require a plugin exposure-field API."""
        population = _FakePopulation(remove_permanently_buried=True)
        manager = _simulation_with_plugin(object())

        with pytest.raises(NotImplementedError, match='get_max_exposure_depth_fields'):
            manager._remove_permanently_buried_populations([population], [_runtime_plan(population)])

    def test_missing_critical_shear_stress_raises(self):
        """Exposure conversion requires critical shear stress from the converter."""
        population = _FakePopulation(remove_permanently_buried=True)
        manager = _simulation_with_plugin(_ExposurePlugin())

        with pytest.raises(ValueError, match='critical_shear_stress'):
            manager._remove_permanently_buried_populations(
                [population],
                [_runtime_plan(population, critical_shear_stress=None)],
            )

class TestSimulationManagerSaveInterval:
    """Tests for save_interval slot-count calculation and boundary logic."""

    @pytest.mark.parametrize(
        'duration_s,interval_s,expected_slots',
        [
            (3600, 3600, 2),        # 1H run, 1H interval  ? slot 0 + 1 boundary
            (4 * 3600, 3600, 5),    # 4H run, 1H interval  ? slot 0 + 4 boundaries
            (86400, 3600, 25),      # 1D run, 1H interval  ? 24 + 1
            (3600, 900, 5),         # 1H run, 15min interval ? 4 + 1
            (3601, 3600, 3),        # just over one interval ? 2 + 1
        ],
    )
    def test_n_output_slots_calculation(self, duration_s, interval_s, expected_slots):
        """Pre-allocated slot count is ceil(duration/interval) + 1 for the initial state."""
        import math
        n_slots = math.ceil(duration_s / interval_s) + 1
        assert n_slots == expected_slots

    def test_save_boundary_triggers_at_next_save_time(self):
        """Record fires exactly when simulation time reaches the boundary, not before."""
        next_save_time = 3600.0
        saved = []

        for t in [0.0, 1200.0, 2400.0, 3600.0, 4800.0]:
            if t >= next_save_time:
                saved.append(t)
                next_save_time += 3600.0

        assert saved == [3600.0]

    def test_final_save_fires_when_sim_ends_between_boundaries(self):
        """A trailing save after the loop captures the final particle state."""
        last_saved_time = 3600.0
        final_time = 4400.0       # simulation ended mid-interval
        slot_idx = 2
        n_output_slots = 5

        should_save = (final_time > last_saved_time) and (slot_idx < n_output_slots)
        assert should_save

    def test_final_save_skipped_when_already_at_boundary(self):
        """No duplicate save when the loop ended exactly on a save boundary."""
        last_saved_time = 7200.0
        final_time = 7200.0
        slot_idx = 3
        n_output_slots = 5

        should_save = (final_time > last_saved_time) and (slot_idx < n_output_slots)
        assert not should_save

    def test_final_save_skipped_when_slots_full(self):
        """No out-of-bounds write when all pre-allocated slots are consumed."""
        last_saved_time = 3600.0
        final_time = 4400.0
        slot_idx = 5
        n_output_slots = 5   # already at capacity

        should_save = (final_time > last_saved_time) and (slot_idx < n_output_slots)
        assert not should_save


class TestSimulationDashboardThrottle:
    """Tests dashboard update cadence and particle payload shape for visualization."""

    def test_dashboard_update_interval_uses_visualization_config(self):
        """Configured dashboard interval should override the default update cadence."""
        class Controller:
            def get(self, key, default=None):
                if key == 'visualization.dashboard.update_interval':
                    return '30S'
                return default

        manager = object.__new__(Simulation)
        manager._controller = Controller()

        assert manager._dashboard_update_interval_seconds() == 30

    def test_dashboard_update_interval_defaults_to_one_hour(self):
        """When unset, dashboard updates should default to a one-hour interval."""
        manager = object.__new__(Simulation)
        manager._controller = type('Controller', (), {'get': lambda self, key, default=None: default})()

        assert manager._dashboard_update_interval_seconds() == 3600

    def test_dashboard_particle_data_uses_arrays_not_single_element_lists(self):
        """Dashboard particle payload should keep vector fields as NumPy arrays."""
        class Population:
            particles = {
                'x': np.array([1.0, 2.0]),
                'y': np.array([3.0, 4.0]),
                'burial_depth': np.array([0.1, 0.2]),
            }

        particle_data = Simulation._dashboard_particle_data(Population())

        assert isinstance(particle_data['burial_depth'], np.ndarray)
        assert isinstance(particle_data['mixing_depth'], np.ndarray)
        assert not isinstance(particle_data['burial_depth'], list)
        assert not isinstance(particle_data['mixing_depth'], list)
        np.testing.assert_array_equal(particle_data['burial_depth'], np.array([0.1, 0.2]))
        np.testing.assert_array_equal(particle_data['mixing_depth'], np.array([np.nan, np.nan]))

    def test_dashboard_particle_data_includes_boundary_statuses(self):
        """Dashboard particle payload should include boundary status arrays for plotting."""
        class Population:
            particles = {
                'x': np.array([1.0, 2.0]),
                'y': np.array([3.0, 4.0]),
                'status_left_domain': np.array([False, True]),
                'status_beached': np.array([True, False]),
            }

        particle_data = Simulation._dashboard_particle_data(Population())

        np.testing.assert_array_equal(particle_data['status_left_domain'], np.array([False, True]))
        np.testing.assert_array_equal(particle_data['status_beached'], np.array([True, False]))

    def test_large_grid_dashboard_updates_are_throttled(self):
        """Large grids should throttle dashboard refreshes to periodic steps."""
        manager = object.__new__(Simulation)
        manager.dashboard = object()
        manager._dashboard_throttle_logged = False
        manager.logger = type('Logger', (), {'info': lambda self, *args, **kwargs: None})()
        manager._controller = type('Controller', (), {'get': lambda self, key, default=None: default})()

        sedtrails_data = type('SedtrailsData', (), {'x': np.arange(100_001)})()
        timer = type('Timer', (), {'step_count': 0})()

        assert manager._should_update_dashboard(sedtrails_data, timer)

        timer.step_count = 1
        assert not manager._should_update_dashboard(sedtrails_data, timer)

        timer.step_count = 10
        assert manager._should_update_dashboard(sedtrails_data, timer)

    def test_small_grid_dashboard_updates_every_step(self):
        """Small grids should keep per-step dashboard updates enabled."""
        manager = object.__new__(Simulation)
        manager.dashboard = object()

        sedtrails_data = type('SedtrailsData', (), {'x': np.arange(100)})()
        timer = type('Timer', (), {'step_count': 1})()

        assert manager._should_update_dashboard(sedtrails_data, timer)


class TestSimulationDomainExitReporting:
    """Tests CLI/log reporting for particles that leave the domain."""

    def test_reports_newly_left_domain_particles(self):
        """Report only particles that newly transition to left-domain status."""
        manager = object.__new__(Simulation)
        manager._report_domain_exits = True
        manager._report_domain_exit_updates = True
        manager.logger = _ListLogger()
        population = SimpleNamespace(
            population_config=SimpleNamespace(population_config={'name': 'sand'}),
            particles={
                'x': np.zeros(3),
                'status_left_domain': np.array([False, True, True]),
            },
        )
        reported_left_domain = np.array([False, False, True])

        newly_left = manager._report_new_domain_exits(
            population,
            population_index=0,
            flow_field_name='bed_load_velocity',
            reported_left_domain=reported_left_domain,
            current_time=10.0,
            current_timestep=2.0,
        )

        assert newly_left == 1
        np.testing.assert_array_equal(reported_left_domain, np.array([False, True, True]))
        assert 'Particles left domain: +1 in sand via bed_load_velocity' in manager.logger.messages[0]
        assert 'population total=2/3' in manager.logger.messages[0]

    def test_new_left_domain_updates_can_be_quiet(self):
        """Intermediate left-domain reporting can be disabled while masks still update."""
        manager = object.__new__(Simulation)
        manager._report_domain_exits = True
        manager._report_domain_exit_updates = False
        manager.logger = _ListLogger()
        population = SimpleNamespace(
            population_config=SimpleNamespace(population_config={'name': 'sand'}),
            particles={
                'x': np.zeros(3),
                'status_left_domain': np.array([False, True, True]),
            },
        )
        reported_left_domain = np.array([False, False, True])

        newly_left = manager._report_new_domain_exits(
            population,
            population_index=0,
            flow_field_name='bed_load_velocity',
            reported_left_domain=reported_left_domain,
            current_time=10.0,
            current_timestep=2.0,
        )

        assert newly_left == 1
        np.testing.assert_array_equal(reported_left_domain, np.array([False, True, True]))
        assert manager.logger.messages == []

    def test_final_summary_reports_total_left_domain_particles(self):
        """Final summary should report aggregate and per-population left-domain totals."""
        manager = object.__new__(Simulation)
        manager._report_domain_exits = True
        manager.logger = _ListLogger()
        populations = [
            SimpleNamespace(
                population_config={'name': 'fine'},
                particles={'x': np.zeros(2), 'status_left_domain': np.array([True, False])},
            ),
            SimpleNamespace(
                population_config={'name': 'medium'},
                particles={'x': np.zeros(3), 'status_left_domain': np.array([False, True, True])},
            ),
        ]

        manager._report_domain_exit_summary(populations)

        assert manager.logger.messages == ['Particles left domain during run: 3/5 (fine=1/2, medium=2/3)']

    def test_reports_newly_beached_particles(self):
        """Report only particles that newly transition to beached status."""
        manager = object.__new__(Simulation)
        manager._report_domain_exits = True
        manager._report_domain_exit_updates = True
        manager.logger = _ListLogger()
        population = SimpleNamespace(
            population_config=SimpleNamespace(population_config={'name': 'sand'}),
            particles={
                'x': np.zeros(3),
                'status_beached': np.array([False, True, True]),
            },
        )
        reported_beached = np.array([False, False, True])

        newly_beached = manager._report_new_beached_particles(
            population,
            population_index=0,
            flow_field_name='bed_load_velocity',
            reported_beached=reported_beached,
            current_time=10.0,
            current_timestep=2.0,
        )

        assert newly_beached == 1
        np.testing.assert_array_equal(reported_beached, np.array([False, True, True]))
        assert 'Particles beached on land: +1 in sand via bed_load_velocity' in manager.logger.messages[0]
        assert 'population total=2/3' in manager.logger.messages[0]

    def test_new_beached_updates_can_be_quiet(self):
        """Intermediate beaching reporting can be disabled while history still updates."""
        manager = object.__new__(Simulation)
        manager._report_domain_exits = True
        manager._report_domain_exit_updates = False
        manager.logger = _ListLogger()
        population = SimpleNamespace(
            population_config=SimpleNamespace(population_config={'name': 'sand'}),
            particles={
                'x': np.zeros(3),
                'status_beached': np.array([False, True, True]),
            },
        )
        reported_beached = np.array([False, False, True])

        newly_beached = manager._report_new_beached_particles(
            population,
            population_index=0,
            flow_field_name='bed_load_velocity',
            reported_beached=reported_beached,
            current_time=10.0,
            current_timestep=2.0,
        )

        assert newly_beached == 1
        np.testing.assert_array_equal(reported_beached, np.array([False, True, True]))
        assert manager.logger.messages == []

    def test_final_summary_reports_total_beached_particles(self):
        """Final summary should report aggregate and per-population beached totals."""
        manager = object.__new__(Simulation)
        manager._report_domain_exits = True
        manager.logger = _ListLogger()
        populations = [
            SimpleNamespace(
                population_config={'name': 'fine'},
                particles={'x': np.zeros(2), 'status_beached': np.array([True, False])},
            ),
            SimpleNamespace(
                population_config={'name': 'medium'},
                particles={'x': np.zeros(3), 'status_beached': np.array([False, True, True])},
            ),
        ]

        manager._report_beached_summary(populations)

        assert manager.logger.messages == ['Particles beached on land during run: 3/5 (fine=1/2, medium=2/3)']

    def test_final_summary_can_use_beached_history(self):
        """Final beached summary should support particles that remobilized later."""
        manager = object.__new__(Simulation)
        manager._report_domain_exits = True
        manager.logger = _ListLogger()
        populations = [
            SimpleNamespace(
                population_config={'name': 'fine'},
                particles={'x': np.zeros(2), 'status_beached': np.array([False, False])},
            ),
        ]
        beached_history = [np.array([True, False])]

        manager._report_beached_summary(populations, beached_history)

        assert manager.logger.messages == ['Particles beached on land during run: 1/2 (fine=1/2)']


class _ListLogger:
    """Minimal logger that stores formatted info messages for assertions."""

    def __init__(self):
        self.messages = []

    def info(self, message, *args):
        self.messages.append(message % args if args else message)
