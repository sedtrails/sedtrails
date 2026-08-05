"""Population schema regression tests."""

import yaml
import pytest

from sedtrails.application_interfaces.validator import YAMLConfigValidator
from sedtrails.exceptions import YamlValidationError


def test_population_schema_rejects_multiple_tracer_methods(tmp_path):
    """Reject populations that configure more than one tracer method."""
    config = _base_config()
    config['particles']['populations'][0]['tracer_methods'] = {
        'vanwesten': {'flow_field_name': ['bed_load_velocity']},
        'soulsby': {'flow_field_name': ['grain_velocity']},
    }

    with pytest.raises(YamlValidationError, match='YAML config validation error'):
        _validate_config(tmp_path, config)


def test_population_schema_requires_flow_field_name(tmp_path):
    """Require flow_field_name for the selected tracer method."""
    config = _base_config()
    config['particles']['populations'][0]['tracer_methods'] = {'vanwesten': {'beta': 0.2}}

    with pytest.raises(YamlValidationError, match='YAML config validation error'):
        _validate_config(tmp_path, config)


def test_population_schema_accepts_one_method_with_flow_fields(tmp_path):
    """Accept a valid single tracer method with flow field names."""
    config = _base_config()

    validated = _validate_config(tmp_path, config)

    tracer_methods = validated['particles']['populations'][0]['tracer_methods']
    assert tracer_methods == {
        'vanwesten': {
            'flow_field_name': ['bed_load_velocity'],
            'beta': 0.2,
            'suspended_velocity_method': 'soulsby_2011',
        }
    }


def test_population_schema_accepts_passive_tracer_default_flow_field(tmp_path):
    """Accept passive tracer configs and apply the default flow field."""
    config = _base_config()
    config['particles']['populations'][0]['particle_type'] = 'passive'
    config['particles']['populations'][0]['characteristics'] = {'diffusion_coefficient': 0.0}
    config['particles']['populations'][0]['tracer_methods'] = {'passive_tracer': {}}

    validated = _validate_config(tmp_path, config)

    tracer_methods = validated['particles']['populations'][0]['tracer_methods']
    assert tracer_methods == {'passive_tracer': {'flow_field_name': ['depth_avg_flow_velocity']}}


def test_population_schema_rejects_vanwesten_bl(tmp_path):
    """Reject stale tracer methods that are not implemented at runtime."""
    config = _base_config()
    config['particles']['populations'][0]['tracer_methods'] = {
        'vanwesten_bl': {'flow_field_name': ['bed_load_velocity']}
    }

    with pytest.raises(YamlValidationError, match='YAML config validation error'):
        _validate_config(tmp_path, config)


@pytest.mark.parametrize(
    ('strategy_name', 'settings'),
    [
        ('random', {'pol_file': './release_area.pol', 'nlocations': 10}),
        ('grid', {'pol_file': './release_area.pol', 'separation': {'dx': 100.0, 'dy': 100.0}}),
    ],
)
def test_population_schema_rejects_legacy_pol_file_for_area_strategies(tmp_path, strategy_name, settings):
    """Reject legacy pol_file for random and grid release areas.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory used for writing the test configuration file.
    strategy_name : str
        Area seeding strategy to validate.
    settings : dict
        Strategy settings containing the legacy ``pol_file`` key.
    """
    config = _base_config()
    config['particles']['populations'][0]['seeding']['strategy'] = {strategy_name: settings}

    with pytest.raises(YamlValidationError, match='YAML config validation error'):
        _validate_config(tmp_path, config)


@pytest.mark.parametrize(
    ('strategy_name', 'settings'),
    [
        ('random', {'poly': './release_area.pol', 'nlocations': 10}),
        ('grid', {'poly': './release_area.pol', 'separation': {'dx': 100.0, 'dy': 100.0}}),
    ],
)
def test_population_schema_accepts_poly_file_path_for_area_strategies(tmp_path, strategy_name, settings):
    """Accept poly file paths for random and grid release areas.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory used for writing the test configuration file.
    strategy_name : str
        Area seeding strategy to validate.
    settings : dict
        Strategy settings containing a ``poly`` file path.
    """
    config = _base_config()
    config['particles']['populations'][0]['seeding']['strategy'] = {strategy_name: settings}

    validated = _validate_config(tmp_path, config)

    strategy = validated['particles']['populations'][0]['seeding']['strategy']
    expected_settings = _with_strategy_defaults(strategy_name, settings)
    assert strategy == {strategy_name: expected_settings}


@pytest.mark.parametrize(
    ('strategy_name', 'settings'),
    [
        ('random', {'poly': ['0,0', '1,0', '0,1'], 'nlocations': 10}),
        ('grid', {'poly': ['0,0', '1,0', '0,1'], 'separation': {'dx': 100.0, 'dy': 100.0}}),
    ],
)
def test_population_schema_accepts_inline_poly_for_area_strategies(tmp_path, strategy_name, settings):
    """Accept inline polygon vertex arrays for random and grid areas.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory used for writing the test configuration file.
    strategy_name : str
        Area seeding strategy to validate.
    settings : dict
        Strategy settings containing inline ``poly`` vertices.
    """
    config = _base_config()
    config['particles']['populations'][0]['seeding']['strategy'] = {strategy_name: settings}

    validated = _validate_config(tmp_path, config)

    strategy = validated['particles']['populations'][0]['seeding']['strategy']
    expected_settings = _with_strategy_defaults(strategy_name, settings)
    assert strategy == {strategy_name: expected_settings}


@pytest.mark.parametrize(
    ('strategy_name', 'settings'),
    [
        ('random', {'poly': ['0,0', '1,0'], 'nlocations': 10}),
        ('grid', {'poly': ['0,0', '1,0'], 'separation': {'dx': 100.0, 'dy': 100.0}}),
    ],
)
def test_population_schema_rejects_inline_poly_with_too_few_vertices(tmp_path, strategy_name, settings):
    """Reject inline polygons with fewer than three vertices.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory used for writing the test configuration file.
    strategy_name : str
        Area seeding strategy to validate.
    settings : dict
        Strategy settings containing an invalid inline ``poly`` value.
    """
    config = _base_config()
    config['particles']['populations'][0]['seeding']['strategy'] = {strategy_name: settings}

    with pytest.raises(YamlValidationError, match='YAML config validation error'):
        _validate_config(tmp_path, config)


def test_population_schema_accepts_constant_burial_depth(tmp_path):
    """Accept configured constant burial depth values."""
    config = _base_config()
    config['particles']['populations'][0]['seeding']['burial_depth'] = {'constant': 1.25}

    validated = _validate_config(tmp_path, config)

    assert validated['particles']['populations'][0]['seeding']['burial_depth'] == {'constant': 1.25}

def test_population_schema_accepts_remove_permanently_buried_boolean(tmp_path):
    """Accept the permanent-burial optimization flag when it is boolean."""
    config = _base_config()
    config['particles']['populations'][0]['seeding']['remove_permanently_buried'] = True

    validated = _validate_config(tmp_path, config)

    assert validated['particles']['populations'][0]['seeding']['remove_permanently_buried'] is True


def test_population_schema_rejects_non_boolean_remove_permanently_buried(tmp_path):
    """Reject non-boolean permanent-burial optimization values."""
    config = _base_config()
    config['particles']['populations'][0]['seeding']['remove_permanently_buried'] = 'yes'

    with pytest.raises(YamlValidationError, match='YAML config validation error'):
        _validate_config(tmp_path, config)


def test_population_schema_accepts_sediment_fraction_index(tmp_path):
    """Accept per-population sediment fraction selection by index."""
    config = _base_config()
    config['particles']['populations'][0]['sediment_fraction_index'] = 2

    validated = _validate_config(tmp_path, config)

    assert validated['particles']['populations'][0]['sediment_fraction_index'] == 2


def test_population_schema_rejects_negative_sediment_fraction_index(tmp_path):
    """Reject invalid negative per-population sediment fraction index values."""
    config = _base_config()
    config['particles']['populations'][0]['sediment_fraction_index'] = -1

    with pytest.raises(YamlValidationError, match='YAML config validation error'):
        _validate_config(tmp_path, config)


def test_population_schema_leaves_fraction_index_unset_without_an_explicit_override(tmp_path):
    """Leave absent population selection available for a global default."""
    config = _base_config()

    validated = _validate_config(tmp_path, config)

    assert 'sediment_fraction_index' not in validated['particles']['populations'][0]


def test_population_schema_accepts_sediment_fraction_name(tmp_path):
    """Accept per-population sediment fraction selection by label."""
    config = _base_config()
    config['particles']['populations'][0]['sediment_fraction_name'] = 'sediment300_nat'

    validated = _validate_config(tmp_path, config)

    assert validated['particles']['populations'][0]['sediment_fraction_name'] == 'sediment300_nat'


@pytest.mark.parametrize('method', ['none', 'brownian'])
def test_population_schema_accepts_population_diffusion_methods(tmp_path, method):
    """Accept both supported per-population diffusion methods."""
    config = _base_config()
    config['particles']['populations'][0]['diffusion'] = {
        'method': method,
        'coefficient': 0.05,
        'seed': 1234,
    }

    validated = _validate_config(tmp_path, config)

    assert validated['particles']['populations'][0]['diffusion']['method'] == method


@pytest.mark.parametrize(
    'diffusion',
    [
        {'method': 'random', 'coefficient': 0.1},
        {'method': 'brownian', 'coefficient': -0.1},
        {'method': 'brownian', 'coefficient': 0.1, 'seed': True},
        {'method': 'brownian', 'coefficient': 0.1, 'seed': 1.5},
    ],
)
def test_population_schema_rejects_invalid_diffusion_config(tmp_path, diffusion):
    """Reject unsupported methods and invalid diffusion parameters."""
    config = _base_config()
    config['particles']['populations'][0]['diffusion'] = diffusion

    with pytest.raises(YamlValidationError, match='YAML config validation error'):
        _validate_config(tmp_path, config)


def _validate_config(tmp_path, config):
    """Write a temporary config file and validate it with the schema validator."""
    config_file = tmp_path / 'sedtrails.yml'
    config_file.write_text(yaml.dump(config))
    return YAMLConfigValidator().validate_yaml(str(config_file))


def _with_strategy_defaults(strategy_name, settings):
    """Return expected defaults for a selected seeding strategy."""
    defaults = {
        'show_check_plots': False,
        'save_check_plots': False,
    }
    if strategy_name == 'random':
        defaults['seed'] = 42
    return {**settings, **defaults}


def _base_config():
    """Build a minimal valid base configuration for population schema tests."""
    return {
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
        }
    }


def test_population_schema_accepts_macdonald_2d_deposition_options(tmp_path):
    config = _base_config()
    config['particles']['populations'][0]['tracer_methods'] = {
        'macdonald': {
            'computationType': '2D',
            'entrainment': {
                'method': 'entrainment_frequency',
                'probability_law': 'linear',
            },
            'deposition': {
                'method': 'markov_settling',
                'settling_height_field': 'water_depth',
                'minimum_settling_height': 0.002,
            },
        }
    }

    validated = _validate_config(tmp_path, config)

    macdonald = validated['particles']['populations'][0]['tracer_methods']['macdonald']
    assert macdonald['entrainment'] == {
        'method': 'entrainment_frequency',
        'probability_law': 'linear',
    }
    deposition = macdonald['deposition']
    assert deposition == {
        'method': 'markov_settling',
        'settling_height_field': 'water_depth',
        'minimum_settling_height': pytest.approx(0.002),
    }


@pytest.mark.parametrize('computation_type', ['2D', 'Q3D'])
def test_population_schema_accepts_implemented_macdonald_computation_types(
    tmp_path,
    computation_type,
):
    """Accept only MacDonald modes implemented by the physics plugin."""
    config = _base_config()
    config['particles']['populations'][0]['tracer_methods'] = {
        'macdonald': {'computationType': computation_type}
    }

    validated = _validate_config(tmp_path, config)

    method = validated['particles']['populations'][0]['tracer_methods']['macdonald']
    assert method['computationType'] == computation_type


@pytest.mark.parametrize('computation_type', ['3D', 'q3d', 'unsupported'])
def test_population_schema_rejects_unimplemented_macdonald_computation_types(
    tmp_path,
    computation_type,
):
    """Reject modes that would fail only after physics conversion."""
    config = _base_config()
    config['particles']['populations'][0]['tracer_methods'] = {
        'macdonald': {'computationType': computation_type}
    }

    with pytest.raises(YamlValidationError, match='YAML config validation error'):
        _validate_config(tmp_path, config)


def test_population_schema_accepts_q3d_vertical_diffusion_factor(tmp_path):
    """Accept a nonnegative dimensionless Q3D vertical diffusion factor."""
    config = _base_config()
    config['particles']['populations'][0]['tracer_methods'] = {
        'macdonald': {
            'computationType': 'Q3D',
            'q3d_vertical_diffusion_factor': 0.2,
        }
    }

    validated = _validate_config(tmp_path, config)

    method = validated['particles']['populations'][0]['tracer_methods']['macdonald']
    assert method['q3d_vertical_diffusion_factor'] == pytest.approx(0.2)


def test_population_schema_rejects_negative_q3d_vertical_diffusion_factor(tmp_path):
    """Reject negative Q3D vertical diffusivity scaling."""
    config = _base_config()
    config['particles']['populations'][0]['tracer_methods'] = {
        'macdonald': {
            'computationType': 'Q3D',
            'q3d_vertical_diffusion_factor': -0.1,
        }
    }

    with pytest.raises(YamlValidationError, match='YAML config validation error'):
        _validate_config(tmp_path, config)


@pytest.mark.parametrize(
    'entrainment',
    [
        {'method': 'threshold'},
        {'method': 'entrainment_frequency', 'probability_law': 'unsupported'},
        {'method': 'shields_threshold', 'probability_law': 'poisson'},
        {'method': 'non_zero_particle_velocity', 'probability_law': 'linear'},
    ],
)
def test_population_schema_rejects_invalid_macdonald_2d_entrainment(tmp_path, entrainment):
    config = _base_config()
    config['particles']['populations'][0]['tracer_methods'] = {
        'macdonald': {'computationType': '2D', 'entrainment': entrainment}
    }

    with pytest.raises(YamlValidationError, match='YAML config validation error'):
        _validate_config(tmp_path, config)

@pytest.mark.parametrize(
    'deposition',
    [
        {'method': 'threshold'},
        {'method': 'markov_settling', 'settling_height_field': 'total_transport_centroid_elevation'},
        {'method': 'markov_settling', 'minimum_settling_height': 0.0},
    ],
)
def test_population_schema_rejects_invalid_macdonald_2d_deposition(tmp_path, deposition):
    config = _base_config()
    config['particles']['populations'][0]['tracer_methods'] = {
        'macdonald': {'computationType': '2D', 'deposition': deposition}
    }

    with pytest.raises(YamlValidationError, match='YAML config validation error'):
        _validate_config(tmp_path, config)
