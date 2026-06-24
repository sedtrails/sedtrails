"""Output schema regression tests."""

import pytest
import yaml

from sedtrails.application_interfaces.validator import YAMLConfigValidator
from sedtrails.exceptions import YamlValidationError


def test_outputs_netcdf_schema_applies_defaults_when_block_is_present(tmp_path):
    """Apply NetCDF output defaults when the block is configured."""
    config = _base_config()
    config['outputs'] = {'store_tracks': True, 'netcdf': {}}

    validated = _validate_config(tmp_path, config)

    assert validated['outputs']['netcdf'] == {
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
        'checkpoint': True,
        'checkpoint_interval': 0,
    }


def test_outputs_netcdf_schema_applies_defaults_when_block_is_omitted(tmp_path):
    """Apply the same NetCDF defaults when the block is omitted."""
    config = _base_config()
    config['outputs'] = {'store_tracks': True}

    validated = _validate_config(tmp_path, config)

    assert validated['outputs']['netcdf'] == {
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
        'checkpoint': True,
        'checkpoint_interval': 0,
    }


def test_outputs_netcdf_schema_accepts_non_default_values(tmp_path):
    """Accept all supported non-default NetCDF output options."""
    config = _base_config()
    config['outputs'] = {
        'store_tracks': True,
        'netcdf': {
            'coordinate_dtype': 'float64',
            'status_dtype': 'int32',
            'compression': 'auto',
            'compression_auto_threshold_mb': 512,
            'compression_level': 9,
            'shuffle': False,
            'time_chunk': 4,
            'particle_chunk': 128,
            'sync_interval': 0,
            'reopen_interval': None,
            'checkpoint': False,
            'checkpoint_interval': 12,
        },
    }

    validated = _validate_config(tmp_path, config)

    assert validated['outputs']['netcdf'] == config['outputs']['netcdf']


@pytest.mark.parametrize('compression', [True, False])
def test_outputs_netcdf_schema_accepts_boolean_compression(tmp_path, compression):
    """Accept explicit boolean compression policies for NetCDF output."""
    config = _base_config()
    config['outputs'] = {'store_tracks': True, 'netcdf': {'compression': compression}}

    validated = _validate_config(tmp_path, config)

    assert validated['outputs']['netcdf']['compression'] is compression


@pytest.mark.parametrize(
    ('key', 'value'),
    [
        ('coordinate_dtype', 'float16'),
        ('status_dtype', 'bool'),
        ('compression', 'gzip'),
        ('compression_auto_threshold_mb', -1),
        ('compression_level', 10),
        ('compression_level', -1),
        ('time_chunk', 0),
        ('particle_chunk', 0),
        ('sync_interval', -1),
        ('reopen_interval', -1),
        ('checkpoint_interval', -1),
    ],
)
def test_outputs_netcdf_schema_rejects_invalid_values(tmp_path, key, value):
    """Reject invalid enum and range values in NetCDF output options."""
    config = _base_config()
    config['outputs'] = {'store_tracks': True, 'netcdf': {key: value}}

    with pytest.raises(YamlValidationError, match='YAML config validation error'):
        _validate_config(tmp_path, config)


def test_outputs_netcdf_schema_rejects_unknown_option(tmp_path):
    """Reject unrecognized NetCDF output options."""
    config = _base_config()
    config['outputs'] = {
        'store_tracks': True,
        'netcdf': {'coordinate_dtype': 'float32', 'unknown_option': True},
    }

    with pytest.raises(YamlValidationError, match='YAML config validation error'):
        _validate_config(tmp_path, config)


def test_outputs_schema_rejects_unknown_top_level_option(tmp_path):
    """Reject unrecognized output options."""
    config = _base_config()
    config['outputs'] = {'store_tracks': True, 'unexpected': True}

    with pytest.raises(YamlValidationError, match='YAML config validation error'):
        _validate_config(tmp_path, config)


def _validate_config(tmp_path, config):
    """Write a temporary config file and validate it with the schema validator."""
    config_file = tmp_path / 'sedtrails.yml'
    config_file.write_text(yaml.dump(config))
    return YAMLConfigValidator().validate_yaml(str(config_file))


def _base_config():
    """Build a minimal valid base configuration for output schema tests."""
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
