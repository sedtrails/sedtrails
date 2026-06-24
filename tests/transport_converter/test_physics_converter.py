"""Tests for physics converter configuration handling."""

from sedtrails.transport_converter.physics_converter import PhysicsConfig, PhysicsConverter


def test_nested_tracer_config_uses_resolved_method_from_dict_config():
    config = PhysicsConfig.from_dict(
        config={'tracer_method': 'soulsby', 'gravity': 9.8},
        tracer_config={
            'soulsby': {
                'flow_field_name': ['grain_velocity'],
                'tracer_grain_size': 0.0001,
                'background_grain_size': 0.0002,
                'soulsby_mu_d': 0.6,
            },
            'vanwesten': {
                'flow_field_name': ['bed_load_velocity'],
                'beta': 0.3,
            },
        },
    )

    assert config.tracer_method == 'soulsby'
    assert config.gravity == 9.8
    assert config.tracer_grain_size == 0.0001
    assert config.background_grain_size == 0.0002
    assert config.soulsby_mu_d == 0.6
    assert not hasattr(config, 'beta')


def test_flat_tracer_config_is_still_applied():
    config = PhysicsConfig.from_dict(
        config={'tracer_method': 'vanwesten'},
        tracer_config={'flow_field_name': ['bed_load_velocity'], 'beta': 0.4},
    )

    assert config.tracer_method == 'vanwesten'
    assert config.beta == 0.4
    assert config.flow_field_name == ['bed_load_velocity']


def test_converter_instances_do_not_share_config_or_caches():
    first = PhysicsConverter(
        {'tracer_method': 'soulsby'},
        {'soulsby': {'flow_field_name': ['grain_velocity'], 'tracer_grain_size': 0.0001}},
    )
    second = PhysicsConverter(
        {'tracer_method': 'soulsby'},
        {'soulsby': {'flow_field_name': ['grain_velocity'], 'tracer_grain_size': 0.0005}},
    )

    assert first.config.tracer_grain_size == 0.0001
    assert second.config.tracer_grain_size == 0.0005
    assert first.config is not second.config
    assert first._grain_properties is not second._grain_properties
    assert first._physics_plugin is None
    assert second._physics_plugin is None

    first._grain_properties['sentinel'] = 1
    first._physics_plugin = object()

    assert second._grain_properties == {}
    assert second._physics_plugin is None
