"""Tests for population tracer runtime planning."""

import numpy as np
import pytest

from sedtrails.exceptions.exceptions import ConfigurationError
from sedtrails.simulation_orchestrator.runtime_plan import (
    DEFAULT_TRANSPORT_PROBABILITY_METHOD,
    TracerRuntimePlan,
    build_plan_sedtrails_data,
    build_population_runtime_plans,
    required_physics_fields,
    unique_flow_field_names,
)


def _population_config(
    tracer_methods,
    *,
    characteristics=None,
    transport_probability=DEFAULT_TRANSPORT_PROBABILITY_METHOD,
):
    """Build a minimal population configuration for runtime plan tests."""
    return {
        'name': 'sand',
        'particle_type': 'sand',
        'characteristics': characteristics or {'density': 2650.0, 'grain_size': 0.00025},
        'tracer_methods': tracer_methods,
        'transport_probability': transport_probability,
    }


def test_vanwesten_population_creates_population_scoped_converter():
    """Build a van Westen plan with population-specific converter settings."""
    population_config = _population_config(
        {
            'vanwesten': {
                'flow_field_name': ['bed_load_velocity', 'suspended_velocity'],
                'beta': 0.3,
                'suspended_velocity_method': 'macdonald_2006',
            }
        },
        transport_probability='stochastic_transport',
    )

    runtime_plan = build_population_runtime_plans(
        [population_config],
        [object()],
        {'gravity': 9.8, 'bertin_coefficient': 0.123},
    )[0]

    assert runtime_plan.population_index == 0
    assert runtime_plan.tracer.method_name == 'vanwesten'
    assert runtime_plan.tracer.flow_field_names == ('bed_load_velocity', 'suspended_velocity')
    assert runtime_plan.tracer.transport_probability_method == 'stochastic_transport'
    assert runtime_plan.tracer.converter.config.tracer_method == 'vanwesten'
    assert runtime_plan.tracer.converter.config.gravity == 9.8
    assert runtime_plan.tracer.converter.config.bertin_coefficient == 0.123
    assert runtime_plan.tracer.converter.config.beta == 0.3
    assert runtime_plan.tracer.converter.config.suspended_velocity_method == 'macdonald_2006'
    assert runtime_plan.tracer.converter.config.grain_diameter == 0.00025


def test_soulsby_population_creates_population_scoped_converter():
    """Build a Soulsby plan and map population and physics values to converter config."""
    population_config = _population_config(
        {
            'soulsby': {
                'flow_field_name': ['grain_velocity'],
                'tracer_grain_size': 0.0001,
                'background_grain_size': 0.0002,
                'soulsby_mu_d': 0.6,
            }
        },
        characteristics={'density': 2600.0, 'grain_size': 0.0004},
    )

    runtime_plan = build_population_runtime_plans([population_config], [object()], {'water_density': 1025.0})[0]

    assert runtime_plan.tracer.method_name == 'soulsby'
    assert runtime_plan.tracer.flow_field_names == ('grain_velocity',)
    assert runtime_plan.tracer.converter.config.tracer_method == 'soulsby'
    assert runtime_plan.tracer.converter.config.water_density == 1025.0
    assert runtime_plan.tracer.converter.config.particle_density == 2600.0
    assert runtime_plan.tracer.converter.config.tracer_grain_size == 0.0001
    assert runtime_plan.tracer.converter.config.background_grain_size == 0.0002
    assert runtime_plan.tracer.converter.config.soulsby_mu_d == 0.6


def test_mixed_populations_preserve_each_population_method_and_flow_fields():
    """Keep tracer method and flow fields isolated per population in mixed runs."""
    vanwesten_config = _population_config({'vanwesten': {'flow_field_name': ['bed_load_velocity']}})
    soulsby_config = _population_config({'soulsby': {'flow_field_name': ['grain_velocity']}})

    runtime_plans = build_population_runtime_plans(
        [vanwesten_config, soulsby_config],
        [object(), object()],
        {},
    )

    assert [runtime_plan.tracer.method_name for runtime_plan in runtime_plans] == ['vanwesten', 'soulsby']
    assert [runtime_plan.tracer.flow_field_names for runtime_plan in runtime_plans] == [
        ('bed_load_velocity',),
        ('grain_velocity',),
    ]
    assert unique_flow_field_names(runtime_plans) == ['bed_load_velocity', 'grain_velocity']


def test_same_method_populations_keep_separate_method_configs():
    """Ensure same-method populations do not share converter instances or config."""
    fine_sand_config = _population_config(
        {'vanwesten': {'flow_field_name': ['bed_load_velocity'], 'beta': 0.1}},
        characteristics={'density': 2650.0, 'grain_size': 0.0001},
    )
    coarse_sand_config = _population_config(
        {'vanwesten': {'flow_field_name': ['bed_load_velocity'], 'beta': 0.4}},
        characteristics={'density': 2650.0, 'grain_size': 0.0005},
    )

    runtime_plans = build_population_runtime_plans(
        [fine_sand_config, coarse_sand_config],
        [object(), object()],
        {},
    )

    fine_converter = runtime_plans[0].tracer.converter
    coarse_converter = runtime_plans[1].tracer.converter
    assert fine_converter is not coarse_converter
    assert fine_converter.config.beta == 0.1
    assert coarse_converter.config.beta == 0.4
    assert fine_converter.config.grain_diameter == 0.0001
    assert coarse_converter.config.grain_diameter == 0.0005


def test_multiple_methods_in_one_population_raises_configuration_error():
    """Reject a population that declares more than one tracer method."""
    population_config = _population_config(
        {
            'vanwesten': {'flow_field_name': ['bed_load_velocity']},
            'soulsby': {'flow_field_name': ['grain_velocity']},
        }
    )

    with pytest.raises(ConfigurationError, match='multiple tracer methods'):
        build_population_runtime_plans([population_config], [object()], {})


def test_unknown_method_raises_configuration_error():
    """Raise a configuration error for unsupported tracer methods."""
    population_config = _population_config({'unknown': {'flow_field_name': ['some_velocity']}})

    with pytest.raises(ConfigurationError, match='unsupported tracer method'):
        build_population_runtime_plans([population_config], [object()], {})


def test_missing_flow_field_name_raises_configuration_error():
    """Raise a configuration error when flow_field_name is missing."""
    population_config = _population_config({'vanwesten': {'beta': 0.3}})

    with pytest.raises(ConfigurationError, match='flow_field_name'):
        build_population_runtime_plans([population_config], [object()], {})


def test_population_count_mismatch_raises_configuration_error():
    """Raise a configuration error when seeded and configured population counts differ."""
    population_config = _population_config({'vanwesten': {'flow_field_name': ['bed_load_velocity']}})

    with pytest.raises(ConfigurationError, match='does not match seeded population count'):
        build_population_runtime_plans([population_config], [], {})


def test_required_physics_fields_are_method_specific_and_unique():
    """Return per-method required physics fields with duplicate flow fields removed."""
    assert required_physics_fields('vanwesten', ['bed_load_velocity', 'bed_load_velocity']) == (
        'bed_load_velocity',
        'mixing_layer_thickness',
        'bed_load_probability',
    )
    assert required_physics_fields('soulsby', ['grain_velocity']) == (
        'grain_velocity',
        'mixing_layer_thickness',
        'soulsby_a',
        'soulsby_b',
    )


def test_build_plan_sedtrails_data_copies_only_required_physics_fields():
    """Copy only required converted physics fields into plan-local sedtrails data."""
    source_data = _FakeSedtrailsData()
    converter = _FakePhysicsConverter()
    tracer_plan = TracerRuntimePlan(
        method_name='vanwesten',
        method_config={'flow_field_name': ['bed_load_velocity']},
        flow_field_names=('bed_load_velocity',),
        transport_probability_method='stochastic_transport',
        required_physics_fields=('bed_load_velocity', 'mixing_layer_thickness'),
        converter=converter,
    )

    plan_data = build_plan_sedtrails_data(source_data, tracer_plan)

    assert converter.transport_probability_method == 'stochastic_transport'
    assert source_data.get_physics_fields() == []
    assert plan_data.get_physics_fields() == ['bed_load_velocity', 'mixing_layer_thickness']
    assert not plan_data.has_physics_field('ignored_field')
    # The dict wrapper is a new object (shallow copy), but the arrays inside are shared
    # by reference. Physics arrays are read-only after computation, so sharing is safe
    # and avoids ~3.5 GB of duplicate allocations per parallel worker.
    assert plan_data.bed_load_velocity is not converter.generated_velocity
    assert plan_data.bed_load_velocity['x'] is converter.generated_velocity['x']
    np.testing.assert_array_equal(plan_data.bed_load_velocity['x'], np.array([1.0, 2.0]))


def test_plan_local_physics_data_keeps_same_named_fields_from_overwriting():
    """Keep same-named converted fields isolated between separate runtime plans."""
    first_tracer_plan = TracerRuntimePlan(
        method_name='vanwesten',
        method_config={'flow_field_name': ['bed_load_velocity']},
        flow_field_names=('bed_load_velocity',),
        transport_probability_method='stochastic_transport',
        required_physics_fields=('mixing_layer_thickness',),
        converter=_NamedScalarPhysicsConverter('mixing_layer_thickness', np.array([0.1, 0.2])),
    )
    second_tracer_plan = TracerRuntimePlan(
        method_name='soulsby',
        method_config={'flow_field_name': ['grain_velocity']},
        flow_field_names=('grain_velocity',),
        transport_probability_method='no_probability',
        required_physics_fields=('mixing_layer_thickness',),
        converter=_NamedScalarPhysicsConverter('mixing_layer_thickness', np.array([9.0, 10.0])),
    )
    source_data = _FakeSedtrailsData()

    first_plan_data = build_plan_sedtrails_data(source_data, first_tracer_plan)
    second_plan_data = build_plan_sedtrails_data(source_data, second_tracer_plan)

    np.testing.assert_array_equal(first_plan_data.mixing_layer_thickness, np.array([0.1, 0.2]))
    np.testing.assert_array_equal(second_plan_data.mixing_layer_thickness, np.array([9.0, 10.0]))
    assert source_data.get_physics_fields() == []


class _FakeSedtrailsData:
    """Minimal sedtrails-data test double storing physics fields by name."""

    def __init__(self):
        """Initialize an empty physics field store."""
        self._physics_fields = {}

    def add_physics_field(self, name, data):
        """Store a named physics field and expose it as an attribute."""
        self._physics_fields[name] = data
        setattr(self, name, data)

    def has_physics_field(self, name):
        """Return whether a physics field exists in this test double."""
        return name in self._physics_fields

    def get_physics_fields(self):
        """List stored physics field names in insertion order."""
        return list(self._physics_fields)


class _FakePhysicsConverter:
    """Physics converter test double that writes deterministic vector/scalar fields."""

    def __init__(self):
        """Initialize converter state captured during conversion."""
        self.generated_velocity = None
        self.transport_probability_method = None

    def convert_physics(self, sedtrails_data, transport_probability_method):
        """Populate test physics fields and record transport probability mode."""
        self.transport_probability_method = transport_probability_method
        self.generated_velocity = {
            'x': np.array([1.0, 2.0]),
            'y': np.array([3.0, 4.0]),
            'magnitude': np.array([5.0, 6.0]),
        }
        sedtrails_data.add_physics_field('bed_load_velocity', self.generated_velocity)
        sedtrails_data.add_physics_field('mixing_layer_thickness', np.array([0.1, 0.2]))
        sedtrails_data.add_physics_field('ignored_field', np.array([9.0, 9.0]))


class _NamedScalarPhysicsConverter:
    """Converter test double that writes one named scalar field."""

    def __init__(self, field_name, field_value):
        """Store the field name and value that will be injected during conversion."""
        self.field_name = field_name
        self.field_value = field_value

    def convert_physics(self, sedtrails_data, transport_probability_method):
        """Add the configured scalar field to the provided sedtrails data."""
        sedtrails_data.add_physics_field(self.field_name, self.field_value)
