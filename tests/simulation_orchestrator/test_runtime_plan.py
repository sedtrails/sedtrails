"""Tests for population tracer runtime planning."""

from pathlib import Path

import numpy as np
import pytest

from sedtrails.application_interfaces.validator import YAMLConfigValidator
from sedtrails.exceptions.exceptions import ConfigurationError
from sedtrails.simulation_orchestrator.runtime_plan import (
    DEFAULT_TRANSPORT_PROBABILITY_METHOD,
    TracerRuntimePlan,
    build_plan_sedtrails_data,
    _resolve_fraction_selection,
    _select_fraction_value,
    build_population_runtime_plans,
    required_physics_fields,
    validate_population_runtime_configurations,
    unique_flow_field_names,
)


def _population_config(
    tracer_methods,
    *,
    characteristics=None,
    transport_probability=DEFAULT_TRANSPORT_PROBABILITY_METHOD,
    particle_type='sand',
    seeding=None,
):
    """Build a minimal population configuration for runtime plan tests."""
    return {
        'name': 'sand',
        'particle_type': particle_type,
        'characteristics': characteristics or {'density': 2650.0, 'grain_size': 0.00025},
        'tracer_methods': tracer_methods,
        'transport_probability': transport_probability,
        'seeding': seeding or {'quantity': 1, 'strategy': {'point': {'locations': ['0,0']}}},
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


def test_passive_tracer_population_defaults_to_depth_averaged_velocity():
    """Build a passive tracer plan with the default depth-averaged flow field."""
    population_config = _population_config(
        {'passive_tracer': {}},
        particle_type='passive',
        characteristics={'diffusion_coefficient': 0.0},
    )

    runtime_plan = build_population_runtime_plans([population_config], [object()], {})[0]

    assert runtime_plan.tracer.method_name == 'passive_tracer'
    assert runtime_plan.tracer.flow_field_names == ('depth_avg_flow_velocity',)
    assert runtime_plan.tracer.required_physics_fields == ('depth_avg_flow_velocity',)


def test_passive_example_preflight_does_not_invent_burial_depth():
    """Validate the passive example without materializing burial depth."""
    repository_root = Path(__file__).resolve().parents[2]
    config_path = repository_root / 'examples' / 'sedtrails-example-passive.yaml'
    config = YAMLConfigValidator().validate_yaml(str(config_path))
    population_configs = config['particles']['populations']

    assert 'burial_depth' not in population_configs[0]['seeding']
    validate_population_runtime_configurations(population_configs)


def test_macdonald_example_uses_registered_centroid_velocity():
    """Build a runtime plan from the shipped MacDonald example."""
    repository_root = Path(__file__).resolve().parents[2]
    config_path = repository_root / 'examples' / 'sedtrails-example-macdonald.yaml'
    config = YAMLConfigValidator().validate_yaml(str(config_path))
    population_configs = config['particles']['populations']

    runtime_plan = build_population_runtime_plans(
        population_configs,
        [object()],
        {},
    )[0]

    assert runtime_plan.tracer.flow_field_names == ('centroid_particle_velocity',)


def test_passive_tracer_rejects_non_default_transport_probability_methods():
    """Passive tracer should fail fast when stochastic/reduced probability modes are configured."""
    population_config = _population_config(
        {'passive_tracer': {}},
        particle_type='passive',
        characteristics={'diffusion_coefficient': 0.0},
        transport_probability='stochastic_transport',
    )

    with pytest.raises(ConfigurationError, match='Only "no_probability" is allowed'):
        build_population_runtime_plans([population_config], [object()], {})


def test_passive_tracer_requires_passive_particle_type():
    """Passive tracer mode should require particle_type='passive'."""
    population_config = _population_config(
        {'passive_tracer': {}},
        particle_type='sand',
        characteristics={'density': 2650.0, 'grain_size': 0.00025},
    )

    with pytest.raises(ConfigurationError, match='Set particle_type to "passive"'):
        build_population_runtime_plans([population_config], [object()], {})


def test_passive_tracer_rejects_explicit_burial_depth():
    """Passive tracer validation should reject an explicit seeding burial depth."""
    population_config = _population_config(
        {'passive_tracer': {}},
        particle_type='passive',
        characteristics={'diffusion_coefficient': 0.0},
        seeding={
            'quantity': 1,
            'strategy': {'point': {'locations': ['0,0']}},
            'burial_depth': {'constant': 0.0},
        },
    )

    with pytest.raises(ConfigurationError, match='seeding.burial_depth'):
        build_population_runtime_plans([population_config], [object()], {})


def test_passive_tracer_preflight_rejects_explicit_burial_depth():
    """Config-only validation should reject passive burial depth before seeding."""
    population_config = _population_config(
        {'passive_tracer': {}},
        particle_type='passive',
        characteristics={'diffusion_coefficient': 0.0},
        seeding={
            'quantity': 1,
            'strategy': {'point': {'locations': ['0,0']}},
            'burial_depth': {'constant': 0.0},
        },
    )

    with pytest.raises(ConfigurationError, match='seeding.burial_depth'):
        validate_population_runtime_configurations([population_config])


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


@pytest.mark.parametrize('transport_probability', ['stochastic_transport', 'reduced_velocity'])
def test_macdonald_rejects_unsupported_transport_probability(transport_probability):
    """MacDonald must not create a zero mixing layer for burial probability modes."""
    population_config = _population_config(
        {
            'macdonald': {
                'flow_field_name': ['centroid_particle_velocity'],
                'computationType': '2D',
            }
        },
        transport_probability=transport_probability,
    )

    with pytest.raises(ConfigurationError, match='Only "no_probability"'):
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


def test_macdonald_required_fields_include_default_transition_inputs():
    """Preserve fields consumed by default MacDonald 2D transitions."""
    fields = required_physics_fields(
        'macdonald',
        ['centroid_particle_velocity'],
        {'computationType': '2D'},
    )

    assert 'max_shields_number' in fields


@pytest.mark.parametrize(
    ('settling_height_field', 'expected_field'),
    [
        (None, 'suspended_transport_centroid_elevation'),
        ('water_depth', 'water_depth'),
    ],
)
def test_macdonald_required_fields_include_markov_settling_height(
    settling_height_field,
    expected_field,
):
    """Preserve the configured MacDonald Markov settling-height field."""
    deposition = {'method': 'markov_settling'}
    if settling_height_field is not None:
        deposition['settling_height_field'] = settling_height_field

    fields = required_physics_fields(
        'macdonald',
        ['centroid_particle_velocity'],
        {'computationType': '2D', 'deposition': deposition},
    )

    assert expected_field in fields


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
    assert plan_data.bed_load_velocity is converter.generated_velocity
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


def test_build_plan_sedtrails_data_selects_population_fraction_before_conversion():
    """Select the configured population fraction before running tracer physics conversion."""
    source_data = _FakeSedtrailsData(
        fractions=3,
        bed_load_transport={
            'x': np.array([[[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]]]),
            'y': np.array([[[3.0, 4.0], [30.0, 40.0], [300.0, 400.0]]]),
            'magnitude': np.array([[[5.0, 6.0], [50.0, 60.0], [500.0, 600.0]]]),
        },
    )
    converter = _InspectingFractionPhysicsConverter(expected=np.array([[100.0, 200.0]]))
    tracer_plan = TracerRuntimePlan(
        method_name='vanwesten',
        method_config={'flow_field_name': ['bed_load_velocity']},
        flow_field_names=('bed_load_velocity',),
        transport_probability_method='stochastic_transport',
        required_physics_fields=('mixing_layer_thickness',),
        converter=converter,
    )

    plan_data = build_plan_sedtrails_data(
        source_data,
        tracer_plan,
        population_config={'sediment_fraction_index': 2},
    )

    assert converter.saw_fraction_shape == (1, 2)
    assert plan_data.fractions == 1
    np.testing.assert_array_equal(plan_data.bed_load_transport['x'], np.array([[100.0, 200.0]]))


def test_build_plan_sedtrails_data_rejects_out_of_bounds_population_fraction():
    """Fail fast when a population requests a sediment fraction index outside available bounds."""
    source_data = _FakeSedtrailsData(
        fractions=2,
        bed_load_transport={
            'x': np.array([[[1.0], [2.0]]]),
            'y': np.array([[[1.0], [2.0]]]),
            'magnitude': np.array([[[1.0], [2.0]]]),
        },
    )
    tracer_plan = TracerRuntimePlan(
        method_name='vanwesten',
        method_config={'flow_field_name': ['bed_load_velocity']},
        flow_field_names=('bed_load_velocity',),
        transport_probability_method='stochastic_transport',
        required_physics_fields=('mixing_layer_thickness',),
        converter=_NamedScalarPhysicsConverter('mixing_layer_thickness', np.array([0.1])),
    )

    with pytest.raises(ConfigurationError, match='out of bounds'):
        build_plan_sedtrails_data(
            source_data,
            tracer_plan,
            population_config={'sediment_fraction_index': 3},
        )


def test_population_fraction_selection_uses_global_default_and_population_override():
    """Resolve one population choice before applying a global default."""
    global_selection = _resolve_fraction_selection(
        {},
        default_fraction_index=2,
        default_fraction_name=None,
    )
    assert global_selection == (2, None)

    index_selection = _resolve_fraction_selection(
        {'sediment_fraction_index': 2},
        default_fraction_index=0,
        default_fraction_name='sediment100_nat',
    )
    assert index_selection == (2, None)

    name_selection = _resolve_fraction_selection(
        {'sediment_fraction_name': 'sediment300_nat', 'sediment_fraction_index': 0},
        default_fraction_index=2,
        default_fraction_name='sediment100_nat',
    )
    assert name_selection == (0, 'sediment300_nat')


def test_fraction_selection_preserves_component_shapes_when_one_component_is_missing():
    """Select only vector components that carry a fraction axis."""
    fractional_values = np.arange(6.0).reshape(1, 3, 2)
    selected = _select_fraction_value(
        {
            'x': fractional_values,
            'y': np.zeros((1, 2)),
            'magnitude': fractional_values + 10.0,
        },
        fractions=3,
        fraction_index=2,
    )

    assert selected['x'].shape == (1, 2)
    assert selected['y'].shape == (1, 2)
    assert selected['magnitude'].shape == (1, 2)
    np.testing.assert_array_equal(selected['y'], np.zeros((1, 2)))


def test_build_plan_sedtrails_data_selects_population_fraction_by_name():
    """Resolve population fraction labels from metadata when selecting by name."""
    source_data = _FakeSedtrailsData(
        fractions=3,
        bed_load_transport={
            'x': np.array([[[1.0], [10.0], [100.0]]]),
            'y': np.array([[[1.0], [10.0], [100.0]]]),
            'magnitude': np.array([[[1.0], [10.0], [100.0]]]),
        },
        metadata=_FakeMetadata(['sediment100_nat', 'sediment200_nat', 'sediment300_nat']),
    )
    converter = _InspectingFractionPhysicsConverter(expected=np.array([[10.0]]))
    tracer_plan = TracerRuntimePlan(
        method_name='vanwesten',
        method_config={'flow_field_name': ['bed_load_velocity']},
        flow_field_names=('bed_load_velocity',),
        transport_probability_method='stochastic_transport',
        required_physics_fields=('mixing_layer_thickness',),
        converter=converter,
    )

    plan_data = build_plan_sedtrails_data(
        source_data,
        tracer_plan,
        population_config={'sediment_fraction_name': 'sediment200_nat'},
    )

    assert converter.saw_fraction_shape == (1, 1)
    np.testing.assert_array_equal(plan_data.bed_load_transport['x'], np.array([[10.0]]))


def test_build_plan_sedtrails_data_reports_namcon_labels_for_invalid_name():
    """Invalid fraction names should report the available NAMCON labels."""
    source_data = _FakeSedtrailsData(
        fractions=3,
        bed_load_transport={
            'x': np.array([[[1.0], [10.0], [100.0]]]),
            'y': np.array([[[1.0], [10.0], [100.0]]]),
            'magnitude': np.array([[[1.0], [10.0], [100.0]]]),
        },
        metadata=_FakeMetadata(['sediment100_nat', 'sediment200_nat', 'sediment300_nat']),
    )
    converter = _InspectingFractionPhysicsConverter(expected=np.array([[10.0]]))
    tracer_plan = TracerRuntimePlan(
        method_name='vanwesten',
        method_config={'flow_field_name': ['bed_load_velocity']},
        flow_field_names=('bed_load_velocity',),
        transport_probability_method='stochastic_transport',
        required_physics_fields=('mixing_layer_thickness',),
        converter=converter,
    )

    with pytest.raises(ConfigurationError, match=r"Available NAMCON labels: \['sediment100_nat', 'sediment200_nat', 'sediment300_nat'\]"):
        build_plan_sedtrails_data(
            source_data,
            tracer_plan,
            population_config={'sediment_fraction_name': 'not_a_fraction'},
        )


def test_build_plan_sedtrails_data_requires_index_when_labels_unavailable():
    """Require explicit index selection when labels are unavailable for name lookup."""
    source_data = _FakeSedtrailsData(
        fractions=3,
        bed_load_transport={
            'x': np.array([[[1.0], [10.0], [100.0]]]),
            'y': np.array([[[1.0], [10.0], [100.0]]]),
            'magnitude': np.array([[[1.0], [10.0], [100.0]]]),
        },
        metadata=None,
    )
    converter = _InspectingFractionPhysicsConverter(expected=np.array([[100.0]]))
    tracer_plan = TracerRuntimePlan(
        method_name='vanwesten',
        method_config={'flow_field_name': ['bed_load_velocity']},
        flow_field_names=('bed_load_velocity',),
        transport_probability_method='stochastic_transport',
        required_physics_fields=('mixing_layer_thickness',),
        converter=converter,
    )

    with pytest.raises(ConfigurationError, match='Detected 3 sediment fractions'):
        build_plan_sedtrails_data(
            source_data,
            tracer_plan,
            population_config={
                'sediment_fraction_name': 'sediment300_nat',
                'sediment_fraction_index': 2,
            },
        )


class _FakeSedtrailsData:
    """Minimal sedtrails-data test double storing physics fields by name."""

    def __init__(self, fractions=1, bed_load_transport=None, metadata=None):
        """Initialize an empty physics field store."""
        self._physics_fields = {}
        self.fractions = fractions
        self.bed_load_transport = bed_load_transport
        self.metadata = metadata

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


class _InspectingFractionPhysicsConverter:
    """Converter test double that records the fraction-selected transport slice."""

    def __init__(self, expected):
        self.expected = expected
        self.saw_fraction_shape = None

    def convert_physics(self, sedtrails_data, transport_probability_method):
        bed_load_x = np.asarray(sedtrails_data.bed_load_transport['x'])
        self.saw_fraction_shape = bed_load_x.shape
        np.testing.assert_array_equal(bed_load_x, self.expected)
        sedtrails_data.add_physics_field('mixing_layer_thickness', np.array([0.1]))


class _FakeMetadata:
    """Metadata test double exposing sediment fraction labels through get()."""

    def __init__(self, labels):
        self._labels = labels

    def get(self, key, default=None):
        if key == 'sediment_fraction_labels':
            return self._labels
        return default
