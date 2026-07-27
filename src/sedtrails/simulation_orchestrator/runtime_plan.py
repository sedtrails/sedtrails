"""Runtime planning for population-specific tracer physics."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from sedtrails.exceptions.exceptions import ConfigurationError
from sedtrails.transport_converter.physics_converter import PhysicsConfig, PhysicsConverter


DEFAULT_PASSIVE_TRACER_FLOW_FIELDS = ('depth_avg_flow_velocity',)
SUPPORTED_TRACER_METHODS = frozenset({'macdonald', 'passive_tracer', 'soulsby', 'vanwesten'})
DEFAULT_TRANSPORT_PROBABILITY_METHOD = 'no_probability'


@dataclass(frozen=True)
class TracerRuntimePlan:
    """Runtime state for one population tracer method."""

    method_name: str
    method_config: Mapping[str, Any]
    flow_field_names: tuple[str, ...]
    transport_probability_method: str
    required_physics_fields: tuple[str, ...]
    converter: PhysicsConverter


@dataclass(frozen=True)
class PopulationRuntimePlan:
    """Runtime state for one particle population."""

    population_index: int
    population_config: Mapping[str, Any]
    population: Any
    tracer: TracerRuntimePlan


def build_population_runtime_plans(
    population_configs: Sequence[Mapping[str, Any]],
    populations: Sequence[Any],
    base_physics_config: PhysicsConfig | Mapping[str, Any],
) -> tuple[PopulationRuntimePlan, ...]:
    """
    Build population-scoped tracer runtime plans.

    Parameters
    ----------
    population_configs : Sequence[Mapping[str, Any]]
        Population configuration mappings.
    populations : Sequence[Any]
        Particle populations to process.
    base_physics_config : PhysicsConfig | Mapping[str, Any]
        Base physics configuration shared by populations.

    Returns
    -------
    tuple[PopulationRuntimePlan, ...]
        Tuple containing the computed values.
    """

    if len(population_configs) != len(populations):
        raise ConfigurationError(
            f'Population config count ({len(population_configs)}) does not match seeded population count '
            f'({len(populations)}).'
        )

    return tuple(
        _build_population_runtime_plan(index, population_config, population, base_physics_config)
        for index, (population_config, population) in enumerate(zip(population_configs, populations, strict=True))
    )


def unique_flow_field_names(runtime_plans: Sequence[PopulationRuntimePlan]) -> list[str]:
    """
    Return configured flow field names across plans, preserving first-seen order.

    Parameters
    ----------
    runtime_plans : Sequence[PopulationRuntimePlan]
        Population runtime plans to inspect.

    Returns
    -------
    list[str]
        String result of the conversion.
    """

    return _unique_preserving_order(
        flow_field_name
        for runtime_plan in runtime_plans
        for flow_field_name in runtime_plan.tracer.flow_field_names
    )


def build_plan_sedtrails_data(sedtrails_data: Any, tracer_plan: TracerRuntimePlan) -> Any:
    """
    Run base plan physics and return a clone containing only plan-required physics fields.
    """

    working_data = _shallow_sedtrails_data_clone(sedtrails_data)
    tracer_plan.converter.convert_physics(
        sedtrails_data=working_data,
        transport_probability_method=tracer_plan.transport_probability_method,
    )

    return _copy_required_plan_fields(sedtrails_data, working_data, tracer_plan)


def add_plan_timestep_physics(
    sedtrails_data: Any,
    tracer_plan: TracerRuntimePlan,
    current_timestep: float,
) -> Any:
    """
    Add timestep-dependent plan physics and return a clone with required fields preserved.
    """

    working_data = _shallow_sedtrails_data_clone(sedtrails_data)
    tracer_plan.converter.convert_timestep_physics(
        sedtrails_data=working_data,
        current_timestep=current_timestep,
    )

    return _copy_required_plan_fields(sedtrails_data, working_data, tracer_plan)


def _copy_required_plan_fields(sedtrails_data: Any, working_data: Any, tracer_plan: TracerRuntimePlan) -> Any:
    plan_data = _shallow_sedtrails_data_clone(sedtrails_data)
    for field_name in tracer_plan.required_physics_fields:
        if working_data.has_physics_field(field_name):
            plan_data.add_physics_field(field_name, _copy_physics_value(getattr(working_data, field_name)))
    return plan_data

def _build_population_runtime_plan(
    population_index: int,
    population_config: Mapping[str, Any],
    population: Any,
    base_physics_config: PhysicsConfig | Mapping[str, Any],
) -> PopulationRuntimePlan:
    tracer_methods = population_config.get('tracer_methods')
    if not isinstance(tracer_methods, Mapping) or not tracer_methods:
        raise ConfigurationError(f'Population {population_index} must define exactly one tracer method.')

    if len(tracer_methods) != 1:
        raise ConfigurationError(
            f'Population {population_index} defines multiple tracer methods '
            f'({", ".join(tracer_methods.keys())}); only one method per population is supported.'
        )

    method_name, method_config = next(iter(tracer_methods.items()))
    if method_name not in SUPPORTED_TRACER_METHODS:
        raise ConfigurationError(
            f'Population {population_index} uses unsupported tracer method {method_name!r}. '
            f'Supported methods: {", ".join(sorted(SUPPORTED_TRACER_METHODS))}.'
        )

    if not isinstance(method_config, Mapping):
        raise ConfigurationError(f'Population {population_index} tracer method {method_name!r} must be a mapping.')

    flow_field_names = _get_flow_field_names(population_index, method_name, method_config)
    transport_probability_method = population_config.get(
        'transport_probability', DEFAULT_TRANSPORT_PROBABILITY_METHOD
    )
    physics_config = build_physics_config(base_physics_config, population_config, method_name, method_config)
    tracer_config = {method_name: dict(method_config)}
    converter = PhysicsConverter(physics_config, tracer_config)

    return PopulationRuntimePlan(
        population_index=population_index,
        population_config=population_config,
        population=population,
        tracer=TracerRuntimePlan(
            method_name=method_name,
            method_config=method_config,
            flow_field_names=flow_field_names,
            transport_probability_method=transport_probability_method,
            required_physics_fields=required_physics_fields(method_name, flow_field_names, method_config),
            converter=converter,
        ),
    )


def build_physics_config(
    base_physics_config: PhysicsConfig | Mapping[str, Any],
    population_config: Mapping[str, Any],
    method_name: str,
    method_config: Mapping[str, Any],
) -> PhysicsConfig:
    """
    Build method-specific physics config for one population.

    Parameters
    ----------
    base_physics_config : PhysicsConfig | Mapping[str, Any]
        Base physics configuration shared by populations.
    population_config : Mapping[str, Any]
        Configuration for a single particle population.
    method_name : str
        Name of the tracer or physics method.
    method_config : Mapping[str, Any]
        Configuration mapping for the selected method.

    Returns
    -------
    PhysicsConfig
        Constructed physics configuration.
    """

    base_config = _physics_config_to_dict(base_physics_config)
    base_config['tracer_method'] = method_name

    characteristics = population_config.get('characteristics', {})
    if isinstance(characteristics, Mapping):
        if 'density' in characteristics:
            base_config['particle_density'] = characteristics['density']
        if 'grain_size' in characteristics:
            base_config['grain_diameter'] = characteristics['grain_size']
        elif 'size' in characteristics:
            base_config['grain_diameter'] = characteristics['size']

    return PhysicsConfig.from_dict(config=base_config, tracer_config={method_name: dict(method_config)})


def required_physics_fields(
    method_name: str,
    flow_field_names: Sequence[str],
    method_config: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """
    Return physics fields that must be preserved for a method plan.

    Parameters
    ----------
    method_name : str
        Name of the tracer or physics method.
    flow_field_names : Sequence[str]
        Flow-field names required by the runtime plans.

    Returns
    -------
    tuple[str, ...]
        Tuple containing the computed values.
    """

    if method_name == 'vanwesten':
        return tuple(
            _unique_preserving_order(
                (
                    *flow_field_names,
                    'mixing_layer_thickness',
                    *(flow_field_name.replace('velocity', 'probability') for flow_field_name in flow_field_names),
                )
            )
        )

    if method_name == 'soulsby':
        return tuple(_unique_preserving_order((*flow_field_names, 'mixing_layer_thickness', 'soulsby_a', 'soulsby_b')))

    if method_name == 'macdonald':
        method_config = method_config or {}
        computation_type = str(method_config.get('computationType', '2D')).upper()
        fields = (
            *flow_field_names,
            'mixing_layer_thickness',
            'particle_advection_velocity',
            'max_shear_velocity',
            'mean_shear_velocity',
            'selected_shear_velocity',
            'selected_bed_shear_stress',
            'rouse_number',
            'skin_roughness_height',
            'profile_roughness_height',
            'total_transport_centroid_elevation',
            'effective_chezy_coefficient',
            'chezy_current_shear_velocity',
            'chezy_equivalent_roughness_height',
        )
        entrainment_config = method_config.get('entrainment', {}) or {}
        entrainment_method = str(entrainment_config.get('method', 'shields_threshold')).lower().replace('-', '_')
        if entrainment_method == 'entrainment_frequency':
            fields = (*fields, 'macdonald_entrainment_frequency')
        if computation_type == 'Q3D':
            fields = (
                *fields,
                'q3d_velocity_deficit_coefficient',
                'q3d_vertical_velocity_gradient',
                'turbulent_shields_number',
                'q3d_entrainment_height_above_bed',
            )
        return tuple(_unique_preserving_order(fields))
    if method_name == 'passive_tracer':
        return tuple(_unique_preserving_order(flow_field_names))

    raise ConfigurationError(f'Unsupported tracer method {method_name!r}.')


def _get_flow_field_names(
    population_index: int, method_name: str, method_config: Mapping[str, Any]
) -> tuple[str, ...]:
    flow_field_names = method_config.get('flow_field_name')
    if method_name == 'passive_tracer' and flow_field_names is None:
        return DEFAULT_PASSIVE_TRACER_FLOW_FIELDS

    if not isinstance(flow_field_names, Sequence) or isinstance(flow_field_names, str) or not flow_field_names:
        raise ConfigurationError(
            f'Population {population_index} tracer method {method_name!r} must define a non-empty '
            '`flow_field_name` list.'
        )

    if not all(isinstance(flow_field_name, str) and flow_field_name for flow_field_name in flow_field_names):
        raise ConfigurationError(
            f'Population {population_index} tracer method {method_name!r} has invalid flow field names.'
        )

    return tuple(flow_field_names)


def _physics_config_to_dict(config: PhysicsConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, PhysicsConfig):
        return asdict(config)
    if is_dataclass(config):
        return asdict(config)
    return dict(config)


def _unique_preserving_order(values: Sequence[str] | Any) -> list[str]:
    unique_values = []
    seen = set()
    for value in values:
        if value not in seen:
            unique_values.append(value)
            seen.add(value)
    return unique_values


def _shallow_sedtrails_data_clone(sedtrails_data: Any) -> Any:
    cloned_data = copy.copy(sedtrails_data)
    cloned_data._physics_fields = {}
    for field_name, value in getattr(sedtrails_data, '_physics_fields', {}).items():
        cloned_data.add_physics_field(field_name, _copy_physics_value(value))
    return cloned_data


def _copy_physics_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _copy_physics_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if hasattr(value, 'copy'):
        return value.copy()
    return copy.deepcopy(value)
