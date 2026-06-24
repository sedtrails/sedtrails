"""
SedTRAILS particle seeding example.

This script demonstrates how a population configuration dictionary becomes a
ParticlePopulation through ParticleSeeder.

The real simulation manager builds these dictionaries from a YAML file. Here we
write them directly so the expected structure is visible.
"""

from dataclasses import dataclass

import numpy as np

from sedtrails.particle_tracer import ParticleSeeder


@dataclass
class ExampleFieldData:
    """
    Minimal object accepted by ParticleSeeder.seed().

    Real SedtrailsData objects also expose x and y coordinate arrays. The seeder
    uses these coordinates to build grid geometry and check whether seed points
    are inside the input model domain.
    """

    x: np.ndarray
    y: np.ndarray
    reference_date: np.datetime64


# ===== STEP 1: Create minimal field coordinates =====

field_data = ExampleFieldData(
    x=np.array([0.0, 10.0, 10.0, 0.0]),
    y=np.array([0.0, 0.0, 10.0, 10.0]),
    reference_date=np.datetime64('2025-06-18T00:00:00'),
)


# ===== STEP 2: Define population configurations =====

point_population_config = {
    'name': 'point_release',
    'particle_type': 'passive',
    'tracer_methods': {
        'passive_tracer': {
            'flow_field_name': ['depth_avg_flow_velocity'],
        }
    },
    'transport_probability': 'no_probability',
    'seeding': {
        'burial_depth': {'constant': 0.0},
        'release_start': '2025-06-18 01:00:00',
        'quantity': 2,
        'strategy': {
            'point': {
                'locations': ['5.0,5.0'],
            }
        },
    },
}

random_population_config = {
    'name': 'random_release',
    'particle_type': 'passive',
    'tracer_methods': {
        'passive_tracer': {
            'flow_field_name': ['depth_avg_flow_velocity'],
        }
    },
    'transport_probability': 'no_probability',
    'seeding': {
        'burial_depth': {'constant': 0.0},
        'quantity': 1,
        'strategy': {
            'random': {
                'bbox': '2.0,2.0 8.0,8.0',
                'seed': 42,
                'nlocations': 3,
            }
        },
    },
}


# ===== STEP 3: Seed the configured populations =====

seeder = ParticleSeeder([point_population_config, random_population_config])
populations = seeder.seed(field_data)


# ===== STEP 4: Inspect the created particle arrays =====

for population in populations:
    config = population.population_config.population_config
    particles = population.particles

    print(f'\nPopulation: {config["name"]}')
    print(f'Particle type: {config["particle_type"]}')
    print(f'Number of particles: {len(particles["x"])}')
    print(f'x coordinates: {particles["x"]}')
    print(f'y coordinates: {particles["y"]}')
    print(f'release time, seconds since reference date: {particles["release_time"]}')
    print(f'burial depth: {particles["burial_depth"]}')
