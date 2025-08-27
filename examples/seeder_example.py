from sedtrails.particle_tracer import PopulationConfig, ParticlePopulation
import numpy as np

# Example usage of the ParticleFactory  to create population of particles
config_random = PopulationConfig(
    {
        'name': 'my_population1',
        'particle_type': 'sand',
        'characteristics': {'grain_size': 0.01, 'density': 2650.0},
        'tracer_methods': {'vanwesten': {'beta': 0.3}},
        'seeding': {
            'burial_depth': {'constant': 3.0},  # constant burial depth in meters
            'release_start': '2025-06-18 13:00:00',
            'quantity': 1,
            'strategy': {'point': {'locations': ['40000,17000']}},
        },
    }
)


population = ParticlePopulation(
    field_x=np.array([0.0, 1.0, 2.5, 5.0]),
    field_y=np.array([0.0, 1.0, 2.0, 3.0]),
    population_config=config_random,
)

print(population.particles)  # Output the x-coordinates of the particles
