from sedtrails.particle_tracer import PopulationConfig, ParticlePopulation
import numpy as np

# Example usage of the ParticleFactory  to create population of particles
config_random = PopulationConfig(
    {
        'population': {
            'particle_type': 'sand',
            'seeding': {
                'strategy': {'random': {'bbox': '1.0,2.0, 3.0,4.0', 'nlocations': 1, 'seed': 42}},
                'quantity': 2,
                'release_start': '2025-06-18 13:00:00',
            },
        }
    }
)


population = ParticlePopulation(
    field_x=np.array([0.0, 1.0, 2.5, 5.0]),
    field_y=np.array([0.0, 1.0, 2.0, 3.0]),
    population_config=config_random,
)

print(population.particles)  # Output the x-coordinates of the particles
