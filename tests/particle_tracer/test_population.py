"""
Tests for the population.py module in the sedtrails package.
"""

import pytest
from sedtrails.particle_tracer import PopulationConfig, ParticlePopulation
import numpy as np


@pytest.fixture
def population_config():
    return PopulationConfig(
        {
            'population': {
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'random': {'bbox': '1.0,2.0, 3.0,4.0', 'nlocations': 2, 'seed': 42}},
                    'quantity': 5,
                    'release_start': '2025-06-18 13:00:00',
                },
            }
        }
    )


class TestParticlePopulation:
    def test_create_population(self, population_config):
        """Test creating a ParticlePopulation with a valid configuration."""
        population = ParticlePopulation(
            field_x=np.array([0.0, 1.0, 2.5, 5.0]),
            field_y=np.array([0.0, 1.0, 2.0, 3.0]),
            population_config=population_config,
        )
        assert population is not None
        assert len(population.particles['x']) == 10  # 2 nlocations * 5 quantity
        assert len(population.particles['y']) == 10  # 2 nlocations * 5 quantity
