"""
Unit tests for particle seeding strategies.
"""

import pytest
from sedtrails.particle_tracer.particle_seeder import (
    SeedingConfig,
    PointStrategy,
    RandomStrategy,
    GridStrategy,
    TransectStrategy,
    ParticleFactory,
)
from sedtrails.exceptions import MissingConfigurationParameter


# Strategy fixtures
@pytest.fixture
def point_strategy():
    return PointStrategy()


@pytest.fixture
def random_strategy():
    return RandomStrategy()


@pytest.fixture
def grid_strategy():
    return GridStrategy()


@pytest.fixture
def transect_strategy():
    return TransectStrategy()


# Config fixtures
@pytest.fixture
def point_config_basic():
    return SeedingConfig(
        {
            'population': {
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'point': {'locations': ['1.0,2.0', '3.0,4.0']}},
                    'quantity': 10,
                    'release_start': '2025-06-18 13:00:00',
                },
            }
        }
    )


@pytest.fixture
def point_config_simple():
    return SeedingConfig(
        {
            'population': {
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'point': {'locations': ['0,0']}},
                    'quantity': 1,
                    'release_start': '2025-06-18 13:00:00',
                },
            }
        }
    )


@pytest.fixture
def point_config_dual():
    return SeedingConfig(
        {
            'population': {
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'point': {'locations': ['1.0,2.0', '3.0,4.0']}},
                    'quantity': 2,
                    'release_start': '2025-06-18 13:00:00',
                },
            }
        }
    )


@pytest.fixture
def random_config():
    return SeedingConfig(
        {
            'population': {
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'random': {'bbox': '1.0,2.0, 3.0,4.0', 'seed': 42}},
                    'quantity': 5,
                    'release_start': '2025-06-18 13:00:00',
                },
            }
        }
    )


@pytest.fixture
def grid_config():
    return SeedingConfig(
        {
            'population': {
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {
                        'grid': {
                            'separation': {'dx': 1.0, 'dy': 1.0},
                            'bbox': {'xmin': 0.0, 'xmax': 2.0, 'ymin': 0.0, 'ymax': 2.0},
                        }
                    },
                    'quantity': 2,
                    'release_start': '2025-06-18 13:00:00',
                },
            }
        }
    )


@pytest.fixture
def grid_config_single():
    return SeedingConfig(
        {
            'population': {
                'particle_type': 'mud',
                'seeding': {
                    'strategy': {
                        'grid': {
                            'separation': {'dx': 1.0, 'dy': 1.0},
                            'bbox': {'xmin': 0.0, 'xmax': 1.0, 'ymin': 0.0, 'ymax': 1.0},
                        }
                    },
                    'quantity': 1,
                    'release_start': '2025-06-18 13:00:00',
                },
            }
        }
    )


@pytest.fixture
def transect_config():
    return SeedingConfig(
        {
            'population': {
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {
                        'transect': {
                            'segments': ['0,0 2,0'],
                            'k': 3,
                        }
                    },
                    'quantity': 5,
                    'release_start': '2025-06-18 13:00:00',
                },
            }
        }
    )


@pytest.fixture
def transect_config_multi():
    return SeedingConfig(
        {
            'population': {
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {
                        'transect': {
                            'segments': ['0,0 1,0', '1,0 1,1'],
                            'k': 2,
                        }
                    },
                    'quantity': 1,
                    'release_start': '2025-06-18 13:00:00',
                },
            }
        }
    )


# Particle classes fixture
@pytest.fixture
def particle_classes():
    from sedtrails.particle_tracer.particle import Sand, Mud, Passive

    return {'Sand': Sand, 'Mud': Mud, 'Passive': Passive}


class TestPointStrategy:
    """Test cases for PointStrategy."""

    def test_point_strategy(self, point_strategy, point_config_basic):
        """Test basic point strategy functionality."""
        result = point_strategy.seed(point_config_basic)

        assert len(result) == 2
        assert result[0] == (10, 1.0, 2.0)
        assert result[1] == (10, 3.0, 4.0)

    def test_point_strategy_missing_locations(self, point_strategy):
        """Test point strategy with missing locations."""
        # Since SeedingConfig validates that strategy settings exist,
        # we need to create a config that passes validation but missing locations
        config = SeedingConfig(
            {
                'population': {
                    'particle_type': 'sand',
                    'seeding': {
                        'strategy': {'point': {'not_locations': 'invalid'}},
                        'quantity': 10,
                        'release_start': '2025-06-18 13:00:00',
                    },
                }
            }
        )

        with pytest.raises(MissingConfigurationParameter, match='"locations" must be provided'):
            point_strategy.seed(config)

    def test_point_strategy_invalid_location_format(self, point_strategy):
        """Test point strategy with invalid location format."""
        config = SeedingConfig(
            {
                'population': {
                    'particle_type': 'sand',
                    'seeding': {
                        'strategy': {'point': {'locations': ['invalid_format']}},
                        'quantity': 10,
                        'release_start': '2025-06-18 13:00:00',
                    },
                }
            }
        )

        with pytest.raises(ValueError, match='Invalid location string'):
            point_strategy.seed(config)


class TestRandomStrategy:
    """Test cases for RandomStrategy."""

    def test_random_strategy(self, random_strategy, random_config):
        """Test basic random strategy functionality."""
        result = random_strategy.seed(random_config)

        assert len(result) == 5
        # Check all particles have quantity 5 and coordinates within bounds
        for qty, x, y in result:
            assert qty == 5
            assert 1.0 <= x <= 3.0
            assert 2.0 <= y <= 4.0

    def test_random_strategy_missing_bbox(self, random_strategy):
        """Test random strategy with missing bounding box."""
        # Since SeedingConfig validates that strategy settings exist,
        # we need to create a config that passes validation but missing bbox
        config = SeedingConfig(
            {
                'population': {
                    'particle_type': 'sand',
                    'seeding': {
                        'strategy': {'random': {'not_bbox': 'invalid'}},
                        'quantity': 5,
                        'release_start': '2025-06-18 13:00:00',
                    },
                }
            }
        )

        with pytest.raises(MissingConfigurationParameter, match='"bbox" must be provided'):
            random_strategy.seed(config)


class TestGridStrategy:
    """Test cases for GridStrategy."""

    def test_grid_strategy(self, grid_strategy, grid_config):
        """Test basic grid strategy functionality."""
        result = grid_strategy.seed(grid_config)

        # Should generate a 3x3 grid (0,1,2 in both directions)
        assert len(result) == 9
        # Check first and last points
        assert (2, 0.0, 0.0) in result
        assert (2, 2.0, 2.0) in result

    def test_grid_strategy_no_bbox(self, grid_strategy):
        """Test grid strategy without bounding box."""
        config = SeedingConfig(
            {
                'population': {
                    'particle_type': 'sand',
                    'seeding': {
                        'strategy': {
                            'grid': {
                                'separation': {'dx': 1.0, 'dy': 1.0},
                                'not_bbox': 'invalid',  # Missing bbox
                            }
                        },
                        'quantity': 2,
                        'release_start': '2025-06-18 13:00:00',
                    },
                }
            }
        )
        with pytest.raises(RuntimeError, match='Bounding box must be provided'):
            grid_strategy.seed(config)

    def test_grid_strategy_missing_separation(self, grid_strategy):
        """Test grid strategy with missing separation parameters."""
        config = SeedingConfig(
            {
                'population': {
                    'particle_type': 'sand',
                    'seeding': {
                        'strategy': {
                            'grid': {
                                'bbox': {'xmin': 0.0, 'xmax': 2.0, 'ymin': 0.0, 'ymax': 2.0},
                                'not_separation': 'invalid',
                            }
                        },
                        'quantity': 2,
                        'release_start': '2025-06-18 13:00:00',
                    },
                }
            }
        )

        with pytest.raises(MissingConfigurationParameter, match='"grid" must be provided'):
            grid_strategy.seed(config)


class TestTransectStrategy:
    """Test cases for TransectStrategy."""

    def test_transect_strategy(self, transect_strategy, transect_config):
        """Test basic transect strategy functionality."""
        result = transect_strategy.seed(transect_config)

        # Should generate 3 points along the line from (0,0) to (2,0)
        assert len(result) == 3
        assert result[0] == (5, 0.0, 0.0)  # Start point
        assert result[1] == (5, 1.0, 0.0)  # Middle point
        assert result[2] == (5, 2.0, 0.0)  # End point

    def test_transect_strategy_multiple_segments(self, transect_strategy, transect_config_multi):
        """Test transect strategy with multiple segments."""
        result = transect_strategy.seed(transect_config_multi)

        # Should generate 2 points per segment = 4 total points
        assert len(result) == 4
        # First segment: (0,0) to (1,0)
        assert (1, 0.0, 0.0) in result
        assert (1, 1.0, 0.0) in result
        # Second segment: (1,0) to (1,1)
        assert (1, 1.0, 0.0) in result
        assert (1, 1.0, 1.0) in result

    def test_transect_strategy_missing_segments(self, transect_strategy):
        """Test transect strategy with missing segments."""
        config = SeedingConfig(
            {
                'population': {
                    'particle_type': 'sand',
                    'seeding': {
                        'strategy': {'transect': {'k': 3}},
                        'quantity': 5,
                        'release_start': '2025-06-18 13:00:00',
                    },
                }
            }
        )

        with pytest.raises(MissingConfigurationParameter, match='"segments" must be provided'):
            transect_strategy.seed(config)

    def test_transect_strategy_invalid_segment_format(self, transect_strategy):
        """Test transect strategy with invalid segment format."""
        config = SeedingConfig(
            {
                'population': {
                    'particle_type': 'sand',
                    'seeding': {
                        'strategy': {
                            'transect': {
                                'segments': ['invalid_format'],
                                'k': 2,
                            }
                        },
                        'quantity': 1,
                        'release_start': '2025-06-18 13:00:00',
                    },
                }
            }
        )

        with pytest.raises(ValueError, match='Invalid segment string'):
            transect_strategy.seed(config)


class TestParticleFactory:
    """Test cases for ParticleFactory."""

    def test_create_particles_point_strategy(self, particle_classes):
        """Test particle creation with PointStrategy."""
        Sand = particle_classes['Sand']

        config = SeedingConfig(
            {
                'population': {
                    'particle_type': 'sand',
                    'seeding': {
                        'strategy': {'point': {'locations': ['1.0,2.0', '3.0,4.0']}},
                        'quantity': 2,
                        'release_start': '2025-06-18 13:00:00',
                    },
                }
            }
        )

        particles = ParticleFactory.create_particles(config)

        # Should create 2 particles per location (2 locations * 2 particles = 4 total)
        assert len(particles) == 4
        # Check all particles are Sand type
        assert all(isinstance(p, Sand) for p in particles)
        # Check positions
        positions = [(p.x, p.y) for p in particles]
        assert positions.count((1.0, 2.0)) == 2  # 2 particles at first location
        assert positions.count((3.0, 4.0)) == 2  # 2 particles at second location
        # Check release times
        assert all(p.release_time == '2025-06-18 13:00:00' for p in particles)

    def test_create_particles_grid_strategy(self, particle_classes):
        """Test particle creation with GridStrategy."""
        Mud = particle_classes['Mud']

        config = SeedingConfig(
            {
                'population': {
                    'particle_type': 'mud',
                    'seeding': {
                        'strategy': {
                            'grid': {
                                'separation': {'dx': 1.0, 'dy': 1.0},
                                'bbox': {'xmin': 0.0, 'xmax': 1.0, 'ymin': 0.0, 'ymax': 1.0},
                            }
                        },
                        'quantity': 1,
                        'release_start': '2025-06-18 13:00:00',
                    },
                }
            }
        )

        particles = ParticleFactory.create_particles(config)

        # Should create 4 particles (2x2 grid)
        assert len(particles) == 4
        # Check all particles are Mud type
        assert all(isinstance(p, Mud) for p in particles)
        # Check positions include corners
        positions = [(p.x, p.y) for p in particles]
        assert (0.0, 0.0) in positions
        assert (1.0, 1.0) in positions

    def test_create_particles_different_particle_types(self, particle_classes):
        """Test creating different particle types."""
        Sand, Mud, Passive = particle_classes['Sand'], particle_classes['Mud'], particle_classes['Passive']

        # Test Sand particles
        sand_config = SeedingConfig(
            {
                'population': {
                    'particle_type': 'sand',
                    'seeding': {
                        'strategy': {'point': {'locations': ['0,0']}},
                        'quantity': 1,
                        'release_start': '2025-06-18 13:00:00',
                    },
                }
            }
        )
        sand_particles = ParticleFactory.create_particles(sand_config)
        assert len(sand_particles) == 1
        assert isinstance(sand_particles[0], Sand)

        # Test Mud particles
        mud_config = SeedingConfig(
            {
                'population': {
                    'particle_type': 'mud',
                    'seeding': {
                        'strategy': {'point': {'locations': ['0,0']}},
                        'quantity': 1,
                        'release_start': '2025-06-18 13:00:00',
                    },
                }
            }
        )
        mud_particles = ParticleFactory.create_particles(mud_config)
        assert len(mud_particles) == 1
        assert isinstance(mud_particles[0], Mud)

        # Test Passive particles
        passive_config = SeedingConfig(
            {
                'population': {
                    'particle_type': 'passive',
                    'seeding': {
                        'strategy': {'point': {'locations': ['0,0']}},
                        'quantity': 1,
                        'release_start': '2025-06-18 13:00:00',
                    },
                }
            }
        )
        passive_particles = ParticleFactory.create_particles(passive_config)
        assert len(passive_particles) == 1
        assert isinstance(passive_particles[0], Passive)

    def test_create_particles_invalid_particle_type(self):
        """Test error handling for invalid particle type."""
        config = SeedingConfig(
            {
                'population': {
                    'particle_type': 'invalid_type',
                    'seeding': {
                        'strategy': {'point': {'locations': ['0,0']}},
                        'quantity': 1,
                        'release_start': '2025-06-18 13:00:00',
                    },
                }
            }
        )

        with pytest.raises(ValueError, match='Unknown particle type'):
            ParticleFactory.create_particles(config)

    def test_create_particles_release_time_set(self):
        """Test that release time is set correctly."""
        config = SeedingConfig(
            {
                'population': {
                    'particle_type': 'sand',
                    'seeding': {
                        'strategy': {'point': {'locations': ['0,0']}},
                        'quantity': 1,
                        'release_start': '2025-06-18 13:00:00',
                    },
                }
            }
        )
        particles = ParticleFactory.create_particles(config)

        # Should have the correct release time
        assert particles[0].release_time == '2025-06-18 13:00:00'
