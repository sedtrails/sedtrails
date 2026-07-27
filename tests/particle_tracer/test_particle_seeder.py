"""
Unit tests for particle seeding strategies.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from sedtrails.exceptions import MissingConfigurationParameter
from sedtrails.exceptions.exceptions import ConfigurationError, DateFormatError
from sedtrails.particle_tracer.particle_seeder import (
    FilePointsStrategy,
    GridStrategy,
    ParticleFactory,
    ParticlePopulation,
    ParticleSeeder,
    PointStrategy,
    PopulationConfig,
    RandomStrategy,
    TransectStrategy,
    _compute_seeding_area,
    _log_seeding_box_volume,
    _parse_polygon,
    _read_polygon_file,
)


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


@pytest.fixture
def file_points_strategy():
    return FilePointsStrategy()


# Config fixtures
@pytest.fixture
def point_config_basic():
    return PopulationConfig(
        {
            'name': 'Basic Point Config',
            'particle_type': 'sand',
            'seeding': {
                'strategy': {'point': {'locations': ['1.0,2.0', '3.0,4.0']}},
                'quantity': 10,
                'release_start': '2025-06-18 13:00:00',
                'burial_depth': {
                    'constant': 1.0,
                },
            },
        }
    )


@pytest.fixture
def point_config_simple():
    return PopulationConfig(
        {
            'name': 'Basic Point Config',
            'particle_type': 'sand',
            'seeding': {
                'strategy': {'point': {'locations': ['0,0']}},
                'quantity': 1,
                'release_start': '2025-06-18 13:00:00',
                'burial_depth': {
                    'constant': 1.0,
                },
            },
        }
    )


@pytest.fixture
def point_config_dual():
    return PopulationConfig(
        {
            'name': 'Basic Point Config',
            'particle_type': 'sand',
            'seeding': {
                'strategy': {'point': {'locations': ['1.0,2.0', '3.0,4.0']}},
                'quantity': 2,
                'release_start': '2025-06-18 13:00:00',
                'burial_depth': {
                    'constant': 1.0,
                },
            },
        }
    )


@pytest.fixture
def random_config():
    return PopulationConfig(
        {
            'name': 'Basic Point Config',
            'particle_type': 'sand',
            'seeding': {
                'strategy': {'random': {'bbox': '1.0,2.0, 3.0,4.0', 'nlocations': 2, 'seed': 42}},
                'quantity': 5,
                'release_start': '2025-06-18 13:00:00',
                'burial_depth': {
                    'constant': 1.0,
                },
            },
        }
    )


@pytest.fixture
def grid_config():
    return PopulationConfig(
        {
            'name': 'Basic Grid Config',
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
                'burial_depth': {
                    'constant': 1.0,
                },
            },
        }
    )


@pytest.fixture
def grid_config_single():
    return PopulationConfig(
        {
            'name': 'Single Grid Config',
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
                'burial_depth': {
                    'constant': 1.0,
                },
            },
        }
    )


@pytest.fixture
def transect_config():
    return PopulationConfig(
        {
            'name': 'Basic Point Config',
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
                'burial_depth': {
                    'constant': 1.0,
                },
            },
        }
    )


@pytest.fixture
def transect_config_multi():
    return PopulationConfig(
        {
            'name': 'Basic Point Config',
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
                'burial_depth': {
                    'constant': 1.0,
                },
            },
        }
    )


@pytest.fixture
def file_points_config_basic(tmp_path):
    """Basic file_points config with CSV file."""
    # Create a temporary CSV file
    csv_file = tmp_path / 'test_points.csv'
    csv_file.write_text('x,y\n1.0,2.0\n3.0,4.0\n5.0,6.0\n')

    return PopulationConfig(
        {
            'name': 'File Points Config',
            'particle_type': 'sand',
            'seeding': {
                'strategy': {
                    'file_points': {
                        'path': str(csv_file),
                        'x_col': 'x',
                        'y_col': 'y',
                        'has_header': True,
                    }
                },
                'quantity': 2,
                'release_start': '2025-06-18 13:00:00',
                'burial_depth': {
                    'constant': 1.0,
                },
            },
        }
    )


@pytest.fixture
def file_points_config_no_header(tmp_path):
    """File_points config with no header."""
    # Create a temporary file without header
    txt_file = tmp_path / 'test_points_no_header.txt'
    txt_file.write_text('1.0 2.0\n3.0 4.0\n')

    return PopulationConfig(
        {
            'name': 'File Points Config No Header',
            'particle_type': 'mud',
            'seeding': {
                'strategy': {
                    'file_points': {
                        'path': str(txt_file),
                        'x_col': 0,
                        'y_col': 1,
                        'has_header': False,
                    }
                },
                'quantity': 1,
                'release_start': '2025-06-18 13:00:00',
                'burial_depth': {
                    'constant': 1.0,
                },
            },
        }
    )


@pytest.fixture
def file_points_config_with_bbox(tmp_path):
    """File_points config with bounding box filtering."""
    # Create a file with points both inside and outside bbox
    csv_file = tmp_path / 'test_points_bbox.csv'
    csv_file.write_text('longitude,latitude\n1.0,1.0\n2.0,2.0\n5.0,5.0\n10.0,10.0\n')

    return PopulationConfig(
        {
            'name': 'File Points Config With BBox',
            'particle_type': 'passive',
            'seeding': {
                'strategy': {
                    'file_points': {
                        'path': str(csv_file),
                        'x_col': 'longitude',
                        'y_col': 'latitude',
                        'has_header': True,
                        'bbox': '0,0 3,3',  # Only first two points should be kept
                    }
                },
                'quantity': 3,
                'release_start': '2025-06-18 13:00:00',
                'burial_depth': {
                    'constant': 1.0,
                },
            },
        }
    )


# Particle classes fixture
@pytest.fixture
def particle_classes():
    from sedtrails.particle_tracer.particle import Mud, Passive, Sand

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
        # Since PopulationConfig validates that strategy settings exist,
        # we need to create a config that passes validation but missing locations
        config = PopulationConfig(
            {
                'name': 'Basic Point Config',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'point': {'not_locations': 'invalid'}},
                    'quantity': 10,
                    'release_start': '2025-06-18 13:00:00',
                    'burial_depth': {
                        'constant': 1.0,
                    },
                },
            }
        )

        with pytest.raises(MissingConfigurationParameter, match='"locations" must be provided'):
            point_strategy.seed(config)

    def test_point_strategy_invalid_location_format(self, point_strategy):
        """Test point strategy with invalid location format."""
        config = PopulationConfig(
            {
                'name': 'Basic Point Config',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'point': {'locations': ['invalid_format']}},
                    'quantity': 10,
                    'release_start': '2025-06-18 13:00:00',
                    'burial_depth': {
                        'constant': 1.0,
                    },
                },
            }
        )

        with pytest.raises(ValueError, match='Invalid location string'):
            point_strategy.seed(config)


class TestRandomStrategy:
    """Test cases for RandomStrategy."""

    def test_random_strategy(self, random_strategy, random_config):
        """Test basic random strategy functionality."""
        result = random_strategy.seed(random_config)

        assert len(result) == 2  # 2 nlocations
        # Check all particles have quantity 5 and coordinates within bounds
        for qty, x, y in result:
            assert qty == 5
            assert 1.0 <= x <= 3.0
            assert 2.0 <= y <= 4.0

    def test_random_strategy_missing_bbox(self, random_strategy):
        """Test random strategy with missing bounding box."""
        # Since PopulationConfig validates that strategy settings exist,
        # we need to create a config that passes validation but missing bbox
        config = PopulationConfig(
            {
                'name': 'Basic Point Config',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'random': {'not_bbox': 'invalid', 'nlocations': 1}},
                    'quantity': 5,
                    'release_start': '2025-06-18 13:00:00',
                    'burial_depth': {
                        'constant': 1.0,
                    },
                },
            }
        )

        with pytest.raises(MissingConfigurationParameter, match='"bbox" or "poly" must be provided'):
            random_strategy.seed(config)


class TestGridStrategy:
    """Test cases for GridStrategy."""

    def test_grid_strategy(self, grid_strategy, grid_config):
        """Test basic grid strategy functionality."""
        result = grid_strategy.seed(grid_config)

        # Should generate a 3x3 grid (0, 1, 2 in both directions)
        # Points: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)
        assert len(result) == 9

        # Check that all points have the correct quantity
        assert all(qty == 2 for qty, *_ in result)

        # Check specific points
        positions = [(x, y) for _, x, y in result]
        assert (0.0, 0.0) in positions
        assert (1.0, 1.0) in positions
        assert (2.0, 2.0) in positions
        assert (0.0, 2.0) in positions  # Top-left
        assert (2.0, 0.0) in positions  # Bottom-right

    def test_grid_strategy_single_point(self, grid_strategy, grid_config_single):
        """Test grid strategy with a single grid point."""
        result = grid_strategy.seed(grid_config_single)

        # Should generate a 2x2 grid: (0,0), (0,1), (1,0), (1,1)
        assert len(result) == 4

        # Check positions
        positions = [(x, y) for _, x, y in result]
        expected_positions = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
        assert set(positions) == set(expected_positions)

    def test_grid_strategy_no_bbox(self, grid_strategy):
        """Test grid strategy without bounding box."""
        config = PopulationConfig(
            {
                'name': 'Grid Config No BBox',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {
                        'grid': {
                            'separation': {'dx': 1.0, 'dy': 1.0},
                            # Missing bbox
                        }
                    },
                    'quantity': 2,
                    'release_start': '2025-06-18 13:00:00',
                    'burial_depth': {
                        'constant': 1.0,
                    },
                },
            }
        )
        with pytest.raises(MissingConfigurationParameter, match='"bbox" or "poly" must be provided'):
            grid_strategy.seed(config)

    def test_grid_strategy_missing_separation(self, grid_strategy):
        """Test grid strategy with missing separation parameters."""
        config = PopulationConfig(
            {
                'name': 'Grid Config Missing Separation',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {
                        'grid': {
                            'bbox': {'xmin': 0.0, 'xmax': 2.0, 'ymin': 0.0, 'ymax': 2.0},
                            # Missing separation
                        }
                    },
                    'quantity': 2,
                    'release_start': '2025-06-18 13:00:00',
                    'burial_depth': {
                        'constant': 1.0,
                    },
                },
            }
        )

        with pytest.raises(MissingConfigurationParameter, match='"separation" with "dx" and "dy" must be provided'):
            grid_strategy.seed(config)

    def test_grid_strategy_string_bbox(self, grid_strategy):
        """Test grid strategy with string bbox format."""
        config = PopulationConfig(
            {
                'name': 'Grid Config String BBox',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {
                        'grid': {
                            'separation': {'dx': 0.5, 'dy': 0.5},
                            'bbox': '0,0 1,1',  # String format
                        }
                    },
                    'quantity': 1,
                    'release_start': '2025-06-18 13:00:00',
                    'burial_depth': {
                        'constant': 1.0,
                    },
                },
            }
        )

        result = grid_strategy.seed(config)

        # Should generate a 3x3 grid (0, 0.5, 1.0 in both directions)
        assert len(result) == 9
        positions = [(x, y) for _, x, y in result]
        assert (0.0, 0.0) in positions
        assert (0.5, 0.5) in positions
        assert (1.0, 1.0) in positions


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
        config = PopulationConfig(
            {
                'name': 'Basic Point Config',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'transect': {'k': 3}},
                    'quantity': 5,
                    'release_start': '2025-06-18 13:00:00',
                    'burial_depth': {
                        'constant': 1.0,
                    },
                },
            }
        )

        with pytest.raises(MissingConfigurationParameter, match='"segments" must be provided'):
            transect_strategy.seed(config)

    def test_transect_strategy_invalid_segment_format(self, transect_strategy):
        """Test transect strategy with invalid segment format."""
        config = PopulationConfig(
            {
                'name': 'Basic Point Config',
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
                    'burial_depth': {
                        'constant': 1.0,
                    },
                },
            }
        )

        with pytest.raises(ValueError, match='Invalid segment string'):
            transect_strategy.seed(config)


class TestFilePointsStrategy:
    """Test cases for FilePointsStrategy."""

    def test_file_points_strategy_basic(self, file_points_strategy, file_points_config_basic):
        """Test basic file_points strategy functionality."""
        result = file_points_strategy.seed(file_points_config_basic)

        # Should generate 3 points from CSV file
        assert len(result) == 3
        assert result[0] == (2, 1.0, 2.0)
        assert result[1] == (2, 3.0, 4.0)
        assert result[2] == (2, 5.0, 6.0)

    def test_file_points_strategy_no_header(self, file_points_strategy, file_points_config_no_header):
        """Test file_points strategy with no header."""
        result = file_points_strategy.seed(file_points_config_no_header)

        # Should generate 2 points from text file
        assert len(result) == 2
        assert result[0] == (1, 1.0, 2.0)
        assert result[1] == (1, 3.0, 4.0)

    def test_file_points_strategy_with_bbox(self, file_points_strategy, file_points_config_with_bbox):
        """Test file_points strategy with bounding box filtering."""
        result = file_points_strategy.seed(file_points_config_with_bbox)

        # Should generate 2 points (only those within bbox 0,0 3,3)
        assert len(result) == 2
        assert result[0] == (3, 1.0, 1.0)
        assert result[1] == (3, 2.0, 2.0)
        # Points (5.0,5.0) and (10.0,10.0) should be filtered out by bbox

    def test_file_points_strategy_stride(self, file_points_strategy, tmp_path):
        """Test file_points strategy with stride parameter."""
        # Create a file with many points
        csv_file = tmp_path / 'test_stride.csv'
        csv_file.write_text('x,y\n1,1\n2,2\n3,3\n4,4\n5,5\n6,6\n')

        config = PopulationConfig(
            {
                'name': 'File Points Stride Config',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {
                        'file_points': {
                            'path': str(csv_file),
                            'stride': 2,  # Keep every 2nd point
                        }
                    },
                    'quantity': 1,
                    'release_start': '2025-06-18 13:00:00',
                    'burial_depth': {'constant': 1.0},
                },
            }
        )

        result = file_points_strategy.seed(config)

        # Should keep every 2nd point: (1,1), (3,3), (5,5)
        assert len(result) == 3
        assert result[0] == (1, 1.0, 1.0)
        assert result[1] == (1, 3.0, 3.0)
        assert result[2] == (1, 5.0, 5.0)

    def test_file_points_strategy_deduplicate(self, file_points_strategy, tmp_path):
        """Test file_points strategy with deduplication."""
        # Create a file with duplicate points
        csv_file = tmp_path / 'test_duplicates.csv'
        csv_file.write_text('x,y\n1,1\n2,2\n1,1\n3,3\n2,2\n')

        config = PopulationConfig(
            {
                'name': 'File Points Dedupe Config',
                'particle_type': 'mud',
                'seeding': {
                    'strategy': {
                        'file_points': {
                            'path': str(csv_file),
                            'deduplicate': True,
                        }
                    },
                    'quantity': 1,
                    'release_start': '2025-06-18 13:00:00',
                    'burial_depth': {'constant': 1.0},
                },
            }
        )

        result = file_points_strategy.seed(config)

        # Should have only unique points: (1,1), (2,2), (3,3)
        assert len(result) == 3
        positions = [(x, y) for _, x, y in result]
        assert set(positions) == {(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)}

    def test_file_points_strategy_missing_path(self, file_points_strategy):
        """Test file_points strategy with missing path."""
        config = PopulationConfig(
            {
                'name': 'File Points Missing Path',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'file_points': {'not_path': 'invalid'}},  # Missing path but has settings
                    'quantity': 1,
                    'release_start': '2025-06-18 13:00:00',
                    'burial_depth': {'constant': 1.0},
                },
            }
        )

        with pytest.raises(MissingConfigurationParameter, match='"path" must be provided'):
            file_points_strategy.seed(config)

    def test_file_points_strategy_file_not_found(self, file_points_strategy):
        """Test file_points strategy with non-existent file."""
        config = PopulationConfig(
            {
                'name': 'File Points Non-existent File',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {
                        'file_points': {
                            'path': '/non/existent/file.csv',
                        }
                    },
                    'quantity': 1,
                    'release_start': '2025-06-18 13:00:00',
                    'burial_depth': {'constant': 1.0},
                },
            }
        )

        with pytest.raises(FileNotFoundError, match='Could not find coordinates file'):
            file_points_strategy.seed(config)

    def test_file_points_strategy_invalid_columns(self, file_points_strategy, tmp_path):
        """Test file_points strategy with invalid column specification."""
        csv_file = tmp_path / 'test_invalid_cols.csv'
        csv_file.write_text('a,b\n1,2\n')

        config = PopulationConfig(
            {
                'name': 'File Points Invalid Cols',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {
                        'file_points': {
                            'path': str(csv_file),
                            'x_col': 'invalid_col',
                            'y_col': 'another_invalid_col',
                        }
                    },
                    'quantity': 1,
                    'release_start': '2025-06-18 13:00:00',
                    'burial_depth': {'constant': 1.0},
                },
            }
        )

        with pytest.raises(ValueError, match='Columns not found'):
            file_points_strategy.seed(config)

    def test_file_points_strategy_bbox_object_format(self, file_points_strategy, tmp_path):
        """Test file_points strategy with bbox as object."""
        csv_file = tmp_path / 'test_bbox_obj.csv'
        csv_file.write_text('x,y\n1,1\n2,2\n5,5\n')

        config = PopulationConfig(
            {
                'name': 'File Points BBox Object',
                'particle_type': 'passive',
                'seeding': {
                    'strategy': {
                        'file_points': {
                            'path': str(csv_file),
                            'bbox': {
                                'xmin': 0.5,
                                'ymin': 0.5,
                                'xmax': 2.5,
                                'ymax': 2.5,
                            },
                        }
                    },
                    'quantity': 2,
                    'release_start': '2025-06-18 13:00:00',
                    'burial_depth': {'constant': 1.0},
                },
            }
        )

        result = file_points_strategy.seed(config)

        # Should keep only points (1,1) and (2,2)
        assert len(result) == 2
        assert result[0] == (2, 1.0, 1.0)
        assert result[1] == (2, 2.0, 2.0)

    def test_file_points_strategy_invalid_stride(self, file_points_strategy, tmp_path):
        """Test file_points strategy with invalid stride."""
        csv_file = tmp_path / 'test_stride.csv'
        csv_file.write_text('x,y\n1,1\n')

        config = PopulationConfig(
            {
                'name': 'File Points Invalid Stride',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {
                        'file_points': {
                            'path': str(csv_file),
                            'stride': 0,  # Invalid stride
                        }
                    },
                    'quantity': 1,
                    'release_start': '2025-06-18 13:00:00',
                    'burial_depth': {'constant': 1.0},
                },
            }
        )

        with pytest.raises(ValueError, match='"stride" must be >= 1'):
            file_points_strategy.seed(config)

    def test_file_points_strategy_empty_after_filtering(self, file_points_strategy, tmp_path):
        """Test file_points strategy when all points are filtered out."""
        csv_file = tmp_path / 'test_empty_filter.csv'
        csv_file.write_text('x,y\n10,10\n20,20\n')

        config = PopulationConfig(
            {
                'name': 'File Points Empty Filter',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {
                        'file_points': {
                            'path': str(csv_file),
                            'bbox': '0,0 1,1',  # Bbox that excludes all points
                        }
                    },
                    'quantity': 1,
                    'release_start': '2025-06-18 13:00:00',
                    'burial_depth': {'constant': 1.0},
                },
            }
        )

        with pytest.raises(ValueError, match='No valid \\(x, y\\) points found after filtering'):
            file_points_strategy.seed(config)


class TestParticleFactory:
    """Test cases for ParticleFactory."""

    def test_create_particles_point_strategy(self, particle_classes):
        """Test particle creation with PointStrategy."""
        Sand = particle_classes['Sand']

        config = PopulationConfig(
            {
                'name': 'Basic Point Config',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'point': {'locations': ['1.0,2.0', '3.0,4.0']}},
                    'quantity': 2,
                    'release_start': '2025-06-18 13:00:00',
                    'burial_depth': {
                        'constant': 1.0,
                    },
                },
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

        config = PopulationConfig(
            {
                'name': 'Grid Particle Creation Test',
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
                    'burial_depth': {
                        'constant': 1.0,
                    },
                },
            }
        )

        particles = ParticleFactory.create_particles(config)

        # Should create 4 particles (2x2 grid): (0,0), (0,1), (1,0), (1,1)
        # Each location gets 1 particle, so 4 total
        assert len(particles) == 4
        # Check all particles are Mud type
        assert all(isinstance(p, Mud) for p in particles)
        # Check positions include all corners
        positions = [(p.x, p.y) for p in particles]
        expected_positions = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
        assert set(positions) == set(expected_positions)

    def test_create_particles_random_strategy(self, particle_classes):
        """Test particle creation with RandomStrategy."""
        Sand = particle_classes['Sand']

        config = PopulationConfig(
            {
                'name': 'Basic Point Config',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'random': {'bbox': '1.0,2.0, 3.0,4.0', 'nlocations': 2, 'seed': 42}},
                    'quantity': 5,
                    'release_start': '2025-06-18 13:00:00',
                    'burial_depth': {
                        'constant': 1.0,
                    },
                },
            }
        )

        particles = ParticleFactory.create_particles(config)

        # Should create 10 particles (2 locations * 5 quantity)
        assert len(particles) == 10
        # Check all particles are Sand type
        assert all(isinstance(p, Sand) for p in particles)
        # Check positions are within bbox
        positions = [(p.x, p.y) for p in particles]
        for x, y in positions:
            assert 1.0 <= x <= 3.0
            assert 2.0 <= y <= 4.0

    def test_create_particles_preserves_vertical_position_configuration(self):
        config = PopulationConfig(
            {
                'name': 'Vertical Position Config',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'point': {'locations': ['0,0']}},
                    'quantity': 1,
                    'release_start': '2025-06-18 13:00:00',
                    'burial_depth': {'constant': 1.25},
                    'vertical_position': {'mode': 'height_above_bed', 'value': 0.4},
                    'remove_permanently_buried': True,
                },
            }
        )

        particles = ParticleFactory.create_particles(config)

        assert config.remove_permanently_buried is True
        assert particles[0].burial_depth == pytest.approx(0.0)
        assert particles[0].vertical_position_mode == 'height_above_bed'
        assert particles[0].vertical_position_value == pytest.approx(0.4)

    def test_burial_depth_mode_rejects_unused_vertical_position_value(self):
        with pytest.raises(ValueError, match='vertical_position.value.*not used.*seeding.burial_depth'):
            PopulationConfig(
                {
                    'name': 'Invalid burial-depth value config',
                    'particle_type': 'sand',
                    'seeding': {
                        'strategy': {'point': {'locations': ['0,0']}},
                        'quantity': 1,
                        'release_start': '2025-06-18 13:00:00',
                        'burial_depth': {'constant': 0.1},
                        'vertical_position': {'mode': 'burial_depth', 'value': 0.1},
                    },
                }
            )

    def test_constant_burial_depth_is_preserved_in_population_config(self):
        """Configured constant burial depths should remain population input."""
        config = PopulationConfig(
            {
                'name': 'Scalar Burial Config',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'point': {'locations': ['0,0']}},
                    'quantity': 1,
                    'release_start': '2025-06-18 13:00:00',
                    'burial_depth': {'constant': 1.25},
                },
            }
        )

        particles = ParticleFactory.create_particles(config)

        assert config.burial_depth == {'constant': 1.25}
        assert particles[0].burial_depth == pytest.approx(1.25)

    def test_scalar_burial_depth_remains_supported(self):
        """Legacy scalar burial depths should remain valid population input."""
        config = PopulationConfig(
            {
                'name': 'Scalar Burial Config',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'point': {'locations': ['0,0']}},
                    'quantity': 1,
                    'release_start': '2025-06-18 13:00:00',
                    'burial_depth': 1.25,
                },
            }
        )

        particles = ParticleFactory.create_particles(config)

        assert config.burial_depth == pytest.approx(1.25)
        assert particles[0].burial_depth == pytest.approx(1.25)

    def test_random_burial_depth_is_sampled_with_strategy_seed(self):
        """Seeded random strategies should reproduce stochastic burial depths."""
        config = PopulationConfig(
            {
                'name': 'Random Burial Config',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'random': {'bbox': '0,0 1,1', 'nlocations': 3, 'seed': 7}},
                    'quantity': 2,
                    'release_start': '2025-06-18 13:00:00',
                    'burial_depth': {'random': 4.0},
                },
            }
        )

        first = ParticleFactory.create_particles(config)
        second = ParticleFactory.create_particles(config)
        first_depths = np.array([particle.burial_depth for particle in first])
        second_depths = np.array([particle.burial_depth for particle in second])

        assert len(first_depths) == 6
        assert np.all(first_depths >= 0.0)
        assert np.all(first_depths <= 4.0)
        assert np.unique(first_depths).size > 1
        np.testing.assert_allclose(first_depths, second_depths)

    def test_create_particles_different_particle_types(self, particle_classes):
        """Test creating different particle types."""
        Sand, Mud, Passive = particle_classes['Sand'], particle_classes['Mud'], particle_classes['Passive']

        # Test Sand particles
        sand_config = PopulationConfig(
            {
                'name': 'Basic Point Config',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'point': {'locations': ['0,0']}},
                    'quantity': 1,
                    'release_start': '2025-06-18 13:00:00',
                    'burial_depth': {
                        'constant': 1.0,
                    },
                },
            }
        )
        sand_particles = ParticleFactory.create_particles(sand_config)
        assert len(sand_particles) == 1
        assert isinstance(sand_particles[0], Sand)

        # Test Mud particles
        mud_config = PopulationConfig(
            {
                'name': 'Basic Point Config',
                'particle_type': 'mud',
                'seeding': {
                    'strategy': {'point': {'locations': ['0,0']}},
                    'quantity': 1,
                    'release_start': '2025-06-18 13:00:00',
                    'burial_depth': {
                        'constant': 1.0,
                    },
                },
            }
        )
        mud_particles = ParticleFactory.create_particles(mud_config)
        assert len(mud_particles) == 1
        assert isinstance(mud_particles[0], Mud)

        # Test Passive particles
        passive_config = PopulationConfig(
            {
                'name': 'Basic Point Config',
                'particle_type': 'passive',
                'seeding': {
                    'strategy': {'point': {'locations': ['0,0']}},
                    'quantity': 1,
                    'release_start': '2025-06-18 13:00:00',
                    'burial_depth': {
                        'constant': 1.0,
                    },
                },
            }
        )
        passive_particles = ParticleFactory.create_particles(passive_config)
        assert len(passive_particles) == 1
        assert isinstance(passive_particles[0], Passive)

    def test_create_particles_invalid_particle_type(self):
        """Test error handling for invalid particle type."""
        config = PopulationConfig(
            {
                'name': 'Basic Point Config',
                'particle_type': 'invalid_type',
                'seeding': {
                    'strategy': {'point': {'locations': ['0,0']}},
                    'quantity': 1,
                    'release_start': '2025-06-18 13:00:00',
                    'burial_depth': {
                        'constant': 1.0,
                    },
                },
            }
        )

        with pytest.raises(ValueError, match='Unknown particle type'):
            ParticleFactory.create_particles(config)

    def test_zero_quantity_creates_no_particles_without_strategy_seeding(self):
        """Zero quantity disables seeding for restart populations."""
        config = PopulationConfig(
            {
                'name': 'Disabled Restart Population',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'random': {'nlocations': 1}},
                    'quantity': 0,
                    'release_start': '2025-06-18 13:00:00',
                    'burial_depth': {
                        'constant': 0.0,
                    },
                },
            }
        )

        assert config.quantity == 0
        assert ParticleFactory.create_particles(config) == []

    def test_create_particles_release_time_set(self):
        """Test that release time is set correctly."""
        config = PopulationConfig(
            {
                'name': 'Basic Point Config',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'point': {'locations': ['0,0']}},
                    'quantity': 1,
                    'release_start': '2025-06-18 13:00:00',
                    'burial_depth': {
                        'constant': 1.0,
                    },
                },
            }
        )
        particles = ParticleFactory.create_particles(config)

        # Should have the correct release time
        assert particles[0].release_time == '2025-06-18 13:00:00'

    def test_create_particles_file_points_strategy(self, particle_classes, tmp_path):
        """Test particle creation with FilePointsStrategy."""
        Passive = particle_classes['Passive']

        # Create a temporary CSV file
        csv_file = tmp_path / 'test_particles.csv'
        csv_file.write_text('x,y\n1.5,2.5\n3.5,4.5\n')

        config = PopulationConfig(
            {
                'name': 'File Points Particle Creation Test',
                'particle_type': 'passive',
                'seeding': {
                    'strategy': {
                        'file_points': {
                            'path': str(csv_file),
                            'x_col': 'x',
                            'y_col': 'y',
                        }
                    },
                    'quantity': 2,
                    'release_start': '2025-06-18 13:00:00',
                    'burial_depth': {
                        'constant': 1.0,
                    },
                },
            }
        )

        particles = ParticleFactory.create_particles(config)

        # Should create 4 particles (2 locations * 2 quantity)
        assert len(particles) == 4
        # Check all particles are Passive type
        assert all(isinstance(p, Passive) for p in particles)
        # Check positions
        positions = [(p.x, p.y) for p in particles]
        assert positions.count((1.5, 2.5)) == 2  # 2 particles at first location
        assert positions.count((3.5, 4.5)) == 2  # 2 particles at second location


@pytest.fixture
def population_config():
    return PopulationConfig(
        {
            'name': 'Basic Random Config',
            'particle_type': 'sand',
            'seeding': {
                'strategy': {'random': {'bbox': '1.0,2.0, 3.0,4.0', 'nlocations': 2, 'seed': 42}},
                'quantity': 5,
                'release_start': '2025-06-18 13:00:00',
                'burial_depth': {
                    'constant': 1.0,
                },
            },
        }
    )


class TestParticlePopulation:

    @staticmethod
    def _diffusing_passive_population(diffusion_coefficient=0.5):
        """Create mobile passive particles on a unit-square grid."""
        config = PopulationConfig(
            {
                'name': 'Diffusing Passive Particles',
                'particle_type': 'passive',
                'characteristics': {'diffusion_coefficient': diffusion_coefficient},
                'transport_probability': 'no_probability',
                'seeding': {
                    'strategy': {'point': {'locations': ['0.5,0.5', '0.6,0.6']}},
                    'quantity': 1,
                    'release_start': '0',
                    'burial_depth': {'constant': 0.0},
                },
            }
        )
        population = ParticlePopulation(
            field_x=np.array([0.0, 1.0, 1.0, 0.0]),
            field_y=np.array([0.0, 0.0, 1.0, 1.0]),
            population_config=config,
        )
        population.particles['status_mobile'] = np.ones(2, dtype=bool)
        return population

    def test_passive_diffusion_is_applied_after_advection(self, monkeypatch):
        """Configured passive diffusivity adds one Brownian displacement per axis."""
        population = self._diffusing_passive_population(diffusion_coefficient=0.5)

        draws = iter((np.array([1.0, -1.0]), np.array([0.5, -0.5])))
        monkeypatch.setattr(np.random, 'standard_normal', lambda size: next(draws))

        population.update_position(
            flow_field={'u': np.zeros(4), 'v': np.zeros(4)},
            current_timestep=0.02,
        )

        sigma = np.sqrt(2.0 * 0.5 * 0.02)
        np.testing.assert_allclose(population.particles['x'], np.array([0.5, 0.6]) + sigma * np.array([1.0, -1.0]))
        np.testing.assert_allclose(population.particles['y'], np.array([0.5, 0.6]) + sigma * np.array([0.5, -0.5]))

    def test_diffusion_receives_velocity_arrays_matching_particle_coordinates(self, monkeypatch):
        """Diffusion strategies always receive coordinate-shaped velocity inputs."""
        population = self._diffusing_passive_population()
        observed = {}

        def check_velocity_shapes(x, y, u, v, kh, dt):
            observed['x_shape'] = x.shape
            observed['y_shape'] = y.shape
            observed['u_shape'] = u.shape
            observed['v_shape'] = v.shape
            np.testing.assert_allclose(u, 0.0)
            np.testing.assert_allclose(v, 0.0)
            return x, y

        monkeypatch.setattr(population._diffusion_calculator, 'calc_diffusion', check_velocity_shapes)
        population.update_position({'u': np.zeros(4), 'v': np.zeros(4)}, 0.02)

        assert observed['u_shape'] == observed['x_shape']
        assert observed['v_shape'] == observed['y_shape']

    def test_zero_diffusivity_skips_random_draws(self, monkeypatch):
        """The zero-diffusion default keeps the optimized advection-only path."""
        population = self._diffusing_passive_population(diffusion_coefficient=0.0)
        monkeypatch.setattr(
            np.random,
            'standard_normal',
            lambda size: pytest.fail('zero diffusivity must not draw random values'),
        )

        population.update_position(
            flow_field={'u': np.ones(4), 'v': np.zeros(4)},
            current_timestep=0.1,
        )

        np.testing.assert_allclose(population.particles['x'], np.array([0.6, 0.7]))
        np.testing.assert_allclose(population.particles['y'], np.array([0.5, 0.6]))

    def test_diffusion_open_boundary_exit_marks_particle_left_domain(self, monkeypatch):
        """A Brownian step through an open edge removes the particle."""
        config = _boundary_action_config('0.2,0.2')
        config['particle_type'] = 'passive'
        config['characteristics'] = {'diffusion_coefficient': 0.5}
        population = ParticleSeeder([config]).seed(_boundary_action_field_data())[0]
        population._current_time = 0.0
        population.update_status()
        # Standard non-MacDonald tracers derive mobility directly from eligibility.
        population.particles['status_mobile'] = population.particles['status_eligible'].copy()
        draws = iter((np.zeros(1), -np.ones(1)))
        monkeypatch.setattr(np.random, 'standard_normal', lambda size: next(draws))

        population.update_position({'u': np.zeros(3), 'v': np.zeros(3)}, 0.5)

        assert population.particles['status_left_domain'].tolist() == [True]
        assert population.particles['status_alive'].tolist() == [False]
        assert population.particles['status_domain'].tolist() == [False]

    def test_diffusion_land_boundary_exit_rolls_back_to_advected_position(self, monkeypatch):
        """A Brownian step through land keeps the particle at its prior position."""
        config = _boundary_action_config('0.2,0.2')
        config['particle_type'] = 'passive'
        config['characteristics'] = {'diffusion_coefficient': 0.5}
        population = ParticleSeeder([config]).seed(_boundary_action_field_data())[0]
        population._current_time = 0.0
        population.update_status()
        # Standard non-MacDonald tracers derive mobility directly from eligibility.
        population.particles['status_mobile'] = population.particles['status_eligible'].copy()
        draws = iter((-np.ones(1), np.zeros(1)))
        monkeypatch.setattr(np.random, 'standard_normal', lambda size: next(draws))

        population.update_position({'u': np.zeros(3), 'v': np.zeros(3)}, 0.5)

        np.testing.assert_allclose(population.particles['x'], [0.2])
        np.testing.assert_allclose(population.particles['y'], [0.2])
        assert population.particles['status_beached'].tolist() == [True]
        assert population.particles['status_domain'].tolist() == [True]
        assert population.particles['status_mobile'].tolist() == [False]

    def test_advection_land_contact_skips_diffusion(self, monkeypatch):
        """A land contact during advection cannot receive a later Brownian step."""
        config = _boundary_action_config('0.2,0.2')
        config['particle_type'] = 'passive'
        config['characteristics'] = {'diffusion_coefficient': 0.5}
        population = ParticleSeeder([config]).seed(_boundary_action_field_data())[0]
        population._current_time = 0.0
        population.update_status()
        # Standard non-MacDonald tracers derive mobility directly from eligibility.
        population.particles['status_mobile'] = population.particles['status_eligible'].copy()
        monkeypatch.setattr(np.random, 'standard_normal', lambda size: pytest.fail('land contact must skip diffusion'))

        population.update_position({'u': -np.ones(3), 'v': np.zeros(3)}, 0.5)

        np.testing.assert_allclose(population.particles['x'], [0.2])
        np.testing.assert_allclose(population.particles['y'], [0.2])

    @pytest.mark.parametrize('coefficient', [-1.0, np.inf, np.nan, True])
    def test_invalid_diffusion_coefficient_is_rejected(self, coefficient):
        """Diffusivity must be finite and non-negative before tracing starts."""
        with pytest.raises(ValueError, match='diffusion_coefficient'):
            self._diffusing_passive_population(diffusion_coefficient=coefficient)

    @pytest.mark.parametrize(
        ('particle_type', 'tracer_methods', 'characteristics'),
        [
            ('passive', {'passive_tracer': {}}, {}),
            (
                'sand',
                {'vanwesten': {'flow_field_name': ['bed_load_velocity']}},
                {'density': 2650.0, 'grain_size': 0.00025},
            ),
            ('mud', {'soulsby': {'flow_field_name': ['grain_velocity']}}, {'density': 2000.0, 'size': 0.00005}),
        ],
    )
    def test_top_level_brownian_diffusion_applies_to_all_population_methods(
        self, particle_type, tracer_methods, characteristics
    ):
        """Top-level diffusion is independent of particle and tracer method."""
        population = _diffusion_population(
            particle_type, tracer_methods, characteristics, {'method': 'brownian', 'coefficient': 0.1, 'seed': 7}
        )

        assert population.population_config.diffusion_method == 'brownian'
        assert population._diffusion_calculator is not None

    @pytest.mark.parametrize('diffusion', [{'method': 'none', 'coefficient': 0.1}, {'coefficient': 0.0}, None])
    def test_top_level_diffusion_noop_modes_skip_calculator(self, diffusion):
        """None, zero coefficient, and omitted diffusion retain the fast path."""
        population = _diffusion_population('passive', {'passive_tracer': {}}, {}, diffusion)

        assert population._diffusion_calculator is None

    def test_population_diffusion_seed_is_reproducible_over_multiple_steps(self):
        """Same per-population seeds give identical multi-step tracks."""
        diffusion = {'method': 'brownian', 'coefficient': 0.1, 'seed': 42}
        first = _diffusion_population('passive', {'passive_tracer': {}}, {}, diffusion)
        second = _diffusion_population('passive', {'passive_tracer': {}}, {}, diffusion)
        third = _diffusion_population('passive', {'passive_tracer': {}}, {}, {**diffusion, 'seed': 43})

        for population in (first, second, third):
            population.particles['status_mobile'] = np.ones(1, dtype=bool)
            for _ in range(2):
                population.update_position({'u': np.zeros(4), 'v': np.zeros(4)}, 0.01)

        np.testing.assert_allclose(first.particles['x'], second.particles['x'])
        np.testing.assert_allclose(first.particles['y'], second.particles['y'])
        assert not np.allclose(first.particles['x'], third.particles['x'])

    @staticmethod
    def _status_test_population(current_time=0.0):

        config = PopulationConfig(
            {
                'name': 'Status Test Config',
                'particle_type': 'sand',
                'transport_probability': 'stochastic_transport',
                'seeding': {
                    'strategy': {'point': {'locations': ['0.5,0.5']}},
                    'quantity': 4,
                    'release_start': '0',
                    'burial_depth': {
                        'constant': 0.0,
                    },
                },
            }
        )
        population = ParticlePopulation(
            field_x=np.array([0.0, 1.0, 1.0, 0.0]),
            field_y=np.array([0.0, 0.0, 1.0, 1.0]),
            population_config=config,
        )
        population._current_time = current_time
        population.particles['x'] = np.array([0.5, 0.5, 2.0, 0.5])
        population.particles['y'] = np.array([0.5, 0.5, 2.0, 0.5])
        population._particle_simplices = population.grid_geometry.locate_points(
            population.particles['x'],
            population.particles['y'],
        )
        population.particles['burial_depth'] = np.array([0.1, 2.0, 0.1, 0.1])
        population.particles['mixing_depth'] = np.ones(4)
        population.particles['transport_probability'] = np.array([1.0, 1.0, 1.0, 0.0])
        return population

    def test_create_population(self, population_config):
        """Test creating a ParticlePopulation with a valid configuration."""
        population = ParticlePopulation(
            field_x=np.array([0.0, 5.0, 5.0, 0.0]),
            field_y=np.array([0.0, 0.0, 5.0, 5.0]),
            population_config=population_config,
        )
        assert population is not None
        assert len(population.particles['x']) == 10  # 2 nlocations * 5 quantity
        assert len(population.particles['y']) == 10  # 2 nlocations * 5 quantity

    def test_repr_volume_uses_random_burial_bbox_area(self):
        """Representative volume should be area times max depth per particle."""
        config = PopulationConfig(
            {
                'name': 'Representative Volume BBox',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'random': {'bbox': '0,0 4,3', 'nlocations': 6, 'seed': 3}},
                    'quantity': 2,
                    'release_start': '2025-06-18 13:00:00',
                    'burial_depth': {'random': 5.0},
                },
            }
        )

        population = ParticlePopulation(
            field_x=np.array([0.0, 4.0, 4.0, 0.0]),
            field_y=np.array([0.0, 0.0, 3.0, 3.0]),
            population_config=config,
        )

        assert population.repr_volume == pytest.approx((4.0 * 3.0 * 5.0) / 12)

    def test_repr_volume_uses_random_burial_poly_area(self):
        """Polygon area should feed representative-volume metadata."""
        config = PopulationConfig(
            {
                'name': 'Representative Volume Poly',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'grid': {'poly': ['0,0', '4,0', '0,3'], 'separation': {'dx': 1.0, 'dy': 1.0}}},
                    'quantity': 1,
                    'release_start': '2025-06-18 13:00:00',
                    'burial_depth': {'random': 2.0},
                },
            }
        )

        population = ParticlePopulation(
            field_x=np.array([0.0, 4.0, 4.0, 0.0]),
            field_y=np.array([0.0, 0.0, 3.0, 3.0]),
            population_config=config,
        )

        expected_volume = (0.5 * 4.0 * 3.0 * 2.0) / len(population.particles['x'])
        assert population.repr_volume == pytest.approx(expected_volume)

    def test_repr_volume_is_nan_for_constant_burial(self, population_config):
        """Constant burial depths do not define a representative seeding volume."""
        population = ParticlePopulation(
            field_x=np.array([0.0, 5.0, 5.0, 0.0]),
            field_y=np.array([0.0, 0.0, 5.0, 5.0]),
            population_config=population_config,
        )

        assert np.isnan(population.repr_volume)

    def test_zero_particle_population_updates_are_noops(self):
        """Disabled restart populations should not require particle fields."""
        config = PopulationConfig(
            {
                'name': 'Disabled Restart Population',
                'particle_type': 'sand',
                'transport_probability': 'stochastic_transport',
                'seeding': {
                    'strategy': {'random': {'nlocations': 1}},
                    'quantity': 0,
                    'release_start': '0',
                    'burial_depth': {
                        'constant': 0.0,
                    },
                },
            }
        )
        population = ParticlePopulation(
            field_x=np.array([0.0, 1.0, 1.0, 0.0]),
            field_y=np.array([0.0, 0.0, 1.0, 1.0]),
            population_config=config,
        )

        population.update_burial_depth()
        population.update_status()
        population.update_position(
            flow_field={'u': np.ones(4), 'v': np.zeros(4)},
            current_timestep=1.0,
        )

        assert len(population.particles['x']) == 0
        assert len(population.particles['y']) == 0
        assert population.particles['status_eligible'].dtype == bool
        assert len(population.particles['status_eligible']) == 0

    def test_update_information_accepts_scalar_transport_probability(self, point_config_simple):
        """Scalar fields should update particles without allocating full grid fields."""
        population = ParticlePopulation(
            field_x=np.array([0.0, 1.0, 1.0, 0.0]),
            field_y=np.array([0.0, 0.0, 1.0, 1.0]),
            population_config=point_config_simple,
        )
        bed_level = np.array([0.0, 1.0, 2.0, 1.0])

        population.update_information(
            current_time=0.0,
            mixing_depth=None,
            transport_probability=1.0,
            bed_level=bed_level,
        )

        np.testing.assert_allclose(population.particles['transport_probability'], 1.0)
        np.testing.assert_allclose(population.particles['bed_level'], 0.0)
        assert 'mixing_depth' not in population.particles

    def test_update_information_batches_static_grid_fields(self, point_config_simple):
        """Static grid fields should share one particle-location pass."""
        population = ParticlePopulation(
            field_x=np.array([0.0, 1.0, 1.0, 0.0]),
            field_y=np.array([0.0, 0.0, 1.0, 1.0]),
            population_config=point_config_simple,
        )
        n_particles = len(population.particles['x'])
        calls = []

        def fail_single_field(*args, **kwargs):
            pytest.fail('static grid fields should be batched')

        def interpolate_multi(fields, x_points, y_points, simplex_ids=None):
            calls.append(len(fields))
            assert len(fields) == 3
            np.testing.assert_array_equal(simplex_ids, population._particle_simplices)
            particle_values = tuple(np.full(len(x_points), value, dtype=float) for value in (0.5, 0.75, 1.25))
            return particle_values, np.zeros(len(x_points), dtype=np.int64)

        population._field_interpolator = fail_single_field
        population._field_interpolator_multi = fail_single_field
        population._field_interpolator_multi_with_simplex = interpolate_multi

        population.update_information(
            current_time=0.0,
            mixing_depth=np.arange(4.0),
            transport_probability=np.ones(4),
            bed_level=np.full(4, 2.0),
        )

        assert calls == [3]
        np.testing.assert_allclose(population.particles['mixing_depth'], np.full(n_particles, 0.5))
        np.testing.assert_allclose(population.particles['transport_probability'], np.full(n_particles, 0.75))
        np.testing.assert_allclose(population.particles['bed_level'], np.full(n_particles, 1.25))
        np.testing.assert_allclose(population.particles['bed_level_previous'], np.full(n_particles, 1.25))
        np.testing.assert_array_equal(population._particle_simplices, np.zeros(n_particles, dtype=np.int64))

    def test_update_information_accepts_temporal_scalar_bounds(self, point_config_simple):
        """Temporal scalar bounds should match preblended-grid interpolation."""
        population = ParticlePopulation(
            field_x=np.array([0.0, 1.0, 1.0, 0.0]),
            field_y=np.array([0.0, 0.0, 1.0, 1.0]),
            population_config=point_config_simple,
        )
        lower = np.array([0.0, 1.0, 2.0, 1.0])
        upper = np.array([2.0, 3.0, 4.0, 3.0])

        population.update_information(
            current_time=0.0,
            mixing_depth=None,
            transport_probability=1.0,
            bed_level={'lower': lower, 'upper': upper, 'weight': 0.25},
        )

        np.testing.assert_allclose(population.particles['bed_level'], 0.5)

    def test_update_burial_depth_tracks_temporal_bed_level_change(self):
        """Accretion should increase burial depth, while erosion clamps at zero."""
        config = PopulationConfig(
            {
                'name': 'Bed Level Change Config',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'point': {'locations': ['0.25,0.25', '0.75,0.75']}},
                    'quantity': 1,
                    'release_start': '2025-06-18 13:00:00',
                    'burial_depth': {'constant': 1.0},
                },
            }
        )
        population = ParticlePopulation(
            field_x=np.array([0.0, 1.0, 1.0, 0.0]),
            field_y=np.array([0.0, 0.0, 1.0, 1.0]),
            population_config=config,
        )
        population.particles['bed_level_previous'] = np.array([10.0, 10.0])
        population.particles['bed_level'] = np.array([10.5, 8.0])
        population.particles['burial_depth'] = np.array([1.0, 1.0])

        population.update_burial_depth()

        np.testing.assert_allclose(population.particles['burial_depth'], [1.5, 0.0])
        np.testing.assert_allclose(population.particles['z'], [9.0, 8.0])

    def test_update_bed_level_after_movement_resamples_current_position(self, point_config_simple):
        """Post-move bed levels should update both bed_level and particle z."""
        population = ParticlePopulation(
            field_x=np.array([0.0, 1.0, 1.0, 0.0]),
            field_y=np.array([0.0, 0.0, 1.0, 1.0]),
            population_config=point_config_simple,
        )
        population.particles['x'] = np.array([0.25])
        population.particles['y'] = np.array([0.25])
        population.particles['burial_depth'] = np.array([0.25])

        population.update_bed_level_change_after_movement(np.array([1.0, 2.0, 3.0, 2.0]))

        np.testing.assert_allclose(population.particles['bed_level'], [1.5])
        np.testing.assert_allclose(population.particles['z'], [1.25])

    def test_update_bed_level_after_movement_reuses_cached_simplex_interpolator(self, point_config_simple):
        """Post-move bed-level sampling should stay on the cached interpolation path."""
        population = ParticlePopulation(
            field_x=np.array([0.0, 1.0, 1.0, 0.0]),
            field_y=np.array([0.0, 0.0, 1.0, 1.0]),
            population_config=point_config_simple,
        )
        n_particles = len(population.particles['x'])
        calls = []

        def fail_uncached(*args, **kwargs):
            pytest.fail('bed-level resampling should use cached simplex interpolation')

        def interpolate_multi(fields, x_points, y_points, simplex_ids=None):
            calls.append(len(fields))
            assert len(fields) == 1
            np.testing.assert_array_equal(simplex_ids, population._particle_simplices)
            return (np.full(len(x_points), 1.25, dtype=float),), np.zeros(len(x_points), dtype=np.int64)

        population.particles['burial_depth'] = np.full(n_particles, 0.25)
        population._field_interpolator = fail_uncached
        population._field_interpolator_multi = fail_uncached
        population._field_interpolator_multi_with_simplex = interpolate_multi

        population.update_bed_level_change_after_movement(np.array([1.0, 2.0, 3.0, 2.0]))

        assert calls == [1]
        np.testing.assert_allclose(population.particles['bed_level'], np.full(n_particles, 1.25))
        np.testing.assert_allclose(population.particles['z'], np.full(n_particles, 1.0))
        np.testing.assert_array_equal(population._particle_simplices, np.zeros(n_particles, dtype=np.int64))

    def test_update_information_batches_static_field_interpolation(self, point_config_simple, monkeypatch):
        """Static particle fields should share one cached interpolation pass."""
        population = ParticlePopulation(
            field_x=np.array([0.0, 1.0, 1.0, 0.0]),
            field_y=np.array([0.0, 0.0, 1.0, 1.0]),
            population_config=point_config_simple,
        )
        calls = []
        initial_simplices = population._particle_simplices.copy()

        def fake_interpolate_fields(fields, x_points, y_points, simplex_ids=None):
            calls.append((len(fields), simplex_ids.copy()))
            values = tuple(np.full(len(x_points), float(index + 1)) for index, _ in enumerate(fields))
            return values, np.zeros(len(x_points), dtype=np.int64)

        monkeypatch.setattr(population, '_field_interpolator_multi_with_simplex', fake_interpolate_fields)

        population.update_information(
            current_time=0.0,
            mixing_depth=np.arange(4.0),
            transport_probability=np.arange(4.0) + 10.0,
            bed_level=np.arange(4.0) + 20.0,
        )

        assert len(calls) == 1
        assert calls[0][0] == 3
        np.testing.assert_array_equal(calls[0][1], initial_simplices)
        np.testing.assert_allclose(population.particles['mixing_depth'], 1.0)
        np.testing.assert_allclose(population.particles['transport_probability'], 2.0)
        np.testing.assert_allclose(population.particles['bed_level'], 3.0)
        np.testing.assert_allclose(population.particles['bed_level_previous'], 3.0)

    def test_update_information_batches_temporal_field_interpolation(self, point_config_simple, monkeypatch):
        """Temporal particle fields should use the cached lower/upper interpolation path."""
        population = ParticlePopulation(
            field_x=np.array([0.0, 1.0, 1.0, 0.0]),
            field_y=np.array([0.0, 0.0, 1.0, 1.0]),
            population_config=point_config_simple,
        )
        calls = []

        def fake_interpolate_fields(fields, x_points, y_points, simplex_ids=None):
            calls.append(len(fields))
            raw_values = (0.0, 10.0) if len(calls) == 1 else (20.0, 40.0)
            values = tuple(np.full(len(x_points), value) for value in raw_values)
            return values, np.zeros(len(x_points), dtype=np.int64)

        monkeypatch.setattr(population, '_field_interpolator_multi_with_simplex', fake_interpolate_fields)

        population.update_information(
            current_time=0.0,
            mixing_depth={'lower': np.arange(4.0), 'upper': np.arange(4.0), 'weight': 0.25},
            transport_probability=1.0,
            bed_level={'lower': np.arange(4.0), 'upper': np.arange(4.0), 'weight': 0.5},
        )

        assert calls == [2, 2]
        np.testing.assert_allclose(population.particles['mixing_depth'], 2.5)
        np.testing.assert_allclose(population.particles['transport_probability'], 1.0)
        np.testing.assert_allclose(population.particles['bed_level'], 30.0)

    def test_update_status_uses_status_keys_and_eligibility_composition(self, monkeypatch):
        """Only particles satisfying every general status flag should be eligible."""
        population = self._status_test_population(current_time=0.0)
        monkeypatch.setattr(np.random, 'rand', lambda n_particles: np.array([0.0, 0.0, 0.0, 1.0]))

        population.update_status()

        expected_keys = {
            'status_alive',
            'status_buried',
            'status_domain',
            'status_released',
            'status_transported',
            'status_eligible',
        }
        assert expected_keys.issubset(population.particles)
        assert not any(
            key in population.particles
            for key in {
                'is_alive',
                'is_exposed',
                'is_inside',
                'is_mobile',
                'is_picked_up',
                'is_released',
                'is_suspended',
                'is_deposited',
                'is_buried',
                'is_available_for_entrainment',
            }
        )
        np.testing.assert_array_equal(population.particles['status_alive'], np.array([True, True, True, True]))
        np.testing.assert_array_equal(population.particles['status_buried'], np.array([False, True, False, False]))
        np.testing.assert_array_equal(population.particles['status_domain'], np.array([True, True, False, True]))
        np.testing.assert_array_equal(population.particles['status_released'], np.array([True, True, True, True]))
        np.testing.assert_array_equal(population.particles['status_transported'], np.array([True, True, True, False]))
        np.testing.assert_array_equal(population.particles['status_eligible'], np.array([True, False, False, False]))

    def test_update_status_uses_cached_simplex_ids_for_domain_mask(self, monkeypatch):
        """Domain status should use cached simplex ids instead of polygon scans."""
        population = self._status_test_population(current_time=0.0)
        population._particle_simplices = np.array([0, -1, 3, -1])
        population._mark_particle_simplices_current()
        population._outer_envelope = type(
            'FailingEnvelope',
            (),
            {'contains_points': lambda self, points: pytest.fail('status_domain should use cached simplex ids')},
        )()
        monkeypatch.setattr(np.random, 'rand', lambda n_particles: np.zeros(n_particles))

        population.update_status()

        np.testing.assert_array_equal(population.particles['status_domain'], np.array([True, False, True, False]))

    def test_update_status_refreshes_invalidated_simplex_ids_after_coordinate_mutation(self, monkeypatch):
        """Invalidated coordinate edits should refresh domain status before mobile-mask composition."""
        population = self._status_test_population(current_time=0.0)
        population.particles['x'][:] = 0.5
        population.particles['y'][:] = 0.5
        population._refresh_particle_simplices()
        population.particles['x'][2] = 2.0
        population.particles['y'][2] = 2.0
        population._invalidate_particle_simplices()
        monkeypatch.setattr(np.random, 'rand', lambda n_particles: np.zeros(n_particles))

        population.update_status()

        np.testing.assert_array_equal(population.particles['status_domain'], np.array([True, True, False, True]))
        assert population._particle_simplices[2] == -1
        assert np.all(population._particle_simplices[[0, 1, 3]] >= 0)

    def test_update_status_requires_released_particles_for_eligibility(self, monkeypatch):
        """Particles that are otherwise mobile should not move before release."""
        population = self._status_test_population(current_time=-1.0)
        monkeypatch.setattr(np.random, 'rand', lambda n_particles: np.zeros(n_particles))

        population.update_status()

        np.testing.assert_array_equal(population.particles['status_released'], np.array([False, False, False, False]))
        np.testing.assert_array_equal(population.particles['status_eligible'], np.array([False, False, False, False]))

    def test_update_status_no_probability_skips_random_draw(self, monkeypatch):
        """No-probability transport should mark particles transported without RNG allocation."""
        population = _single_particle_population(release_start='0')
        population._current_time = 0.0
        monkeypatch.setattr(np.random, 'rand', lambda n_particles: pytest.fail('np.random.rand should not be called'))

        population.update_status()

        np.testing.assert_array_equal(population.particles['status_transported'], np.array([True]))
        np.testing.assert_array_equal(population.particles['status_eligible'], np.array([True]))

    def test_no_probability_keeps_burial_and_buried_status_zero_over_time(self, monkeypatch):
        """With no_probability, burial/state stay zero and z follows bed level each timestep."""
        population = _single_particle_population(release_start='0')
        monkeypatch.setattr(np.random, 'rand', lambda n_particles: pytest.fail('np.random.rand should not be called'))

        bed_levels = [0.0, 0.35, -0.2, 0.8]
        for step, bed_level in enumerate(bed_levels):
            current_time = float(step)
            population.update_information(
                current_time=current_time,
                mixing_depth=1.0,
                transport_probability=0.2,
                bed_level=bed_level,
            )
            population.update_status()
            population.update_bed_level_change_after_movement(bed_level)

            np.testing.assert_allclose(population.particles['burial_depth'], np.array([0.0]))
            np.testing.assert_array_equal(population.particles['status_buried'], np.array([False]))
            np.testing.assert_allclose(population.particles['z'], np.array([bed_level]))
            np.testing.assert_allclose(population.particles['bed_level'], np.array([bed_level]))

    def test_update_status_no_probability_does_not_require_mixing_depth(self, monkeypatch):
        """No-probability mode should remain independent from mixing-depth fields."""
        config = PopulationConfig(
            {
                'name': 'No Probability Population',
                'particle_type': 'sand',
                'transport_probability': 'no_probability',
                'seeding': {
                    'strategy': {'point': {'locations': ['0.5,0.5']}},
                    'quantity': 1,
                    'release_start': '0',
                    'burial_depth': {'constant': 0.0},
                },
            }
        )
        population = ParticlePopulation(
            field_x=np.array([0.0, 1.0, 0.0, 1.0]),
            field_y=np.array([0.0, 0.0, 1.0, 1.0]),
            population_config=config,
            reference_date=np.datetime64('1970-01-01T00:00:00', 's'),
        )
        population._current_time = 0.0
        monkeypatch.setattr(np.random, 'rand', lambda n_particles: pytest.fail('np.random.rand should not be called'))

        population.update_status()

        np.testing.assert_array_equal(population.particles['status_transported'], np.array([True]))
        np.testing.assert_array_equal(population.particles['status_buried'], np.array([False]))

    def test_update_status_reuses_cached_particle_locations(self, monkeypatch):
        """Unchanged particles should not be relocated on every status update."""
        population = _single_particle_population(release_start='0')
        population._current_time = 0.0

        def fail_locate_points(*args, **kwargs):
            pytest.fail('locate_points should not be called for unchanged particles')

        monkeypatch.setattr(population.grid_geometry, 'locate_points', fail_locate_points)

        population.update_status()

        np.testing.assert_array_equal(population.particles['status_domain'], np.array([True]))

    def test_update_status_relocates_invalidated_externally_moved_particles(self, monkeypatch):
        """Particles whose coordinates change outside update_position can be relocated once."""
        population = _single_particle_population(release_start='0')
        population._current_time = 0.0
        calls = []

        def fake_locate_points(x_points, y_points, start_simplices=None):
            calls.append((np.asarray(x_points).copy(), np.asarray(y_points).copy()))
            return np.array([0], dtype=np.int64)

        monkeypatch.setattr(population.grid_geometry, 'locate_points', fake_locate_points)
        population.particles['x'][0] = 0.25
        population.particles['y'][0] = 0.25
        population._invalidate_particle_simplices()

        population.update_status()

        assert len(calls) == 1
        np.testing.assert_allclose(calls[0][0], np.array([0.25]))
        np.testing.assert_allclose(calls[0][1], np.array([0.25]))
        np.testing.assert_array_equal(population.particles['status_domain'], np.array([True]))
        assert not population._particle_simplices_stale

    def test_update_information_skips_stale_simplex_ids_after_invalidation(self, monkeypatch):
        """Invalidated particle coordinates should not seed interpolation with stale simplex ids."""
        population = _single_particle_population(release_start='0')
        population._current_time = 0.0
        calls = []

        def fake_interpolate(fields, x_points, y_points, simplex_ids=None):
            calls.append(simplex_ids)
            return (np.array([[1.0], [1.0], [0.0]]), np.array([0], dtype=np.int64))

        monkeypatch.setattr(population, '_field_interpolator_multi_with_simplex', fake_interpolate)
        population.particles['x'][0] = 0.25
        population.particles['y'][0] = 0.25
        population._invalidate_particle_simplices()

        population.update_information(
            current_time=0.0,
            mixing_depth=np.array([1.0]),
            transport_probability=np.array([1.0]),
            bed_level=np.array([0.0]),
        )

        assert calls == [None]
        assert not population._particle_simplices_stale

    def test_update_status_uses_active_connectivity_holes_for_domain_mask(self, monkeypatch):
        """Particles inside a mesh hole should be outside the active domain."""

        config = {
            'name': 'Hole Domain Config',
            'particle_type': 'sand',
            'transport_probability': 'no_probability',
            'seeding': {
                'strategy': {'point': {'locations': ['1.0,1.0', '1.0,0.4']}},
                'quantity': 1,
                'release_start': '0',
                'burial_depth': {'constant': 0.0},
            },
        }
        field_data = SimpleNamespace(
            x=np.array([0.0, 2.0, 2.0, 0.0, 0.8, 1.2, 1.2, 0.8]),
            y=np.array([0.0, 0.0, 2.0, 2.0, 0.8, 0.8, 1.2, 1.2]),
            face_node_connectivity=np.array(
                [
                    [0, 1, 5],
                    [0, 5, 4],
                    [1, 2, 6],
                    [1, 6, 5],
                    [2, 3, 7],
                    [2, 7, 6],
                    [3, 0, 4],
                    [3, 4, 7],
                ],
                dtype=np.int64,
            ),
        )
        population = ParticleSeeder([config]).seed(field_data)[0]
        population._current_time = 0.0
        population.particles['transport_probability'] = np.ones(2)
        monkeypatch.setattr(np.random, 'rand', lambda n_particles: np.zeros(n_particles))

        population.update_status()

        np.testing.assert_array_equal(population.particles['status_domain'], np.array([False, True]))

    def test_open_boundary_exit_marks_particle_left_domain(self, monkeypatch):
        """Particles crossing open boundary edges are removed from later movement."""
        population = ParticleSeeder([_boundary_action_config('0.2,0.2')]).seed(_boundary_action_field_data())[0]
        population._current_time = 0.0
        population.particles['transport_probability'] = np.ones(1)
        monkeypatch.setattr(np.random, 'rand', lambda n_particles: np.zeros(n_particles))

        population.update_status()
        population.particles['status_mobile'] = population.particles['status_eligible'].copy()
        population.update_position(
            flow_field={'u': np.zeros(3), 'v': -np.ones(3)},
            current_timestep=0.5,
        )

        assert population.particles['status_left_domain'].tolist() == [True]
        assert population.particles['status_alive'].tolist() == [False]
        assert population.particles['status_domain'].tolist() == [False]
        assert population.particles['status_mobile'].tolist() == [False]

        population.update_status()

        assert population.particles['status_alive'].tolist() == [False]
        assert population.particles['status_eligible'].tolist() == [False]

    def test_open_boundary_exit_keeps_original_update_mask_shape(self, monkeypatch):
        """Boundary exits should not shrink the mobile mask before position assignment."""
        config = _boundary_action_config('0.2,0.2')
        config['seeding']['strategy']['point']['locations'] = ['0.2,0.2', '0.4,0.2']
        population = ParticleSeeder([config]).seed(_boundary_action_field_data())[0]
        population._current_time = 0.0
        population.particles['transport_probability'] = np.ones(2)
        monkeypatch.setattr(np.random, 'rand', lambda n_particles: np.zeros(n_particles))

        population.update_status()
        population.particles['status_mobile'] = population.particles['status_eligible'].copy()
        population.update_position(
            flow_field={'u': np.zeros(3), 'v': -np.ones(3)},
            current_timestep=0.5,
        )

        assert population.particles['status_left_domain'].tolist() == [True, True]
        assert population.particles['status_mobile'].tolist() == [False, False]

    def test_land_boundary_contact_marks_beached_without_removing_particle(self, monkeypatch):
        """Particles crossing land boundary edges stay put and can reactivate later."""
        population = ParticleSeeder([_boundary_action_config('0.2,0.2')]).seed(_boundary_action_field_data())[0]
        population._current_time = 0.0
        population.particles['transport_probability'] = np.ones(1)
        monkeypatch.setattr(np.random, 'rand', lambda n_particles: np.zeros(n_particles))

        population.update_status()
        population.particles['status_mobile'] = population.particles['status_eligible'].copy()
        population.update_position(
            flow_field={'u': -np.ones(3), 'v': np.zeros(3)},
            current_timestep=0.5,
        )

        np.testing.assert_allclose(population.particles['x'], np.array([0.2]))
        np.testing.assert_allclose(population.particles['y'], np.array([0.2]))
        assert population.particles['status_beached'].tolist() == [True]
        assert population.particles['status_left_domain'].tolist() == [False]
        assert population.particles['status_alive'].tolist() == [True]
        assert population.particles['status_domain'].tolist() == [True]
        assert population.particles['status_mobile'].tolist() == [False]

        population.update_status()

        assert population.particles['status_beached'].tolist() == [False]
        assert population.particles['status_alive'].tolist() == [True]
        assert population.particles['status_eligible'].tolist() == [True]

    def test_update_position_carries_cached_simplex_ids(self, point_config_simple):
        """Position updates should reuse and refresh particle simplex ids."""
        population = ParticlePopulation(
            field_x=np.array([0.0, 1.0, 1.0, 0.0]),
            field_y=np.array([0.0, 0.0, 1.0, 1.0]),
            population_config=point_config_simple,
        )
        old_simplices = population._particle_simplices.copy()
        population.particles['status_mobile'] = np.ones(len(population.particles['x']), dtype=bool)

        population.update_position(
            flow_field={'u': np.ones(4), 'v': np.zeros(4)},
            current_timestep=0.1,
        )

        np.testing.assert_allclose(population.particles['x'], [0.1])
        np.testing.assert_allclose(population.particles['y'], [0.0])
        expected_simplices = population.grid_geometry.locate_points(
            population.particles['x'],
            population.particles['y'],
            old_simplices,
        )
        np.testing.assert_array_equal(population._particle_simplices, expected_simplices)

    def test_all_seed_locations_outside_domain_raise_configuration_error(self):
        config = PopulationConfig(
            {
                'name': 'Outside Domain Config',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'point': {'locations': ['10.0,10.0']}},
                    'quantity': 1,
                    'burial_depth': {
                        'constant': 0.0,
                    },
                },
                'transport_probability': 'no_probability',
            }
        )

        with pytest.raises(ConfigurationError, match='All seeded particles are outside the input field domain'):
            ParticlePopulation(
                field_x=np.array([0.0, 1.0, 1.0, 0.0]),
                field_y=np.array([0.0, 0.0, 1.0, 1.0]),
                population_config=config,
            )


# ---------------------------------------------------------------------------
# Polygon helpers
# ---------------------------------------------------------------------------


class TestParsePolygon:
    """Tests for _parse_polygon and _read_polygon_file."""

    def test_inline_list(self):
        verts = _parse_polygon(['0,0', '1,0', '1,1', '0,1'])
        assert verts.shape == (4, 2)
        np.testing.assert_array_equal(verts[0], [0.0, 0.0])
        np.testing.assert_array_equal(verts[2], [1.0, 1.0])

    def test_inline_list_space_separated(self):
        verts = _parse_polygon(['0 0', '1 0', '0.5 1'])
        assert verts.shape == (3, 2)

    def test_inline_too_few_vertices(self):
        with pytest.raises(ValueError, match='at least 3 vertices'):
            _parse_polygon(['0,0', '1,1'])

    def test_inline_invalid_coord(self):
        with pytest.raises(ValueError):
            _parse_polygon(['0,0', 'bad', '1,1'])

    def test_file_plain_text(self, tmp_path):
        p = tmp_path / 'poly.txt'
        p.write_text('0 0\n1 0\n1 1\n0 1\n')
        verts = _read_polygon_file(str(p))
        assert verts.shape == (4, 2)

    def test_file_csv_with_header(self, tmp_path):
        p = tmp_path / 'poly.csv'
        p.write_text('x,y\n0,0\n1,0\n1,1\n0,1\n')
        verts = _read_polygon_file(str(p))
        assert verts.shape == (4, 2)

    def test_file_pol_format(self, tmp_path):
        p = tmp_path / 'poly.pol'
        p.write_text('my_polygon\n4 2\n0.0 0.0\n2.0 0.0\n2.0 2.0\n0.0 2.0\n')
        verts = _read_polygon_file(str(p))
        assert verts.shape == (4, 2)
        np.testing.assert_array_equal(verts[2], [2.0, 2.0])

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _read_polygon_file(str(tmp_path / 'missing.txt'))

    def test_string_path_dispatches_to_file(self, tmp_path):
        p = tmp_path / 'poly.txt'
        p.write_text('0 0\n4 0\n4 4\n0 4\n')
        verts = _parse_polygon(str(p))
        assert verts.shape == (4, 2)

    def test_invalid_type(self):
        with pytest.raises(ValueError, match='file path string or a list'):
            _parse_polygon(12345)


# ---------------------------------------------------------------------------
# RandomStrategy with poly
# ---------------------------------------------------------------------------


class TestRandomStrategyPoly:
    def _make_config(self, poly, nlocations=10, seed=42):
        return PopulationConfig(
            {
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'random': {'poly': poly, 'nlocations': nlocations, 'seed': seed}},
                    'quantity': 1,
                    'release_start': '2025-01-01 00:00:00',
                    'burial_depth': {'constant': 0.0},
                },
            }
        )

    def test_inline_poly_all_inside(self):
        # Unit square polygon
        poly = ['0,0', '1,0', '1,1', '0,1']
        config = self._make_config(poly, nlocations=20)
        result = RandomStrategy().seed(config)
        assert len(result) == 20
        for qty, x, y in result:
            assert qty == 1
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0

    def test_inline_poly_reproducible(self):
        poly = ['0,0', '2,0', '2,2', '0,2']
        config = self._make_config(poly, nlocations=5, seed=7)
        r1 = RandomStrategy().seed(config)
        r2 = RandomStrategy().seed(config)
        assert r1 == r2

    def test_poly_from_file(self, tmp_path):
        p = tmp_path / 'sq.txt'
        p.write_text('0 0\n1 0\n1 1\n0 1\n')
        config = self._make_config(str(p), nlocations=5)
        result = RandomStrategy().seed(config)
        assert len(result) == 5

    def test_missing_both_bbox_and_poly(self):
        config = PopulationConfig(
            {
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'random': {'nlocations': 1, 'seed': 1}},
                    'quantity': 1,
                    'release_start': '2025-01-01 00:00:00',
                    'burial_depth': {'constant': 0.0},
                },
            }
        )
        with pytest.raises(MissingConfigurationParameter, match='"bbox" or "poly"'):
            RandomStrategy().seed(config)

    def test_poly_triangular_points_inside(self):
        # Right triangle: (0,0), (2,0), (0,2)
        poly = ['0,0', '2,0', '0,2']
        config = self._make_config(poly, nlocations=50, seed=99)
        result = RandomStrategy().seed(config)
        assert len(result) == 50
        for _, x, y in result:
            # All points must satisfy x+y <= 2 (inside triangle, roughly)
            assert x + y <= 2.0 + 1e-9


# ---------------------------------------------------------------------------
# GridStrategy with poly
# ---------------------------------------------------------------------------


class TestGridStrategyPoly:
    def _make_config(self, poly, dx=1.0, dy=1.0):
        return PopulationConfig(
            {
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {
                        'grid': {
                            'poly': poly,
                            'separation': {'dx': dx, 'dy': dy},
                        }
                    },
                    'quantity': 1,
                    'release_start': '2025-01-01 00:00:00',
                    'burial_depth': {'constant': 0.0},
                },
            }
        )

    def test_inline_unit_square_grid(self):
        # 2x2 unit square: grid at 0,0  0,1  1,0  1,1 (corners on boundary)
        poly = ['0,0', '2,0', '2,2', '0,2']
        config = self._make_config(poly, dx=1.0, dy=1.0)
        result = GridStrategy().seed(config)
        positions = {(x, y) for _, x, y in result}
        # All integer grid points inside (or on boundary of) 2x2 square
        assert (0.0, 0.0) in positions
        assert (1.0, 1.0) in positions
        assert (2.0, 2.0) in positions

    def test_triangular_poly_filters_points(self):
        # Triangle (0,0), (4,0), (0,4): diagonal cuts out top-right
        poly = ['0,0', '4,0', '0,4']
        config = self._make_config(poly, dx=1.0, dy=1.0)
        result = GridStrategy().seed(config)
        for _, x, y in result:
            assert x + y <= 4.0 + 1e-9  # must be inside triangle

    def test_poly_fewer_points_than_full_bbox(self):
        # Rectangular bbox would give 3x3=9 points; triangle gives fewer
        poly = ['0,0', '2,0', '0,2']
        config = self._make_config(poly, dx=1.0, dy=1.0)
        result = GridStrategy().seed(config)
        assert len(result) < 9

    def test_poly_from_file(self, tmp_path):
        p = tmp_path / 'sq.csv'
        p.write_text('x,y\n0,0\n3,0\n3,3\n0,3\n')
        config = self._make_config(str(p), dx=1.0, dy=1.0)
        result = GridStrategy().seed(config)
        positions = {(x, y) for _, x, y in result}
        assert (0.0, 0.0) in positions
        assert (3.0, 3.0) in positions

    def test_missing_both_bbox_and_poly(self):
        config = PopulationConfig(
            {
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'grid': {'separation': {'dx': 1.0, 'dy': 1.0}}},
                    'quantity': 1,
                    'release_start': '2025-01-01 00:00:00',
                    'burial_depth': {'constant': 0.0},
                },
            }
        )
        with pytest.raises(MissingConfigurationParameter, match='"bbox" or "poly"'):
            GridStrategy().seed(config)

    def test_pol_file_format(self, tmp_path):
        p = tmp_path / 'area.pol'
        p.write_text('test_polygon\n4 2\n0.0 0.0\n2.0 0.0\n2.0 2.0\n0.0 2.0\n')
        config = self._make_config(str(p), dx=1.0, dy=1.0)
        result = GridStrategy().seed(config)
        positions = {(x, y) for _, x, y in result}
        assert (0.0, 0.0) in positions
        assert (2.0, 2.0) in positions


# ---------------------------------------------------------------------------
# Seeding box volume logging
# ---------------------------------------------------------------------------


class TestComputeSeedingArea:
    def test_bbox_string(self):
        area = _compute_seeding_area('random', {'bbox': '0,0 4,3'})
        assert area == pytest.approx(12.0)

    def test_bbox_dict(self):
        area = _compute_seeding_area('grid', {'bbox': {'xmin': 1.0, 'ymin': 2.0, 'xmax': 5.0, 'ymax': 6.0}})
        assert area == pytest.approx(16.0)

    def test_poly_square(self):
        area = _compute_seeding_area('random', {'poly': ['0,0', '2,0', '2,2', '0,2']})
        assert area == pytest.approx(4.0)

    def test_poly_triangle(self):
        # right triangle base=4, height=4 → area=8
        area = _compute_seeding_area('grid', {'poly': ['0,0', '4,0', '0,4']})
        assert area == pytest.approx(8.0)

    def test_point_strategy_returns_none(self):
        assert _compute_seeding_area('point', {'locations': ['0,0']}) is None

    def test_transect_strategy_returns_none(self):
        assert _compute_seeding_area('transect', {'segments': ['0,0 1,1'], 'k': 2}) is None

    def test_file_points_strategy_returns_none(self):
        assert _compute_seeding_area('file_points', {'path': 'x.csv'}) is None

    def test_no_area_key_returns_none(self):
        assert _compute_seeding_area('random', {'seed': 1, 'nlocations': 5}) is None


class TestLogSeedingBoxVolume:
    def _make_random_config(self, burial_depth, bbox='0,0 4,3', nlocations=6, quantity=2):
        return PopulationConfig(
            {
                'name': 'test_pop',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'random': {'bbox': bbox, 'nlocations': nlocations, 'seed': 1}},
                    'quantity': quantity,
                    'release_start': '2025-01-01 00:00:00',
                    'burial_depth': burial_depth,
                },
            }
        )

    def test_logs_when_random_burial_and_bbox(self, caplog):
        config = self._make_random_config({'random': 3.0})
        positions = [(2, 1.0, 1.0), (2, 2.0, 2.0), (2, 3.0, 3.0)]  # 6 particles total
        with caplog.at_level('INFO', logger='sedtrails.particle_tracer.particle_seeder'):
            _log_seeding_box_volume(config, positions)
        assert len(caplog.records) == 1
        msg = caplog.records[0].message
        # area = 4*3=12, depth=3, volume=36, n=6, repr=6.0
        assert '12' in msg
        assert '36' in msg
        assert '6' in msg
        assert 'test_pop' in msg

    def test_no_log_for_constant_burial(self, caplog):
        config = self._make_random_config({'constant': 1.0})
        positions = [(1, 0.0, 0.0)]
        with caplog.at_level('INFO', logger='sedtrails.particle_tracer.particle_seeder'):
            _log_seeding_box_volume(config, positions)
        assert caplog.records == []

    def test_no_log_for_point_strategy(self, caplog):
        config = PopulationConfig(
            {
                'name': 'pts',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'point': {'locations': ['0,0', '1,1']}},
                    'quantity': 1,
                    'release_start': '2025-01-01 00:00:00',
                    'burial_depth': {'random': 2.0},
                },
            }
        )
        positions = [(1, 0.0, 0.0), (1, 1.0, 1.0)]
        with caplog.at_level('INFO', logger='sedtrails.particle_tracer.particle_seeder'):
            _log_seeding_box_volume(config, positions)
        assert caplog.records == []

    def test_logs_with_poly(self, caplog):
        config = PopulationConfig(
            {
                'name': 'poly_pop',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'random': {'poly': ['0,0', '2,0', '2,2', '0,2'], 'nlocations': 4, 'seed': 1}},
                    'quantity': 1,
                    'release_start': '2025-01-01 00:00:00',
                    'burial_depth': {'random': 5.0},
                },
            }
        )
        # 4 positions × qty 1 = 4 particles; area=4, depth=5, volume=20, repr=5
        positions = [(1, 0.5, 0.5), (1, 1.5, 0.5), (1, 0.5, 1.5), (1, 1.5, 1.5)]
        with caplog.at_level('INFO', logger='sedtrails.particle_tracer.particle_seeder'):
            _log_seeding_box_volume(config, positions)
        assert len(caplog.records) == 1
        msg = caplog.records[0].message
        assert '20' in msg  # volume
        assert '4' in msg  # n_particles or area
        assert '5' in msg  # depth or repr volume

    def test_factory_emits_log_for_random_burial_bbox(self, caplog):
        config = PopulationConfig(
            {
                'name': 'factory_test',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'random': {'bbox': '0,0 10,10', 'nlocations': 5, 'seed': 42}},
                    'quantity': 2,
                    'release_start': '2025-01-01 00:00:00',
                    'burial_depth': {'random': 2.0},
                },
            }
        )
        with caplog.at_level('INFO', logger='sedtrails.particle_tracer.particle_seeder'):
            ParticleFactory.create_particles(config)
        assert any('volume' in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Permanently-buried particle removal
# ---------------------------------------------------------------------------


def _make_population(burial_depth_cfg, remove_flag=False, n_pts=4):
    """Create a minimal ParticlePopulation on a unit-square grid."""
    config = PopulationConfig(
        {
            'name': 'test_pop',
            'particle_type': 'sand',
            'seeding': {
                'strategy': {'point': {'locations': [f'{i},{i}' for i in range(n_pts)]}},
                'quantity': 1,
                'release_start': '2025-01-01 00:00:00',
                'burial_depth': burial_depth_cfg,
                'remove_permanently_buried': remove_flag,
            },
        }
    )
    # Grid: unit square with enough nodes to contain the seed points
    field_x = np.array([0.0, 4.0, 4.0, 0.0])
    field_y = np.array([0.0, 0.0, 4.0, 4.0])
    return ParticlePopulation(field_x=field_x, field_y=field_y, population_config=config)


class TestRemovePermanentlyBuriedParticles:
    def test_flag_off_removes_nothing(self):
        pop = _make_population({'constant': 5.0}, remove_flag=False)
        n_before = len(pop.particles['x'])
        # Even with zero max_exposure, nothing is removed when flag is off
        max_exposure = np.zeros(4)
        removed = pop.remove_permanently_buried_particles(max_exposure)
        assert removed == 0
        assert len(pop.particles['x']) == n_before

    def test_removes_particles_deeper_than_exposure(self):
        # 4 particles, burial_depth = 5.0 (constant)
        # max_exposure at every node = 3.0  →  5.0 > 3.0, all removed
        pop = _make_population({'constant': 5.0}, remove_flag=True)
        n_before = len(pop.particles['x'])
        max_exposure = np.full(4, 3.0)
        removed = pop.remove_permanently_buried_particles(max_exposure)
        assert removed == n_before
        assert len(pop.particles['x']) == 0

    def test_keeps_particles_shallower_than_exposure(self):
        # burial_depth = 1.0, max_exposure = 3.0  →  all kept
        pop = _make_population({'constant': 1.0}, remove_flag=True)
        n_before = len(pop.particles['x'])
        max_exposure = np.full(4, 3.0)
        removed = pop.remove_permanently_buried_particles(max_exposure)
        assert removed == 0
        assert len(pop.particles['x']) == n_before

    def test_partial_removal(self):
        # Place particles exactly at grid node positions so interpolation is exact.
        # Grid nodes: (0,0), (4,0), (4,4), (0,4).
        # burial_depth = 2.0 for all particles.
        # max_exposure at nodes: [10, 10, 0.5, 0.5]
        #   → particles at (0,0),(4,0): 2.0 ≤ 10  → kept
        #   → particles at (4,4),(0,4): 2.0 > 0.5 → removed
        config = PopulationConfig(
            {
                'name': 'partial_test',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'point': {'locations': ['0,0', '4,0', '4,4', '0,4']}},
                    'quantity': 1,
                    'release_start': '2025-01-01 00:00:00',
                    'burial_depth': {'constant': 2.0},
                    'remove_permanently_buried': True,
                },
            }
        )
        field_x = np.array([0.0, 4.0, 4.0, 0.0])
        field_y = np.array([0.0, 0.0, 4.0, 4.0])
        pop = ParticlePopulation(field_x=field_x, field_y=field_y, population_config=config)
        max_exposure = np.array([10.0, 10.0, 0.5, 0.5])
        n_before = len(pop.particles['x'])
        removed = pop.remove_permanently_buried_particles(max_exposure)
        assert 0 < removed < n_before
        assert len(pop.particles['x']) == n_before - removed

    def test_simplices_updated_after_removal(self):
        pop = _make_population({'constant': 5.0}, remove_flag=True)
        n_before = len(pop._particle_simplices)
        max_exposure = np.full(4, 3.0)
        removed = pop.remove_permanently_buried_particles(max_exposure)
        assert len(pop._particle_simplices) == n_before - removed

    def test_all_particle_arrays_trimmed(self):
        pop = _make_population({'constant': 5.0}, remove_flag=True)
        # Manually set extra particle keys to test they are all trimmed
        n = len(pop.particles['x'])
        pop.particles['burial_depth'] = np.full(n, 5.0)
        pop.particles['some_extra_field'] = np.ones(n)
        max_exposure = np.zeros(4)
        pop.remove_permanently_buried_particles(max_exposure)
        for key, arr in pop.particles.items():
            assert len(arr) == 0, f"Key '{key}' not trimmed"

    def test_nan_exposure_keeps_particle(self):
        # NaN max_exposure should be treated as inf (keep particle conservatively)
        pop = _make_population({'constant': 999.0}, remove_flag=True)
        max_exposure = np.full(4, np.nan)
        removed = pop.remove_permanently_buried_particles(max_exposure)
        assert removed == 0

    def test_logs_removal_info(self, caplog):
        pop = _make_population({'constant': 5.0}, remove_flag=True)
        max_exposure = np.zeros(4)
        with caplog.at_level('WARNING', logger='sedtrails.particle_tracer.particle_seeder'):
            pop.remove_permanently_buried_particles(max_exposure)
        assert any('permanently buried' in r.message.lower() for r in caplog.records)

    def test_logs_no_removal_debug(self, caplog):
        pop = _make_population({'constant': 0.5}, remove_flag=True)
        max_exposure = np.full(4, 10.0)
        with caplog.at_level('DEBUG', logger='sedtrails.particle_tracer.particle_seeder'):
            pop.remove_permanently_buried_particles(max_exposure)
        assert any('no permanently buried' in r.message.lower() for r in caplog.records)

    def test_config_flag_read_from_population_config(self):
        config = PopulationConfig(
            {
                'name': 'p',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'point': {'locations': ['0,0']}},
                    'quantity': 1,
                    'release_start': '2025-01-01 00:00:00',
                    'burial_depth': {'constant': 0.0},
                    'remove_permanently_buried': True,
                },
            }
        )
        assert config.remove_permanently_buried is True

    def test_config_flag_defaults_to_false(self):
        config = PopulationConfig(
            {
                'name': 'p',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'point': {'locations': ['0,0']}},
                    'quantity': 1,
                    'release_start': '2025-01-01 00:00:00',
                    'burial_depth': {'constant': 0.0},
                    # remove_permanently_buried not specified
                },
            }
        )
        assert config.remove_permanently_buried is False

    def test_release_time_is_converted_to_seconds_since_reference_date(self):
        population = _single_particle_population(release_start='1970-01-01 00:10:00')

        np.testing.assert_array_equal(population.particles['release_time'], np.array([600.0]))

    def test_update_status_respects_release_time(self):
        population = _single_particle_population(release_start='1970-01-01 00:10:00')
        population.particles['transport_probability'] = np.ones_like(population.particles['x'])

        population._current_time = 599.0
        population.update_status()

        assert population.particles['status_released'].tolist() == [False]
        assert population.particles['status_eligible'].tolist() == [False]
        np.testing.assert_array_equal(population.particles['release_time'], np.array([600.0]))

        population._current_time = 600.0
        population.update_status()

        assert population.particles['status_released'].tolist() == [True]
        assert population.particles['status_eligible'].tolist() == [True]
        np.testing.assert_array_equal(population.particles['release_time'], np.array([600.0]))

    def test_invalid_release_time_raises_date_format_error(self):
        with pytest.raises(DateFormatError):
            _single_particle_population(release_start='1970/01/01 00:10:00')

    def test_release_time_before_reference_date_warns(self):
        with pytest.warns(UserWarning, match='Computed release time is negative'):
            population = _single_particle_population(release_start='1969-12-31 23:50:00')

        np.testing.assert_array_equal(population.particles['release_time'], np.array([-600.0]))

    def test_missing_release_start_defaults_to_simulation_start(self):
        config = PopulationConfig(
            {
                'name': 'Release Time Config',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'point': {'locations': ['0.5,0.5']}},
                    'quantity': 1,
                    'burial_depth': {
                        'constant': 0.0,
                    },
                },
                'transport_probability': 'no_probability',
            }
        )
        population = ParticlePopulation(
            field_x=np.array([0.0, 1.0, 0.0, 1.0]),
            field_y=np.array([0.0, 0.0, 1.0, 1.0]),
            population_config=config,
            reference_date=np.datetime64('1970-01-01T00:00:00', 's'),
        )

        np.testing.assert_array_equal(population.particles['release_time'], np.array([0.0]))

    def test_particle_seeder_uses_field_data_reference_date_for_release_time(self):
        class FieldData:
            x = np.array([0.0, 1.0, 0.0, 1.0])
            y = np.array([0.0, 0.0, 1.0, 1.0])
            reference_date = np.datetime64('2000-01-01T00:00:00', 's')

        populations = ParticleSeeder(
            {
                'name': 'Release Time Config',
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'point': {'locations': ['0.5,0.5']}},
                    'quantity': 1,
                    'release_start': '2000-01-01 01:00:00',
                    'burial_depth': {
                        'constant': 0.0,
                    },
                },
                'transport_probability': 'no_probability',
            }
        ).seed(FieldData())

        np.testing.assert_array_equal(populations[0].particles['release_time'], np.array([3600.0]))


def _single_particle_population(release_start):
    config = PopulationConfig(
        {
            'name': 'Release Time Config',
            'particle_type': 'sand',
            'seeding': {
                'strategy': {'point': {'locations': ['0.5,0.5']}},
                'quantity': 1,
                'release_start': release_start,
                'burial_depth': {
                    'constant': 0.0,
                },
            },
            'transport_probability': 'no_probability',
        }
    )
    return ParticlePopulation(
        field_x=np.array([0.0, 1.0, 0.0, 1.0]),
        field_y=np.array([0.0, 0.0, 1.0, 1.0]),
        population_config=config,
        reference_date=np.datetime64('1970-01-01T00:00:00', 's'),
    )


def _diffusion_population(particle_type, tracer_methods, characteristics, diffusion):
    """Build a one-particle population using the canonical diffusion config."""
    config = {
        'name': 'Configured Diffusion Population',
        'particle_type': particle_type,
        'characteristics': characteristics,
        'tracer_methods': tracer_methods,
        'transport_probability': 'no_probability',
        'seeding': {
            'strategy': {'point': {'locations': ['0.5,0.5']}},
            'quantity': 1,
            'release_start': '0',
            'burial_depth': {'constant': 0.0},
        },
    }
    if diffusion is not None:
        config['diffusion'] = diffusion
    return ParticlePopulation(
        field_x=np.array([0.0, 1.0, 1.0, 0.0]),
        field_y=np.array([0.0, 0.0, 1.0, 1.0]),
        population_config=PopulationConfig(config),
    )


def _boundary_action_config(location):
    return {
        'name': 'Boundary Action Config',
        'particle_type': 'sand',
        'transport_probability': 'no_probability',
        'seeding': {
            'strategy': {'point': {'locations': [location]}},
            'quantity': 1,
            'release_start': '0',
            'burial_depth': {'constant': 0.0},
        },
    }


def _boundary_action_field_data():
    return SimpleNamespace(
        x=np.array([0.0, 1.0, 0.0]),
        y=np.array([0.0, 0.0, 1.0]),
        face_node_connectivity=np.array([[0, 1, 2]], dtype=np.int64),
        boundary_edge_classification={
            'edge_nodes': [[0, 1], [2, 0], [1, 2]],
            'edge_classes': ['open', 'land', 'unclassified'],
        },
    )


def _macdonald_2d_test_population():
    config = PopulationConfig(
        {
            'name': 'MacDonald 2D deposition test',
            'particle_type': 'sand',
            'transport_probability': 'no_probability',
            'tracer_methods': {'macdonald': {'computationType': '2D'}},
            'seeding': {
                'strategy': {'point': {'locations': ['0.5,0.5']}},
                'quantity': 1,
                'release_start': '0',
                'burial_depth': {'constant': 0.0},
                'vertical_position': {'mode': 'height_above_bed', 'value': 0.1},
            },
        }
    )
    population = ParticlePopulation(
        field_x=np.array([0.0, 1.0, 1.0, 0.0]),
        field_y=np.array([0.0, 0.0, 1.0, 1.0]),
        population_config=config,
    )
    population._current_time = 0.0
    population.update_status()
    population.initialize_macdonald_2d_release_state()
    return population


def _constant_flow_field(speed):
    return {
        'u': np.full(4, speed, dtype=float),
        'v': np.zeros(4, dtype=float),
        'magnitude': np.full(4, abs(speed), dtype=float),
    }


def test_macdonald_2d_release_state_maps_bed_and_burial_modes():
    population = _macdonald_2d_test_population()

    population.particles['vertical_position_initialized'][:] = False
    population.particles['vertical_position_mode'][:] = 'bed'
    population.particles['status_deposited'][:] = False
    population.particles['status_suspended'][:] = True
    population.initialize_macdonald_2d_release_state()

    assert population.particles['status_deposited'].tolist() == [True]
    assert population.particles['status_suspended'].tolist() == [False]
    assert population.particles['status_buried'].tolist() == [False]

    population.particles['vertical_position_initialized'][:] = False
    population.particles['vertical_position_mode'][:] = 'burial_depth'
    population.particles['burial_depth'][:] = 0.2
    population.initialize_macdonald_2d_release_state()

    assert population.particles['status_deposited'].tolist() == [True]
    assert population.particles['status_suspended'].tolist() == [False]
    # no_probability deliberately skips the burial-depth/mixing-depth comparison.
    assert population.particles['status_buried'].tolist() == [False]
    assert population.particles['status_eligible'].tolist() == [True]


def test_burial_status_recalculation_uses_initialized_depth_immediately():
    population = _macdonald_2d_test_population()
    population.particles['mixing_depth'] = np.array([0.1])

    population.population_config.population_config['transport_probability'] = 'stochastic_transport'
    assert population.update_status_buried(np.array([0.2])).tolist() == [True]

    population.population_config.population_config['transport_probability'] = 'no_probability'
    assert population.update_status_buried(np.array([0.2])).tolist() == [False]

def test_vanwesten_burial_status_regression_is_unchanged():
    config = PopulationConfig(
        {
            'name': 'Van Westen burial regression',
            'particle_type': 'sand',
            'transport_probability': 'stochastic_transport',
            'tracer_methods': {'vanwesten': {}},
            'seeding': {
                'strategy': {'point': {'locations': ['0.25,0.5', '0.75,0.5']}},
                'quantity': 1,
                'release_start': '0',
                'burial_depth': {'constant': 0.2},
            },
        }
    )
    population = ParticlePopulation(
        field_x=np.array([0.0, 1.0, 1.0, 0.0]),
        field_y=np.array([0.0, 0.0, 1.0, 1.0]),
        population_config=config,
    )
    population._current_time = 0.0
    population.particles['mixing_depth'] = np.array([0.1, 0.3])
    population.particles['transport_probability'] = np.ones(2)
    population.update_status()

    np.testing.assert_allclose(population.particles['burial_depth'], [0.2, 0.2])
    assert population.particles['status_buried'].tolist() == [True, False]
    assert population.particles['status_eligible'].tolist() == [False, True]
    assert population.particles['status_deposited'].tolist() == [True, True]
    assert population.particles['status_suspended'].tolist() == [False, False]
    assert population.particles['vertical_position_initialized'].tolist() == [False, False]

def test_macdonald_2d_shields_threshold_deposits_and_reentrains():
    population = _macdonald_2d_test_population()

    population.sample_macdonald_2d_transition_fields(
        particle_velocity_field=_constant_flow_field(1.0),
        shields_number_field=np.full(4, 0.04),
    )
    population.update_macdonald_2d_deposition(
        method='shields_threshold',
        critical_shields_number=0.05,
        current_timestep=1.0,
    )
    assert population.particles['status_deposited'].tolist() == [True]
    assert population.particles['status_mobile'].tolist() == [False]

    # Shields-only entrainment must not depend on the particle velocity field.
    population.sample_macdonald_2d_transition_fields(
        shields_number_field=np.full(4, 0.06),
    )
    population.update_macdonald_2d_entrainment(
        method='shields_threshold',
        critical_shields_number=0.05,
        current_timestep=1.0,
    )
    assert population.particles['status_deposited'].tolist() == [False]
    assert population.particles['status_suspended'].tolist() == [True]
    assert population.particles['status_mobile'].tolist() == [True]


def test_macdonald_2d_frequency_entrainment_uses_poisson_probability():
    population = _macdonald_2d_test_population()
    population.particles['status_deposited'][:] = True
    population.particles['status_suspended'][:] = False
    population.sample_macdonald_2d_transition_fields(
        entrainment_frequency_field=np.full(4, 1.0),
    )

    population.update_macdonald_2d_entrainment(
        method='entrainment_frequency',
        critical_shields_number=None,
        current_timestep=100.0,
        probability_law='poisson',
        rng=np.random.default_rng(1),
    )

    assert population.particles['macdonald_2d_entrainment_probability'][0] == pytest.approx(1.0)
    assert population.particles['status_entrained_now'].tolist() == [True]
    assert population.particles['status_deposited'].tolist() == [False]
    assert population.particles['status_suspended'].tolist() == [True]
    assert population.particles['status_mobile'].tolist() == [True]

@pytest.mark.parametrize(
    'entrainment_method',
    ['shields_threshold', 'non_zero_particle_velocity', 'entrainment_frequency'],
)
@pytest.mark.parametrize('deposition_method', ['shields_threshold', 'markov_settling'])
def test_macdonald_2d_entrainment_deposition_combinations(
    entrainment_method,
    deposition_method,
):
    population = _macdonald_2d_test_population()
    population.particles['status_deposited'][:] = True
    population.particles['status_suspended'][:] = False
    population.sample_macdonald_2d_transition_fields(
        particle_velocity_field=_constant_flow_field(1.0),
        shields_number_field=np.full(4, 0.06),
        entrainment_frequency_field=np.full(4, 100.0),
        settling_height_field=np.full(4, 1.0),
        shear_velocity_field=np.full(4, 0.05),
    )

    population.update_macdonald_2d_entrainment(
        method=entrainment_method,
        critical_shields_number=0.05,
        current_timestep=1.0,
        rng=np.random.default_rng(1),
    )
    population.update_macdonald_2d_deposition(
        method=deposition_method,
        critical_shields_number=0.05,
        current_timestep=1.0,
        settling_velocity=0.0 if deposition_method == 'markov_settling' else None,
        rng=np.random.default_rng(1),
    )

    assert population.particles['status_deposited'].tolist() == [False]
    assert population.particles['status_suspended'].tolist() == [True]
    assert population.particles['status_mobile'].tolist() == [True]

def test_markov_settling_height_choice_changes_transition_rate():
    shallow_rate, _ = ParticlePopulation._markov_settling_rate(0.01, 0.1, 0.05, 0.001)
    full_depth_rate, _ = ParticlePopulation._markov_settling_rate(0.01, 1.0, 0.05, 0.001)

    assert shallow_rate == pytest.approx(10.0 * full_depth_rate)


def test_macdonald_2d_markov_settling_updates_probability_and_status():
    population = _macdonald_2d_test_population()
    rng = np.random.default_rng(1)

    population.particles['status_mobile'] = population.particles['status_eligible'].copy()
    population.sample_macdonald_2d_transition_fields(
        settling_height_field=np.full(4, 0.001),
        shear_velocity_field=np.zeros(4),
    )
    population.update_macdonald_2d_deposition(
        method='markov_settling',
        critical_shields_number=0.05,
        current_timestep=1000.0,
        settling_velocity=0.01,
        minimum_settling_height=0.001,
        rng=rng,
    )

    assert population.particles['macdonald_2d_settling_probability'][0] == pytest.approx(1.0)
    assert population.particles['status_deposited'].tolist() == [True]
    assert population.particles['status_deposited_now'].tolist() == [True]

def test_macdonald_2d_shields_deposition_prevents_horizontal_movement():
    population = _macdonald_2d_test_population()
    population.particles['status_deposited'][:] = False
    population.particles['status_mobile'] = np.ones(1, dtype=bool)
    flow = _constant_flow_field(0.25)

    population.sample_macdonald_2d_transition_fields(
        particle_velocity_field=flow,
        shields_number_field=np.full(4, 0.01),
    )
    population.update_macdonald_2d_deposition(
        method='shields_threshold',
        critical_shields_number=0.05,
        current_timestep=1.0,
    )
    population.update_position(flow_field=flow, current_timestep=1.0)

    assert population.particles['status_deposited'].tolist() == [True]
    assert population.particles['status_mobile'].tolist() == [False]
    assert population.particles['x'][0] == pytest.approx(0.5)

def test_macdonald_2d_initializes_not_deposited_and_status_only_sets_eligibility():
    population = _macdonald_2d_test_population()

    assert population.particles['status_deposited'].tolist() == [False]
    assert population.particles['status_suspended'].tolist() == [True]

    population.update_status()

    assert population.particles['status_eligible'].tolist() == [True]
    assert 'status_deposition_initialized' not in population.particles

def test_macdonald_2d_markov_deposits_after_completed_movement():
    population = _macdonald_2d_test_population()
    flow = _constant_flow_field(0.25)

    population.particles['status_mobile'] = population.particles['status_eligible'].copy()
    population.sample_macdonald_2d_transition_fields(
        settling_height_field=np.full(4, 0.001),
        shear_velocity_field=np.zeros(4),
    )
    population.update_position(flow_field=flow, current_timestep=1.0)
    population.update_macdonald_2d_deposition(
        method='markov_settling',
        critical_shields_number=None,
        current_timestep=1000.0,
        settling_velocity=0.01,
        minimum_settling_height=0.001,
        rng=np.random.default_rng(1),
    )

    assert population.particles['x'][0] == pytest.approx(0.75)
    assert population.particles['status_deposited'].tolist() == [True]
    assert 'macdonald_2d_particle_velocity_magnitude' not in population.particles
    assert 'macdonald_2d_shields_number' not in population.particles
