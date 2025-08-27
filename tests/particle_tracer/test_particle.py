"""
Unit tests for data classes in the particle.py module of the sedtrails package.
"""

import pytest
from sedtrails.particle_tracer.particle import Mud, Sand, Passive, PhysicalProperties


class TestParticle:
    """
    Test suite for the Particle class and its children.
    """

    def test_mud_initialization(self):
        """
        Test the initialization of the Mud class.
        """

        mud = Mud()
        assert isinstance(mud, Mud)

    def test_sand_initialization(self):
        """
        Test the initialization of the Sand class.
        """
        sand = Sand()
        assert isinstance(sand, Sand)

    def test_passive_initialization(self):
        """
        Test the initialization of the Passive class.
        """
        passive = Passive()
        assert isinstance(passive, Passive)

    def test_name_property_is_optional(self):
        """
        Test that the name property is optional in the Particle class, and
        the default value is an empty string.
        """
        particle = Sand()
        assert particle.name == ''

        particle = Mud()
        assert particle.name == ''

        particle = Passive()
        assert particle.name == ''


class TestMud:
    """
    Test suite for the Mud class.
    """

    # TODO: implement tests for Mud class


class TestSand:
    """
    Test suite for the Sand class.
    """

    # TODO: implement tests for Mud class


class TestPassive:
    """
    Test suite for the Passive class.
    """

    # TODO: implement tests for Mud class


class TestInterpolatedValue:
    """
    Test suite for the InterpolatedValue class.
    """


class TestPhysics:
    """
    Test suite for the Physics class.
    """


class TestPhysicalProperties:
    """
    Test suite for the PhysicalProperties class.
    """

    def test_valid_initialization(self):
        """
        Test initialization with valid values.
        """
        props = PhysicalProperties(density=2650.0, diameter=2e-4)
        assert props.density == 2650.0
        assert props.diameter == 2e-4

    def test_invalid_density(self):
        """
        Test initialization with invalid density values.
        """
        with pytest.raises(ValueError, match='Density must be positive'):
            PhysicalProperties(density=-1.0, diameter=2e-4)

        with pytest.raises(ValueError, match='Density must be positive'):
            PhysicalProperties(density=0.0, diameter=2e-4)

    def test_invalid_diameter(self):
        """
        Test initialization with invalid diameter values.
        """
        with pytest.raises(ValueError, match='Diameter must be positive'):
            PhysicalProperties(density=2650.0, diameter=-2e-4)

        with pytest.raises(ValueError, match='Diameter must be positive'):
            PhysicalProperties(density=2650.0, diameter=0.0)

    def test_particle_physical_properties(self):
        """
        Test physical properties for different particle types.
        """
        # Test Mud properties
        mud = Mud()
        assert mud.physical_properties.density == 2650.0
        assert mud.physical_properties.diameter == 2e-6
