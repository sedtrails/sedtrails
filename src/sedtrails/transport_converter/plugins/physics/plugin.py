"""
A common interface for physics plugins.
Plugins must inherit from this class and implement the convert method.
"""

from abc import ABC, abstractmethod
from sedtrails.transport_converter import SedtrailsData


class BasePhysicsPlugin(ABC):
    """
    Abstract base class for physics plugins.
    """

    def __init__(self):
        return None

    @abstractmethod
    def add_physics(self, sedtrails_data: SedtrailsData, *args, **kwargs):
        """
        Computes the physics of particles in the simulation based on particle types.
        """
        pass

    # additional methods can be added here
