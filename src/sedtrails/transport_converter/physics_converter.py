"""
Physics Converter Module for Sediment Transport

This module adds physics-based calculations to existing SedtrailsData objects
using the physics library functions and allowing method selection.
"""

from typing import Optional
from dataclasses import dataclass

# Import physics library
from sedtrails.transport_converter import physics_lib


@dataclass
class PhysicsConfig:
    """Configuration parameters for physics calculations."""

    # Physics methods
    tracer_method: str = 'van_westen'  # name of method for

    # Physical constants
    gravity: float = 9.81  # m/s^2
    von_karman_constant: float = 0.40  # [-]
    kinematic_viscosity: float = 1.36e-6  # m^2/s (10°C, 35 ppt)
    water_density: float = 1027.0  # kg/m^3 (10°C, 35 ppt)
    particle_density: float = 2650.0  # kg/m^3 (quartz)
    porosity: float = 0.4  # [-]

    # Sediment properties
    grain_diameter: float = 2.5e-4  # m (250 μm)

    # Morphological acceleration factor
    morfac: float = 1.0


class PhysicsConverter:
    """
    Add physics calculations to existing SedtrailsData using different methods.

    Each method (van Westen, Soulsby, etc.) has its own complete workflow.
    """

    def __init__(self, config: Optional[PhysicsConfig] = None):
        """
        Initialize the physics converter.

        Parameters:
        -----------
        config : PhysicsConfig, optional
            Configuration for physics calculations. If None, uses defaults.
        """
        self.config = config or PhysicsConfig()
        self._grain_properties: dict = {}
        self._physics_plugin = None

    @property
    def grain_properties(self):
        """Get the calculated grain properties."""

        # lazy grain properties calculation
        if not self._grain_properties:
            self._calculate_grain_properties()
        return self._grain_properties

    def _calculate_grain_properties(self) -> None:
        """Calculate time-independent grain properties using physics library."""
        self._grain_properties = physics_lib.compute_grain_properties(
            self.config.grain_diameter,
            self.config.gravity,
            self.config.particle_density,
            self.config.water_density,
            self.config.kinematic_viscosity,
        )

    @property
    def physics_plugin(self, tracer_method: Optional[str] = None):
        """Get the physics plugin instance based on the configured method.

        Parameters:
        -----------
        tracer_method : str, optional
            The tracer method to use for physics calculations. If None, uses the configured method.
        """

        import importlib  # lazy import for performance

        if self._physics_plugin is None:
            if tracer_method is None:
                tracer_method = self.config.tracer_method
                # Dynamically import the physics plugin based on the configured method
            plugin_module_name = f'sedtrails.transport_converter.plugins.physics.{tracer_method}'
            try:
                plugin_module = importlib.import_module(plugin_module_name)
            except ImportError as e:
                raise ImportError(
                    f'Failed to import physics plugin module: {plugin_module_name} '
                    f'Ensure the module exists and is correctly named.'
                ) from e
            else:
                self._physics_plugin = plugin_module.PhysicsPlugin(self.config)
        return self._physics_plugin


    def convert_physics(self, sedtrails_data, transport_probability_method: str = None) -> None:
        """
        Converts and adds physics calculations to existing SedtrailsData object using the tracer method.

        Parameters:
        -----------
        sedtrails_data : SedtrailsData
            Existing SedtrailsData object to be enhanced with physics calculations.
        transport_probability_method : str, optional
            Method to use for transport probability effects
        """

        if self._physics_plugin is None:
            plugin = self.physics_plugin
        else:
            plugin = self._physics_plugin

        # Use empty dict as default if no config provided
        plugin.add_physics(sedtrails_data, self.grain_properties, transport_probability_method or {})