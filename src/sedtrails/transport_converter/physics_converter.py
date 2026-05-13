"""
Physics Converter Module for Sediment Transport

This module adds physics-based calculations to existing SedtrailsData objects
using the physics library functions and allowing method selection.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

# Import physics library
from sedtrails.transport_converter import physics_lib

# Physical constants
GRAVITY = 9.81  # m/s^2
VON_KARMAN_CONSTANT = 0.40  # [-]
KINEMATIC_VISCOSITY = 1.36e-6  # m^2/s (10°C, 35 ppt)
WATER_DENSITY = 1027.0  # kg/m^3 (10°C, 35 ppt)
PARTICLE_DENSITY = 2650.0  # kg/m^3 (quartz)
POROSITY = 0.4  # [-]
# Sediment properties
GRAIN_DIAMETER = 2.5e-4  # m (250 μm)
# Morphological acceleration factor
MORFAC = 1.0


@dataclass
class PhysicsConfig:
    """Configuration parameters for physics calculations."""

    # Physics methods
    tracer_method: str = 'vanwesten'  # name of method for
    gravity: float = GRAVITY
    von_karman_constant: float = VON_KARMAN_CONSTANT
    kinematic_viscosity: float = KINEMATIC_VISCOSITY
    water_density: float = WATER_DENSITY
    particle_density: float = PARTICLE_DENSITY
    porosity: float = POROSITY
    grain_diameter: float = GRAIN_DIAMETER
    morfac: float = MORFAC

    @classmethod
    def from_dict(
        cls, config: Optional[Dict[str, Any]] = None, tracer_config: Optional[Dict[str, Any]] = None
    ) -> 'PhysicsConfig':
        """
        Build a PhysicsConfig by merging:
        1) defaults (class fields),
        2) base config dict,
        3) method-specific tracer_config (flattened into attributes).
        Supports tracer_config passed either as a flat dict, or nested under the method name.
        """
        # Start from defaults
        obj = cls()
        # Apply base config
        if config:
            if isinstance(config, cls):
                base = asdict(config)  # deep copy dataclass fields
            elif isinstance(config, dict):
                base = config
            else:
                base = {k: v for k, v in vars(config).items() if not k.startswith('_')}
            for k, v in base.items():
                setattr(obj, k, v)
        # Apply method-specific (flatten) from tracer_config
        if tracer_config:
            method = getattr(config, 'tracer_method', 'vanwesten')
            # If tracer_config is nested like {"soulsby": {...}}, pick the active method
            if isinstance(tracer_config.get(method, None), dict):
                method_params = tracer_config[method]
                for k, v in method_params.items():
                    setattr(obj, k, v)
            else:
                # Otherwise assume tracer_config is already flat
                for k, v in tracer_config.items():
                    setattr(obj, k, v)
        return obj

    # Optional: expose a dict view when needed
    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class PhysicsConverter:
    """
    Add physics calculations to existing SedtrailsData using different methods.

    Each method (van Westen, Soulsby, etc.) has its own complete workflow.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, tracer_config: Optional[dict] = None):
        """
        Initialize the physics converter.

        Parameters:
        -----------
        config : dict, optional
            Base config for physics (e.g., tracer_method, global constants overrides).
        tracer_config : dict, optional
            Method-specific parameters (e.g., soulsby params). These will be flattened into PhysicsConfig.
        """
        self.config = PhysicsConfig.from_dict(config=config or {}, tracer_config=tracer_config or {})
        self._grain_properties: dict = {}
        self._physics_plugin = None
        self.tracer_config = tracer_config

    @property
    def grain_properties(self):
        """Get the calculated grain properties."""

        # lazy grain properties calculation
        if not self._grain_properties:
            self._calculate_grain_properties()
        return self._grain_properties

    def _calculate_grain_properties(self) -> None:
        """Calculate time-independent grain properties using physics library."""
        if self.config.tracer_method == 'soulsby':
            # Soulsby method uses tracer grain size
            self._grain_properties = physics_lib.compute_grain_properties(
                self.config.tracer_grain_size,
                self.config.gravity,
                self.config.particle_density,
                self.config.water_density,
                self.config.kinematic_viscosity,
            )
        elif self.config.tracer_method == 'macdonald':
            self._grain_properties = physics_lib.compute_grain_properties(
                self.config.grain_diameter,
                self.config.gravity,
                self.config.particle_density,
                self.config.water_density,
                self.config.kinematic_viscosity,
                settling_velocity_method="macdonald2006"
            )
        else:
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
                self._physics_plugin = plugin_module.PhysicsPlugin(
                    self.config, self.tracer_config
                )  # all classes should be called the PhysicsPlugin
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
