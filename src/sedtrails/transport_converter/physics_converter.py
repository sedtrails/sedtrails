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
# Bertin (2008) mixing layer empirical coefficient
BERTIN_COEFFICIENT = 0.041


@dataclass
class PhysicsConfig:
    """Configuration parameters for physics calculations.

    Attributes
    ----------
    tracer_method : str
        Name of the active sediment tracer physics method.
    suspended_velocity_method : str
        Method used to compute suspended sediment velocity.
    gravity : float
        Gravitational acceleration in m/s^2.
    von_karman_constant : float
        Von Karman constant.
    kinematic_viscosity : float
        Kinematic viscosity in m^2/s.
    water_density : float
        Water density in kg/m^3.
    particle_density : float
        Particle density in kg/m^3.
    porosity : float
        Bed porosity.
    grain_diameter : float
        Representative grain diameter in meters.
    morfac : float
        Morphological acceleration factor.
    bertin_coefficient : float
        Empirical coefficient for Bertin-style mixing layer thickness.
    """

    # Physics methods
    tracer_method: str = 'vanwesten'  # name of method for
    suspended_velocity_method: str = 'soulsby_2011'  # default suspended velocity method
    gravity: float = GRAVITY
    von_karman_constant: float = VON_KARMAN_CONSTANT
    kinematic_viscosity: float = KINEMATIC_VISCOSITY
    water_density: float = WATER_DENSITY
    particle_density: float = PARTICLE_DENSITY
    porosity: float = POROSITY
    grain_diameter: float = GRAIN_DIAMETER
    morfac: float = MORFAC
    bertin_coefficient: float = BERTIN_COEFFICIENT

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

        Parameters
        ----------
        config : Optional[Dict[str, Any]]
            Configuration mapping used by the operation.
        tracer_config : Optional[Dict[str, Any]]
            Tracer-method-specific configuration mapping.

        Returns
        -------
        'PhysicsConfig'
            Constructed physics configuration.
        """
        # Start from defaults
        obj = cls()

        # Apply base config
        if config:
            if isinstance(config, cls):
                base = asdict(config)
            elif isinstance(config, dict):
                base = config
            else:
                base = {k: v for k, v in vars(config).items() if not k.startswith('_')}
            for k, v in base.items():
                setattr(obj, k, v)

        # Apply method-specific (flatten) from tracer_config
        if tracer_config:
            method = obj.tracer_method
            if isinstance(tracer_config.get(method), dict):
                method_params = tracer_config[method]
            elif any(isinstance(value, dict) for value in tracer_config.values()):
                available_methods = sorted(key for key, value in tracer_config.items() if isinstance(value, dict))
                raise ValueError(
                    f'Nested tracer_config was provided, but it does not contain settings '
                    f"for the active tracer_method '{method}'. "
                    f'Available method keys: {available_methods}'
                )
            else:
                method_params = tracer_config

            for k, v in method_params.items():
                setattr(obj, k, v)
        return obj

    # Optional: expose a dict view when needed
    def as_dict(self) -> Dict[str, Any]:
        """
        Return as dict.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the requested values.
        """
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
        """
        Get the calculated grain properties.

        Returns
        -------
        dict
            Calculated grain-property values.
        """

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
        """
        Get the physics plugin instance based on the configured method.

        Parameters
        ----------
        tracer_method : str, optional
            The tracer method to use for physics calculations. If None, uses the configured method.

        Returns
        -------
        object
            Loaded physics plugin instance.
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
                self._physics_plugin = plugin_module.PhysicsPlugin(self.config, self.tracer_config)
        return self._physics_plugin

    def convert_physics(self, sedtrails_data, transport_probability_method: str = None) -> None:
        """
        Converts and adds base physics calculations to an existing SedtrailsData object.

        Parameters
        ----------
        sedtrails_data : SedtrailsData
            Existing SedtrailsData object to be enhanced with physics calculations.
        transport_probability_method : str, optional
            Method to use for transport probability effects.
        """

        if self._physics_plugin is None:
            plugin = self.physics_plugin
        else:
            plugin = self._physics_plugin

        plugin.add_physics(sedtrails_data, self.grain_properties, transport_probability_method or 'no_probability')

    def convert_timestep_physics(self, sedtrails_data, current_timestep: float) -> None:
        """Add timestep-dependent physics fields when the active plugin provides them."""

        if self._physics_plugin is None:
            plugin = self.physics_plugin
        else:
            plugin = self._physics_plugin

        timestep_method = getattr(plugin, 'add_timestep_physics', None)
        if timestep_method is not None:
            timestep_method(sedtrails_data, self.grain_properties, current_timestep)
