"""A plugin for passive tracer calculations."""

import logging

from sedtrails.transport_converter import SedtrailsData
from sedtrails.transport_converter.plugins import BasePhysicsPlugin

logger = logging.getLogger(__name__)


class PhysicsPlugin(BasePhysicsPlugin):  # all clases should be called the PhysicsPlugin
    """
    Plugin for passive tracer calculations.
    """

    def __init__(self, config, tracer_config):
        """Initialize the passive-tracer physics plugin.

        Parameters
        ----------
        config : PhysicsConfig
            Physics configuration supplied by the converter.
        tracer_config : dict
            Method-specific tracer configuration retained for interface
            compatibility.
        """
        super().__init__()
        self.config = config

    def add_physics(self, sedtrails_data: SedtrailsData, *args, **kwargs) -> None:
        """
        Add flow fields for passive tracer

        Parameters
        ----------
        sedtrails_data : SedtrailsData
            SedTRAILS data object to process.
        *args : object
            Additional positional arguments passed through to the implementation.
        **kwargs : object
            Additional keyword arguments passed through to the implementation.
        """
        flow_velocity_x = sedtrails_data.depth_avg_flow_velocity['x']
        flow_velocity_y = sedtrails_data.depth_avg_flow_velocity['y']
        flow_velocity_magnitude = sedtrails_data.depth_avg_flow_velocity['magnitude']

        logger.info('Adding physics fields to SedtrailsData')

        # Flow velocities (vector fields)
        sedtrails_data.add_physics_field(
            'depth_avg_flow_velocity',
            {'x': flow_velocity_x, 'y': flow_velocity_y, 'magnitude': flow_velocity_magnitude},
        )
