from numpy import ndarray
from dataclasses import dataclass, field
from sedtrails.particle_tracer.position_calculator_numba import create_numba_particle_calculator
from typing import Dict, Any
import numpy as np
from .particle_seeder import PopulationConfig, ParticleFactory
from matplotlib.path import Path
from scipy.spatial import ConvexHull


@dataclass
class ParticlePopulation:
    """
    Class handle operations for population of particles.
    A population is a group of particles that share the same type and seeding strategy.

    Attributes
    ----------
    field_x : ndarray
        The x-coordinates of the flow field data where particles are seeded.
    field_y : ndarray
        The y-coordinates of the flow field data where particles are seeded.
    population_config : PopulationConfig
        Configuration for the particle population, including seeding strategy and parameters.
    particles : Dict
        A dictionary containing particle attributes such as positions and status.
    _field_interpolator : Dict
        A dictionary containing Numba functions for interpolating field data.
    _position_calculator : Dict
        A dictionary containing Numba functions for calculating particle positions.
    _current_time : ndarray
        The current time in the simulation, used for updating particle positions.
    _field_mixing_depth : ndarray
        The mixing depth of the flow field, used to determine particle behavior.
    _field_transport_probability : ndarray
        The probability of particle transport in the flow field, used to determine if particles are picked up.
    """

    field_x: ndarray
    field_y: ndarray
    population_config: PopulationConfig
    particles: Dict = field(init=False, default_factory=dict)  # a dictionary with arrays
    _field_interpolator: Any = field(init=False)  # holds a Numba function
    _position_calculator: Any = field(init=False)  # holds a Numba function
    _current_time: ndarray = field(init=False)
    _field_mixing_depth: ndarray = field(init=False)  # TODO: we're not using this field yet
    _field_transport_probability: ndarray = field(init=False)  # TODO: we're not using this field yet

    def __post_init__(self):
        # Create a Numba calculator for particle operations
        numba_functions = create_numba_particle_calculator(grid_x=self.field_x, grid_y=self.field_y)

        self._field_interpolator = numba_functions['interpolate_field']
        self._position_calculator = numba_functions['update_particles']

        # generate particles based on the configuration
        _particles = ParticleFactory.create_particles(self.population_config)
        self.particles = {
            'x': np.array([p.x for p in _particles]),
            'y': np.array([p.y for p in _particles]),
            'release_time': np.array([p.release_time for p in _particles]),
            'burial_depth': np.array([p.burial_depth for p in _particles]),
        }

        # store the outer envelope of the domain
        coords = np.column_stack((self.field_x, self.field_y))
        hull = ConvexHull(coords)
        self._outer_envelope = Path(coords[hull.vertices])

    def update_information(
        self, current_time: ndarray, mixing_depth: ndarray, transport_probability: ndarray, bed_level: ndarray
    ) -> None:
        """
        Updates field data information for particles in the population.
        Parameters
        ----------
        current_time : ndarray
            The current time in the simulation.
        mixing_depth : ndarray
            The mixing depth of the flow field.
        transport_probability : ndarray
            The probability of particle transport in the flow field.
        bed_level : ndarray
            The bed level of the flow field.
        """

        self._current_time = current_time

        if not np.isnan(mixing_depth).all():
            self.particles['mixing_depth'] = self._field_interpolator(
                mixing_depth, self.particles['x'], self.particles['y']
            )

        if not np.isnan(transport_probability).all():
            """values between 0 and 1"""
            self.particles['transport_probability'] = self._field_interpolator(
                transport_probability, self.particles['x'], self.particles['y']
            )

        if not np.isnan(bed_level).all():
            self.particles['bed_level'] = self._field_interpolator(bed_level, self.particles['x'], self.particles['y'])

    def update_burial_depth(self) -> None:
        """Updates the burial depth of particles in the population.
        This method is a placeholder and should be implemented with the actual logic for updating burial depth.
        Currently, it does not perform any operations.
        """

        # Initialize vertical position ('z') based on bed level and burial depth
        if 'z' not in self.particles:
            self.particles['z'] = self.particles['bed_level'] - self.particles['burial_depth']

        # Make sure particles can never be higher than the bed level
        i_above_bed = self.particles['z'] > self.particles['bed_level']
        self.particles['z'][i_above_bed] = self.particles['bed_level'][i_above_bed]

        # Update burial depth (is always a positive value)
        self.particles['burial_depth'] = self.particles['bed_level'] - self.particles['z']

    def update_status(self) -> None:
        """
        updates status of particles in the population.
        """
        n_particles = len(self.particles['x'])

        # Compute whether particles are picked up (or trapped) based on transport probability
        # Note: If "reduced_velocity" is chosen, "transport_probability" always equals one.
        self.particles['is_picked_up'] = np.random.rand(n_particles) < self.particles['transport_probability']

        # Compute whether particles are inside (or outside) the domain envelope
        self.particles['is_inside'] = self._outer_envelope.contains_points(
            np.column_stack((self.particles['x'], self.particles['y']))
        )

        # Compute whether particles are exposed (or buried)
        self.particles['is_exposed'] = self.particles['burial_depth'] < self.particles['mixing_depth']

        # Compute whether particles are released (or retained)
        # FIXME: Temporary implementation
        self.particles['release_time'] = np.zeros_like(self.particles['x'])
        self.particles['is_released'] = self._current_time >= self.particles['release_time']

        # Compute whether particles are alive (or dead) (still TODO)
        self.particles['is_alive'] = np.ones(n_particles, dtype=bool)

        # Compute whether particles are mobile (or static) - combination of all status flags
        self.particles['is_mobile'] = (
            self.particles['is_inside']
            & self.particles['is_alive']
            & self.particles['is_exposed']
            & self.particles['is_released']
            & self.particles['is_picked_up']
        )

    def update_position(self, flow_field: Dict, current_timestep: float) -> None:
        """
        Update the position of particles in the population based on the flow field.

        Parameters
        ----------
        flow_field : Dict
            A dictionary containing the flow field information.
        current_timestep : float
            The current time step in the simulation in seconds.
        """

        ix = self.particles['is_mobile']  # Get indices of mobile particles

        n_particles = len(self.particles['x'])
        dx = np.zeros(n_particles)
        dy = np.zeros(n_particles)

        new_x, new_y = self._position_calculator(
            self.particles['x'][ix],
            self.particles['y'][ix],
            flow_field['u'],
            flow_field['v'],
            current_timestep,
        )

        dx[ix] = new_x - self.particles['x'][ix]
        dy[ix] = new_y - self.particles['y'][ix]

        self.particles['x'][ix] = new_x
        self.particles['y'][ix] = new_y
