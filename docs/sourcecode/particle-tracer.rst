Particle Tracer
===================

This is an overview of the SedTRAILS particle tracer, which is designed to simulate the transport and dispersion of sediment particles in aquatic environments. The particle tracer is implemented as a Python package, and the source code is available in the `sedtrails.particle_tracer` module. 

Data Retriever
--------------
The data retriever module is responsible for fetching and processing data required for particle tracking simulations. It includes functions to retrieve hydrodynamic data, sediment properties, and other relevant information needed for accurate particle transport modeling.

.. automodule:: sedtrails.particle_tracer.data_retriever
                    :members:


Diffusion Library
----------------
The diffusion library provides functions for simulating the diffusion of particles in water. It includes various models for particle diffusion, allowing users to simulate how particles spread out over time due to random motion in the water column.

.. automodule:: sedtrails.particle_tracer.diffusion_library
                    :members:

Flow Field Buffer
------------------
The flow field buffer is responsible for storing and managing the flow field data used in particle tracking simulations. It allows for efficient access to flow data, which is crucial for accurately simulating particle movement in response to hydrodynamic forces.

.. automodule:: sedtrails.particle_tracer.flow_field_buffer
                    :members:

Particle
--------
The particle module defines the properties and behaviors of individual particles in the simulation. It includes functions for initializing particles, updating their positions based on flow data, and applying diffusion effects.

.. automodule:: sedtrails.particle_tracer.particle
                    :members:

Particle Seeding
----------------
The particle seeding module is responsible for generating initial particle distributions in the simulation area. It includes functions for defining seeding strategies, such as grid-based seeding. 

.. automodule:: sedtrails.particle_tracer.particle_seeding
                    :members:

Pathway Interface
-----------------
The pathway interface module provides functions for managing and analyzing particle pathways during simulations. It includes methods for tracking particle trajectories, calculating distances traveled, and identifying key pathway characteristics.

.. automodule:: sedtrails.particle_tracer.pathway_interface
                    :members:

Position Calculator
-------------------
The position calculator module is responsible for computing the positions of particles at each time step during the simulation. It includes functions for integrating particle motion based on flow data and diffusion effects.

.. automodule:: sedtrails.particle_tracer.position_calculator
                    :members:

Position Calculator (Numba)
---------------------------
The Numba-optimized position calculator module provides high-performance computation of particle positions using Numba, a Just-In-Time (JIT) compiler for Python. This module accelerates the position calculation process, making it suitable for large-scale simulations.

.. automodule:: sedtrails.particle_tracer.position_calculator_numba
                    :members:

Property Interpolator
---------------------
The property interpolator module is responsible for interpolating properties such as sediment concentration and flow velocity at particle locations. It ensures that particles receive accurate property values based on their positions in the flow field.

.. automodule:: sedtrails.particle_tracer.property_interpolator
                    :members:

State Checker
----------------
The state checker module monitors the state of particles during the simulation. It includes functions for checking particle states, such as whether they are still active, have reached a sink, or have been removed from the simulation. This is crucial for managing particle lifecycles and ensuring accurate tracking.

.. automodule:: sedtrails.particle_tracer.state_checker
                    :members:

Status Library
----------------
The status library provides functions for managing the status of particles (e.g., active/inactive, dead/alive) and burial state.

.. automodule:: sedtrails.particle_tracer.status_library
                    :members:

Timer
------
The timer module provides functions for measuring and managing simulation time. It includes methods for starting, stopping, and resetting timers, as well as for tracking elapsed time during simulations.

.. automodule:: sedtrails.particle_tracer.timer
                    :members:   