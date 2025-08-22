# Particle seeding in SedTRAILS

Seeding is the process of adding the particle(s) to the model. In SedTRAILS, the type and initial position(s) of the sediment(s) are set as separate populations. The options for particle types are sand, mud, and passive tracer. Multiple strategies are available to locate the paticles as initial conditions. They can be added as either a single particle (point strategy) or more. In case of adding many particles, they can be located along a spcified line (transect strategy), or within a polygon (grid and random strategies). 

Inputs required for particle types:
| Sand       | Mud        | Passive    |
|------------|------------|------------|
| grain size | grain size | grain size |
| ???        | ???        | ???        |


Inputs required for seeding strategies:
| Point         | Transect         | Grid (uniform)    | Grid (random)  |
|---------------|------------------|-------------------|----------------|
| x, y          | x1, y1           | x1, y1            | ???            |
| ???           | x2, y2           | x2, y2            | ???            | 
| ???           | ???              | particle distance | ???            | 
| ???           | ???              | ???               | ???            | 


Here I copy paste Manuel's comments in Github issue#267. I will revise this part of the documentation while the seeding factory is finalised.

"""
The code in this PR follows this reasoning:

The seeding tool generates n-number of particles for a population using the seeding strategy defined in the configuration for each population.
The number of particle per location are defined by quantity and the number of seeding locations depend on the strategy used. e.g., above only one seed location is specified. The number of seeding locations for some of the strategies are computed at runtime (e.g., grid).
The total number of particle in a population is the product of quantity * seeding locations
When seeding happens a list with the total number of particles in a population is created. If more than one population is specified in the configuration file, than particle creation should be repeated for each one (I will work on automating this). The creation of particle is manage by a ParticleFactory, which takes the configuration for a population and generates the particles, assigns the initial x,y coordinates, release_time (release_start) and quantity.
"""