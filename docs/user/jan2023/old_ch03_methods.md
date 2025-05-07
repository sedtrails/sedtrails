# Chapter 3: Methods

Our approach for determining sediment connectivity has four main steps:

1. Simulate hydrodynamics
2. Simulate sediment transport with an Eulerian model
3. Estimate Lagrangian sediment transport pathways
4. Derive a sediment connectivity network from those pathways

## Eulerian Model

SedTRAILS is coupled offline to an Eulerian hydrodynamic and sediment transport model (e.g., Delft3D 4 or D-FLOW FM). Delft3D uses a structured mesh; FM uses an unstructured mesh. These differences affect pre-processing and interpolation routines.

### From Elias & Pearson (2020)

SedTRAILS uses sediment transports from Delft3D Online Morphology, which couples Delft3D-Wave and Delft3D-Flow. Flow is modeled via unsteady shallow-water equations. Wave effects are incorporated using SWAN. Sediment transport is calculated at each flow timestep using Van Rijn's formulations. Suspended and bed load transport are handled separately.

Delft3D can trace sediment via tagged sediment fractions, but this is computationally expensive and prone to mixing artifacts. Elias et al. (2011) improved this by excluding tracers from bed updating. Still, long runtimes motivated development of SedTRAILS.

### Current Use

SedTRAILS uses outputs from morphostatic Eulerian models. Required parameters:

* Mean and max bed shear stress (\$\tau\_m\$, \$\tau\_{max}\$)
* Current velocity (\$\vec{U}\_c\$)
* Depth (\$h\$)
* Bedload and suspended load transport (\$\vec{q}\_b\$, \$\vec{q}\_s\$)
* Suspended sediment concentration (\$C\$)

Timestep-averaged vector fields and error handling for negative values are noted issues.

## Visualizing Sediment Transport Fields

SedTRAILS visualizes transport fields computed by Delft3D to create sediment pathways. A scaling factor \$a\_f\$ is applied to convert transport fluxes into velocities. Diffusion is modeled via random displacements. Precomputed velocity fields decouple particle computation, improving runtime. The \$a\_f\$ factor has no physical basis but improves visual clarity.

### From Elias & Pearson (2020)

SedTRAILS simulates accurate, efficient sediment trajectories using static transport fields. Repeated tides are assumed to represent long-term dynamics. Particle motion is efficient and spans large areas.

Mass flux \$S\_m\$ is converted to volume flux \$S\_v\$:

$\vec{S}_{v} = \frac{S_{m}}{\rho_b}$

Effective velocity \$u\_{tr}\$ is then:

$u_{tr} = \frac{S_v}{h_{tr}}$

10-min incremental transport is calculated from cumulative values:

$\overline{TT}(\Delta t_n) = \frac{\int TT(t_{n}) - \int TT(t_{n-1})}{(t_{n}-t_{n-1})}$

Particle trajectories use mean vectors for stability and efficiency.

## Direct Computation of Transport Velocities

Instead of using \$a\_f\$, transport fluxes can be converted to velocities via:

$\vec{q}_{tot} = \vec{q}_{bed} + \vec{q}_{sus}$

Bedload velocity:

$u_{bed} = \frac{q_{bed}\rho_s}{c_{bed}\delta_{bed}}$

Where:

$\delta_{bed} = 0.3D_{50} D_{*}^{0.7}T^{0.5}$

Suspended transport velocity:

$u_{sus} = \frac{\int_{z_1}^{z_2}U(z)C(z)dz}{\int_{z_1}^{z_2}C(z)dz}$

Weighted mean velocity:

$U_{tot} = \frac{u_{bed}q_{bed}}{q_{bed}+q_{sus}} + \frac{u_{sus}q_{sus}}{q_{bed}+q_{sus}}$

## Soulsby et al. (2011) Approach

Sediment velocity:

$\vec{U}_{gr} = R \cdot P \cdot F \cdot \vec{U}_{c}$

### Constants

* \$g = 9.81 \text{ m/s}^2\$
* \$\rho\_w = 1027 \text{ kg/m}^3\$
* \$\rho\_s = 2650 \text{ kg/m}^3\$
* \$d\_t = d\_a = 250 \mu m\$

### Probability of Motion (P)

$P = \left[1 + \left( \frac{\pi \mu_d / 6}{\theta_{max} - \theta_{cr,A}} \right)^{4} \right]^{-1/4}$

### Freedom Factor (F)

Randomly updated per timestep.

## Numerical Implementation

Particle advection uses:

$\frac{\partial\vec{x}}{\partial t} = \vec{U}_{transp}(\vec{x},t) + R(\theta)_{diff}\cdot\vec{U}_{transp}(\vec{x},t)$

Implemented using 4th-order Runge-Kutta.

### Unstructured Grids

Support added via `mxparticle_4nd_rk_fm`, using Delaunay triangulation to define `facenodes`. Grids now set with:

```matlab
if strcmpi(input.gridtype,'curvilinear')
    ...
elseif strcmpi(input.gridtype,'unstructured')
    ...
end
```

## Lagrangian Coherent Structure Analysis

FTLE computed via:

* Flow map: \$\Phi^{t0+T}\_{t0}\$
* Jacobian: \$D\Phi^{t0+T}\_{t0}\$
* Cauchy-Green tensor: \$\Delta\_{i,j}\$
* FTLE: \$\sigma\_{i,j}=\frac{1}{|T|}\log\sqrt{(\lambda\_{max})\_{i,j}}\$

Repelling or attracting structures found by direction of time integration.

## Implementation in SedTRAILS

Flow charts to be added.
