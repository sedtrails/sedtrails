Benchmark tests
===============

SedTRAILS includes a small benchmark suite under ``tests/analytical`` based on the idealized and analytic test cases described by Lange & van Sebille (2017).
These analytical tests exercise passive tracer advection only (no sediment physics), so they remain close to the original analytic formulations.

The suite is intentionally split into one pytest per benchmark so that CI can report failures per scenario:

* Radial rotation
* Longitudinal shear
* Time oscillation
* Peninsula flow
* Stommel gyre
* Damped inertial oscillation
* Brownian motion

What the tests exercise
-----------------------

The analytical benchmarks are designed to validate the SedTRAILS particle-tracer
stack (interpolation + time integration), not a bespoke integrator. Specifically:

* ``create_numba_particle_calculator`` is used to build the interpolator and RK4
	kernels for every advection benchmark.
* ``update_particles_with_simplex`` is called via the shared helper integrator for
	steady flows (radial rotation, longitudinal shear, peninsula, Stommel gyre).
* ``update_particles_temporal_with_simplex`` is used for time-dependent flows
	(damped oscillation and the time-oscillation benchmark when temporal blending is
	enabled).
* ``FieldDataRetriever`` (with ``SedtrailsData``) is used in the time-oscillation
	benchmark to exercise the full retrieval + advection path.
* ``BrownianDiffusionStrategy`` is used directly in the Brownian benchmark to validate the
	diffusion operator against analytic moments.

Running the suite
-----------------

From a shell in the repository root:

.. code-block:: bash

	python -m pytest tests/analytical -q

Artifact locations and plotting
-------------------------------

Artifacts are written into ``tests/analytical/output`` by default.

Each benchmark writes:

* ``*_comparison.png`` (or a more specific plot name)
* ``*.txt`` metrics file (key/value pairs)
* ``*.npy`` arrays (errors, relative errors, summary stats)

Psi error plots (peninsula + Stommel)
------------------------------------

The peninsula and Stommel benchmarks use a streamfunction $\psi$ to describe the steady 2D flow. In steady, incompressible flow,
particles should remain on a constant-$\psi$ streamline. The psi error plots show $\psi(t) - \psi(t_0)$ for each particle; each line
represents one particle, and values near zero indicate that particles remain on their initial streamlines.

The streamfunction has units of length$^2$/time. Since these benchmarks plot $x$ and $y$ in kilometers, the psi error axis is in km$^2$/s.
Use the relative error metrics (``*_streamfunction_relative_error.npy`` and ``*_metrics.txt``) to judge whether a deviation is significant.

Small deviations from zero are expected because SedTRAILS uses barycentric interpolation on a discrete grid and a finite RK4 timestep.
In the Stommel case, velocities are derived from finite differences of $\psi$, which adds additional discretization error.

The longitudinal shear benchmark seeds particles from -30 to 60 degrees latitude and plots in degrees to match the original formulation on a flat grid. Peninsula and Stommel cases include additional seed points and longer integration windows to ensure better spatial coverage. The peninsula benchmark also stops early if any particle reaches x=100 km.

Consistency notes
-----------------

The test parameters are aligned with the Lange & van Sebille descriptions (RK4 with 5 min steps, particle counts, and runtime), but SedTRAILS operates on a flat Cartesian grid. Therefore, the longitudinal shear benchmark validates a uniform zonal flow using a degree-based grid and an approximate meters-to-degrees conversion rather than the spherical lon/lat conversion that Parcels performs internally. The Brownian benchmark samples the analytic Gaussian displacement directly to match the stated $K_h$ formulation, and the SedTRAILS random-walk operator uses the same $K_h$ coefficient.

You can load and plot arrays with a short script, for example:

.. code-block:: python

	import numpy as np
	import matplotlib.pyplot as plt

	data = np.load('tests/analytical/output/03_timeoscillation_x_error.npy')
	plt.plot(data)
	plt.title('Time oscillation x-error')
	plt.show()

Paper-ready overview figure + metrics summary
--------------------------------------------

Once the benchmark plots and metrics are generated, you can create a single
landscape overview figure (7 panels, labeled (a) to (g)) and a consolidated
metrics summary text file using:

.. code-block:: bash

	python tests/analytical/benchmark_report.py --input-dir tests/analytical/output

The script writes:

* ``benchmarks_overview.png`` (combined figure)
* ``benchmarks_metrics_summary.txt`` (all metrics in one file)

You can override the output locations if needed:

.. code-block:: bash

	python tests/analytical/benchmark_report.py \
		--input-dir tests/analytical/output \
		--output-figure tests/analytical/output/benchmarks_overview.png \
		--output-metrics tests/analytical/output/benchmarks_metrics_summary.txt


When to add a benchmark
-----------------------

Add a benchmark when a new control-flow path or physics formulation needs regression coverage against a known analytic reference.


References
----------
Lange, M., & van Sebille, E. (2017). Parcels v0. 9: prototyping a Lagrangian ocean analysis framework for the petascale age. Geoscientific Model Development, 10(11), 4175-4186.

Van Sebille, E., Griffies, S. M., Abernathey, R., Adams, T. P., Berloff, P., Biastoch, A., ... & Zika, J. D. (2018). Lagrangian ocean analysis: Fundamentals and practices. Ocean modelling, 121, 49-75.