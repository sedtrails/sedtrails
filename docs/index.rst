.. SedTRAILS documentation master file, created by
   sphinx-quickstart on Wed Feb 19 07:07:49 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SedTRAILS documentation
=======================

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0
   :alt: License: GPLv3

.. image:: https://img.shields.io/badge/tu_delft-DCC-black?style=flat&label=TU%20Delft&labelColor=%23000000%20&color=%2300A6D6
   :target: https://dcc.tudelft.nl
   :alt: TUDelft DCC

.. image:: https://github.com/sedtrails/sedtrails/actions/workflows/publish.yml/badge.svg
   :target: https://github.com/sedtrails/sedtrails/actions/workflows/publish.yml
   :alt: Deploy Sphinx Documentation

.. image:: https://github.com/sedtrails/sedtrails/actions/workflows/ruff.yml/badge.svg?branch=dev
   :target: https://github.com/sedtrails/sedtrails/actions/workflows/ruff.yml
   :alt: Ruff

.. image:: https://github.com/sedtrails/sedtrails/actions/workflows/pytest.yml/badge.svg?branch=dev&event=push
   :target: https://github.com/sedtrails/sedtrails/actions/workflows/pytest.yml
   :alt: Pytest


**Sed**\ iment **TRA**\ nsport v\ **I**\ sualization and **L**\ agrangian **S**\ imulator.

SedTRAILS is an open-source Python package for modeling sediment transport that uses Lagrangian particle tracking to simulate sediment pathways in coastal and estuarine environments. The current version is a **beta release**, bugs and issues are expected. Please report any problems you encounter on the `GitHub Issues page <https://github.com/sedtrails/sedtrails/issues>`_.

Features
--------
* Lagrangian particle tracking for sediment transport simulation.
* Dashboard for interactive visualization of simulation results in real-time.
* Support for D-Flow FM, XBeach, and SFINCS hydrodynamic model outputs.
* Support for various physics convertion methods.
* Terminal user interface (CLI) for easy setup and execution of simulations.
* Modular design for easy integration and extension.
* Comprehensive documentation and examples.

.. toctree::
   :maxdepth: 2


   quickstart
   introduction

.. toctree::
   :maxdepth: 3
   :caption: User Documentation

   user/installation
   user/simulations
   user/output
   user/visualization
   user/dashboard
   user/input-formats
   user/seeding

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/tutorial1
   tutorials/loop-simulations

.. toctree::
   :maxdepth: 2
   :caption: Developer Documentation

   developer/contribution
   developer/dev-environment
   developer/plugins
   developer/benchmarks
   developer/adding-unit-tests
   developer/architecture

.. toctree::
   :maxdepth: 1
   :caption: References

   references/simulation-params
   apidocs/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

ACKNOWLEDGEMENTS
================

SedTRAILS development has been funded in part through the Netherlands Organisation for Scientific Research (NWO) projects "TRAILS" (grant number 17600) in the research programme ‘Living Labs in the Dutch Delta’, "Revealing Hidden Networks of Coastal Sediment Pathways via Laboratory & Numerical Experiments" (grant number 21026), and SEAWAD (grant number 14489) in the research programme ‘Collaboration Program Water’. 
The SedTRAILS code was originally developed under the Deltares-USGS collaborative agreement as a coral larvae and sediment particle tracking module by *Maarten van Ormondt, Edwin Elias,* and *Johan Reyns* (Deltares/Deltares USA), *Andrew Stevens and Curt Storlazzi* from the US Geological Survey (USGS), and *Stuart Pearson* (TU Delft). 
It has since been further developed in part during the KPP Beheer en Onderhoud Kust en Kustgenese projects, in partnership between Deltares, Rijkswaterstaat, and TU Delft.

The Python version of SedTRAILS was supported by the `Digital Competence Centre, Delft University of Technology <https://dcc.tudelft.nl/>`_.
