.. SedTRAILS documentation master file, created by
   sphinx-quickstart on Wed Feb 19 07:07:49 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SedTRAILS documentation
=======================

Add your content using ``reStructuredText`` syntax. See the
`reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
documentation for details.


.. toctree::
   :maxdepth: 2

   README
   quickstart
   introduction

.. toctree::
   :maxdepth: 3
   :caption: USER DOCUMENTATION

   user/installation
.. 
   user/jan2023/old_ch01_intro.md
   user/jan2023/old_ch03_methods.md
   user/jan2023/old_ch04_userManual.md
   user/jan2023/old_overview.md
   user/nov2023/old_sedtrailsPaperDraft.md

.. toctree::
   :maxdepth: 2
   :caption: DEVELOPER DOCUMENTATION

   developer/contribution
   developer/architecture/software-diagrams.md

.. toctree::
   :maxdepth: 2
   :caption: API DOCUMENTATION

   api-refs

.. toctree::
   :maxdepth: 2
   :caption: SOURCE CODE


   sourcecode/transport-converter
   sourcecode/particle-tracer

   


ACKNOWLEDGEMENTS
================

SedTRAILS development has been funded in part through the Netherlands Organisation for Scientific Research (NWO) projects "TRAILS" (grant number 17600) in the research programme ‘Living Labs in the Dutch Delta’, "Revealing Hidden Networks of Coastal Sediment Pathways via Laboratory & Numerical Experiments" (grant number 21026), and SEAWAD (grant number 14489) in the research programme ‘Collaboration Program Water’. The SedTRAILS code was originally developed under the Deltares-USGS collaborative agreement as a coral larvae and sediment particle tracking module by Maarten van Ormondt, Edwin Elias, and Johan Reyns (Deltares/Deltares USA), Andrew Stevens and Curt Storlazzi from the US Geological Survey (USGS), and Stuart Pearson (TU Delft). It has since been further developed in part during the KPP Beheer en Onderhoud Kust en Kustgenese projects, in partnership between Deltares, Rijkswaterstaat, and TU Delft.

The present version of SedTRAILS is supported by the Digital Competence Centre, Delft University of Technology.