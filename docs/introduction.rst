What is SedTRAILS?
====================

SedTRAILS (Sediment TRAnsport vIsualization & Lagrangian Simulator) is a sediment transport model that uses Lagrangian particle tracking to simulate sediment pathways in coastal and estuarine environments. 


Why SedTRAILS?
^^^^^^^^^^^^^^
Estuaries and coasts can be conceptualized as connected networks of water and sediment fluxes. These dynamic geomorphic systems are governed by waves, tides, wind, and river input, and evolve according to complex nonlinear transport processes. To predict their evolution, we need to better understand the pathways that sediment takes from source through temporary storage areas to sink. Knowledge of these pathways is essential for predicting the response of such systems to climate change impacts or human interventions (e.g., dredging and nourishment). The conceptual framework of sediment connectivity has the potential to expand our system understanding and address practical coastal management problems (Pearson, 2020).

Connectivity provides a structured framework for analyzing these sediment pathways, schematizing the system as a series of geomorphic cells or nodes, and the sediment fluxes between those nodes as links (Heckmann, 2015). Once organized in this fashion, the resulting network can be expressed algebraically as an adjacency matrix: sediment moving from a given source to different receptors. There is a wealth of pre-existing statistical tools and techniques that can be used to interpret the data once it is in this form, drawing on developments in other scientific disciplines (Newman, 2018; Rubinov, 2010). Lagrangian flow networks have been increasingly used to analyze flow and transport pathways in oceanographic and geophysical applications (Ser-Giacomi, 2015; Padberg-Gehle, 2017; Reijnders, 2021). However, this approach has not yet been adopted to analyze coastal or estuarine sediment transport, and requires a multitude of field measurements or numerical model simulations.

Lagrangian particle tracking has been widely used to assess connectivity in the context of oceanography and marine ecology (Hufnagl, 2016; Van Sebille, 2018), because the models record the complete history of a particle’s trajectory, not only its start and end points. Particle tracking models are also relatively fast and lend themselves well to parallel computing (Paris, 2013). This approach thus permits a faster and more detailed analysis of sediment connectivity than existing Eulerian approaches (e.g., Pearson, 2020). Although several Lagrangian sediment transport models have been developed (e.g., MacDonald, 2007; Soulsby, 2011), they have not been used to support connectivity studies. Hence, there is a need for Lagrangian sediment particle tracking tools tailored to predicting sediment transport pathways and determining connectivity of complex coastal systems.

To meet this need, we developed a Lagrangian sediment transport model, **SedTRAILS** (Sediment TRAnsport vIsualization & Lagrangian Simulator) and used it to develop a sediment connectivity network. Our approach provides new analytical techniques for distilling relevant patterns from the chaotic, spaghetti-like network of sediment pathways that often characterize estuarine and coastal systems.


Publications 
------------

SedTRAILS and coastal sediment connectivity have been applied in several journal publications:


    .. [1] van Westen, B., de Schipper, M., Pearson, S.G., & Luijendijk, A. (2025). Lagrangian Modelling Reveals Sediment Pathways at Evolving Coasts. Scientific Reports.
           https://doi.org/10.1038/s41598-025-92910-z
    .. [2] Pearson, S.G., van Prooijen, B.C., Elias, E.P., Vitousek, S., Wang, Z.B. (2020). Sediment Connectivity: A Framework for Analyzing Coastal Sediment Transport Pathways.  Journal of Geophysical Research: Earth Surface.  
           https://doi.org/10.1029/2020JF005595 
      
Conference Proceedings/Presentations/Posters
--------------------------------------------

SedTRAILS and coastal sediment connectivity have been shared at several conferences:

    .. [1] fdfsdff

MSc Theses
----------

SedTRAILS has been applied in several master's theses:

    .. [1] Lambregts. P. (2021). Sediment bypassing at Ameland inlet and the role of an ebb-tidal delta nourishment. http://resolver.tudelft.nl/uuid:2e5dfc75-d7b8-44bd-a1f3-99f2b18f3533
    .. [2] Meijers, C. (2021). Sediment transport pathways in Burrard Inlet. http://resolver.tudelft.nl/uuid:ff2b7adf-6c8b-40b4-99e3-216395e890fa
    .. [3] Krikke, L. (2023). Impact of the Eastern Scheldt Storm Surge Barrier on the Morphodynamics of the Ebb-Tidal Delta. http://resolver.tudelft.nl/uuid:9c712275-b01a-436a-b322-398762168053
    .. [4] Bisschop, F. (2023). Modelling sediment and propagule pathways to improve mangrove rehabilitation: A case study of the pilot project in Demak, Indonesia http://resolver.tudelft.nl/uuid:333ddfa1-4ecf-43c0-9f4c-9b8b37972d36
    .. [5] Thillaigovindarasu, N.R. (2023). Mangrove-Sediment Connectivity in the Presence of Structures Used to Aid Restoration. http://resolver.tudelft.nl/uuid:c4dcc4b7-0012-4c41-9f74-9d83d54a1914
    .. [6] Laan, J. (2024). Probabilistic modelling of tidal inlets: Sediment fate estimation in the coastal system using Markov chains http://resolver.tudelft.nl/uuid:47c23f3b-21b6-4111-93a0-d5f41dcb7195
    .. [7] Meijer, M. (2024). Analysing dispersal of the Ameland ebb-tidal delta nourishment using SedTRAILS. http://resolver.tudelft.nl/uuid:d579f176-746e-44bc-9259-8ed98d822868