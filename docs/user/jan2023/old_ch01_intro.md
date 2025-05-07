# Chapter 1: Introduction

## Motivation

Estuaries and coasts can be conceptualized as connected networks of water and sediment fluxes. These dynamic geomorphic systems are governed by waves, tides, wind, and river input, and evolve according to complex nonlinear transport processes. To predict their evolution, we need to better understand the pathways that sediment takes from source through temporary storage areas to sink. Knowledge of these pathways is essential for predicting the response of such systems to climate change impacts or human interventions (e.g., dredging and nourishment). The conceptual framework of sediment connectivity has the potential to expand our system understanding and address practical coastal management problems (Pearson, 2020).

Connectivity provides a structured framework for analyzing these sediment pathways, schematizing the system as a series of geomorphic cells or nodes, and the sediment fluxes between those nodes as links (Heckmann, 2015). Once organized in this fashion, the resulting network can be expressed algebraically as an adjacency matrix: sediment moving from a given source to different receptors. There is a wealth of pre-existing statistical tools and techniques that can be used to interpret the data once it is in this form, drawing on developments in other scientific disciplines (Newman, 2018; Rubinov, 2010). Lagrangian flow networks have been increasingly used to analyze flow and transport pathways in oceanographic and geophysical applications (Ser-Giacomi, 2015; Padberg-Gehle, 2017; Reijnders, 2021). However, this approach has not yet been adopted to analyze coastal or estuarine sediment transport, and requires a multitude of field measurements or numerical model simulations.

Lagrangian particle tracking has been widely used to assess connectivity in the context of oceanography and marine ecology (Hufnagl, 2016; Van Sebille, 2018), because the models record the complete history of a particle’s trajectory, not only its start and end points. Particle tracking models are also relatively fast and lend themselves well to parallel computing (Paris, 2013). This approach thus permits a faster and more detailed analysis of sediment connectivity than existing Eulerian approaches (e.g., Pearson, 2020). Although several Lagrangian sediment transport models have been developed (e.g., MacDonald, 2007; Soulsby, 2011), they have not been used to support connectivity studies. Hence, there is a need for Lagrangian sediment particle tracking tools tailored to predicting sediment transport pathways and determining connectivity of complex coastal systems.

To meet this need, we developed a Lagrangian sediment transport model, **SedTRAILS** (**Sed**iment **TRA**nsport v**I**sualization & **L**agrangian **S**imulator) and used it to develop a sediment connectivity network. Our approach provides new analytical techniques for distilling relevant patterns from the chaotic, spaghetti-like network of sediment pathways that often characterize estuarine and coastal systems.

SedTRAILS and connectivity have already been applied to numerous research and consulting projects to answer questions about coastal sediment pathways. Applications include estimating nourishment or dredge disposal fate (Stevens, 2020; Elias, 2020, 2021; Lambregts, 2021), identifying large-scale sediment transport pathways (Stevens, 2020; Bult, 2021; Van Gijzen, 2020), determining the impact of human interventions in the coastal system on centennial time scales (Meijers, 2021), and identifying sources of sediment to dredging hotspots (Stevens, 2020). These projects have spanned a range of coastal environments around the world, from tidal inlets in the Wadden Sea to alongshore uniform beaches on the Holland coast, and from a muddy estuary in the United States to a fjord on the west coast of Canada. SedTRAILS and connectivity provide added interpretive value to existing process-based models, which makes them readily applicable. Furthermore, these tools have enabled the effective communication of complex numerical model results to a non-scientific community and stakeholders in order to generate meaningful discussions.

### The Need for a "Different" Model Approach

*Note: this text has been taken from Elias (2020).*

Delft3D has been under development at Deltares (and its predecessor WL-Delft / Delft Hydraulics) since the early 1990s and has been applied in complex morphological systems in the past, like tidal inlet systems (e.g., Elias 2006; Lesser 2009; Van der Weegen 2009; Dastgheib 2012; Elias and Hansen 2012). These studies show that process-based model suites like Delft3D have reached the stage that they can be used successfully to investigate tidal inlet processes and greatly improve our fundamental understanding of the processes driving sediment transport and morphodynamic change.

Process-based models seem to perform particularly well at the end nodes of the scale cascade (Figure 2.1), on the short term (small scale) and on the long term (large scale). Both short-term, quasi-real-time models and long-term models seem to produce useful results. Short-term models typically have a high resolution and use complex sediment transport equations. Long-term models are constrained by computational time and use simplified grids, boundaries, and morphodynamic acceleration techniques such as MorFac.

Medium-term simulations (years to decades) are harder: short-term models take too long to run, and long-term models rely heavily on schematization. Running these models can take weeks, limiting how many versions can be tested. If results deviate from expectations — not uncommon — this adds uncertainty.

A key pitfall in morphodynamic modelling is overfitting: "tweaking" parameters and inputs to match observations without understanding the underlying dynamics. This can result in models that look right but do not capture real processes.

Insights from the Kustgenese 2 research project highlighted the importance of small-scale processes (like shoal instabilities) even at large system scales. These subtle dynamics are hard to capture in existing models and are critical to understanding sediment pathways.

This realization motivated the development of **SedTRAILS**, designed to compute inlet-scale sediment transport efficiently while preserving detail and accuracy. SedTRAILS can simulate tidal processes while capturing connectivity patterns across an entire inlet system.

## Objectives

Potential users include coastal managers, consultancies, and academic researchers. Initially, development will focus on a closed research community, with broader release planned later.

We aim for SedTRAILS to be:

- A well-validated scientific tool that can predict sediment pathways  
- An open-source model with an active community  
- An interactive visualization tool with a GUI for stakeholders  

Research questions include:

- How can SedTRAILS form the basis for probabilistic estimates of sediment pathways?  
- How can we use SedTRAILS to predict the bleaching of sand grains for OSL?  
- How can we use SedTRAILS to optimize nourishment or dredge disposal strategies?  
- How can Lagrangian Coherent Structures be used to explain or predict transport pathways and barriers?  
- How can SedTRAILS be used to compare/explore sediment vs. microplastic vs. [living organism] pathways?  
- How can we use SedTRAILS to explore gross vs. net transport pathways?  
