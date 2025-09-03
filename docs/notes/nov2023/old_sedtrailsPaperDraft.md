
# Introduction 

**Need to understand sediment pathways**\\
*Big picture: changing coasts require system understanding, also more human interventions*\\

Scientific Problem:
- Coastal maintenance strategies in the Netherlands and many other places around the 
- We need better analysis and design tools for these nourishments, and to this end have 
-  Our existing SedTRAILS modelling has focused solely on residual transport patterns or identifying sediment pathways, without directly quantifying volumetric transport or the associated dispersal timescales
-  In this study, we seek to develop an approach for modelling the mass or volumetric transport of sediment from a nourishment (morphostatically) using SedTRAILS 

Why SedTRAILS?
- There is a growing need to improving our **understanding of complex coastal systems** like tidal inlets and estuaries and to **assess the impact of human interventions** (e.g., nourishments, dredge disposals, coastal structures),
- Past projects for clients like USGS and Rijkswaterstaat have **demonstrated that there is a strong demand for the kinds of analyses** that SedTRAILS can do
- SedTRAILS can help us **identify and analyze sediment transport pathways/connectivity**, and its visualization capabilities provide a **valuable tool for stakeholder communication** and interaction.
- SedTRAILS also provides **a means of extracting additional interpretive value from existing sediment transport models** (Delft3D/FM)
- SedTRAILS could become a “standard” post-processing step in modelling projects for modellers to understand their results and communicate them to the client

Sandy beaches, estuaries, and barrier coasts

Need to understand and intervene in pathways:
- System understanding
- Coastal erosion etc
- Will intertidal flats or mangrove forests keep up with SLR?
- Where to place a sand nourishment or deposit dredged sediment for beneficial reuse?

Where it gets complicated to model and understand: tidal inlets and estuaries with complex bathymetry and flow patterns (e.g. sub-grid emergent features in Ameland)

**Eulerian Sediment Transport Models**

Key processes in D3D bed models: Delft3D sediment tracing is quite diffusive, depending on active layer thickness, so it is difficult to look at maximum potential pathways. Also highlight Victor's concerns re well-posedness of active layer?

**Problems with Multi-sed-frc approaches**

*Rephrase:* Tracking sediment to illustrate sediment pathways can in principle be performed with the standard Delft3D application. Delft3D will not compute the movement of individual particles, but Delft3D can compute the sediment transport of multiple fractions simultaneously and thereby follow the movement of each fraction. By using multiple sediment fractions with similar sediment characteristics and properties, the movement of each separate fraction can be followed. This approach has been used to track movement of larger scale morphologic features (such as nourishments) in the past. Bak (2017) provides a recent example, tracking the movement of the Ameland ebb-tidal delta nourishment. In this example the sediment of the nourishment was tagged as a separate sediment fraction compared to the sediment of the ebb-tidal delta (both fractions had similar characteristics). This method of sediment tracing has two major limitations. Firstly, these simulations are computationally expensive. To capture e.g. the sediment bypassing cycle, sediment movement from the updrift to the downdrift island, model simulations over longer time-spans of months to years would be needed. Such simulations may take weeks to months of runtime to complete. Secondly,  revealed that if initial tracer deposits are small compared to the total sediment volume, the exchange layer thickness becomes an important calibration parameter for the dispersal rate. Since sediments are assumed to be well-mixed within the bed layers, the ratio of tracer versus background sediment mass available determines the contribution of each sediment class to the sediment transport in subsequent time-steps.

**Existing Lagrangian coastal models**

*General Lagrangian models; specifically sand e.g. Soulsby and MacDonald*

Main approaches to modelling sediment are Eulerian, but here we focus on Lagrangian. At one extreme of Eulerian modelling, we can go full Delft3D morphodynamic with multiple bed layers and sediment fractions and morphodynamic updating. But this takes a long time and is complicated...

Uncertainties and future stuff requires fast models because we need to test lots of different combinations of conditions

With Lagrangian modelling, the focus is not directly on HOW MUCH sediment but WHICH sediment

Particle tracking in coastal systems is well-established . Include fine sediment, far-field dredge plume modelling

 and  exist but we needed something open, fast, and that we could use for visualizing connectivity.

Largely similar to soulsby but with different wave-driven transport approach and different algorithm- pre-computing sediment velocity fields.

**Where those models fall short: what are our needs?**

*Management needs; process understanding; where do those other models not quite cut it?*

Existing PT or ST models don't quite cut it: we need high spatial resolution, need to feed connectivity and LCS analysis

Need to query fast to answer management questions, e.g. MCR questions (see Application Section~)

Critical limitations of previous approaches:
- Not open, not fast or easy to query and develop large datasets
- Usually just used for point sources
- We take our inspiration from larval dispersal and oceanographic or water quality studies (LCS)

We need something to feed connectivity, inform a holistic vision of coastal systems, and understand pathways/link to changes in bathymetry.

**Enter SedTRAILS in all its glory**
*How is it different? More user-friendly? Efficient/open? How do we use it differently?*
SedTRAILS fills these gaps!

*REPHRASE:* To meet this need, we developed a Lagrangian sediment transport model, SedTRAILS (**Sed**iment **TRA**nsport v**I**sualization \& **L**agrangian **S**imulator) and used it to XXX. Our approach provides new analytical techniques for distilling relevant patterns from the chaotic, spaghetti-like network of sediment pathways that often characterize estuarine and coastal systems. We demonstrate a proof of concept for our approach by applying it to XXX, and provide an outlook for future research and practical opportunities using these tools.

Here we outline the physics and numerics behind a Lagrangian sediment mass-transport particle tracking model (SedTRAILS), demonstrating its application to case studies...

This new version of SedTRAILS would allow us to make quick, accurate estimates of sediment dispersal in cases where bathymetric changes are relatively small and morphodynamic timescales are relatively long. For example, the dispersal of small quantities of dredged material (rather than large nourishments that significantly modify flow fields) could be well represented in this manner. It would specifically allow us to determine the fate of nourished sediment and the pathways it takes along the coast.

Furthermore, SedTRAILS can be used as an effective means of visualizing and communicating complex process-based numerical model results to stakeholders.  It can also be used to populate Lagrangian flow networks in order to conduct sediment connectivity analyses.

This approach will change the way we understand coastal sediment pathways and design sand nourishments.

Full morphodynamic models of nourishments take too long and existing Lagrangian models do not meet our functional needs, but by tracking mass transport with an improved version of SedTRAILS that uses the  equations (or similar), we can have a quick and effective tool for nourishment analysis and design and system understanding.

**What do we want?**
- Develop a fast sand particle tracking model and suite of visualization/analysis tools (SedTRAILS). 
- SedTRAILS is **distinct from existing Deltares particle tracking models** in that it focuses on sand transport, rather than cohesive sediment or other passive tracers
- However, as discussed with Johan and Edwin today, we should coordinate with them to see where we can find efficiencies
- Visualization/analysis tools include residual transport visualization (i.e. much of the previous SedTRAILS work), dynamic querying of sources/receptors, connectivity/network analysis
- Share as **open-source community model**

**Relevance**:
*How is this relevant for society/stakeholders?* Better tools for sand nourishment analysis and design will enable more efficient, effective, and environmentally friendly coastal protection
*How we can use it in practice?* Once developed, coastal engineers and scientists can apply the SedTRAILS model to estimate nourishment fate and visualize/communicate this with stakeholders

**Focus of the paper and section outline**

*Purpose of the paper: introducing new software to a wider audience; less so for showing new scientific results*

In Section~ we describe the model. In Section~ we verify the model against idealized and real test cases, then demonstrate applications of SedTRAILS in Section~, and discuss assumptions in Section~. We review the main conclusions and prospects for future development in Section~.

# Model Description 
## Modelling Approach
**Modelling philosophy \& Key assumptions**

We want to model sediment pathways, to visualize how primarily sandy sediment moves through coastal and estuarine systems, and to understand the fate of particles due to human interventions in these coastal systems.

Mass conservation: concrete bed, 2D "columns" of water containing a fixed mass of sediment as per Huib de Swart comments
Morphostatic assumption and timescale limits

How do we propose to get around previous issues? We don't try and resolve everything perfectly yet; we limit our focus such that it is a useful engineering tool.

Earlier versions of SedTRAILS visualized sediment transport fields (e.g., ), but now we directly compute a sediment transport velocity field and use that to advect particles.

Divergent vs non-divergent vector fields

**Main steps (e.g. hydrodynamics then transport then post)**

[Step 1.]
- Run hydrodynamic model
- Compute sediment velocities
- Define sources
- Run SedTRAILS
- Post-process

## Hydrodynamic Model
**Delft3D/DFM**

*Overview of model; different versions and how they generally work;*

The Lagrangian SedTRAILS model must be coupled offline to an Eulerian hydrodynamic and sediment transport model. This Eulerian model solves the equations of motion for water and sediment on a fixed grid, which is then later used to advect moving particles. Here we consider both Delft3D 4  and D-FLOW FM . The most relevant difference for running SedTRAILS is that Delft3D 4 has a structured mesh, while the mesh of D-FLOW FM is unstructured.  

The sediment transports used in the SedTRAILS model are based on Delft3D Online Morphology. The main components of the Delft3D Online Morphology model are the coupled Delft3D-Wave and the Delft3D-Flow modules. Delft3D-Flow forms the core of the model system simulating water motion due to tidal and meteorological forcing by solving the unsteady shallow-water equations. These equations can be resolved on a structured grid (Delft3D 4) or unstructured mesh (Delft3D FM). Wave effects, such as enhanced bed shear stresses and wave forcing due to breaking, are integrated in the flow simulation by running the 3rd generation SWAN wave processor. The SWAN wave model is based on discrete spectral action balance equations, computing the evolution of random, short-crested waves (e.g., ). The results of the wave simulation, such as wave height, peak spectral period, and mass fluxes are stored on the computational flow grid and included in the flow calculations through additional driving terms near surface and bed, enhanced bed shear stress, mass flux and increased turbulence. Wave processes are resolved at the wave time-step, which is typically every 10 to 60 minutes.

**What you need from the model**

*Which fields to output? What timeframe (e.g. representative tidal cycle)*

The main computational bottleneck in running SedTRAILS is the underlying hydrodynamic model, so it is desirable to limit its model run time. If we want to simulate particle pathways over longer time periods than the feasible hydrodynamic model run time, we can loop the hydrodynamic model, as is common practice in Lagrangian ocean modelling .  This looped model could consist of a representative tidal cycle with components chosen to approximate the residual sediment transport at a given site (e.g., ), or else could be selected as a subset of a longer real time series .

This approach thus permits particles to be advected for longer time scales than available from the raw data. However, particle looping can only work if the model has no drift in the velocity or tracer fields, that there are no large unphysical jumps in the fields between the end and the beginning of the model run, and that any unphysical jumps will have a small net effect on the particle pathways.

A key requirement for looping the model is that velocity fields in the area of interest must be virtually identical at the start and end of the cycle. Failing to choose this carefully could lead to an artificial net transport effect or aphysical jumps, and hence incorrect pathways.  Depending on the size of the model domain, the ideal time window will not be the same everywhere. For this reason, SedTRAILS computations based on a looped model should be analyzed critically at the boundaries.  

Output every 10-20 minutes from Eulerian model. This is important where flow varies rapidly in space and time. 

Input file

In order to run SedTRAILS, `_c`, `_a`, 

### Wave-Driven Sediment Velocity 

In nearshore regions, wave nonlinearity also plays an important role in determining sediment transport. We account for this effect on the sediment advection velocity in SedTRAILS using the method of , as implemented in the XBeach model .  estimate wave nonlinearity as a function of the Ursell number:

	Ur = {4} {(kh)^3}

where wave amplitude `a_w = {2}H_{s}` (significant wave height), `k` is the local wave number derived via linear wave theory, and `h` is local water depth. 

The total nonlinearity `B` is given by:

	B = p_1 + {1+{p_4}}

where `p_1=0`,  `p_2=0.857`,  `p_3=0.471`,  `p_4=0.297`,  `p_5=0.815`, and `p_6=0.672`, fit to extensive field dataset under breaking and non-breaking conditions.

The phase of the nonlinearity `` is given by:

	 = -90^{} + 90^{}  (p_5/Ur^{p_6})

From this, it follows that `S_k = B  ` and `A_s = B  `. We can then compute the wave-induced nonlinear wave-averaged (onshore) velocity as:

	_{a} = f_{ua}(S_k - A_s) _{rms}

where `f_{ua}` is an adjustable calibration parameter (default value = 1.0???), and the orbital velocity `_{rms}= H_{rms}/T  (kh)` from linear wave theory (with root-mean-square wave height `H_{rms}` and wave period `T`). A higher value of `_a` corresponds to a stronger wave-induced velocity component in the direction of wave propagation `_{wave}`. 

`_{a}` is vectorially added to the Eulerian current velocity `_c` to give the total flow velocity under combined wave and current conditions `_{flow}` for use in Equation .

## Sediment Advection Velocity
*Different ways to do this, we choose to directly compute sediment velocity field; we use Soulsby and 

Earlier versions of SedTRAILS (e.g., Stevens et al, Pearson 2021) visualized the sediment transport field but did not resolve particle motions (*add Huib de Swart language*). Here we directly compute a sediment velocity field.

*Rephrase:* The SedTRAILS approach was specifically developed to simulate sediment trajectories accurately and computationally efficiently. Runtime efficiency is obtained from decoupling the sediment trajectory computation from the sediment transport vector computation.  We use classic morphodynamic schematisation techniques to derive a morphodynamic tide and/or wave climate. A morphostatic Delft3D model is then run in high resolution over the morphological representative conditions. Since there is no morphodynamic feedback the sediment transport vector fields remain unchanged for repetitive tides. With a careful selection of a representative morphological tide, the Delft3D computation only needs to execute once. The resulting sediment transport vector fields can then be repeated to more efficiently model pathways. The particle motion computations are efficient (fast) as the sediment transport vectors are already resolved. 

*Rephrase:* SedTRAILS (Sediment TRAnsport vIsualization \& Lagrangian Simulator) was applied to visualize, identify, and analyze the pathways along which sand-sized sediment was transported during the coupled hydrodynamic and sediment transport model simulations. Based on the Eulerian sediment  velocity fields, SedTRAILS computes the Lagrangian pathways that idealized particles travel as they pass through a changing vector field. SedTRAILS was adapted from techniques described in  and employs a similar methodology to sediment particle tracking modules such as PTM (MacDonald et al., 2006).

*Rephrase:* Existing particle tracking approaches (like Delft3D PART) use velocity fields coupled with simplified formulas to govern sediment entrainment and settling thresholds based on critical shear stresses, often greatly simplifying the processes of sediment transport, as key behaviour like particle settling may be neglected.

In SedTRAILS, we resolve the sediment advection velocity field `_{sed}` by adapting the approach of :

	_{sed} = R  P  _{flow}

The vector components `{U}_{sed,x}` and `{U}_{sed,y}` of `_{sed}` are scaled such that `_{sed}` is aligned with `_{flow}`. In the absence of waves, `_{flow}` is equivalent to the Eulerian current velocity `_c` obtained from the hydrodynamic model. The effect of wave nonlinearity on `_{flow}` is accounted for in Section~. 

To estimate the forces acting on the bed, the mean bed shear stress (`_m`) and maximum bed shear stress due (`_{max}`) are determined directly from the Eulerian model output. If both waves and currents are present in the Eulerian model, this approach accounts for their combined effect on the seabed. The friction velocities `u_{*m}` and `u_{*max}` are given as `u_{*} = `, and Shields numbers `_{m}` and `_{max}` for a given sediment class are given as ` = /`. Parameters used in the model are further described in .

### Velocity Reduction Factor (R)

To derive the velocity reduction factor `R` of tracer sediment for all points in the model domain, we first consider bed and suspended load transport separately (`R_b` and `R_s`, respectively), and then compare them to determine the total transport velocity reduction factor `R`.

Bedload velocity is determined as a function of critical shields exceedance, with (INTUITIVE EXPLANATION) 

Based on  and , the bedload velocity `U_b` is given by:

	U_{b} = 10 u_{*m} [ 1 - 0.7 (_{cr,A}/_{max})^{1/2}]

The bedload transport velocity reduction factor `R_b=U_b/U_{flow}` is given by:

	R_{b} =

		10 u_{*m} [ 1 - 0.7 (_{cr,A}/_{max})^{1/2}]/U_{flow}, &  > _{cr,A}`}\\
		0, &   _{cr,A}`}

As a rule of thumb,  indicate that the bedload transport velocity is typically between 15-50\

The suspended transport velocity `U_s` is given by:

	U_{s} = ^{z_2}U(z)C(z)dz}{_{z_1}^{z_2}C(z)dz}

Suspended transport velocity reduction factor `R_s=U_s/U_c` is given by:

	R_{s} = {[(8/7)-]} 
	-1]}{[(8/7R_b)^{(7-7)}-1]}

To calculate the total transport velocity reduction factor `R`, we use the Rouse number `=w_s/( u_{*max})` to determine whether bedload or suspended load transport dominate:

	R =

		R_b, & \\
		R_s, & 

*Assumption of Rouse profile: developed log flow profile etc, which may not be valid under wave-driven flows?*

### Probability of Motion (P)

The probability of movement `P` for tracer sediment at all points in the model domain is determined by:

	P =

		^{-1/4} , &  > _{cr,A}`}\\
		0, &   _{cr,A}`}

Where `_d=0.5` is the dynamic friction coefficient defined by . During a given timestep ` t`, `P  t` becomes the amount of time that a grain spends moving.

## Particle Source Definition
**Different types of sources**\\
*point source, cluster, transect, grid*
Defaults

:\\
This file uses a *k*-means clustering algorithm to classify the bathymetry file into *k* clusters with similar XY position and bathymetry (Figure~). The result is an objective a partitioning of the model domain into a series of *k* representative cells. The centroid of each cell is then used as the source location for particles in SedTRAILS. It outputs several plots so that you can verify whether the clusters look reasonable, and saves a  file with the source info for SedTRAILS to use in subsequent steps.

\\
Alternatively, particle sources can be set up in a line by specifying start and end points for each line as well as the desired number of points. This option is particularly useful if using SedTRAILS for (relatively) alongshore uniform settings where the alongshore variation in particle trajectories is small but cross-shore variations may be large (e.g., at a barred beach). 

\\
To analyze Lagrangian Coherent Structures (LCS), Finite Time Lyapunov Exponents (FTLE) are calculated for the domain of interest.  This requires the creation of a regular grid of particles to enable the separation between neighbouring particles to be computed. At present, the user specifies a bounding box specifying the four corners of the grid, as well as the number of points in the X and Y directions. 

**Note: should add function to read in an arbitrary list of points as sources. Original mdrift code had some lines to read in 

**Release timing**\\
*instantaneous; continuous*
Defaults

Ergodicity: especially important for faster conditions where particles can move into a new region on a single tidal cycle (see  for further language on this)

## Sediment Pathway Estimation

### Numerical Scheme
*Bilinear interpolation of field (
Either Delft3D-4 curvilinear grid or FM, both now treated the same way; retriangulated

To estimate particle trajectories, particles are passively advected by the transport velocity fields derived in the previous steps:

	}{ t} = F(t)  

where `` is the horizontal position of a particle, `t` is time, `F(t)` is a freedom factor related to particle burial in the bed, and `` is the spatiotemporally-varying velocity field (e.g, `_{sed}` or `_{c}`). `D(_{diff})` is a random walk scaling factor to estimate the diffusion in random direction `_{diff}`. This equation is solved using a 4th-order Runge Kutta algorithm . 

*Note: there is a Courant-criteria-esque ``Speed Number'' defined in the code, but I have disabled that for FM output because we didn't have a good grid cell area output and I wasn't sure how to go about computing it.*

Timestep limitations

INCLUDE FIGURE SHOWING PARTICLE POSITION ON A TRIANGULAR GRID

Curvilinear Delft3D 4 and unstructured FM output are reformatted into a triangular grid, on which calculations are performed. This also includes nested Delft3D4 models.

**Diffusion**\\
Discuss how diffusion is handled and the implications for reverse PT?

However, diffusion is not usually applied to particle trajectories computed in reverse .

**Larger-scale pathway stuff**\\
*Going from single timesteps to longer pathways*

### Freedom Factor F(t)
If we want to consider the interaction of particles with the bed, then we must determine `F(t)`, a freedom factor representing whether the particle is buried and buried within the bed (`F(t)=0`) or on the bed surface and available for transport (`F(t)=1`). If we assume that particles have no interaction with the bed (as though the bed were made of concrete, or in cases with completely passive tracers), `F=1` for all timesteps.  Otherwise, `F(t)` depends on particle mobility and a series of empirical parameters. At `t=0`, it is assumed that all tracer particles are on the surface (`F(0)=1`). The state of a particle at the current timestep `F(t)` depends on its previous state `F(t-1)`:

	F(t) =

			1  & \\
			0  & 

		&   F(t-1)=1  \\

			1  & \\
			0  & 

		&   F(t-1)=0  

where `` is a random number from 0 to 1. `b` is the transition probability per second that a free particle becomes buried, which depends on the background sediment characteristics:

	b =

		b_e  1 -    , &  > _{cr,a}`}\\
		0, &   _{cr,a}`}

Conversely, `a` is the transition probability per second that a buried particle becomes free:

	a = {(1-_e)}

where `_e` is the long-term equilibrium proportion of particles that are free, `b_e` is the maximum free-to-buried transition probability, and `_s` is a scale value determining the distribution of residence times.  Thus, a representative particle is much more likely to transition states from buried to free or free to buried as `_{max,a}` increases, in proportion to `_e`. These three parameters must be determined via calibration, but as a default we use the values chosen by : `_e=0.1`, `b_e=1.7 10^{-7} s^{-1}`, and `_s=0.1` (based on a sand tracer study carried out on the north Scottish coast). 

## Post-Processing \& Visualization

**Connectivity**\\
*Can compile into connectivity network a la 

**Visualization**\\
*Different tricks we can do like plotting by age or origin to highlight different information*

## Model Availability
*FAIR; QC; Documentation; Code Maintenance etc; see Rolf doc*\\

FAIR:

	-sep 0pt
	 0pt	
	- Findable?
	- Source code openly available? Why/why not?
	- Interoperability/dependencies? OET?
	- Accessible/reusable? License? How can users build on or contribute to software?

Code Quality Control: how do we ensure code quality?

Description:

	-sep 0pt
	 0pt	
	- What can you do with this software?
	- How can you start using this software?
	- How to install?
	- Link to access?
	- Separate documentation?

Maintenance: how will software be maintained? User community?

## How to Run?
Text goes here

# Verification 

In this section, we verify SedTRAILS against three different cases: analytical test cases to show model stability, lab test cases to show sediment velocity calc is correct, and field verification to show how it performs in the real world...

Does the software deliver on its claims? Entry level, typical, and advanced cases

## Analytical Test Cases
*Standard analytical tests for e.g. numerical model stability; see de Vries thesis or PARCELS*

 proposes a series of standard test cases for particle models:

	-sep 0pt
	 0pt	
	- Radial rotation with known period. This setup tests particle trajectories in the simplest-possible flow, without time evolution. 
	- **Longitudinal shear dispersion flow in a pipe (e.g., Fischer et al., 2013) to ensure that shear dispersion effects are properly represented. **
	- Effective lateral diffusion due to an oscillating vertical shear flow (Bowden, 1965) to test particle trajectories in a time-evolving flow. 
	- **Steady-state flow around a peninsula   
	- Steady-state flow in a Stommel gyre and western boundary current (Fabbroni, 2009) to test particle trajectories in a domain with large gradients in flow speed. 
	- Damped inertial oscillation on a geostrophic flow (Fabbroni, 2009; Döös et al., 2013) to appropriately quantify sub-inertial motion, e.g., loopers. 
	- For codes that include diffusivity, a simulation of Brownian motion with a given Kh and Kv to test for sub-grid parameterizations of diffusivity.

Other analytical tests from  available here:\\ 

With these tests, we verify the accuracy of SedTRAILS' numerical scheme and general modelling capabilities.

## Laboratory Test Cases
*Test transport velocity calculations*

Johan see Crickmore \& Lean (1962)

With these tests, we verify the SedTRAILS sediment velocity calculation for simple, well-constrained cases.

## Real-World Test Cases

To verify SedTRAILS in a real-world setting, we consider the sediment tracer study of  at Ameland Inlet in the Netherlands.

STUART TO DISCUSS WITH ROY AND RE-RUN MODEL FROM VAN WEERDENBURG (2021)

With this test, we verify that SedTRAILS can reproduce the distribution of sediment ... maximum potential pathways

# Application 
*What kinds of questions can we answer with SedTRAILS that we couldn't answer before? Querying (permit area, eroding beach, navigation channel, then combination); gross vs net; reverse particle tracking; connectivity input; check prev studies like Carlijn's to see what else we tried*

Reflect back on questions posed in Section~.

 use an earlier version of SedTRAILS to inform coastal sediment connectivity;  use SedTRAILS to investigate Lagrangian coherent structures in sediment velocity fields.

# Discussion 

**Big New Awesome Things**\\

DISTINGUISH FROM PARCELS ETC ESP IF WE GO GMD ROUTE

MAYBE FOR GMD WE DON'T NEED TO DO THIS SO MUCH?

*The fancy novel thing about SedTRAILS that makes it better: fast/flexible/fun*
Successful verification and application shows the model's potential
Limit of concrete bed approach
No transport gradients presented therefore no morphological change
Potential next steps for dealing with mixing in the bed

Pragmatic approach: maximum potential pathways, didn't try to quantify rates, morphological change
Validation of the model for different cases is ongoing

What kind of measurements do we need to make useful comparisons?

	-sep 0pt
	 0pt	
	- Tracer studies (fluorescent-magnetic, OSL)
	- Repeat bathymetric surveys
	- Potential for validation with remote sensing (sediment streaks?)
	- Laboratory studies with tracers

**Education \& Stakeholder Communication**\\
*This stuff is really complicated and this way maybe we can help explain it better*
Just beware of danger of attractive model results; make sure you use it for reliable models

NOT JUST A WAY TO COMMUNICATE TO STAKEHOLDERS: IT GIVES A NEW WAY FOR MODELLERS TO SEE NEW PATTERNS IN THEIR MODELS AND MAKE NEW INTERPRETATIONS

CAN WE COME UP WITH A CONTRASTING EXAMPLE WHERE YOU WOULD DO IT THE OLD WAY? DO IN EXAMPLE MODELS AND THEN DISCUSS?  old way being vector fields, sediment budgets, erosion sedimentation maps (no idea where coming from or going to), but this tells you potential sources or sinks for sediment from a site of interest, which could inform things like dredging strategies,

So maybe we can do our examples with a bit of this in mind
Ameland: where is sediment from island tips eroding to? How does it circulate on a larger scale? Only needs like 5 sentences, not a detailed case study. That is not the purpose of this example, it's more a proof of concept.

## Outlook
*Prospects for future development; end with potential impact*\\

**Next Steps: Representation of Physical Processes**\\
Next steps to address: mixing in the bed, fine sediment, and lab experiment with tracers in 2D
Physical processes that we neglect: mixing in the bed, 3D processes, fine sediment processes (e.g. flocculation and vertical settling)
Aeolian transport to address sediment transport pathways across full continuum
Other comments on improvements we could make based on case study

Additional functions for coral larvae or mangrove propagule dispersal (e.g. Storlazzi 2017, Bisschop 2023, Thillaigovindarasu 2023)
Coastal structures (e.g. permeable/impermeable dams) should be further developed and tested

**Next Steps: Software Improvements**\\
Interactive GUI and visualization tool; open source community model;
Model performance and efficiency; parallelization, adaptive timesteps etc
Potential for coupling with other models (e.g. ShorelineS)

**Next Steps: Applications**\\
*Connectivity and probabilistic approaches; Lagrangian Coherent Structures; dredge disposal and nourishment planning; morphodynamics*

# Conclusions 
*Overview of main takeaways*\\
We made a new model and tested it! It worked
These are the key assumptions
It is useful for a bunch of things
Outlook for future

*Impact!*

 # Model Constants  

To apply the method of Soulsby (2011), several constant parameters are assumed based on Soulsby (1997). This approach is intended for sand-sized particles ($63$–$2000~\mu\text{m}$). 

Gravity: $g = 9.81~\text{m/s}^2$  
von Karman's constant: $\kappa = 0.40$  
Kinematic viscosity: $\nu = 1.36 \times 10^{-6}~\text{m}^2/\text{s}$  
Water density: $\rho_w = 1027~\text{kg/m}^3$ (valid for $T = 10^\circ$C and $S = 35$ ppt)  
Sediment particle density: $\rho_s = 2650~\text{kg/m}^3$  
Tracer grain size: $d_t$  
Background grain size: $d_a$  

The dimensionless grain size is defined as:

$$
D_{*} = d \left( \frac{g(\rho_s/\rho_w - 1)}{\nu^2} \right)^{1/3}
$$

To determine the potential for particle mobility, the critical Shields number is calculated:

$$
\theta_{cr} = \frac{0.3}{1 + 1.2 D_{*}} + 0.055 \left(1 - \exp(-0.020 D_{*})\right)
$$

Settling velocity of suspended particles is estimated:

$$
w_s = \frac{\nu}{d} \sqrt{10.36^2 + 1.049 D_{*}^3} - 10.36
$$

The ratio of tracer grain size to background grain size is:

$$
A = \frac{d_t}{d_a}
$$

If tracer sediment is a different size from background sediment, we apply an adjustment to the critical Shields number to account for hiding and exposure effects:

$$
\theta_{cr,A} = \theta_{cr,t} \sqrt{\frac{8}{3A^2 + 6A - 1}} \cdot \frac{3.2260A}{\left\lbrace 4A - 2 \left[ A + 1 - \sqrt{A^2 + 2A - \frac{1}{3}} \right] \right\rbrace}
$$
