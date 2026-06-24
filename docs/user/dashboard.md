# Monitoring Dashboard

The purpose of the dashboard is to provide a real-time visualization of the simulation as it runs. It can help users to monitor the progress of the simulation, identify potential issues, and gain insights into sediment transport and deposition. 

![SedTRAILS dashboard example](../_static/img/dashboard-example.png)

## Navigating the Dashboard

- **(a) Flow Field (Latest)**: we see the latest flow field timestep. The colour scale corresponds to the magnitude of the flow field and vectors indicating direction are displayed on top.
- **(b) Bathymetry + Particles**: the bathymetry and particle positions overlaid. Initial positions are marked with Xs, and the position at the current timestep is indicated with a circle. 
- **(c) Longshore Velocity**: the average and maximum magnitude (?) of the alongshore velocity of displayed particles. For now it is assumed that the alongshore velocity is in the X direction.
- **(d) Cross-shore Velocity**: the average and maximum magnitude (?) of the cross-shore velocity of displayed particles. For now it is assumed that the cross-shore velocity is in the Y direction.
- **(e) Average Distance per Timestep**: the average distance travelled by displayed particles in each output (not computational!) timestep. This is perhaps the most important diagnostic tool: if the travel distance of your particles is zero and you expect them to move (given your particle characteristics and flow conditions), something may be wrong!
- **(f) Burial Depth**: the average and maximum depths at which displayed particles are buried ($\delta_{burial}$) and the mixing depth ($\delta_{mix}$ in [van Westen et al (2025)](https://doi.org/10.1038/s41598-025-92910-z))
- The progress bar at the bottom indicates the current simulation timestep and the relative percentage of your total run that is complete.

Note that it is possible to zoom and pan using the buttons in the bottom right corner. Pressing the "home" icon will return the default view. Clicking the "save" button gives the option to save the view as a ``*.png`` file.

The dashboard uses adaptive date formatting, automatically adjusting based on simulation duration:
  - **Short simulations** (< 1 day): Shows time only in HH:MM format
  - **Multi-day simulations** (1-7 days): Shows date and time in MM-DD HH:MM format  
  - **Long simulations** (> 7 days): Shows date only in MM-DD format


Future implementations will allow for more customization of the dashboard and optimize it for speed. However, it is primarily intended as an "is my model actually running as I intend it?" diagnostic, so users are advised to use the model output files and other visualization utilities to plot their results.

To keep redraws responsive, the live dashboard automatically uses a deterministic sample once a run has more than 5,000 particles. The particle map and dashboard time-series diagnostics are then based on that displayed sample. This does not change the simulation or the NetCDF output; use the output files for complete post-processing statistics.

## Turning Off the Dashboard
At the moment, the dashboard has been included as a quick check and has not been optimized for speed. Once you are satisfied that the model is running, you may wish to disable the dashboard for "production runs". To do so, change ``enable: true`` to ``enable: false`` in the config file ([see here](../user/simulations.md)):

```yaml
visualization:
  dashboard:
    enable: true
    update_interval: 1H
```
It is also possible to change the interval at which the dashboard is updated by modifying ``update_interval: 1H``. The default is 1 hour because of the timescale of our example simulation, but it may be useful to change this if your simulation is much longer/shorter.
