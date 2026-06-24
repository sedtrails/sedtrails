# Looping Eulerian Forcing Fields

## Overview

SedTRAILS can run simulations longer than the available flow-model forcing by looping the input Eulerian fields in time. This mirrors a common workflow in Lagrangian particle tracking where a representative cycle (tidal, spring-neap, seasonal) is repeated to extend trajectories. This was a "widely used" feature of the old MATLAB sedtrails code.

## When to Use Looping

Use looping when:
- Your forcing data represent a repeatable cycle.
- You want to study transport over longer periods than the forcing duration.

Avoid looping when:
- There is a trend or drift in the input flow fields.
- The end and start of the forcing period have large, unphysical jumps.

## Configuration

Enable looping by setting `inputs.repeat_eulerian_fields` to `true`.

```yaml
inputs:
  data: ./sample-data/inlet_sedtrails.nc
  read_interval: 10D
  repeat_eulerian_fields: true

# Run longer than the input forcing duration
# (example: 60 days of particles from a 15 day forcing file)

time:
  start: 2016-09-21 19:20:00
  duration: 60D
  timestep: 60S
```

If `repeat_eulerian_fields` is `false`, SedTRAILS will keep using the final available Eulerian fields once the end of the forcing period is reached.

## How Time Mapping Works

SedTRAILS first validates that `time.start` is inside the input forcing window. Looping is not used to rescue a simulation that starts before the first forcing timestamp or after the last forcing timestamp.

After that valid start:
- In-window simulation times use the matching Eulerian forcing timestamp.
- If `repeat_eulerian_fields` is `true`, simulation times after the forcing end are mapped back into the forcing cycle.
- If `repeat_eulerian_fields` is `false`, SedTRAILS keeps using the final available Eulerian fields once the forcing period is exhausted.

This time mapping is applied consistently across velocity and tracer fields so that particle advection remains synchronized with the Eulerian data.

## Morfac Decompression

When `general.input_model.morfac` is greater than 1, the forcing time axis is decompressed before mapping. This means:
- A morphologically accelerated input model is interpreted at its real-time scale.
- Looping and interpolation operate on the decompressed timeline.

This keeps the Eulerian time base consistent with the Lagrangian integration when morphological acceleration has been applied to the input model output.

## Practical Tips

- Test for discontinuities by plotting a few grid-point time series across the final and initial forcing steps.
- For long loops, confirm that the cycle length (e.g., 14.76 days for spring-neap) is represented accurately in the input data.
- Consider comparing short (single-cycle) and long (multi-cycle) runs to quantify sensitivity to looping.
