# Input Formats

This page collects the practical differences between the input model formats SedTRAILS can work with. The goal is to document the assumptions that matter for loading data, choosing variables, and interpreting model output.

Note: structured input grids are flattened to 1D spatial vectors so downstream modules can treat all input formats consistently.


## Input Models

SedTRAILS computes particle pathways based on pre-existing hydrodynamic and/or sediment transport model output. At present we support output from several models:
1. D-Flow FM
2. Delft3D-4
3. XBeach
4. SFINCS

In the sections below we provide information about the output settings required in these models in order to be used in SedTRAILS.


## D-Flow FM
```
format: fm_netcdf
```
D-Flow FM input is the most direct path through SedTRAILS. The map output from D-Flow FM is written to binary `*.nc` files which can be directly read by SedTRAILS using the `fm_netcdf.py` format converter plugin. The seeding GUI and the simulation workflow expect a NetCDF-based flow field with cell-center coordinates and a bathymetry variable that can be used to preview the domain.

Typical variables used by the GUI and runtime are:

- `net_xcc`
- `net_ycc`
- `bedlevel` or `bed_level`

## Delft3D 4
```
format: d3d4_netcdf
```

By default, the map output from Delft3D-4 is written to binary `trim-*.dat` files. At present it is not possible to directly read these files directly with Python, so it is better to instead write the map output as `*.nc` files. To enable `*.nc` output in Delft3D-4, add the following lines to the `*.mdf` file:
```
FlNcdf= #maphis#
ncFormat=4
```
The resulting  `*.nc` file can then be directly read by SedTRAILS using the `d3d4_netcdf.py` format converter plugin.

The Delft3D4 `U1/V1`, `TAUKSI/TAUETA`, `SBUU/SBVV`, and `SSUU/SSVV`
variables are local xi/eta vector components, rather than global x/y
components. SedTRAILS centers each pair with the time-dependent `KFU` and
`KFV` wet masks, or the static `KCU` and `KCV` masks when the dynamic
masks are unavailable, then rotates the result with static `ALFAS` into
global x/y. Incomplete vector pairs are rejected. Faces where `KCS=0` are
removed before conversion and particle seeding.

Delft3D4 `DPS0`/`DP0` are bottom depths positive down. SedTRAILS negates
face-centered `DPS0` (or a face-aligned `DP0` with no non-face location
metadata) to obtain bed elevation. A node-located or edge-located `DP0`
without `DPS0` is rejected. If `DPS` is
absent, water depth is derived as `S1 - bed_level`.

SedTRAILS uses coordinate differences as planar metric distances. Use
Delft3D4 output in a projected coordinate system with consistent length units.
A geographic `XZ/YZ` grid in longitude/latitude degrees must be reprojected
before it can produce physically meaningful particle distances and velocities
in SedTRAILS.

For multi-fraction Delft3D-4 inputs, SedTRAILS resolves `sediment_fraction_name` values against the `NAMCON` field in the NetCDF file. If the input only contains a single sediment fraction, NAMCON is not used for fraction selection. For example, a label such as `sediment100_nat` can be selected by name instead of by zero-based index when multiple fractions are present. This name-based lookup is not yet implemented for FM or XBeach inputs because their transport variable naming conventions differ and still need dedicated mapping.


## XBeach
```
format: xbeach
```
The map output from XBeach is written to binary `*.nc` files which can  be directly read by SedTRAILS using the `xbeach.py` format converter plugin. We wrote this explicitly to work with `*_mean` variables only. Otherwise, especially in surfbeat mode, the global output provides randomly-phased long wave flow fields, which don't provide meaningful information on flow or transport for SedTRAILS- just instantaneous snapshots. This means that we work with `zb_mean`, which is the average bed level over that time avg interval (defined by `tintg` in the params file).

The variables expected in the output section of the XBeach `params.txt` file are:

```text
nmeanvar =13
thetamean
zb
hh
ue
ve
taubx
tauby
Susg
Svsg
Subg
Svbg
ua
cctot
```

In practice, the GUI uses the spatial coordinates from `globalx` and `globaly`, and bathymetry preview defaults to `zb_mean` when it is available.

## SFINCS
```
format: sfincs
```
The map output from SFINCS is written to binary `*.nc` files which can  be directly read by SedTRAILS using the `sfincs.py` format converter plugin. SFINCS input is supported through the same general configuration flow, but it has its own conventions for reference dates, grid layout, and file naming. The main thing to watch is whether the file contains the coordinate and time variables required by the converter and by any visualization path you plan to use.

For configuration details, start from the SFINCS example and the converter-specific documentation, then verify the exact variable names in your NetCDF file.

## Other models
To add other models as input, you need to construct a format converter plugin as per the
[Plugin Guidelines](../developer/plugins.md).