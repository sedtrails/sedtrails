import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

def macdonald_z_over_h(rouse_number):
    """
    MacDonald et al. (2006) Eq. 27 expressed as z_s/H = f(Rouse).
    (Depth-independent.)
    """
    log_arg = np.log(rouse_number) - 0.4
    tanh_arg = 1.2 * log_arg
    return 0.0398 * (10 ** (-1.08 * np.tanh(tanh_arg)))

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

# Project root = current working directory
root = Path(__file__).resolve().parents[1]

# Target directory for NetCDF
nc_dir = (
    root
    / "src"
    / "sedtrails"
    / "transport_converter"
    / "plugins"
    / "physics"
)

nc_dir.mkdir(parents=True, exist_ok=True)

out_nc = nc_dir / "macdonald_z_over_h_lookup.nc"

# PNG in *present working directory*
out_png = root / "macdonald_zs_over_H_vs_rouse.png"

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

rouse = np.geomspace(0.01, 50.0, 600)
z_over_h = macdonald_z_over_h(rouse).astype(np.float32)

ds = xr.Dataset(
    data_vars={"z_over_h": (("rouse",), z_over_h)},
    coords={"rouse": rouse},
    attrs={
        "description": "MacDonald et al. (2006) Eq. 27 suspended-load centroid height fraction (z_s/H)",
        "units": "-"
    },
)

encoding = {"z_over_h": {"zlib": True, "complevel": 4}}
ds.to_netcdf(out_nc, encoding=encoding)

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------

plt.figure(figsize=(7, 4))
plt.plot(rouse, z_over_h, linewidth=2)
plt.xscale("log")
plt.xlabel("Rouse number (-)")
plt.ylabel("z_s/H (-)")
plt.title("MacDonald Eq. 27: z_s/H vs Rouse")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()

plt.savefig(out_png, dpi=300)
plt.show()