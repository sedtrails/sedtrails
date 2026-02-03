import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def macdonald_z_over_h(rouse_number):
    """
    MacDonald et al. (2006) Eq. 27 expressed as z_s/H = f(Rouse).
    (Depth-independent.)
    """
    log_arg = np.log(rouse_number) - 0.4
    tanh_arg = 1.2 * log_arg
    return 0.0398 * (10 ** (-1.08 * np.tanh(tanh_arg)))

# Rouse grid (log-spaced)
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
out_nc = r"c:\Users\dagalaki\Documents\GitHub\sedtrails\vassia_mcdonald\src\sedtrails\transport_converter\plugins\physics\macdonald_z_over_h_lookup.nc"
ds.to_netcdf(out_nc, encoding=encoding)

# Optional: quick plot
plt.figure(figsize=(7,4))
plt.plot(rouse, z_over_h, linewidth=2)
plt.xscale("log")
plt.xlabel("Rouse number (-)")
plt.ylabel("z_s/H (-)")
plt.title("MacDonald Eq. 27: z_s/H vs Rouse")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

plt.savefig(
    r"c:\Users\dagalaki\Documents\GitHub\sedtrails\vassia_mcdonald\src\sedtrails\transport_converter\plugins\physics\macdonald_zs_over_H_vs_rouse.png",
    dpi=300
)