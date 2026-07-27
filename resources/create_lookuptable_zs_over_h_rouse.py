import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

def macdonald_z_over_h(rouse_number):
    """
    MacDonald et al. (2006) Eq. 27 expressed as z_s/H = f(Rouse):

        z_s/h = 0.0398 * 10 ** ( -1.08 * tanh[ 1.2 * ln(w_s / (kappa*u_star)) - 0.4 ] )

    (ERDC/CHL TR-06-20, p.25, Eq. 27; rouse_number here stands in for
    w_s / (kappa*u_star)). The -0.4 offset is applied AFTER multiplying by 1.2,
    i.e. tanh(1.2*ln(rouse) - 0.4), not tanh(1.2*(ln(rouse) - 0.4)).
    (Depth-independent.)
    """
    tanh_arg = 1.2 * np.log(rouse_number) - 0.4
    return 0.0398 * (10 ** (-1.08 * np.tanh(tanh_arg)))


def _build_lookup_table():
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


if __name__ == "__main__":
    _build_lookup_table()