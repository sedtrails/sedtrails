import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sedtrails.transport_converter.physics_lib import *

def make_parameter_space_plots_uc_uw(
    compute_grain_properties,
    calculate_equilibrium_bedform_height,
    calculate_current_related_bed_roughness_vanrijn2007,
    calculate_apparent_bed_roughness_vanrijn2007,
    calculate_skin_roughness,
    # ranges
    uc_range=(0.1, 3.0),
    uw_range=(0.0, 3.0),
    d50_range=(5e-5, 2e-3),
    depths=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
    n_uc=25,
    n_uw=25,
    n_d=35,
    # constants
    phi_deg=90.0,
    gravity=9.81,
    water_density=1025.0,
    sediment_density=2650.0,
    kinematic_viscosity=1.3e-6,
    # diagnostic friction factors for theta_max mapping
    f_c=0.004,
    f_w=0.02,
    theta_cr_min=1e-6,
    # plotting
    gridsize=60,
    mincnt=1,
    save_path="roughness_parameter_space",
    dpi=300,
    show=True,
):
    """
    Parameter-space comparison:
      Uc, Uw -> tau_c, tau_w -> tau_max -> theta_max
    then compares:
      ks_skin, eta_b, ks_c, ka
    in (theta_max, d50) space using hexbin.

    This version:
      - enforces one GLOBAL LogNorm (same vmin/vmax everywhere),
      - adds gridlines,
      - saves the figure (png + pdf).
    """

    rho = water_density
    rho_s = sediment_density
    s = rho_s / rho

    # ----------------------------
    # Grids and broadcasting
    # ----------------------------
    uc = np.linspace(uc_range[0], uc_range[1], n_uc)
    uw = np.linspace(uw_range[0], uw_range[1], n_uw)
    d50 = np.logspace(np.log10(d50_range[0]), np.log10(d50_range[1]), n_d)

    D50 = d50[:, None, None]      # (d,1,1)
    UC  = uc[None, :, None]       # (1,uc,1)
    UW  = uw[None, None, :]       # (1,1,uw)

    # ----------------------------
    # Grain properties (theta_cr)
    # ----------------------------
    theta_cr = np.zeros_like(d50)
    d_star = np.zeros_like(d50)
    for i, d in enumerate(d50):
        props = compute_grain_properties(
            grain_diameter=d,
            gravity=gravity,
            sediment_density=rho_s,
            water_density=rho,
            kinematic_viscosity=kinematic_viscosity,
        )
        theta_cr[i] = max(props["critical_shields"], theta_cr_min)
        d_star[i] = props["dimensionless_grain_size"]
    TH_CR = theta_cr[:, None, None]  # shape (d,1,1)
    D_STAR = d_star[:, None, None]  # shape (d,1,1)

    # Plot theta_cr vs d50
    # flatten grain-size dimension only
    Dstar_1d = D_STAR[:, 0, 0]
    d50_1d   = D50[:, 0, 0]
    theta_cr_1d = TH_CR[:, 0, 0]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(Dstar_1d, theta_cr_1d, "o-", lw=2)
    ax.set_xlabel(r"Dimensionless grain size $D_*$ [-]")
    ax.set_ylabel(r"Critical Shields parameter $\theta_{cr}$ [-]")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    def Dstar_to_d50(Dstar):
        return np.interp(Dstar, Dstar_1d, d50_1d)
    def d50_to_Dstar(d50):
        return np.interp(d50, d50_1d, Dstar_1d)

    secax = ax.secondary_xaxis(
        "top",
        functions=(Dstar_to_d50, d50_to_Dstar),
    )

    secax.set_xlabel(r"$d_{50}$ [$\mu$m]")
    secax.xaxis.set_major_formatter(
        lambda x, pos: f"{x*1e6:.0f}"
    )

    # ----------------------------
    # Skin roughness (grain-only)
    # ----------------------------
    ks_skin_1d = np.array([
        calculate_skin_roughness(method="soulsby_d50", d50=d)
        for d in d50
    ])
    KS_SKIN = ks_skin_1d[:, None, None]  # (d,1,1)

    # ----------------------------
    # Stresses and theta_max
    # ----------------------------
    tau_c = 0.5 * rho * f_c * UC**2
    tau_w = 0.5 * rho * f_w * UW**2

    phi = np.deg2rad(np.minimum(phi_deg, 360.0 - phi_deg))
    tau_max = np.sqrt(tau_c**2 + tau_w**2 + 2.0 * tau_c * tau_w * np.cos(phi))

    theta_max = tau_max / ((rho_s - rho) * gravity * D50)   # (d,uc,uw)

    # flattened x/y for hexbin
    x = theta_max.reshape(-1)
    y = np.broadcast_to(D50, theta_max.shape).reshape(-1)
    # Global log-safe limits
    xpos = x[np.isfinite(x) & (x > 0)]
    ypos = y[np.isfinite(y) & (y > 0)]

    xlim = (xpos.min(), xpos.max())
    ylim = (ypos.min(), ypos.max())


    # ----------------------------
    # Helper: positive min/max
    # ----------------------------
    def positive_minmax(arr):
        arr = np.asarray(arr)
        pos = arr[np.isfinite(arr) & (arr > 0)]
        if pos.size == 0:
            return np.inf, -np.inf
        return float(pos.min()), float(pos.max())

    # ----------------------------
    # PASS 1: global vmin/vmax over ALL fields & depths
    # ----------------------------
    global_min = np.inf
    global_max = -np.inf

    # include skin (broadcast to full shape)
    skin3d = np.broadcast_to(KS_SKIN, theta_max.shape)
    mn, mx = positive_minmax(skin3d)
    global_min = min(global_min, mn)
    global_max = max(global_max, mx)

    for h in depths:
        H = np.full_like(theta_max, h)

        eta_b = calculate_equilibrium_bedform_height(theta_max, TH_CR, D50, H)

        ks_c, ks_r, ks_mr, ks_d = calculate_current_related_bed_roughness_vanrijn2007(
            flow_velocity_magnitude=UC,
            near_bed_peak_orbital_velocity=UW,
            grain_diameter=D50,
            water_depth=H,
            relative_density=s,
            gravity=gravity,
        )

        ka, ka_ksc, gamma = calculate_apparent_bed_roughness_vanrijn2007(
            ks_c=ks_c,
            near_bed_peak_orbital_velocity=UW,
            flow_velocity_magnitude=UC,
            phi_deg=np.full_like(UW, phi_deg),
        )

        for arr in (eta_b, ks_c, ka):
            mn, mx = positive_minmax(arr)
            global_min = min(global_min, mn)
            global_max = max(global_max, mx)

    # guard for LogNorm
    if not np.isfinite(global_min) or global_min <= 0:
        global_min = 1e-12
    if not np.isfinite(global_max) or global_max <= global_min:
        global_max = global_min * 10

    global_norm = LogNorm(vmin=global_min, vmax=global_max)
    # from matplotlib.colors import PowerNorm
    # global_norm = PowerNorm(
    #     gamma=1.4, # > 1 stretches low values # 0.6,              # < 1 stretches high values
    #     vmin=global_min,
    #     vmax=global_max,
    # )


    # ----------------------------
    # Plot helper (gridlines + global norm)
    # ----------------------------
    def hexplot(ax, field, *, show_ylabel=False, show_xlabel=False, ylabel=None):
        field3d = np.broadcast_to(field, theta_max.shape)
        z = field3d.reshape(-1)

        m = np.isfinite(z) & (z > 0) & (x > 0) & (y > 0)

        hb = ax.hexbin(
            x[m], y[m],
            C=z[m],
            reduce_C_function=np.nanmedian,
            gridsize=gridsize,
            mincnt=mincnt,
            xscale="log",   # important for log-log hexbin rendering [1](https://www.coastalwiki.org/wiki/Sediment_transport_formulas_for_the_coastal_environment)[2](https://apps.dtic.mil/sti/tr/pdf/AD1005459.pdf)
            yscale="log",
            cmap="inferno",
            norm=global_norm,
        )

        ax.set_ylabel(ylabel if show_ylabel else "")
        ax.set_xlabel(r"$\theta_{\max}$ [-]" if show_xlabel else "")

        # gridlines (major + minor)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(True, which="major", alpha=0.35, linewidth=0.6)
        ax.grid(True, which="minor", alpha=0.15, linewidth=0.4)
        ax.tick_params(direction="in", which="both")


        return hb

    # ----------------------------
    # PASS 2: plot with clean labels + one colorbar per column (same limits)
    # ----------------------------
    fig, axes = plt.subplots(
        nrows=len(depths),
        ncols=4,
        figsize=(18, 3.8 * len(depths)),
        constrained_layout=True,
    )

    col_titles = [
        r"Skin roughness ($2.5d_{50}$)",
        r"MacDonald bedform height $\eta_b$",
        r"van Rijn (2007) $k_{s,c}$",
        r"van Rijn (2007) $k_a$ (apparent)",
    ]
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=11)

    hb_col = [None, None, None, None]

    for r, h in enumerate(depths):
        H = np.full_like(theta_max, h)

        eta_b = calculate_equilibrium_bedform_height(theta_max, TH_CR, D50, H)

        ks_c, ks_r, ks_mr, ks_d = calculate_current_related_bed_roughness_vanrijn2007(
            flow_velocity_magnitude=UC,
            near_bed_peak_orbital_velocity=UW,
            grain_diameter=D50,
            water_depth=H,
            relative_density=s,
            gravity=gravity,
        )

        ka, ka_ksc, gamma = calculate_apparent_bed_roughness_vanrijn2007(
            ks_c=ks_c,
            near_bed_peak_orbital_velocity=UW,
            flow_velocity_magnitude=UC,
            phi_deg=np.full_like(UW, phi_deg),
        )

        ks_skin3d = np.broadcast_to(KS_SKIN, theta_max.shape)
        show_x = (r == len(depths) - 1)

        hb0 = hexplot(
            axes[r, 0], ks_skin3d,
            show_ylabel=True,
            ylabel=rf"$d_{{50}}$ [m]" + "\n" + rf"$h={h}$ m",
            show_xlabel=show_x,
        )
        hb1 = hexplot(axes[r, 1], eta_b, show_xlabel=show_x)
        hb2 = hexplot(axes[r, 2], ks_c,  show_xlabel=show_x)
        hb3 = hexplot(axes[r, 3], ka,    show_xlabel=show_x)

        if hb_col[0] is None:
            hb_col = [hb0, hb1, hb2, hb3]

    # One colorbar per column, but all share identical vmin/vmax via global_norm
    fig.colorbar(hb_col[0], ax=axes[:, 0], label="value [m] (global log scale)", pad=0.02)
    fig.colorbar(hb_col[1], ax=axes[:, 1], label="value [m] (global log scale)", pad=0.02)
    fig.colorbar(hb_col[2], ax=axes[:, 2], label="value [m] (global log scale)", pad=0.02)
    fig.colorbar(hb_col[3], ax=axes[:, 3], label="value [m] (global log scale)", pad=0.02)

    # overall title (optional)
    fig.suptitle(
        rf"Parameter space roughness comparison (global color scale), $\phi={phi_deg:.0f}^\circ$",
        fontsize=12,
    )

    # ----------------------------
    # Save figure
    # ----------------------------
    png_name = f"{save_path}_phi{phi_deg:.0f}.png"
    # pdf_name = f"{save_path}_phi{phi_deg:.0f}.pdf"

    fig.savefig(png_name, dpi=dpi, bbox_inches="tight")
    # fig.savefig(pdf_name, bbox_inches="tight")
    print("Saved:", png_name)
    # print("Saved:", pdf_name)

    if show:
        plt.show()

    return fig, axes


make_parameter_space_plots_uc_uw(
    compute_grain_properties=compute_grain_properties,
    calculate_equilibrium_bedform_height=calculate_equilibrium_bedform_height,
    calculate_current_related_bed_roughness_vanrijn2007=calculate_current_related_bed_roughness_vanrijn2007,
    calculate_apparent_bed_roughness_vanrijn2007=calculate_apparent_bed_roughness_vanrijn2007,
    calculate_skin_roughness=calculate_skin_roughness,

    # your requested ranges
    uc_range=(0.1, 3.0),
    uw_range=(0.0, 3.0),
    d50_range=(5e-5, 2e-3),
    depths=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0),

    # resolution (lower these if it’s heavy)
    n_uc=25,
    n_uw=25,
    n_d=35,

    # physics constants
    water_density=1025.0,
    sediment_density=2650.0,
    kinematic_viscosity=1.3e-6,
    gravity=9.81,

    # combined-flow direction
    phi_deg=90.0,

    # diagnostic friction factors for stress mapping (you can tune)
    f_c=0.004,
    f_w=0.02,
)


# ============================================================
# DIAGNOSTIC: from tau -> roughness -> velocity profile
# Compare ks_skin+eta_b vs ks_c vs k_a for several depths.
# Two cases:
#   (A) Fixed tau  (SedTRAILS-style: tau from hydro -> u* fixed)
#   (B) Fixed Ubar (your request: unchanged depth-averaged velocity)
# ============================================================
def fmt(x):
    """Short scientific formatting for annotations."""
    if x == 0:
        return "0"
    if abs(x) < 1e-2 or abs(x) > 1e2:
        return f"{x:.1e}"
    return f"{x:.2f}"

def log_profile_u(z, ustar, ks, kappa=0.4):
    """
    Simple log-law velocity profile using equivalent sand roughness ks.
    Uses z0 = ks/30.
    u(z) = u*/kappa * ln(z/z0)
    """
    z0 = ks / 30.0
    z = np.asarray(z)
    # avoid invalid region (roughness sublayer / below z0)
    z_eff = np.maximum(z, 1.05 * z0)
    return (ustar / kappa) * np.log(z_eff / z0)

def ustar_for_depthavg_Ubar(Ubar, h, ks, kappa=0.4):
    """
    Given depth-averaged velocity Ubar and roughness ks, estimate u* assuming a log-profile.

    If u(z) = u*/kappa ln(z/z0) with z0=ks/30, then depth-mean over z0..h gives approx:
      Ubar ≈ (u*/kappa) * ( ln(h/z0) - 1 )
    for h >> z0.

    => u* ≈ kappa * Ubar / ( ln(h/z0) - 1 )
    """
    z0 = ks / 30.0
    denom = (np.log(h / z0) - 1.0)
    # guard against tiny denom (very rough / shallow)
    denom = np.maximum(denom, 1e-6)
    return kappa * Ubar / denom

def compute_ks_variants(d50, h, Uc, Uw, phi_deg, rho, rho_s, g):
    """
    Compute ks choices:
      ks1 = ks_skin + eta_b
      ks2 = ks_c
      ks3 = k_a
    using your existing functions.
    """
    # grain properties
    props = compute_grain_properties(
        grain_diameter=d50,
        gravity=g,
        sediment_density=rho_s,
        water_density=rho,
        kinematic_viscosity=1.3e-6,
    )
    theta_cr = props["critical_shields"]

    # stresses (same as your parameter-space logic)
    fc = 0.004
    fw = 0.02
    tau_c = 0.5 * rho * fc * Uc**2
    tau_w = 0.5 * rho * fw * Uw**2
    phi = np.deg2rad(np.minimum(phi_deg, 360.0 - phi_deg))
    tau_max = np.sqrt(tau_c**2 + tau_w**2 + 2.0 * tau_c * tau_w * np.cos(phi))

    theta_max = tau_max / ((rho_s - rho) * g * d50)

    # roughness ingredients
    ks_skin = calculate_skin_roughness(method="soulsby_d50", d50=d50)

    eta_b = calculate_equilibrium_bedform_height(
        theta_max, theta_cr, d50, h
    )

    # van Rijn ks_c and apparent ka (functions expect array-like; scalars work too)
    s = rho_s / rho
    ks_c, ks_r, ks_mr, ks_d = calculate_current_related_bed_roughness_vanrijn2007(
        flow_velocity_magnitude=Uc,
        near_bed_peak_orbital_velocity=Uw,
        grain_diameter=d50,
        water_depth=h,
        relative_density=s,
        gravity=g,
    )

    ka, ka_ksc, gamma = calculate_apparent_bed_roughness_vanrijn2007(
        ks_c=ks_c,
        near_bed_peak_orbital_velocity=Uw,
        flow_velocity_magnitude=Uc,
        phi_deg=phi_deg,
    )

    ks_skin_plus_eta = ks_skin + eta_b

    return {
        "tau_max": float(tau_max),
        "theta_max": float(theta_max),
        "theta_cr": float(theta_cr),
        "ks_skin_plus_eta": float(ks_skin_plus_eta),
        "ks_c": float(ks_c),
        "k_a": float(ka),
    }


#----------------------------------------------------------------------------------------------------------


# ----------------------------
# Choose demonstration scenarios
# ----------------------------
rho = 1025.0
rho_s = 2650.0
g = 9.81

# Use one representative forcing state (you can change these easily)
Uc_demo = 1.0     # depth-avg current [m/s]
Uw_demo = 1.0     # near-bed orbital [m/s]
phi_demo = 0.0   # wave-current angle [deg]

# Compare across depths; keep one d50 (change to 1e-4 or 2e-3 to see grain-size effect)
d50_demo = 3e-4   # 300 micron
depths_demo = [0.5, 5.0, 20.0]

# For "fixed Ubar" case, set the depth-averaged velocity we want to hold constant
Ubar_target = 1.0

# Heights to highlight in the profile (arbitrary z locations)
z_rel_marks = [0.05, 0.2]  # z/h = 5% and 20%

# ----------------------------
# Plot
# ----------------------------
fig, axes = plt.subplots(
    nrows=len(depths_demo),
    ncols=2,
    figsize=(12, 3.6 * len(depths_demo)),
    constrained_layout=True
)

for r, h in enumerate(depths_demo):
    vals = compute_ks_variants(
        d50=d50_demo,
        h=h,
        Uc=Uc_demo,
        Uw=Uw_demo,
        phi_deg=phi_demo,
        rho=rho, rho_s=rho_s, g=g,
    )

    theta_max = vals["theta_max"]
    theta_cr  = vals["theta_cr"]
    mobility  = theta_max / theta_cr

    ks_c = vals["ks_c"]
    k_a  = vals["k_a"]
    ka_over_kc = k_a / ks_c if ks_c > 0 else np.nan

    tau_max = vals["tau_max"]
    ks1 = vals["ks_skin_plus_eta"]
    ks2 = vals["ks_c"]
    ks3 = vals["k_a"]

    # z grid (avoid z=0 on log)
    z = np.linspace(1e-4 * h, h, 300)

    # -----------------------------------
    # (A) FIXED TAU (u* fixed)
    # -----------------------------------
    ustar_A = np.sqrt(tau_max / rho)

    u1A = log_profile_u(z, ustar_A, ks1)
    u2A = log_profile_u(z, ustar_A, ks2)
    u3A = log_profile_u(z, ustar_A, ks3)

    axA = axes[r, 0]
    axA.plot(u1A, z / h, label=r"$k_s=k_{skin}+\eta_b$")
    axA.plot(u2A, z / h, label=r"$k_s=k_{s,c}$", linestyle="--")
    axA.plot(u3A, z / h, label=r"$k_s=k_a$", linestyle="-.")

    # # mark z/h points
    # for zr in z_rel_marks:
    #     zmark = zr * h
    #     axA.scatter(
    #         [log_profile_u(zmark, ustar_A, ks1), log_profile_u(zmark, ustar_A, ks2), log_profile_u(zmark, ustar_A, ks3)],
    #         [zr, zr, zr],
    #         s=10
    #     )

    axA.set_title(
        rf"Fixed $\tau$ (u* fixed), $h={h}$ m" + "\n" +
        rf"$\theta_{{\max}}={fmt(theta_max)}$, "
        rf"$\theta_c={fmt(theta_cr)}$, "
        rf"$\theta_{{\max}}/\theta_c={fmt(mobility)}$",
        fontsize=10
    )    
    axA.set_xlabel("u(z) [m/s]")
    axA.set_ylabel("z/h [-]")
    axA.grid(True, which="both", alpha=0.3)


    # -----------------------------------
    # (B) FIXED Ubar (depth-mean fixed)
    # -----------------------------------
    ustar1B = ustar_for_depthavg_Ubar(Ubar_target, h, ks1)
    ustar2B = ustar_for_depthavg_Ubar(Ubar_target, h, ks2)
    ustar3B = ustar_for_depthavg_Ubar(Ubar_target, h, ks3)

    u1B = log_profile_u(z, ustar1B, ks1)
    u2B = log_profile_u(z, ustar2B, ks2)
    u3B = log_profile_u(z, ustar3B, ks3)

    axB = axes[r, 1]
    axB.plot(u1B, z / h, label=rf"$k_s=k_{{skin}}+\eta_b$, $u_*={ustar1B:.3f}$")
    axB.plot(u2B, z / h, label=rf"$k_s=k_{{s,c}}$, $u_*={ustar2B:.3f}$", linestyle="--")
    axB.plot(u3B, z / h, label=rf"$k_s=k_a$, $u_*={ustar3B:.3f}$", linestyle="-.")

    # for zr in z_rel_marks:
    #     zmark = zr * h
    #     axB.scatter(
    #         [log_profile_u(zmark, ustar1B, ks1), log_profile_u(zmark, ustar2B, ks2), log_profile_u(zmark, ustar3B, ks3)],
    #         [zr, zr, zr],
    #         s=10
    #     )

    axB.set_title(
        rf"Fixed $\bar{{U}}={Ubar_target}$ m/s, $h={h}$ m" + "\n" +
        rf"$\theta_{{\max}}/\theta_c={fmt(mobility)}$, "
        rf"$k_a/k_{{s,c}}={fmt(ka_over_kc)}$",
        fontsize=10
    )
    axB.set_xlabel("u(z) [m/s]")
    axB.set_ylabel("z/h [-]")
    axB.grid(True, which="both", alpha=0.3)

    # Legend only on first row to reduce clutter
    if r == 0:
        axA.legend(loc="lower right", fontsize=9)
        axB.legend(loc="lower right", fontsize=9)

fig.suptitle(
    rf"Effect of roughness choice on log-law velocity profile (d50={d50_demo*1e6:.0f} μm, Uc={Uc_demo}, Uw={Uw_demo}, φ={phi_demo}°)",
    fontsize=12
)

outname = f"diagnostic_velocity_profile_roughness_switch_d50_{d50_demo*1e6:.0f}um_uc_{Uc_demo}_uw_{Uw_demo}_phi_{phi_demo:.0f}.png"
fig.savefig(outname, dpi=300, bbox_inches="tight")
print("Saved:", outname)
plt.show()
