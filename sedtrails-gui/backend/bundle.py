"""Results netCDF -> viewer bundle importer, registry, mesh, classification.

Extracted verbatim from sedtrails-viewer/sedtrails_viewer.py (Phase 0 split).
"""

import hashlib
import json
import math
import os
import re
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from .settings import (APP_DIR, BG_DZ_PCTL, BG_ELEV_PCTL, CMAP_NPY,
                       FORMAT_VERSION, LOAD_WHOLE_LIMIT, PAD_FRAC, QMAX,
                       REFERENCE_DATE_FALLBACK, SENTINEL, SLAB_PARTICLES,
                       SLAB_STEPS, VIEWER_APP_DIR)
from .util import NC_LOCK, _log, _nanfilled, _read_json, _write_json

_registry_lock = threading.Lock()
_polygons_lock = threading.Lock()

# bumped when only meta-derived fields change (reference-date time-of-day fix);
# lets cached bundles refresh meta.json cheaply without re-writing the bins
META_VERSION = 2

# bumped when the model-layer mesh construction changes; cached bundles then
# rebuild mesh.bin on their next load (v2: cell cutoff vs 8th neighbor)
MESH_VERSION = 2


def _registry_path():
    return APP_DIR / "registry.json"


def _polygons_path():
    return APP_DIR / "polygons.json"


def load_registry():
    reg = _read_json(_registry_path(), {"simulations": []})
    # Merge bundles imported with the standalone viewer (read-only; entries
    # forgotten in the GUI are remembered in reg["hidden"]).
    old = _read_json(VIEWER_APP_DIR / "registry.json", None)
    if old:
        known = {s["id"] for s in reg["simulations"]}
        hidden = set(reg.get("hidden", []))
        reg["simulations"] += [s for s in old.get("simulations", [])
                               if s["id"] not in known and s["id"] not in hidden]
    return reg


def save_registry(reg):
    _write_json(_registry_path(), reg)


def registry_add(entry):
    with _registry_lock:
        reg = load_registry()
        reg["simulations"] = [s for s in reg["simulations"] if s["id"] != entry["id"]]
        reg["simulations"].insert(0, entry)
        save_registry(reg)


def registry_find(bundle_id):
    for s in load_registry()["simulations"]:
        if s["id"] == bundle_id:
            return s
    return None


# ── netCDF track reading (both layouts) ───────────────────────────────────────
def open_tracks(nc_path):
    """Open a SedTRAILS results netCDF; return (ds, info dict).

    Layout handling mirrors westerschelde/scripts/evaluate_source_sink.py
    load_tracks(): time-major files have x.dims[0] in (n_timesteps, time, obs);
    legacy files are particle-major with a 2D time variable.
    """
    import netCDF4 as nc

    with NC_LOCK:
        return _open_tracks_locked(nc.Dataset(nc_path, "r"))


def _open_tracks_locked(ds):
    x = ds.variables["x"]
    time_major = x.dimensions[0] in ("n_timesteps", "time", "obs")
    T_raw = x.shape[0] if time_major else x.shape[1]
    N = x.shape[1] if time_major else x.shape[0]

    t_var = ds.variables["time"]
    t = _nanfilled(t_var[:] if t_var.ndim == 1 else t_var[0, :]).ravel()
    finite = np.isfinite(t)
    T = int(np.argmin(finite)) if not finite.all() else len(t)  # longest finite prefix
    T = min(T if T > 0 else len(t), T_raw)
    t = t[:T]

    units = getattr(t_var, "units", "") or ""
    # keep the time-of-day of the reference date (e.g. "seconds since
    # 2025-08-03 21:40:00") — dropping it shifted the particle timeline vs the
    # forcing timeline by up to a day
    m = re.search(r"since\s+(\d{4}-\d{2}-\d{2})[T ]?(\d{2}:\d{2}:\d{2})?", units)
    reference_date = m.group(1) if m else REFERENCE_DATE_FALLBACK
    if m and m.group(2) and m.group(2) != "00:00:00":
        reference_date += " " + m.group(2)
    time_days = ((t - 0.0) / 86400.0).tolist()  # days since reference_date

    has_burial = "burial_depth" in ds.variables
    has_flags = "status_mobile" in ds.variables
    return ds, {
        "time_major": time_major,
        "T": T,
        "N": N,
        "reference_date": reference_date,
        "time_days": time_days,
        "has_burial": has_burial,
        "has_flags": has_flags,
    }


def iter_slabs(ds, info, names):
    """Yield (axis, i0, i1, {name: float array (t, n) time-major-normalized}).

    axis is 'time' (i0:i1 timesteps, all particles) or 'particle'
    (all timesteps, i0:i1 particles). Arrays are always (n_timesteps, n_particles).
    """
    T, N, time_major = info["T"], info["N"], info["time_major"]
    variables = {name: ds.variables[name] for name in names}
    if time_major:
        for i0 in range(0, T, SLAB_STEPS):
            i1 = min(i0 + SLAB_STEPS, T)
            with NC_LOCK:
                arrs = {n: _nanfilled(v[i0:i1, :]) for n, v in variables.items()}
            yield "time", i0, i1, arrs
    else:
        per_var_bytes = T * N * 8
        step = N if per_var_bytes < LOAD_WHOLE_LIMIT else SLAB_PARTICLES
        for i0 in range(0, N, step):
            i1 = min(i0 + step, N)
            with NC_LOCK:
                arrs = {n: _nanfilled(v[i0:i1, :T]).T for n, v in variables.items()}
            yield "particle", i0, i1, arrs


# ── Importer ───────────────────────────────────────────────────────────────────
def _quantize_pair(x, y, quant):
    qx = np.round((x - quant["xmin"]) / quant["sx"])
    qy = np.round((y - quant["ymin"]) / quant["sy"])
    bad = (~np.isfinite(qx) | ~np.isfinite(qy)
           | (qx < 0) | (qx > QMAX) | (qy < 0) | (qy > QMAX))
    qx = np.where(bad, SENTINEL, qx).astype("<u2")
    qy = np.where(bad, SENTINEL, qy).astype("<u2")
    return qx, qy


def bundle_dir_for(nc_path):
    p = Path(nc_path)
    return p.parent / (p.stem + ".viewer")


def bundle_id_for(nc_path):
    p = Path(nc_path).resolve()
    h = hashlib.sha1(str(p).lower().encode("utf-8")).hexdigest()[:10]
    label = p.parent.name if p.stem == "sedtrails_results" else p.stem
    return f"{label}-{h}"


def import_run(nc_path, forcing_path=None, name=None, job=None, grid_path=None):
    """Convert a results netCDF to a viewer bundle (cached). Returns meta."""
    def prog(frac, msg):
        if job is not None:
            job["progress"] = round(float(frac), 3)
            job["message"] = msg
        _log(f"import: {msg}")

    nc_path = str(Path(nc_path).resolve())
    bdir = bundle_dir_for(nc_path)
    bid = bundle_id_for(nc_path)
    meta_path = bdir / "meta.json"
    src_stat = os.stat(nc_path)
    src_sig = [int(src_stat.st_size), int(src_stat.st_mtime)]

    meta = _read_json(meta_path, None)
    fresh = (meta is not None and meta.get("format_version") == FORMAT_VERSION
             and meta.get("source_signature") == src_sig
             and (bdir / "positions.bin").exists())

    if not fresh:
        ds, info = open_tracks(nc_path)
        try:
            T, N = info["T"], info["N"]
            prog(0.02, f"scanning extent ({T} steps x {N} particles)")

            names = ["x", "y"] + (["burial_depth"] if info["has_burial"] else [])
            xmin = ymin = np.inf
            xmax = ymax = -np.inf
            burial_max = 0.0
            if info["time_major"]:
                n_slabs = math.ceil(T / SLAB_STEPS)
            else:
                p_step = N if T * N * 8 < LOAD_WHOLE_LIMIT else SLAB_PARTICLES
                n_slabs = math.ceil(N / p_step)
            for k, (_, _, _, arrs) in enumerate(iter_slabs(ds, info, names)):
                with np.errstate(all="ignore"):
                    xmin = min(xmin, np.nanmin(arrs["x"])) if np.isfinite(arrs["x"]).any() else xmin
                    xmax = max(xmax, np.nanmax(arrs["x"])) if np.isfinite(arrs["x"]).any() else xmax
                    ymin = min(ymin, np.nanmin(arrs["y"])) if np.isfinite(arrs["y"]).any() else ymin
                    ymax = max(ymax, np.nanmax(arrs["y"])) if np.isfinite(arrs["y"]).any() else ymax
                    if "burial_depth" in arrs and np.isfinite(arrs["burial_depth"]).any():
                        burial_max = max(burial_max, float(np.nanmax(arrs["burial_depth"])))
                del arrs
                prog(0.02 + 0.28 * (k + 1) / n_slabs, f"scanning extent (slab {k + 1})")

            if not (np.isfinite(xmin) and np.isfinite(ymin)):
                raise ValueError("no finite particle positions found")
            pad_x = max((xmax - xmin) * PAD_FRAC, 1.0)
            pad_y = max((ymax - ymin) * PAD_FRAC, 1.0)
            xmin, xmax = xmin - pad_x, xmax + pad_x
            ymin, ymax = ymin - pad_y, ymax + pad_y
            quant = {"xmin": xmin, "ymin": ymin,
                     "sx": (xmax - xmin) / QMAX, "sy": (ymax - ymin) / QMAX,
                     "sentinel": SENTINEL}
            # Floor avoids amplifying numerical noise into the color range on
            # runs where burial is physically (near-)zero.
            burial_max = max(burial_max, 0.01)

            bdir.mkdir(parents=True, exist_ok=True)
            pos_mm = np.memmap(bdir / "positions.bin", dtype="<u2", mode="w+", shape=(T, N, 2))
            bur_mm = (np.memmap(bdir / "burial.bin", dtype="<u2", mode="w+", shape=(T, N))
                      if info["has_burial"] else None)

            for k, (axis, i0, i1, arrs) in enumerate(iter_slabs(ds, info, names)):
                qx, qy = _quantize_pair(arrs["x"], arrs["y"], quant)
                if axis == "time":
                    pos_mm[i0:i1, :, 0] = qx
                    pos_mm[i0:i1, :, 1] = qy
                else:
                    pos_mm[:, i0:i1, 0] = qx
                    pos_mm[:, i0:i1, 1] = qy
                if bur_mm is not None:
                    b = arrs["burial_depth"]
                    qb = np.round(np.clip(b, 0.0, burial_max) / burial_max * QMAX)
                    qb = np.where(np.isfinite(qb), qb, SENTINEL).astype("<u2")
                    if axis == "time":
                        bur_mm[i0:i1, :] = qb
                    else:
                        bur_mm[:, i0:i1] = qb
                    del b, qb
                del arrs, qx, qy
                prog(0.30 + 0.55 * (k + 1) / n_slabs, f"writing bundle (slab {k + 1})")

            pos_mm.flush()
            del pos_mm
            if bur_mm is not None:
                bur_mm.flush()
                del bur_mm

            meta = {
                "format_version": FORMAT_VERSION,
                "meta_version": META_VERSION,
                "id": bid,
                "run_name": name or Path(nc_path).parent.name,
                "source_nc": nc_path,
                "source_signature": src_sig,
                "n_particles": N,
                "n_timesteps": T,
                "reference_date": info["reference_date"],
                "time_days": info["time_days"],
                "quant": quant,
                "burial_max": burial_max,
                "has_burial": info["has_burial"],
                "extent": [xmin, xmax, ymin, ymax],
                "backgrounds": [],
                "files": {"positions": "positions.bin",
                          **({"burial": "burial.bin"} if info["has_burial"] else {})},
            }
            _write_json(meta_path, meta)
        finally:
            with NC_LOCK:
                ds.close()
    else:
        prog(0.85, "bundle up to date (cached)")
        if name:
            meta["run_name"] = name
        if meta.get("meta_version") != META_VERSION:
            # meta-only refresh: time handling changed but the bins are fine
            ds, info = open_tracks(nc_path)
            with NC_LOCK:
                ds.close()
            meta["reference_date"] = info["reference_date"]
            meta["time_days"] = info["time_days"]
            meta["meta_version"] = META_VERSION

    _ensure_flags(nc_path, bdir, meta, prog)
    if forcing_path:
        meta["forcing_nc"] = str(Path(forcing_path).resolve())
    if grid_path:
        meta["grid_nc"] = str(Path(grid_path).resolve())
    prog(0.86, "building model-layer mesh")
    make_mesh(bdir, meta, meta.get("forcing_nc"), meta.get("grid_nc"))
    _write_json(meta_path, meta)

    registry_add({
        "id": bid, "name": meta["run_name"], "nc_path": nc_path,
        "bundle_dir": str(bdir), "forcing_path": meta.get("forcing_nc"),
        "grid_path": meta.get("grid_nc"),
        "n_particles": meta["n_particles"], "n_timesteps": meta["n_timesteps"],
        "imported_at": time.strftime("%Y-%m-%d %H:%M"),
    })
    prog(1.0, "done")

    # Verification print: dequantized first/last frame round-trip error bound.
    q = meta["quant"]
    _log(f"import: quantization step sx={q['sx']:.3f} m, sy={q['sy']:.3f} m "
         f"(max round-trip error {max(q['sx'], q['sy']) / 2:.3f} m)")
    return meta


def _ensure_flags(nc_path, bdir, meta, prog):
    """Extract per-particle status flags (flags.bin, uint8 (T,N)) from the
    results netCDF: 1 = mobile, 0 = immobile, 255 = unknown. Streamed by the
    viewer on demand for the 'fade immobile particles' option. Runs as a
    cheap separate pass so existing bundles are upgraded without re-import."""
    if meta.get("has_flags") and (bdir / "flags.bin").exists():
        return
    ds, info = open_tracks(nc_path)
    try:
        if not info["has_flags"]:
            meta["has_flags"] = False
            return
        T, N = meta["n_timesteps"], meta["n_particles"]
        prog(0.85, "extracting particle status flags")
        flg_mm = np.memmap(bdir / "flags.bin", dtype="u1", mode="w+", shape=(T, N))
        for axis, i0, i1, arrs in iter_slabs(ds, info, ["status_mobile"]):
            f = arrs["status_mobile"]
            qf = np.full(f.shape, 255, np.uint8)
            fin = np.isfinite(f) & (f < 254)
            qf[fin] = f[fin] > 0.5
            if axis == "time":
                flg_mm[i0:i1, :] = qf
            else:
                flg_mm[:, i0:i1] = qf
            del arrs, f, qf
        flg_mm.flush()
        del flg_mm
        meta["has_flags"] = True
        meta["files"]["flags"] = "flags.bin"
        _log("flags: extracted status_mobile -> flags.bin")
    except Exception as e:
        traceback.print_exc()
        _log(f"flags: FAILED ({e})")
    finally:
        with NC_LOCK:
            ds.close()


# ── Model-layer mesh (rendered live in the viewer, like the analysis scripts) ──
def _cmap_triplets(name_or_npy, n=256):
    """256 RGB triplets (0-255) for a matplotlib colormap or a .npy colormap."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import colormaps
    from matplotlib.colors import ListedColormap

    if name_or_npy and os.path.exists(str(name_or_npy)):
        cmap = ListedColormap(np.load(name_or_npy))
    else:
        cmap = colormaps[name_or_npy]
    cols = cmap(np.linspace(0.0, 1.0, n))[:, :3]
    return (cols * 255).round().astype(int).tolist()


def _cells_geometry(grid_path, forcing_path):
    """Exact model cells: faces from the grid/net file, values mapped from the
    forcing's cell-centre nodes via cKDTree — the same technique as the
    analysis scripts' PolyCollection. Values are FLAT per face (each face's
    corners are emitted as dedicated vertices carrying the face value)."""
    import netCDF4 as nc
    from scipy.spatial import cKDTree

    with NC_LOCK, nc.Dataset(grid_path, "r") as gm:
        node_x = _nanfilled(gm.variables["mesh2d_node_x"][:])
        node_y = _nanfilled(gm.variables["mesh2d_node_y"][:])
        face_x = _nanfilled(gm.variables["mesh2d_face_x"][:])
        face_y = _nanfilled(gm.variables["mesh2d_face_y"][:])
        fn_var = gm.variables["mesh2d_face_nodes"]
        fill = int(getattr(fn_var, "_FillValue", -999))
        fn = np.ma.filled(fn_var[:], fill).astype(np.int64)
    with NC_LOCK, nc.Dataset(forcing_path, "r") as ds:
        xn = _nanfilled(ds.variables["net_xcc"][:])
        yn = _nanfilled(ds.variables["net_ycc"][:])
        bed0 = _nanfilled(ds.variables["bedlevel"][0, :])
        nt = ds.variables["bedlevel"].shape[0]
        bed1 = _nanfilled(ds.variables["bedlevel"][nt - 1, :]) if nt > 1 else bed0

    dist, idx = cKDTree(np.column_stack([xn, yn])).query(np.column_stack([face_x, face_y]))
    face_bed = np.where(dist < 500.0, bed0[idx], np.nan)
    face_dz = np.where(dist < 500.0, bed1[idx] - bed0[idx], np.nan)

    max_nodes = fn.shape[1]
    counts = (fn != fill).sum(axis=1)
    keep = counts >= 3
    fn = fn[keep] - 1                    # 1-based -> 0-based
    counts = counts[keep]
    face_bed = face_bed[keep]
    face_dz = face_dz[keep]

    n_verts = int(counts.sum())
    n_tris = int((counts - 2).sum())
    xy = np.empty((n_verts, 2), "<f4")
    bed_v = np.empty(n_verts, "<f4")
    dz_v = np.empty(n_verts, "<f4")
    tris = np.empty((n_tris, 3), "<u4")

    vo = 0
    to = 0
    for c in range(3, max_nodes + 1):    # vectorized per corner-count group
        sel = counts == c
        m = int(sel.sum())
        if m == 0:
            continue
        nodes = fn[sel][:, :c]                     # (m, c)
        base = vo + np.arange(m, dtype=np.int64) * c
        xy[vo:vo + m*c, 0] = node_x[nodes].reshape(-1)
        xy[vo:vo + m*c, 1] = node_y[nodes].reshape(-1)
        fb = np.where(np.isfinite(face_bed[sel]), face_bed[sel], -9999.0)
        fd = np.where(np.isfinite(face_dz[sel]), face_dz[sel], -9999.0)
        bed_v[vo:vo + m*c] = np.repeat(fb, c)
        dz_v[vo:vo + m*c] = np.repeat(fd, c)
        for k in range(c - 2):                     # fan triangulation
            tris[to + k*m:to + (k+1)*m, 0] = base
            tris[to + k*m:to + (k+1)*m, 1] = base + k + 1
            tris[to + k*m:to + (k+1)*m, 2] = base + k + 2
        vo += m * c
        to += m * (c - 2)
    return xy, bed_v, dz_v, tris, face_bed, face_dz


def _voronoi_cells_geometry(forcing_path):
    """Approximate model cells from the forcing file ALONE: the Voronoi
    tessellation of the cell centres (net_xcc/net_ycc). Exact topology needs
    the grid/net file, but this already gives crisp, flat-colored cells
    instead of a smooth interpolation — one input file suffices."""
    import netCDF4 as nc
    from scipy.spatial import Voronoi, cKDTree

    with NC_LOCK, nc.Dataset(forcing_path, "r") as ds:
        xn = _nanfilled(ds.variables["net_xcc"][:])
        yn = _nanfilled(ds.variables["net_ycc"][:])
        bed0 = _nanfilled(ds.variables["bedlevel"][0, :])
        nt = ds.variables["bedlevel"].shape[0]
        bed1 = _nanfilled(ds.variables["bedlevel"][nt - 1, :]) if nt > 1 else bed0

    ok = np.isfinite(xn) & np.isfinite(yn)
    pts = np.column_stack([xn[ok], yn[ok]])
    v_bed = bed0[ok]
    v_dz = (bed1 - bed0)[ok]
    n_real = len(pts)

    # guard ring closes off the outer cells; per-point cutoff (3x the local
    # spacing) trims the remaining slivers along the concave coastline
    x0, x1 = pts[:, 0].min(), pts[:, 0].max()
    y0, y1 = pts[:, 1].min(), pts[:, 1].max()
    mx, my = (x1 - x0) * 0.2 + 1.0, (y1 - y0) * 0.2 + 1.0
    ts = np.linspace(0.0, 1.0, 100, endpoint=False)
    ring = np.concatenate([
        np.column_stack([x0 - mx + ts * (x1 - x0 + 2*mx), np.full(100, y0 - my)]),
        np.column_stack([x0 - mx + ts * (x1 - x0 + 2*mx), np.full(100, y1 + my)]),
        np.column_stack([np.full(100, x0 - mx), y0 - my + ts * (y1 - y0 + 2*my)]),
        np.column_stack([np.full(100, x1 + mx), y0 - my + ts * (y1 - y0 + 2*my)]),
    ])
    vor = Voronoi(np.vstack([pts, ring]))
    # cutoff radius from the 8th neighbor, not the 1st: on strongly anisotropic
    # grids (alongshore spacing >> cross-shore) the 1st neighbor is the short
    # axis and whole rows of valid cells would be cut (sandengine grid)
    tree = cKDTree(pts)
    k8 = min(9, n_real)
    nn = tree.query(pts, k=k8)[0][:, k8 - 1]
    r_cut2 = (3.0 * np.maximum(nn, 1e-3)) ** 2

    verts_l, bed_l, dz_l, tris_l = [], [], [], []
    vo = 0
    vv = vor.vertices
    for i in range(n_real):
        region = vor.regions[vor.point_region[i]]
        c = len(region)
        if c < 3 or -1 in region:
            continue
        cell = vv[region]
        d2 = (cell[:, 0] - pts[i, 0])**2 + (cell[:, 1] - pts[i, 1])**2
        if d2.max() > r_cut2[i]:
            continue
        verts_l.append(cell)
        bed_l.append(np.full(c, v_bed[i] if np.isfinite(v_bed[i]) else -9999.0))
        dz_l.append(np.full(c, v_dz[i] if np.isfinite(v_dz[i]) else -9999.0))
        fan = np.empty((c - 2, 3), np.int64)
        fan[:, 0] = vo
        fan[:, 1] = vo + np.arange(1, c - 1)
        fan[:, 2] = vo + np.arange(2, c)
        tris_l.append(fan)
        vo += c

    xy = np.concatenate(verts_l).astype("<f4")
    bed_v = np.concatenate(bed_l).astype("<f4")
    dz_v = np.concatenate(dz_l).astype("<f4")
    tris = np.concatenate(tris_l).astype("<u4")
    return xy, bed_v, dz_v, tris, v_bed, v_dz


def make_mesh(bdir, meta, forcing_path, grid_path=None):
    """Write mesh.bin for the model layer (rendered live on the GPU).

    With a grid/net file: the TRUE model cells, flat-colored per face — the
    same look as the analysis scripts. With only the forcing file: Voronoi
    cells of the cell centres (flat-colored approximation, single file)."""
    if not (forcing_path and os.path.exists(forcing_path)):
        return
    style = "cells" if (grid_path and os.path.exists(grid_path)) else "voronoi"
    if meta.get("mesh") and meta["mesh"].get("style") == style \
            and meta["mesh"].get("mesh_version") == MESH_VERSION \
            and (bdir / "mesh.bin").exists():
        return
    try:
        if style == "cells":
            xy, bed_v, dz_v, tris, fb, fd = _cells_geometry(grid_path, forcing_path)
            clim_src, dz_src = fb, fd
        else:
            xy, bed_v, dz_v, tris, fb, fd = _voronoi_cells_geometry(forcing_path)
            clim_src, dz_src = fb, fd

        q = meta["quant"]
        xy[:, 0] -= q["xmin"]
        xy[:, 1] -= q["ymin"]
        fin_dz = dz_src[np.isfinite(dz_src)]
        has_dz = bool(fin_dz.size and np.max(np.abs(fin_dz)) > 1e-6)
        clim_bed = [float(v) for v in np.nanpercentile(clim_src, BG_ELEV_PCTL)]
        dz_lim = float(np.nanpercentile(np.abs(fin_dz), BG_DZ_PCTL)) if has_dz else 1.0

        with open(bdir / "mesh.bin", "wb") as f:
            f.write(np.ascontiguousarray(xy).tobytes())
            f.write(np.ascontiguousarray(bed_v).tobytes())
            f.write(np.ascontiguousarray(dz_v).tobytes())
            f.write(np.ascontiguousarray(tris).tobytes())
        meta["mesh"] = {
            "file": "mesh.bin",
            "style": style,
            "mesh_version": MESH_VERSION,
            "n_nodes": int(len(bed_v)),
            "n_tris": int(len(tris)),
            "clim_bed": clim_bed,
            "clim_dz": [-dz_lim, dz_lim],
            "has_dz": has_dz,
            "cmap_bed": _cmap_triplets(CMAP_NPY if CMAP_NPY else "gist_earth"),
            "cmap_dz": _cmap_triplets("RdBu_r"),
        }
        _log(f"mesh layer ({style}): {len(bed_v)} vertices, {len(tris)} triangles"
             + ("" if has_dz else " (static bed — no bed-change layer)"))
    except Exception as e:
        traceback.print_exc()
        _log(f"mesh layer: FAILED ({e})")


# ── Polygon classification ─────────────────────────────────────────────────────
def _poly_paths(polygons):
    from matplotlib.path import Path as MplPath
    out = []
    for p in polygons:
        verts = np.asarray(p["verts"], dtype=np.float64)
        if verts.ndim != 2 or len(verts) < 3:
            continue
        out.append({
            "key": p["key"],
            "geom_key": hashlib.sha1(b"v3" + verts.round(2).tobytes()).hexdigest()[:16],
            "path": MplPath(verts),
            "bbox": (verts[:, 0].min(), verts[:, 0].max(),
                     verts[:, 1].min(), verts[:, 1].max()),
        })
    return out


def _contains(poly, x, y, idx):
    """Indices (subset of idx) inside poly, with bbox prefilter. x/y are metres."""
    bx0, bx1, by0, by1 = poly["bbox"]
    xi = x[idx]
    yi = y[idx]
    cand = (xi >= bx0) & (xi <= bx1) & (yi >= by0) & (yi <= by1)
    if not cand.any():
        return idx[:0]
    sub = idx[cand]
    inside = poly["path"].contains_points(np.column_stack([x[sub], y[sub]]))
    return sub[inside]


_classify_progress = {}   # bundle_id -> {"t": step, "T": total, "n": n_polygons}


def _classify_missing(bdir, meta, missing, bundle_id=""):
    """One streaming pass computing per-polygon particle flags for every
    polygon in `missing` simultaneously. Cached per polygon GEOMETRY, so a
    newly drawn polygon costs one polygon — not a full re-classification.

    Per polygon and particle: at0 (inside at the first position), at_end
    (inside at the last finite position), visited (ever inside, with the
    re-entry rule: a particle starting inside only counts after leaving
    once), first (timestep of first counted entry).

    The hot path stays in quantized uint16 space: a per-polygon quantized
    bounding box selects candidates with cheap integer compares, and only
    those few positions are dequantized for the exact point-in-polygon test.
    Being outside the bbox also proves 'left the start polygon' for free.
    """
    T, N = meta["n_timesteps"], meta["n_particles"]
    q = meta["quant"]
    pos = np.memmap(bdir / "positions.bin", dtype="<u2", mode="r", shape=(T, N, 2))

    def qbox(poly):
        bx0, bx1, by0, by1 = poly["bbox"]
        c = lambda v, vmin, s: np.uint16(min(max((v - vmin) / s, 0), QMAX))
        return (c(bx0, q["xmin"], q["sx"]), c(bx1, q["xmin"], q["sx"]) + 1,
                c(by0, q["ymin"], q["sy"]), c(by1, q["ymin"], q["sy"]) + 1)

    st = [{"at0": np.zeros(N, bool), "visited": np.zeros(N, bool),
           "left": np.zeros(N, bool), "first": np.full(N, SENTINEL, np.uint16),
           "qbox": qbox(p)} for p in missing]
    last_qx = np.full(N, SENTINEL, np.uint16)
    last_qy = np.full(N, SENTINEL, np.uint16)

    t0 = time.time()
    for t in range(T):
        _classify_progress[bundle_id] = {"t": t, "T": T, "n": len(missing)}
        frame = np.asarray(pos[t])
        qx = frame[:, 0]
        qy = frame[:, 1]
        valid = qx != SENTINEL
        if not valid.any():
            continue
        last_qx[valid] = qx[valid]
        last_qy[valid] = qy[valid]

        for poly, s in zip(missing, st):
            bx0, bx1, by0, by1 = s["qbox"]
            inbox = (qx >= bx0) & (qx < bx1) & (qy >= by0) & (qy < by1)
            track = s["at0"] & ~s["left"]
            if t > 0:
                # outside the bbox ⇒ certainly outside the polygon ⇒ has left
                s["left"][track & valid & ~inbox] = True
                track = s["at0"] & ~s["left"]
            cand = np.flatnonzero(inbox & valid & (~s["visited"] | track))
            if cand.size == 0:
                continue
            x = qx[cand].astype(np.float64) * q["sx"] + q["xmin"]
            y = qy[cand].astype(np.float64) * q["sy"] + q["ymin"]
            inside_l = poly["path"].contains_points(np.column_stack([x, y]))
            inside_idx = cand[inside_l]
            if t == 0:
                s["at0"][inside_idx] = True
            else:
                visit = inside_idx[(~s["at0"][inside_idx]) | s["left"][inside_idx]]
                fresh = visit[~s["visited"][visit]]
                s["visited"][fresh] = True
                s["first"][fresh] = t
                # inside the bbox but outside the polygon ⇒ also left
                leavers = cand[~inside_l]
                leavers = leavers[s["at0"][leavers] & ~s["left"][leavers]]
                s["left"][leavers] = True

    fx = last_qx.astype(np.float64) * q["sx"] + q["xmin"]
    fy = last_qy.astype(np.float64) * q["sy"] + q["ymin"]
    has_last = np.flatnonzero(last_qx != SENTINEL)
    for poly, s in zip(missing, st):
        at_end = np.zeros(N, bool)
        at_end[_contains(poly, fx, fy, has_last)] = True
        flags = (s["at0"].astype(np.uint8)
                 | (at_end.astype(np.uint8) << 1)
                 | (s["visited"].astype(np.uint8) << 2))
        (bdir / f"polycls_{poly['geom_key']}.bin").write_bytes(
            flags.tobytes() + s["first"].astype("<u2").tobytes())
    _classify_progress.pop(bundle_id, None)
    _log(f"classify: {len(missing)} new polygon(s), {T}x{N} in {time.time() - t0:.1f} s")


def classify(bundle_id, polygons):
    """Per-particle polygon attributes for the given polygon list.

    Combines per-polygon cached results into (polygon ids = list indices):
      origin_id u8  — first polygon containing the first position
      final_id  u8  — first polygon containing the last finite position
      first     u16 — timestep of first counted polygon entry (65535 never)
      visited   u32 — bit j: ever inside polygon j (re-entry rule applied)
      at0       u32 — bit j: inside polygon j at the first position
      at_end    u32 — bit j: inside polygon j at the last finite position
    """
    sim = registry_find(bundle_id)
    if sim is None:
        raise ValueError(f"unknown bundle: {bundle_id}")
    bdir = Path(sim["bundle_dir"])
    meta = _read_json(bdir / "meta.json", None)
    if meta is None:
        raise ValueError(f"bundle has no meta.json: {bdir}")

    polys = _poly_paths(polygons)
    if len(polys) > 32:
        raise ValueError(f"at most 32 polygons supported ({len(polys)} given)")

    missing = [p for p in polys if not (bdir / f"polycls_{p['geom_key']}.bin").exists()]
    if missing:
        _classify_missing(bdir, meta, missing, bundle_id)

    N = meta["n_particles"]
    origin_id = np.full(N, 255, np.uint8)
    final_id = np.full(N, 255, np.uint8)
    first = np.full(N, SENTINEL, np.uint16)
    visited = np.zeros(N, np.uint32)
    at0_m = np.zeros(N, np.uint32)
    at_end_m = np.zeros(N, np.uint32)

    for j, poly in enumerate(polys):
        raw = (bdir / f"polycls_{poly['geom_key']}.bin").read_bytes()
        flags = np.frombuffer(raw, np.uint8, N)
        pfirst = np.frombuffer(raw, "<u2", N, N)
        at0 = (flags & 1) != 0
        at_end = (flags & 2) != 0
        vis = (flags & 4) != 0
        bit = np.uint32(1 << j)
        np.copyto(origin_id, j, where=at0 & (origin_id == 255))
        np.copyto(final_id, j, where=at_end & (final_id == 255))
        visited |= np.where(vis, bit, np.uint32(0))
        at0_m |= np.where(at0, bit, np.uint32(0))
        at_end_m |= np.where(at_end, bit, np.uint32(0))
        first = np.where(vis, np.minimum(first, pfirst), first)

    header = json.dumps({
        "n": N,
        "arrays": ["origin_id:u1", "final_id:u1", "first:u2",
                   "visited:u4", "at0:u4", "at_end:u4"],
        "origin_counts": np.bincount(origin_id, minlength=256).tolist()[:len(polys)],
        "final_counts": np.bincount(final_id, minlength=256).tolist()[:len(polys)],
        "visited_counts": [int(((visited >> j) & 1).sum()) for j in range(len(polys))],
    }).encode("utf-8")
    return (header + b"\n" + origin_id.tobytes() + final_id.tobytes()
            + first.astype("<u2").tobytes() + visited.astype("<u4").tobytes()
            + at0_m.astype("<u4").tobytes() + at_end_m.astype("<u4").tobytes())


# ── Polygon txt import (westerschelde format) ──────────────────────────────────
def load_poly_txt(path):
    """Parse a polygon txt: latin-1, '#' comments, comma/space separated x y."""
    verts = []
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                try:
                    verts.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    continue
    return verts


def import_polygon_dir(directory, pattern="*.txt"):
    directory = Path(directory)
    polys = []
    for f in sorted(directory.glob(pattern)):
        verts = load_poly_txt(f)
        if len(verts) >= 3:
            stem = f.stem
            label = stem.split("_", 1)[1].replace("_", " ") if "_" in stem else stem
            polys.append({"name": label, "verts": verts})
    return polys
