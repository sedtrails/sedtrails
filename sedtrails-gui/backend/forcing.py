"""Forcing tab backend: open a sedtrails-compatible forcing netCDF (D-Flow FM
"sedtrails_nc" node-cloud layout), discover its variables and vector pairs,
build a Voronoi cell mesh with a per-vertex node index (cached on disk next to
the file), and stream per-timestep float32 value slabs to the frontend.

The Voronoi construction mirrors bundle._voronoi_cells_geometry, but instead of
baking values per vertex it emits the ORIGINAL node index per vertex, so any
variable/timestep can be gathered client-side onto the same static geometry.
"""

import datetime
import hashlib
import json
import re
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np

from .util import (NC_LOCK, _jobs, _jobs_lock, _log, _nanfilled, _read_json,
                   _write_json, resolve_input_path)

MESH_FORMAT_VERSION = 4          # v4: cell cutoff vs 8th neighbor (anisotropic grids)
_store = {}          # id -> {"path": str, "meta": dict, "cache_dir": Path}
_store_lock = threading.Lock()
_range_cache = {}    # (id, var) -> [lo, hi] global 2-98 percentile of first/mid/last frame

# variable-name pairs known from transport_converter fm_netcdf conventions
KNOWN_PAIRS = [
    ("sea_water_x_velocity", "sea_water_y_velocity", "flow velocity"),
    ("bedload_x_comp", "bedload_y_comp", "bedload transport"),
    ("susload_x_comp", "susload_y_comp", "suspended load transport"),
]


def forcing_id(path):
    p = Path(path).resolve()
    return "f" + hashlib.sha1(str(p).lower().encode("utf-8")).hexdigest()[:10]


def cache_dir_for(path):
    p = Path(path)
    return p.parent / (p.stem + ".forcing_cache")


def _open_ds(path):
    import netCDF4 as nc
    return nc.Dataset(path, "r")


def _reference_datetime(t_var):
    units = getattr(t_var, "units", "") or ""
    m = re.search(r"since\s+(\d{4}-\d{2}-\d{2})[T ]?(\d{2}:\d{2}:\d{2})?", units)
    if not m:
        return datetime.datetime(1970, 1, 1)
    d = datetime.datetime.strptime(m.group(1), "%Y-%m-%d")
    if m.group(2):
        h, mi, s = (int(x) for x in m.group(2).split(":"))
        d = d.replace(hour=h, minute=mi, second=s)
    return d


def open_forcing(path, base=None):
    """Register a forcing file; return its id + meta (times, variables, pairs)."""
    p = resolve_input_path(path, base)
    if not p.is_file():
        extra = f" (resolved to {p})" if str(p) != str(path) else ""
        raise FileNotFoundError(f"file not found: {path}{extra}")
    fid = forcing_id(p)

    with NC_LOCK:
        ds = _open_ds(p)
        return _open_forcing_locked(ds, p, fid)


def _open_forcing_locked(ds, p, fid):
    try:
        if "net_xcc" not in ds.variables:
            raise ValueError("not a sedtrails forcing file (net_xcc/net_ycc missing) — "
                             "convert the model output first")
        x = _nanfilled(ds.variables["net_xcc"][:])
        y = _nanfilled(ds.variables["net_ycc"][:])
        node_dim = ds.variables["net_xcc"].dimensions[0]
        n_nodes = len(x)
        ok = np.isfinite(x) & np.isfinite(y)
        extent = [float(np.min(x[ok])), float(np.max(x[ok])),
                  float(np.min(y[ok])), float(np.max(y[ok]))]

        t_var = ds.variables.get("time")
        times, ref = [], None
        if t_var is not None:
            ref = _reference_datetime(t_var)
            tsec = _nanfilled(t_var[:]).ravel()
            times = [(ref + datetime.timedelta(seconds=float(s))).strftime("%Y-%m-%d %H:%M")
                     if np.isfinite(s) else "?" for s in tsec]

        vector_names = set()
        pairs = []
        for xn, yn, label in KNOWN_PAIRS:
            if xn in ds.variables and yn in ds.variables:
                pairs.append({"label": label, "x": xn, "y": yn})
                vector_names |= {xn, yn}
        for name, v in ds.variables.items():          # generic *_x_* / x_comp fallback
            if name in vector_names or node_dim not in v.dimensions:
                continue
            partner = None
            if "_x_" in name:
                partner = name.replace("_x_", "_y_")
            elif name.endswith("x_comp"):
                partner = name[:-6] + "y_comp"
            if partner and partner in ds.variables and partner not in vector_names:
                pairs.append({"label": name.replace("_x_", " ").replace("x_comp", "").strip("_"),
                              "x": name, "y": partner})
                vector_names |= {name, partner}

        variables = []
        for name, v in ds.variables.items():
            if name in ("net_xcc", "net_ycc", "time") or node_dim not in v.dimensions:
                continue
            variables.append({
                "name": name,
                "time_varying": "time" in v.dimensions,
                "units": getattr(v, "units", ""),
                "long_name": getattr(v, "long_name", name),
                "is_vector_comp": name in vector_names,
            })

        meta = {"id": fid, "path": str(p), "n_nodes": n_nodes, "extent": extent,
                "n_timesteps": len(times), "times": times,
                "reference_date": ref.strftime("%Y-%m-%d %H:%M") if ref else None,
                "variables": variables, "vector_pairs": pairs}
    finally:
        ds.close()

    cdir = cache_dir_for(p)
    with _store_lock:
        _store[fid] = {"path": str(p), "meta": meta, "cache_dir": cdir}

    result = {"id": fid, "meta": meta}
    mesh_meta = _read_json(cdir / "mesh.json", None)
    if mesh_meta and mesh_meta.get("format_version") == MESH_FORMAT_VERSION \
            and (cdir / "mesh.bin").exists():
        result["mesh"] = mesh_meta
    else:
        job_id = uuid.uuid4().hex[:12]
        job = {"id": job_id, "status": "running", "progress": 0.0,
               "message": "building forcing mesh", "bundle": None}
        with _jobs_lock:
            _jobs[job_id] = job

        def work():
            try:
                mm = build_mesh(str(p), cdir, job)
                job["status"] = "done"
                job["mesh"] = mm
            except Exception as e:
                traceback.print_exc()
                job["status"] = "error"
                job["message"] = f"{type(e).__name__}: {e}"

        threading.Thread(target=work, daemon=True).start()
        result["mesh_job"] = job_id
    return result


def build_mesh(path, cdir, job=None):
    """Voronoi cells of the node cloud; per-vertex ORIGINAL node index.

    mesh.bin layout: xy f4 (Nv,2) ABSOLUTE coords | nodeIndex u4 (Nv,) |
    tris u4 (Nt,3) | node xy f4 (n_nodes, 2).
    """
    from scipy.spatial import Voronoi, cKDTree

    def prog(f, msg):
        if job is not None:
            job["progress"] = round(f, 3)
            job["message"] = msg

    t0 = time.time()
    with NC_LOCK:
        ds = _open_ds(path)
        try:
            xn = _nanfilled(ds.variables["net_xcc"][:])
            yn = _nanfilled(ds.variables["net_ycc"][:])
        finally:
            ds.close()

    ok = np.isfinite(xn) & np.isfinite(yn)
    orig_idx = np.flatnonzero(ok)
    pts = np.column_stack([xn[ok], yn[ok]])
    n_real = len(pts)
    prog(0.1, f"Voronoi tessellation of {n_real} nodes")

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
    tree = cKDTree(pts)
    nn = tree.query(pts, k=2)[0][:, 1]
    # cutoff radius from the 8th neighbor, not the 1st: on strongly anisotropic
    # grids (alongshore spacing >> cross-shore) the 1st neighbor is the short
    # axis and whole rows of valid cells would be cut (sandengine grid)
    k8 = min(9, n_real)
    loc = tree.query(pts, k=k8)[0][:, k8 - 1] if k8 >= 2 else nn
    r_cut2 = (3.0 * np.maximum(loc, 1e-3)) ** 2
    prog(0.5, "assembling cells")

    verts_l, nidx_l, tris_l = [], [], []
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
        nidx_l.append(np.full(c, orig_idx[i], np.uint32))
        fan = np.empty((c - 2, 3), np.int64)
        fan[:, 0] = vo
        fan[:, 1] = vo + np.arange(1, c - 1)
        fan[:, 2] = vo + np.arange(2, c)
        tris_l.append(fan)
        vo += c

    xy = np.concatenate(verts_l).astype("<f4")
    nidx = np.concatenate(nidx_l).astype("<u4")
    tris = np.concatenate(tris_l).astype("<u4")
    nodes = np.column_stack([xn, yn]).astype("<f4")
    nodes[~np.isfinite(nodes)] = -1e30

    prog(0.9, "writing mesh cache")
    cdir.mkdir(parents=True, exist_ok=True)
    with open(cdir / "mesh.bin", "wb") as f:
        f.write(np.ascontiguousarray(xy).tobytes())
        f.write(np.ascontiguousarray(nidx).tobytes())
        f.write(np.ascontiguousarray(tris).tobytes())
        f.write(np.ascontiguousarray(nodes).tobytes())
    mesh_meta = {"format_version": MESH_FORMAT_VERSION, "n_verts": int(len(xy)),
                 "n_tris": int(len(tris)), "n_nodes": int(len(nodes)),
                 # spacing of the finest cells (1st pctl nearest-neighbor
                 # distance) — the frontend's world-based arrow-scale anchor
                 "cell_p01": float(np.percentile(nn, 1)),
                 "file": "mesh.bin"}
    _write_json(cdir / "mesh.json", mesh_meta)
    _log(f"forcing mesh: {len(xy)} vertices, {len(tris)} triangles "
         f"in {time.time() - t0:.1f} s -> {cdir / 'mesh.bin'}")
    return mesh_meta


def get_store(fid):
    with _store_lock:
        s = _store.get(fid)
    if s is None:
        raise ValueError(f"unknown forcing id: {fid} (open it first)")
    return s


def mesh_payload(fid):
    """mesh.json header + mesh.bin bytes, as one JSON-line + binary payload."""
    s = get_store(fid)
    mesh_meta = _read_json(s["cache_dir"] / "mesh.json", None)
    if mesh_meta is None:
        raise ValueError("mesh not built yet")
    header = json.dumps(mesh_meta).encode("utf-8")
    return header + b"\n" + (s["cache_dir"] / "mesh.bin").read_bytes()


def _slab(ds, name, t):
    """float32[n_nodes] for one variable/timestep. Robust to extra dimensions
    (sediment fractions, layers): selects the requested time along the time
    axis and index 0 along any remaining non-node axis."""
    v = ds.variables[name]
    if "time" in v.dimensions and v.ndim >= 2:
        idx = [slice(None)] * v.ndim
        idx[v.dimensions.index("time")] = int(t)
        arr = v[tuple(idx)]
    else:
        arr = v[:]
    arr = np.asarray(_nanfilled(arr)).squeeze()
    while arr.ndim > 1:            # e.g. (n_fractions, n_nodes) → first fraction
        arr = arr[0]
    return arr.astype("<f4")


def field_payload(fid, var, t):
    """float32[n_nodes] for one variable/timestep + a stable global color range."""
    s = get_store(fid)
    with NC_LOCK:
        ds = _open_ds(s["path"])
        try:
            arr = _slab(ds, var, t)
            key = (fid, var)
            if key not in _range_cache:
                nt = s["meta"]["n_timesteps"]
                probes = [arr] if "time" not in ds.variables[var].dimensions else \
                    [_slab(ds, var, k) for k in {0, nt // 2, nt - 1}]
                allv = np.concatenate(probes)
                fin = allv[np.isfinite(allv)]
                _range_cache[key] = ([float(np.percentile(fin, 2)), float(np.percentile(fin, 98))]
                                     if fin.size else [0.0, 1.0])
            lo, hi = _range_cache[key]
        finally:
            ds.close()
    arr[~np.isfinite(arr)] = -1e30
    return arr.tobytes(), f"{lo:.6g},{hi:.6g}"


def vector_payload(fid, xvar, yvar, t):
    """interleaved float32[n_nodes, 2] (u, v) + 98th-percentile magnitude."""
    s = get_store(fid)
    with NC_LOCK:
        ds = _open_ds(s["path"])
        try:
            u = _slab(ds, xvar, t)
            v = _slab(ds, yvar, t)
        finally:
            ds.close()
    mag = np.hypot(u, v)
    fin = mag[np.isfinite(mag)]
    p98 = float(np.percentile(fin, 98)) if fin.size else 1.0
    # same -1e30 sentinel as field_payload: lets the client discard dry/masked
    # nodes in vector-derived maps and skip them in the arrow shader
    bad = ~(np.isfinite(u) & np.isfinite(v))
    u[bad] = -1e30
    v[bad] = -1e30
    out = np.empty((len(u), 2), "<f4")
    out[:, 0] = u
    out[:, 1] = v
    return out.tobytes(), f"{p98:.6g}"
