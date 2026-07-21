"""Connectivity tab backend (v0): polygon-to-polygon connectivity matrix.

Builds on the viewer's cached per-polygon classification (bundle.classify
machinery): M[i][j] = number (or fraction) of particles starting in polygon i
that end in / ever visit polygon j. New metrics later = new `mode` values here.
"""

from pathlib import Path

import numpy as np

from .bundle import _classify_missing, _poly_paths, registry_find
from .util import _read_json


def compute(bundle_id, polygons, mode="start_end", normalize="count"):
    sim = registry_find(bundle_id)
    if sim is None:
        raise ValueError(f"unknown bundle: {bundle_id} (import it in the Viewer tab first)")
    bdir = Path(sim["bundle_dir"])
    meta = _read_json(bdir / "meta.json", None)
    if meta is None:
        raise ValueError(f"bundle has no meta.json: {bdir}")

    polys = _poly_paths(polygons)
    if len(polys) < 2:
        raise ValueError("connectivity needs at least 2 polygons")
    if len(polys) > 200:
        # uint8 origin/final ids allow up to 254 polygons (255 = unassigned)
        raise ValueError("at most 200 polygons supported")

    missing = [p for p in polys if not (bdir / f"polycls_{p['geom_key']}.bin").exists()]
    if missing:
        _classify_missing(bdir, meta, missing, bundle_id)

    N = meta["n_particles"]
    n = len(polys)
    origin = np.full(N, 255, np.uint8)
    at_end = np.zeros((n, N), bool)
    visited = np.zeros((n, N), bool)
    for j, poly in enumerate(polys):
        raw = (bdir / f"polycls_{poly['geom_key']}.bin").read_bytes()
        flags = np.frombuffer(raw, np.uint8, N)
        at0 = (flags & 1) != 0
        at_end[j] = (flags & 2) != 0
        visited[j] = (flags & 4) != 0
        np.copyto(origin, j, where=at0 & (origin == 255))

    target = at_end if mode == "start_end" else visited
    matrix = np.zeros((n, n), np.float64)
    row_counts = []
    for i in range(n):
        src = origin == i
        row_counts.append(int(src.sum()))
        for j in range(n):
            matrix[i, j] = int((src & target[j]).sum())
    if normalize == "fraction":
        for i in range(n):
            if row_counts[i]:
                matrix[i] /= row_counts[i]

    return {
        "labels": [p["key"] for p in polys],
        "matrix": [[round(float(v), 4) for v in row] for row in matrix],
        "row_counts": row_counts,
        "n_unassigned": int((origin == 255).sum()),
        "n_particles": N,
        "mode": mode,
        "normalize": normalize,
    }


def auto_polygons(forcing_id, n=12, var=None):
    """Geomorphic cells à la Pearson et al. (2021): k-means clustering of the
    forcing node cloud on (x, y, bed level) — each feature min-max normalized so
    position and bathymetry weigh equally — then the Voronoi cells of the
    cluster centroids, closed against the domain bounding box (mirror trick):
    n simple polygons covering the domain, ready for connectivity analysis."""
    from scipy.cluster.vq import kmeans2
    from scipy.spatial import Voronoi

    from .forcing import NC_LOCK, _open_ds, _slab, get_store

    n = int(n)
    if not 2 <= n <= 200:
        raise ValueError("number of cells must be between 2 and 200")
    s = get_store(forcing_id)
    with NC_LOCK:
        ds = _open_ds(s["path"])
        try:
            x = np.asarray(ds.variables["net_xcc"][:], float).ravel()
            y = np.asarray(ds.variables["net_ycc"][:], float).ravel()
            z = np.asarray(_slab(ds, var, 0), float).ravel() if var else np.zeros_like(x)
        finally:
            ds.close()

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z > -1e29)
    if ok.sum() < n:
        raise ValueError("not enough valid nodes for that many cells")
    pts = np.column_stack([x[ok], y[ok]])
    zz = z[ok]
    if len(pts) > 200_000:                       # keep k-means quick on big grids
        step = len(pts) // 200_000 + 1
        pts, zz = pts[::step], zz[::step]

    feats = np.column_stack([pts, zz])
    lo, hi = feats.min(axis=0), feats.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    _, labels = kmeans2((feats - lo) / span, n, minit="++", seed=0)
    cents = np.array([pts[labels == k].mean(axis=0)
                      for k in range(n) if (labels == k).any()])

    # Voronoi of the centroids; reflecting them across the four bbox edges makes
    # every real region finite and clipped exactly at the bounding box
    x0, x1 = pts[:, 0].min(), pts[:, 0].max()
    y0, y1 = pts[:, 1].min(), pts[:, 1].max()
    mirrors = np.vstack([
        np.column_stack([2 * x0 - cents[:, 0], cents[:, 1]]),
        np.column_stack([2 * x1 - cents[:, 0], cents[:, 1]]),
        np.column_stack([cents[:, 0], 2 * y0 - cents[:, 1]]),
        np.column_stack([cents[:, 0], 2 * y1 - cents[:, 1]]),
    ])
    vor = Voronoi(np.vstack([cents, mirrors]))
    polygons = []
    for k in range(len(cents)):
        region = vor.regions[vor.point_region[k]]
        if not region or -1 in region:
            continue
        verts = [[round(float(vor.vertices[i][0]), 1),
                  round(float(vor.vertices[i][1]), 1)] for i in region]
        polygons.append({"name": f"auto_cell_{len(polygons) + 1:02d}", "verts": verts})
    return {"polygons": polygons, "n_nodes_used": int(len(pts))}
