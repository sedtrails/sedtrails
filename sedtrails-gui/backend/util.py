"""Small shared helpers: logging, JSON files, browse, native dialogs, jobs.

Extracted verbatim from sedtrails-viewer/sedtrails_viewer.py (Phase 0 split).
"""

import json
import os
import re
import string
import threading
import time
from pathlib import Path

import numpy as np

from . import settings as cfg

_jobs = {}
_video_jobs = {}
_jobs_lock = threading.Lock()

# HDF5 (and therefore netCDF4) is NOT thread-safe: concurrent reads from the
# threaded HTTP server crash the whole process with an access violation.
# Every netCDF4 open/read/close in this backend must hold this lock.
NC_LOCK = threading.RLock()


# ── Small helpers ──────────────────────────────────────────────────────────────
def resolve_input_path(path, base=None):
    """Resolve a user/config-supplied file path robustly.

    - POSIX drive mounts from HPC configs (``/p/...``) map to the drive letter
      on Windows (``P:\\...``) — the H7 cluster mounts network drives that way.
    - Absolute or drive-anchored paths (``p:\\...``, ``\\\\server\\...``) are
      NEVER joined onto ``base`` (joining used to mangle ``p:\\x`` → ``C:\\p\\x``).
    - Only genuinely relative paths resolve against ``base`` (the config dir).
    """
    raw = str(path).strip()
    p = Path(raw)
    fwd = raw.replace("\\", "/")
    if os.name == "nt":
        m = re.match(r"^/([A-Za-z])(/.*)?$", fwd)
        if m:
            p = Path(m.group(1).upper() + ":" + (m.group(2) or "/"))
    anchored = p.is_absolute() or re.match(r"^[A-Za-z]:", raw) or fwd.startswith("/")
    if not anchored and base:
        p = Path(base) / p
    return p.resolve()


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _nanfilled(var_slice):
    """netCDF4 slice → plain float array with masked/fill values as NaN.

    Kept memory-lean (in-place fill-value replacement, no masked-array
    wrappers): import slabs on big runs are hundreds of MB each and the
    viewer must survive on low-RAM machines.
    """
    arr = var_slice
    if isinstance(arr, np.ma.MaskedArray):
        arr = arr.filled(np.nan)
    arr = np.asarray(arr, dtype=np.float64)
    arr[arr > 1e30] = np.nan
    arr[arr < -1e30] = np.nan
    return arr


# ── Native file picker ─────────────────────────────────────────────────────────
_picker_lock = threading.Lock()


def native_save(ext, suggest=None, initial_dir=None):
    """Native 'Save as…' dialog on the server machine. '' = cancelled."""
    import tkinter as tk
    from tkinter import filedialog

    with _picker_lock:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            return filedialog.asksaveasfilename(
                parent=root, initialdir=initial_dir or None,
                initialfile=suggest or None, defaultextension=ext,
                filetypes=[(ext.upper().strip('.') + " file", "*" + ext)]) or ""
        finally:
            root.destroy()


def native_pick(kind, initial_dir=None):
    """Open a native OS file dialog (the server runs on the user's machine).

    kind: 'nc' (single netCDF), 'txt' (multiple polygon txts), 'yaml' (single
    yaml config), 'dir' (folder). Returns a list of paths (empty = cancelled).
    Raises if tkinter unavailable (headless session) — the viewer then falls
    back to its built-in browser.
    """
    import tkinter as tk
    from tkinter import filedialog

    with _picker_lock:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            if kind == "dir":
                p = filedialog.askdirectory(initialdir=initial_dir or None, parent=root)
                return [p] if p else []
            if kind == "yaml":
                p = filedialog.askopenfilename(
                    initialdir=initial_dir or None, parent=root,
                    title="Select SedTRAILS config",
                    filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")])
                return [p] if p else []
            if kind == "txt":
                ps = filedialog.askopenfilenames(
                    initialdir=initial_dir or None, parent=root,
                    title="Select polygon file(s)",
                    filetypes=[("Polygon text files", "*.txt"), ("All files", "*.*")])
                return list(ps)
            p = filedialog.askopenfilename(
                initialdir=initial_dir or None, parent=root,
                title="Select netCDF file",
                filetypes=[("netCDF files", "*.nc"), ("All files", "*.*")])
            return [p] if p else []
        finally:
            root.destroy()


def _browse(dir_param):
    if not dir_param:
        roots = []
        for d in string.ascii_uppercase:
            drive = f"{d}:\\"
            if os.path.exists(drive):
                roots.append(drive)
        roots.append(str(Path.home()))
        roots.extend(str(Path(r)) for r in cfg.DATA_ROOTS if os.path.exists(r))
        return {"dir": "", "parent": None,
                "dirs": [{"name": r, "path": r} for r in dict.fromkeys(roots)],
                "files": []}
    d = Path(dir_param)
    if not d.is_dir():
        raise ValueError(f"not a directory: {dir_param}")
    dirs, files = [], []
    try:
        entries = sorted(os.scandir(d), key=lambda e: e.name.lower())
    except PermissionError:
        entries = []
    for e in entries:
        try:
            if e.is_dir():
                if not e.name.startswith("."):
                    dirs.append({"name": e.name, "path": str(Path(d) / e.name)})
            elif e.name.lower().endswith((".nc", ".txt", ".yaml", ".yml")):
                files.append({"name": e.name, "path": str(Path(d) / e.name),
                              "size": e.stat().st_size})
        except OSError:
            continue
    parent = str(d.parent) if d.parent != d else ""
    return {"dir": str(d), "parent": parent, "dirs": dirs, "files": files}
