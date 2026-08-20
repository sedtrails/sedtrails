"""Shared configuration constants for the SedTRAILS GUI backend."""

from pathlib import Path

DATA_ROOTS = []                  # extra browse roots (absolute paths); drives + home are added automatically

BG_WIDTH_PX = 6144               # width of generated background images (height from aspect)
BG_ELEV_PCTL = (2.0, 98.0)       # percentile color limits for the bathymetry background
BG_DZ_PCTL = 98.0                # symmetric percentile limit for the bed-change background
CMAP_NPY = None                  # optional path to an (N,3)/(N,4) .npy colormap for the bathymetry

SLAB_STEPS = 8                   # timesteps per read slab for time-major files
SLAB_PARTICLES = 50_000          # particles per read slab for particle-major files
LOAD_WHOLE_LIMIT = 2 * 1024**3   # particle-major files smaller than this (per variable) load in one go
PAD_FRAC = 0.02                  # extent padding around the data bounding box
REFERENCE_DATE_FALLBACK = "2017-12-01"  # used when the time variable has no CF units

APP_DIR = Path.home() / ".sedtrails_gui"
VIEWER_APP_DIR = Path.home() / ".sedtrails_viewer"  # old standalone-viewer registry, merged read-only
FORMAT_VERSION = 1
SENTINEL = 65535
QMAX = 65534

TOOL_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = TOOL_DIR / "web"
