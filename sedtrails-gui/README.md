# SedTRAILS GUI

A local GUI covering the full SedTRAILS workflow in five tabs:

| Tab | What it does |
|---|---|
| **Settings** | Open/edit/save a simulation yaml. The form is generated from the packaged JSON schemas (`sedtrails/config/*.schema.json`), so every option shows its documentation (ⓘ), defaults and allowed values, with live validation and an estimated output-file size. Saves preserve your comments and layout (ruamel.yaml). |
| **Populations** | Draw seeding per population directly on the map — points, transects, boxes, polygons — and preview the *exact* seed locations (computed by sedtrails' own seeding strategies, including RNG seeds). Values are written into the loaded yaml. |
| **Run** | Start/stop `sedtrails run` on the loaded config, with live log, progress bar (output slots from the streaming netCDF) and ETA. One click imports the finished result into the Viewer. |
| **Viewer** | Particles and forcing in one map: the full standalone particle viewer (trajectories, paths, polygon classification, rule-based filtering/coloring, PNG/MP4 export) plus netCDF forcing layers — Voronoi-cell scalar fields (bed level, water depth, BSS, …) with time scrubbing and quiver arrows for the vector pairs. The forcing file follows Settings → inputs → data. |
| **Connectivity** | v0: polygon→polygon connectivity matrix (start→end or start→ever-visited, counts or fractions) over an imported run, with CSV export and a network overlay on the map. Built on the viewer's cached classification; new metrics slot in as new modes on the same endpoint. |

## Run it

Select the sedtrails venv interpreter and run directly (VS Code or terminal):

```
python sedtrails_gui.py
```

Opens its own app window (Edge/Chrome `--app` mode, no address bar) serving
`http://127.0.0.1:8766/`; set `APP_WINDOW = False` for a plain browser tab.
The standalone viewer (`sedtrails-viewer/`, port 8765) is untouched and can run
at the same time; runs imported there show up here too.

## Dependencies

- required: `numpy`, `netCDF4`
- viewer extras: `matplotlib` + `scipy` (model layers, classification), `tkinter`
  (native file dialogs), `ffmpeg` on PATH (MP4 export)
- GUI extras: the `sedtrails` package + `jsonschema` (Settings/Seeding/Run tabs),
  `ruamel.yaml` (comment-preserving saves; without it saves fall back to a
  schema-ordered dump that drops comments)

## Configuration

`HOST` / `PORT` / `OPEN_BROWSER` / `APP_WINDOW` / `DATA_ROOTS` at the top of
`sedtrails_gui.py`;
import slab sizes and the app dir (`~/.sedtrails_gui`) in `backend/settings.py`.

## Layout

- `sedtrails_gui.py` — entry point (constants + server startup)
- `backend/` — HTTP routes (`httpd.py`), results-bundle importer/classifier
  (`bundle.py`, verbatim from the standalone viewer), config API (`config_api.py`),
  forcing reader/mesh (`forcing.py`), seeding preview (`seeding_api.py`),
  run manager (`run_manager.py`), connectivity (`connectivity.py`)
- `web/` — `index.html` + `css/app.css` + `js/core/*` (shared WebGL map, tabs,
  playback) + `js/tabs/*` (one file per section; `viewer*.js` are the split-up
  standalone viewer, kept byte-identical where possible)
