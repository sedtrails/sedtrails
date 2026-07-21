"""
SedTRAILS GUI — configure, seed, run and inspect SedTRAILS simulations.

Run directly (VS Code / terminal):

    python sedtrails_gui.py

Opens its own app window (Edge/Chrome, no address bar) serving
http://127.0.0.1:8766/ — set APP_WINDOW = False for a plain browser tab.
Tabs: Settings, Populations, Run, Viewer (particles + forcing), Connectivity.
The Viewer tab is the full standalone sedtrails-viewer; the standalone tool
remains untouched next to this one and both can run side by side (different
ports, separate registries — bundles imported with the old viewer show up
here automatically).

Dependencies: numpy, netCDF4. Optional: matplotlib + scipy (model layers,
classification), tkinter (native file dialogs), ffmpeg (MP4 export), and the
sedtrails package + jsonschema (Settings / Seeding / Run sections).

More constants (slab sizes, browse roots, app dir): backend/settings.py.
"""

import os
import sys
import threading
import webbrowser

# Let the GUI peek at run progress (written_slots) in the output netCDF while
# sedtrails is still writing it — Windows HDF5 locking would refuse otherwise.
# The launched run itself gets this stripped from its environment (run_manager).
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# ── Configuration ──────────────────────────────────────────────────────────────
HOST = "127.0.0.1"
PORT = 8766                      # auto-increments if taken
OPEN_BROWSER = True
APP_WINDOW = True                # own window without address bar (Edge/Chrome); False = normal browser tab
DATA_ROOTS = []                  # extra browse roots (absolute paths); drives + home are added automatically

from backend import settings

settings.DATA_ROOTS = DATA_ROOTS

from backend.bundle import load_registry
from backend.httpd import GuiHandler, QuietServer
from backend.util import _log


def _open_window(url):
    """Open the GUI in its own app window (no tabs / address bar) via a
    Chromium browser's --app mode; falls back to a normal browser tab."""
    import shutil
    import subprocess
    candidates = [
        shutil.which("msedge"),
        shutil.which("chrome"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for exe in candidates:
        if exe and os.path.exists(exe):
            subprocess.Popen([exe, f"--app={url}"])
            return
    webbrowser.open(url)


def main():
    import faulthandler
    faulthandler.enable()  # a hard crash (e.g. in the netCDF C library) prints a trace
    settings.APP_DIR.mkdir(parents=True, exist_ok=True)
    port = PORT
    for _ in range(21):
        try:
            server = QuietServer((HOST, port), GuiHandler)
            break
        except OSError:
            port += 1
    else:
        _log("no free port found")
        sys.exit(1)

    url = f"http://{HOST}:{port}/"
    reg = load_registry()
    _log("SedTRAILS GUI")
    _log(f"  serving {settings.WEB_DIR / 'index.html'}")
    _log(f"  {len(reg['simulations'])} simulation(s) in registry")
    _log(f"  open {url}")
    if OPEN_BROWSER:
        opener = _open_window if APP_WINDOW else webbrowser.open
        threading.Timer(0.4, lambda: opener(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("stopped")


if __name__ == "__main__":
    main()
