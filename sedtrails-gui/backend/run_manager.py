"""Run tab backend: launch `sedtrails run` as a subprocess, stream its log,
and report progress from the incrementally-written output netCDF.

Progress source (in order of preference):
1. global attrs `written_slots` / `estimated_output_slots` on sedtrails_results.nc
   (updated per save slot by sedtrails' NetCDFWriter),
2. the tqdm progress bar sedtrails prints to stderr (parsed live from the log
   stream — tqdm updates with carriage returns, so the reader splits on \\r too),
3. output file size vs. the estimated uncompressed payload,
while the log ring buffer feeds the live log pane. HDF5 may refuse to open a
file the writer holds (Windows locking) — every probe is try/except.
"""

import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from . import config_api
from .util import NC_LOCK, _log

MAX_LOG_LINES = 8000
_TQDM_PCT = re.compile(r"(\d+(?:\.\d+)?)%\|")
_TQDM_ETA = re.compile(r"<((?:\d+:)?\d+:\d+)")   # [elapsed<remaining, rate]


def _hms_to_s(txt):
    parts = [int(x) for x in txt.split(":")]
    s = 0
    for p in parts:
        s = s * 60 + p
    return s


class RunManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.proc = None
        self.state = "idle"          # idle | running | done | error | stopped
        self.returncode = None
        self.started = None
        self.ended = None
        self.config_path = None
        self.result_path = None
        self.estimate = None         # config_api.estimate() result at start
        self.lines = []              # ring buffer
        self.base_offset = 0         # offset of lines[0]
        self._probe = {"at": 0.0, "progress": None}
        self._rate = deque(maxlen=24)   # (t, written_slots) for ETA
        self._brate = deque(maxlen=24)  # (t, bytes) for the size-fallback ETA
        self._tqdm = None               # {"pct", "eta_s", "at"} parsed from the log
        self._cr_open = False           # last ring-buffer line is a \r-updated line

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self, config_path):
        with self._lock:
            if self.proc is not None and self.proc.poll() is None:
                raise RuntimeError("a run is already in progress")
            cfg_path = Path(config_path).resolve()
            if not cfg_path.is_file():
                raise FileNotFoundError(f"config not found: {cfg_path}")

            loaded = config_api.load_config(str(cfg_path))
            cfg = loaded["config"]
            out_dir = (cfg.get("outputs", {}) or {}).get("directory", "./output")
            out_dir = (cfg_path.parent / out_dir).resolve()
            self.result_path = str(out_dir / "sedtrails_results.nc")
            self.estimate = config_api.estimate(cfg)

            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            env = dict(os.environ)
            env.pop("HDF5_USE_FILE_LOCKING", None)   # writer keeps default locking
            # sedtrails disables its tqdm bar on non-TTY stderr; this opt-in
            # keeps it printing so the GUI can parse live progress from the log
            env["SEDTRAILS_FORCE_PROGRESS"] = "1"
            self.proc = subprocess.Popen(
                [sys.executable, "-u", "-c",
                 "from sedtrails.application_interfaces.cli import app; app()",
                 "run", "--config", str(cfg_path)],
                cwd=str(cfg_path.parent),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                creationflags=creationflags, env=env)
            self.state = "running"
            self.returncode = None
            self.started = time.time()
            self.ended = None
            self.config_path = str(cfg_path)
            self.lines = []
            self.base_offset = 0
            self._probe = {"at": 0.0, "progress": None}
            self._rate.clear()
            self._brate.clear()
            self._tqdm = None
            self._cr_open = False
            threading.Thread(target=self._reader, daemon=True).start()
            _log(f"run started: {cfg_path} (pid {self.proc.pid}) -> {self.result_path}")
            return {"ok": True, "pid": self.proc.pid, "result_path": self.result_path}

    def _ingest(self, text, ends_with_cr):
        """One log segment (split on \\n OR \\r). \\r segments (tqdm updates)
        replace the previous \\r segment in the ring buffer instead of
        appending, and are parsed for pct/ETA."""
        m = _TQDM_PCT.search(text)
        if m:
            eta = _TQDM_ETA.search(text)
            self._tqdm = {"pct": float(m.group(1)),
                          "eta_s": float(_hms_to_s(eta.group(1))) if eta else None,
                          "at": time.time()}
        if self._cr_open and self.lines:
            if text:
                self.lines[-1] = text
            self._cr_open = ends_with_cr
            return
        if text or not ends_with_cr:
            self.lines.append(text)
            if len(self.lines) > MAX_LOG_LINES:
                drop = len(self.lines) - MAX_LOG_LINES
                del self.lines[:drop]
                self.base_offset += drop
        self._cr_open = ends_with_cr

    def _reader(self):
        proc = self.proc
        try:
            fd = proc.stdout.fileno()
            buf = b""
            while True:
                chunk = os.read(fd, 8192)
                if not chunk:
                    break
                buf += chunk
                while True:
                    i_n = buf.find(b"\n")
                    i_r = buf.find(b"\r")
                    idx = min(x for x in (i_n, i_r) if x >= 0) if max(i_n, i_r) >= 0 else -1
                    if idx < 0:
                        break
                    seg = buf[:idx].decode("utf-8", "replace")
                    is_cr = buf[idx:idx + 1] == b"\r"
                    buf = buf[idx + 1:]
                    self._ingest(seg, is_cr)
            if buf:
                self._ingest(buf.decode("utf-8", "replace"), False)
        except (ValueError, OSError):
            pass
        proc.wait()
        self.returncode = proc.returncode
        self.ended = time.time()
        if self.state == "running":
            self.state = "done" if proc.returncode == 0 else "error"
        _log(f"run finished: rc={proc.returncode} ({self.state})")

    def stop(self):
        with self._lock:
            if self.proc is None or self.proc.poll() is not None:
                return {"ok": True, "note": "no run in progress"}
            self.state = "stopped"
            pid = self.proc.pid
            if os.name == "nt":
                # /T kills the whole tree (parallel runs fork worker processes)
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)],
                               capture_output=True)
            else:
                self.proc.terminate()
            _log(f"run stopped (pid {pid})")
            return {"ok": True}

    # ── progress ───────────────────────────────────────────────────────────
    def _probe_progress(self):
        now = time.time()
        if now - self._probe["at"] < 2.0:
            return self._probe["progress"]
        prog = {"written_slots": None, "total_slots": None, "pct": None,
                "eta_s": None, "bytes": None, "est_bytes": None}
        est = self.estimate or {}
        prog["est_bytes"] = est.get("bytes")
        prog["total_slots"] = est.get("slots")
        rp = self.result_path
        # ignore a result file left over from a PREVIOUS run (its attrs would
        # show stale/complete progress until the new writer recreates it)
        if rp and self.started and os.path.exists(rp):
            try:
                if os.path.getmtime(rp) < self.started - 1.0:
                    rp = None
            except OSError:
                rp = None
        if rp and os.path.exists(rp):
            try:
                prog["bytes"] = os.path.getsize(rp)
            except OSError:
                pass
            try:
                import netCDF4 as nc
                with NC_LOCK, nc.Dataset(rp, "r") as ds:
                    w = getattr(ds, "written_slots", None)
                    t = getattr(ds, "estimated_output_slots", None)
                    if w is not None:
                        prog["written_slots"] = int(w)
                    if t is not None:
                        prog["total_slots"] = int(t)
            except Exception:
                pass                       # writer holds the file — size fallback
        w, t = prog["written_slots"], prog["total_slots"]
        if w is not None and t:
            prog["pct"] = min(100.0, w / max(t, 1) * 100.0)
            self._rate.append((now, w))
            if len(self._rate) >= 2:
                (t0, w0), (t1, w1) = self._rate[0], self._rate[-1]
                if w1 > w0:
                    prog["eta_s"] = max(0.0, (t - w1) * (t1 - t0) / (w1 - w0))
        elif self._tqdm is not None:
            # tqdm bar parsed from the live log (% of simulated time)
            prog["pct"] = min(100.0, self._tqdm["pct"])
            prog["eta_s"] = self._tqdm["eta_s"]
            prog["source"] = "log"
        elif prog["bytes"] and prog["est_bytes"]:
            prog["pct"] = min(99.0, prog["bytes"] / prog["est_bytes"] * 100.0)
            self._brate.append((now, prog["bytes"]))
            if len(self._brate) >= 2:
                (t0, b0), (t1, b1) = self._brate[0], self._brate[-1]
                if b1 > b0:
                    prog["eta_s"] = max(0.0, (prog["est_bytes"] - b1) * (t1 - t0) / (b1 - b0))
        self._probe = {"at": now, "progress": prog}
        return prog

    def status(self):
        running = self.proc is not None and self.proc.poll() is None
        if not running and self.state == "running":
            time.sleep(0.1)               # reader thread is finishing up
        st = {"state": self.state, "returncode": self.returncode,
              "config_path": self.config_path, "result_path": self.result_path,
              "started": self.started,
              "elapsed": (self.ended or time.time()) - self.started if self.started else None,
              "log_length": self.base_offset + len(self.lines)}
        st["progress"] = self._probe_progress() if self.started else None
        if self.state in ("done", "error", "stopped") and st["progress"]:
            st["progress"]["eta_s"] = None
        return st

    def get_log(self, offset):
        offset = max(int(offset), self.base_offset)
        rel = offset - self.base_offset
        lines = self.lines[rel:]
        return {"offset": offset + len(lines), "lines": lines,
                "truncated": offset > 0 and rel == 0 and self.base_offset > 0}


MANAGER = RunManager()
