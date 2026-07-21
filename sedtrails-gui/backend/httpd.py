"""HTTP server: static frontend files + JSON/binary API routes.

Extracted verbatim from sedtrails-viewer/sedtrails_viewer.py (Phase 0 split).
"""

import json
import os
import sys
import threading
import traceback
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import config_api, connectivity, forcing, seeding_api
from .run_manager import MANAGER as run_manager
from .bundle import (_classify_progress, _polygons_lock, _polygons_path,
                     _registry_lock, classify, import_polygon_dir, import_run,
                     load_poly_txt, load_registry, registry_find, save_registry)
from .settings import WEB_DIR
from .util import (_browse, _jobs, _jobs_lock, _log, _read_json, _video_jobs,
                   _write_json, native_pick, native_save)


class GuiHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet: only log errors
        if args and str(args[1] if len(args) > 1 else "").startswith(("4", "5")):
            _log(f"http: {fmt % args}")

    # -- helpers --
    def _send(self, code, body, ctype="application/json", cache=False, headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (ConnectionAbortedError, BrokenPipeError):
            pass

    def _send_file(self, path, ctype, cache=True):
        try:
            size = os.path.getsize(path)
        except OSError:
            return self._send(404, {"error": "not found"})
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "max-age=60" if cache else "no-store")
        self.end_headers()
        try:
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except (ConnectionAbortedError, BrokenPipeError):
            pass

    def _json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    # -- routing --
    def do_GET(self):
        try:
            parsed = urllib.parse.urlparse(self.path)
            route = parsed.path
            qs = urllib.parse.parse_qs(parsed.query)

            if route in ("/", "/index.html"):
                return self._send_file(WEB_DIR / "index.html", "text/html; charset=utf-8", cache=False)
            if route.startswith(("/css/", "/js/")):
                fpath = (WEB_DIR / route.lstrip("/")).resolve()
                if WEB_DIR.resolve() not in fpath.parents or not fpath.is_file():
                    return self._send(404, {"error": "not found"})
                ctype = ("text/css; charset=utf-8" if fpath.suffix == ".css"
                         else "text/javascript; charset=utf-8")
                return self._send_file(fpath, ctype, cache=False)
            if route == "/api/registry":
                return self._send(200, load_registry())
            if route == "/api/schema":
                return self._send(200, config_api.get_schemas())
            if route == "/api/run/status":
                return self._send(200, run_manager.status())
            if route == "/api/run/log":
                return self._send(200, run_manager.get_log(qs.get("offset", ["0"])[0]))
            if route.startswith("/api/forcing/"):
                parts = route.split("/")          # /api/forcing/<id>/<what>
                if len(parts) == 5:
                    fid, what = parts[3], parts[4]
                    if what == "mesh":
                        return self._send(200, forcing.mesh_payload(fid),
                                          ctype="application/octet-stream")
                    if what == "field":
                        data, rng = forcing.field_payload(
                            fid, qs.get("var", [""])[0], qs.get("t", ["0"])[0])
                        return self._send(200, data, ctype="application/octet-stream",
                                          headers={"X-Data-Range": rng})
                    if what == "vector":
                        data, p98 = forcing.vector_payload(
                            fid, qs.get("x", [""])[0], qs.get("y", [""])[0],
                            qs.get("t", ["0"])[0])
                        return self._send(200, data, ctype="application/octet-stream",
                                          headers={"X-Mag-P98": p98})
                return self._send(404, {"error": "not found"})
            if route == "/api/browse":
                return self._send(200, _browse(qs.get("dir", [""])[0]))
            if route == "/api/pickfile":
                try:
                    paths = native_pick(qs.get("kind", ["nc"])[0],
                                        qs.get("dir", [""])[0] or None)
                    return self._send(200, {"paths": paths})
                except Exception as e:
                    return self._send(200, {"paths": None, "error": str(e)})
            if route == "/api/picksave":
                try:
                    path = native_save(qs.get("ext", [".png"])[0],
                                       qs.get("name", [""])[0] or None,
                                       qs.get("dir", [""])[0] or None)
                    return self._send(200, {"path": path})
                except Exception as e:
                    return self._send(200, {"path": None, "error": str(e)})
            if route.startswith("/api/job/"):
                with _jobs_lock:
                    job = _jobs.get(route.rsplit("/", 1)[1])
                return self._send(200, job or {"status": "unknown"})
            if route == "/api/polygons":
                with _polygons_lock:
                    data = _read_json(_polygons_path(), {"polygons": []})
                    if "groups" in data:   # migrate the old grouped format
                        flat = []
                        for g in data.get("groups", []):
                            for p in g.get("polygons", []):
                                if not any(q["name"] == p["name"] for q in flat):
                                    flat.append({k: p[k] for k in ("name", "verts", "color") if k in p})
                        data = {"polygons": flat}
                        _write_json(_polygons_path(), data)
                    return self._send(200, data)
            if route == "/api/classify/progress":
                return self._send(200, _classify_progress.get(qs.get("bundle", [""])[0], {}))
            if route.startswith("/data/"):
                parts = route.split("/")
                if len(parts) == 4:
                    sim = registry_find(urllib.parse.unquote(parts[2]))
                    fname = os.path.basename(urllib.parse.unquote(parts[3]))
                    if sim:
                        fpath = Path(sim["bundle_dir"]) / fname
                        ctype = ("application/json" if fname.endswith(".json")
                                 else "image/png" if fname.endswith(".png")
                                 else "application/octet-stream")
                        return self._send_file(fpath, ctype, cache=not fname.endswith(".json"))
                return self._send(404, {"error": "not found"})
            return self._send(404, {"error": "not found"})
        except Exception as e:
            traceback.print_exc()
            return self._send(500, {"error": str(e)})

    def _raw_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length)

    def do_POST(self):
        try:
            parsed = urllib.parse.urlparse(self.path)
            route = parsed.path
            qs = urllib.parse.parse_qs(parsed.query)

            # raw-binary routes (no JSON body)
            if route == "/api/savefile":
                path = qs.get("path", [""])[0]
                if not path or Path(path).suffix.lower() not in (".png", ".mp4", ".webm", ".txt", ".json"):
                    return self._send(400, {"error": "invalid save path"})
                data = self._raw_body()
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_bytes(data)
                _log(f"saved {len(data)/1e6:.1f} MB -> {path}")
                return self._send(200, {"ok": True, "bytes": len(data)})
            if route == "/api/video/frame":
                job = _video_jobs.get(qs.get("job", [""])[0])
                if not job:
                    return self._send(400, {"error": "unknown video job"})
                i = int(qs.get("i", ["0"])[0])
                (Path(job["dir"]) / f"{i:06d}.png").write_bytes(self._raw_body())
                job["frames"] = max(job["frames"], i + 1)
                return self._send(200, {"ok": True})

            body = self._json_body()

            if route == "/api/connectivity/compute":
                return self._send(200, connectivity.compute(
                    body["bundle_id"], body.get("polygons", []),
                    body.get("mode", "start_end"), body.get("normalize", "count")))
            if route == "/api/polygons/auto":
                return self._send(200, connectivity.auto_polygons(
                    body["forcing_id"], body.get("n", 12), body.get("var")))
            if route == "/api/run/start":
                try:
                    return self._send(200, run_manager.start(body["config_path"]))
                except RuntimeError as e:
                    return self._send(409, {"error": str(e)})
            if route == "/api/run/stop":
                return self._send(200, run_manager.stop())
            if route == "/api/seeding/preview":
                return self._send(200, seeding_api.preview(
                    body.get("population", {}), body.get("base"),
                    bool(body.get("count_only"))))
            if route == "/api/forcing/open":
                return self._send(200, forcing.open_forcing(body["path"],
                                                            body.get("base")))
            if route == "/api/config/load":
                return self._send(200, config_api.load_config(body["path"]))
            if route == "/api/config/save":
                return self._send(200, config_api.save_config(body["path"], body["config"]))
            if route == "/api/config/validate":
                return self._send(200, config_api.validate_config(body.get("config", {})))
            if route == "/api/config/template":
                return self._send(200, config_api.create_template())
            if route == "/api/config/estimate":
                return self._send(200, config_api.estimate(body.get("config", {})))
            if route == "/api/config/result":
                # where a run of this config writes its output (mirrors run_manager.start)
                cfg_path = Path(body.get("config_path", "")).resolve()
                if not cfg_path.is_file():
                    return self._send(400, {"error": f"config not found: {cfg_path}"})
                loaded = config_api.load_config(str(cfg_path))
                out_dir = (loaded["config"].get("outputs", {}) or {}).get("directory", "./output")
                rp = (cfg_path.parent / out_dir).resolve() / "sedtrails_results.nc"
                out = {"path": str(rp), "exists": rp.is_file()}
                if out["exists"]:
                    st = rp.stat()          # matches bundle source_signature
                    out["size"] = int(st.st_size)
                    out["mtime"] = int(st.st_mtime)
                return self._send(200, out)

            if route == "/api/import":
                nc_path = body.get("nc_path", "")
                if not os.path.isfile(nc_path):
                    return self._send(400, {"error": f"file not found: {nc_path}"})
                job_id = uuid.uuid4().hex[:12]
                job = {"id": job_id, "status": "running", "progress": 0.0,
                       "message": "starting", "bundle": None}
                with _jobs_lock:
                    _jobs[job_id] = job

                def work():
                    try:
                        meta = import_run(nc_path, body.get("forcing_path") or None,
                                          body.get("name") or None, job,
                                          body.get("grid_path") or None)
                        job["bundle"] = meta["id"]
                        job["status"] = "done"
                    except MemoryError:
                        traceback.print_exc()
                        job["status"] = "error"
                        job["message"] = ("not enough free memory for the import — close "
                                          "other applications and retry, or lower SLAB_STEPS "
                                          "in backend/settings.py")
                    except Exception as e:
                        traceback.print_exc()
                        job["status"] = "error"
                        job["message"] = f"{type(e).__name__}: {e}"

                threading.Thread(target=work, daemon=True).start()
                return self._send(200, {"job_id": job_id})

            if route == "/api/polygons":
                with _polygons_lock:
                    _write_json(_polygons_path(), {"polygons": body.get("polygons", [])})
                return self._send(200, {"ok": True})

            if route == "/api/polygons/import":
                path = body.get("path", "")
                if os.path.isdir(path):
                    polys = import_polygon_dir(path, body.get("pattern", "*.txt"))
                elif os.path.isfile(path):
                    verts = load_poly_txt(path)
                    polys = ([{"name": Path(path).stem, "verts": verts}]
                             if len(verts) >= 3 else [])
                else:
                    return self._send(400, {"error": f"path not found: {path}"})
                return self._send(200, {"polygons": polys})

            if route == "/api/classify":
                payload = classify(body["bundle_id"], body.get("polygons", []))
                return self._send(200, payload, ctype="application/octet-stream")

            if route == "/api/video/start":
                path = body.get("path", "")
                if not path or not path.lower().endswith(".mp4"):
                    return self._send(400, {"error": "choose an .mp4 output path first"})
                import shutil as _sh
                import tempfile
                if not _sh.which("ffmpeg"):
                    return self._send(400, {"error": "ffmpeg not found on PATH — "
                                            "install ffmpeg to export MP4"})
                job_id = uuid.uuid4().hex[:12]
                _video_jobs[job_id] = {"dir": tempfile.mkdtemp(prefix="stviewer_vid_"),
                                       "path": path, "fps": int(body.get("fps", 12)),
                                       "frames": 0}
                return self._send(200, {"job": job_id})

            if route == "/api/video/finish":
                import shutil as _sh
                import subprocess
                job = _video_jobs.pop(body.get("job", ""), None)
                if not job:
                    return self._send(400, {"error": "unknown video job"})
                try:
                    if job["frames"] == 0:
                        raise ValueError("no frames received")
                    cmd = ["ffmpeg", "-y", "-framerate", str(job["fps"]),
                           "-i", str(Path(job["dir"]) / "%06d.png"),
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                           job["path"]]
                    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                    if res.returncode != 0:
                        raise RuntimeError("ffmpeg failed: " + res.stderr[-400:])
                    _log(f"video: {job['frames']} frames @ {job['fps']} fps -> {job['path']}")
                    return self._send(200, {"ok": True, "frames": job["frames"]})
                except Exception as e:
                    return self._send(500, {"error": str(e)})
                finally:
                    _sh.rmtree(job["dir"], ignore_errors=True)

            if route == "/api/forget":
                with _registry_lock:
                    reg = load_registry()
                    reg["simulations"] = [s for s in reg["simulations"]
                                          if s["id"] != body.get("bundle_id")]
                    hid = reg.setdefault("hidden", [])
                    if body.get("bundle_id") and body["bundle_id"] not in hid:
                        hid.append(body["bundle_id"])
                    save_registry(reg)
                return self._send(200, {"ok": True})

            return self._send(404, {"error": "not found"})
        except Exception as e:
            traceback.print_exc()
            return self._send(500, {"error": str(e)})


class QuietServer(ThreadingHTTPServer):
    # On Windows, SO_REUSEADDR lets a second instance silently share the port
    # with a stale one (requests then hit an unpredictable server). Fail
    # instead — main() moves to the next port.
    allow_reuse_address = False

    def handle_error(self, request, client_address):
        # Browsers slam their keep-alive sockets on tab close/reload — routine.
        exc = sys.exception()
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)
