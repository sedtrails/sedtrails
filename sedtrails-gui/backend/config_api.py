"""Settings tab backend: schemas, yaml load/save (comment-preserving), validation,
template creation and output-size estimation.

Mirrors sedtrails' own machinery where possible:
- schemas come from the installed sedtrails package (fallback: the repo checkout),
- validation replicates YAMLConfigValidator's Draft 2020-12 registry,
- the size estimate mirrors Simulation._estimate_netcdf_payload_bytes /
  _estimate_output_timesteps (5 coordinate + 6 status fields per particle-slot),
- duration parsing mirrors particle_tracer/timer.py convert_duration_string_to_seconds.
"""

import json
import math
import re
import threading
from pathlib import Path

from .settings import TOOL_DIR
from .util import _log

SCHEMA_FILES = {
    "main": "main.schema.json",
    "population": "population.schema.json",
    "visualization": "visualization.schema.json",
}
_URN = "urn:sedtrails:config:{}"

_lock = threading.RLock()   # RLock: _get_validator holds it while calling get_schemas
_schemas = None          # name -> dict
_validator = None        # jsonschema Draft202012Validator
_ruamel = None           # ruamel.yaml.YAML round-trip instance, or False if unavailable
_open_docs = {}          # abs path -> (mtime, ruamel CommentedMap)

_DTYPE_BYTES = {"float32": 4, "f4": 4, "float64": 8, "f8": 8, "uint8": 1, "u1": 1,
                "int32": 4, "i4": 4}
_COORD_FIELDS = 5        # x, y, z, burial_depth, mixing_depth
_STATUS_FIELDS = 6       # alive, buried, domain, transported, released, mobile
_DUR_RE = re.compile(r"(?:(\d+)D)?\s*(?:(\d+)H)?\s*(?:(\d+)M)?\s*(?:(\d+)S)?")


# ── schemas & validator ────────────────────────────────────────────────────────
def _schema_dir():
    try:
        from importlib.resources import files
        d = files("sedtrails") / "config"
        if (d / SCHEMA_FILES["main"]).is_file():
            return d
    except Exception:
        pass
    d = TOOL_DIR.parent / "sedtrails" / "src" / "sedtrails" / "config"
    if (d / SCHEMA_FILES["main"]).is_file():
        return d
    raise FileNotFoundError(
        "sedtrails config schemas not found — install the sedtrails package in this "
        "environment, or keep the sedtrails repo checkout next to sedtrails-gui")


def get_schemas():
    global _schemas
    with _lock:
        if _schemas is None:
            d = _schema_dir()
            _schemas = {name: json.loads((d / fn).read_text(encoding="utf-8"))
                        for name, fn in SCHEMA_FILES.items()}
    return _schemas


def _get_validator():
    """Draft 2020-12 validator with the same urn registry as YAMLConfigValidator."""
    global _validator
    with _lock:
        if _validator is None:
            import jsonschema
            from referencing import Registry, Resource
            from referencing.jsonschema import DRAFT202012

            schemas = dict(get_schemas())
            registry = Registry()
            for name in ("population", "visualization"):
                registry = registry.with_resource(
                    uri=_URN.format(SCHEMA_FILES[name]),
                    resource=Resource(contents=schemas[name], specification=DRAFT202012))
            _validator = jsonschema.Draft202012Validator(
                schema=schemas["main"], registry=registry)
    return _validator


def _error_entry(err):
    pointer = "/".join(str(p) for p in err.absolute_path)
    msg = err.message
    if err.context:  # oneOf/anyOf: pick the most relevant sub-error
        from jsonschema.exceptions import best_match
        best = best_match(err.context)
        if best is not None:
            sub = "/".join(str(p) for p in best.absolute_path)
            msg = best.message + (f" (at {sub})" if sub and sub != pointer else "")
            if sub:
                pointer = sub
    if len(msg) > 300:
        msg = msg[:300] + "…"
    return {"pointer": pointer, "message": msg}


def validate_config(cfg):
    errors = [_error_entry(e) for e in _get_validator().iter_errors(cfg)]
    return {"valid": not errors, "errors": errors[:50]}


# ── yaml load / save ───────────────────────────────────────────────────────────
def _get_ruamel():
    global _ruamel
    if _ruamel is None:
        try:
            from ruamel.yaml import YAML
            from ruamel.yaml.constructor import RoundTripConstructor

            # keep timestamps as plain strings (same intent as SedtrailsYamlLoader)
            RoundTripConstructor.add_constructor(
                "tag:yaml.org,2002:timestamp",
                lambda self, node: self.construct_scalar(node))
            yml = YAML()  # round-trip mode: preserves comments, order, quoting
            yml.preserve_quotes = True
            yml.width = 4096
            yml.indent(mapping=2, sequence=4, offset=2)   # matches the example configs
            _ruamel = yml
        except ImportError:
            _ruamel = False
    return _ruamel


def _plain(obj):
    """ruamel/pyyaml containers -> plain JSON-able python."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _pyyaml_load(text):
    import yaml

    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_constructor("tag:yaml.org,2002:timestamp",
                            lambda loader, node: loader.construct_scalar(node))
    return yaml.load(text, Loader=_Loader)


def load_config(path):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    text = p.read_text(encoding="utf-8")
    yml = _get_ruamel()
    if yml:
        doc = yml.load(text)
        _open_docs[str(p.resolve())] = doc
        return {"config": _plain(doc) or {}, "roundtrip": "ruamel", "path": str(p.resolve())}
    data = _pyyaml_load(text)
    return {"config": _plain(data) or {}, "roundtrip": "pyyaml", "path": str(p.resolve())}


def _patch(node, new, yml):
    """Recursively apply `new` (plain dict/list) into a ruamel node in place,
    keeping comments/order of everything that did not change."""
    if isinstance(node, dict) and isinstance(new, dict):
        for k in [k for k in node.keys() if k not in new]:
            del node[k]
        for k, v in new.items():
            if k in node and isinstance(node[k], dict) and isinstance(v, dict):
                _patch(node[k], v, yml)
            elif k in node and isinstance(node[k], list) and isinstance(v, list):
                _patch(node[k], v, yml)
            elif k in node and not isinstance(node[k], (dict, list)) and not isinstance(v, (dict, list)):
                if node[k] != v or type(node[k]) is not type(v):
                    node[k] = v
            else:
                node[k] = _to_ruamel(v, yml)
        return
    if isinstance(node, list) and isinstance(new, list):
        common = min(len(node), len(new))
        for i in range(common):
            if isinstance(node[i], (dict, list)) and isinstance(new[i], (dict, list)) \
                    and isinstance(node[i], dict) == isinstance(new[i], dict):
                _patch(node[i], new[i], yml)
            elif node[i] != new[i]:
                node[i] = _to_ruamel(new[i], yml)
        del node[common:len(node)]
        for v in new[common:]:
            node.append(_to_ruamel(v, yml))


def _to_ruamel(v, yml):
    from ruamel.yaml.comments import CommentedMap, CommentedSeq

    if isinstance(v, dict):
        m = CommentedMap()
        for k, x in v.items():
            m[k] = _to_ruamel(x, yml)
        return m
    if isinstance(v, list):
        s = CommentedSeq()
        for x in v:
            s.append(_to_ruamel(x, yml))
        return s
    return v


def _schema_key_order(schema, root):
    """Property order of an object schema (resolving one $ref level)."""
    if "$ref" in schema:
        ref = schema["$ref"]
        if ref.startswith("#/"):
            target = root
            for part in ref[2:].split("/"):
                target = target.get(part, {})
            schema = target
        else:
            for name, fn in SCHEMA_FILES.items():
                if ref == _URN.format(fn):
                    schema = get_schemas()[name]
    return schema


def _order_by_schema(data, schema, root):
    """Recursively order dict keys by schema property order (pyyaml fallback path)."""
    schema = _schema_key_order(schema, root)
    if schema.get("$id"):
        root = schema
    props = schema.get("properties", {})
    if isinstance(data, dict):
        ordered = {}
        for k in list(props.keys()) + [k for k in data if k not in props]:
            if k in data:
                ordered[k] = _order_by_schema(data[k], props.get(k, {}), root)
        return ordered
    if isinstance(data, list):
        item_schema = schema.get("items", {})
        return [_order_by_schema(v, item_schema, root) for v in data]
    return data


def save_config(path, cfg):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    validation = validate_config(cfg)
    yml = _get_ruamel()
    if yml:
        key = str(p.resolve())
        doc = _open_docs.get(key)
        if doc is None and p.is_file():
            doc = yml.load(p.read_text(encoding="utf-8"))
        if doc is None:
            doc = _to_ruamel(cfg, yml)
        else:
            _patch(doc, cfg, yml)
        with open(p, "w", encoding="utf-8") as f:
            yml.dump(doc, f)
        _open_docs[key] = doc
        mode = "ruamel"
    else:
        import yaml as pyyaml
        main = get_schemas()["main"]
        ordered = _order_by_schema(cfg, main, main)
        with open(p, "w", encoding="utf-8") as f:
            pyyaml.safe_dump(ordered, f, default_flow_style=False, sort_keys=False,
                             indent=2, allow_unicode=True)
        mode = "pyyaml"
    _log(f"config saved ({mode}) -> {p}")
    return {"ok": True, "path": str(p.resolve()), "roundtrip": mode,
            "validation": validation}


def create_template():
    """Defaults-applied starter config with one population (uses sedtrails' own
    defaults applier when importable, else a static minimal template)."""
    skeleton = {
        "general": {},
        "inputs": {"data": ""},
        "domain": {"subset_x": "0:100000", "subset_y": "0:100000"},
        "time": {"timestep": "30S"},
        "physics": {},
        "particles": {"populations": [{}]},
        "outputs": {},
        "compute": {},
    }
    try:
        from sedtrails.application_interfaces.validator import YAMLConfigValidator
        v = YAMLConfigValidator()
        skeleton = v._apply_defaults(v.schema_content, skeleton)
    except Exception as e:
        _log(f"template: sedtrails defaults applier unavailable ({e}); using skeleton")
    pop = skeleton["particles"]["populations"][0]
    pop.setdefault("name", "population_1")
    pop.setdefault("particle_type", "passive")
    if not pop.get("characteristics"):
        pop["characteristics"] = {"diffusion_coefficient": 0.0}
    seeding = pop.setdefault("seeding", {})
    seeding.setdefault("quantity", 1)
    if not seeding.get("strategy"):   # defaults applier leaves an empty {} here
        seeding["strategy"] = {"random": {"bbox": "0,0 1000,1000",
                                          "nlocations": 100, "seed": 42}}
    if not pop.get("barriers"):
        pop.pop("barriers", None)
    return {"config": skeleton}


# ── size estimate ──────────────────────────────────────────────────────────────
def duration_seconds(s):
    """'3D 2H1M3S' -> seconds; None if unparsable/empty."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    m = _DUR_RE.fullmatch(str(s).strip())
    if not m or not any(m.groups()):
        return None
    d, h, mi, sec = (int(g) if g else 0 for g in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + sec


def _population_count(pop):
    """(n_particles, exact) best-effort mirror of ParticleFactory semantics:
    total = quantity x number of seed locations."""
    seeding = pop.get("seeding", {}) or {}
    q = seeding.get("quantity", 1) or 1
    strat = seeding.get("strategy", {}) or {}
    if "point" in strat:
        locs = strat["point"].get("locations")
        return (q * len(locs), True) if locs else (None, False)
    if "transect" in strat:
        t = strat["transect"]
        segs = t.get("segments")
        k = t.get("k", 100.0)
        return (int(q * k * len(segs)), True) if segs else (None, False)
    if "random" in strat:
        n = strat["random"].get("nlocations", 1)
        return (int(q * n), True)
    if "grid" in strat:
        g = strat["grid"]
        sep = g.get("separation", {}) or {}
        dx, dy = sep.get("dx", 100.0) or 100.0, sep.get("dy", 100.0) or 100.0
        bbox = g.get("bbox")
        if bbox:
            try:
                parts = [float(v) for v in str(bbox).replace(",", " ").split()]
                nx = max(1, int((parts[2] - parts[0]) / dx) + 1)
                ny = max(1, int((parts[3] - parts[1]) / dy) + 1)
                return (int(q * nx * ny), False)   # poly mask may trim this
            except Exception:
                return (None, False)
        return (None, False)
    if "file_points" in strat:
        path = strat["file_points"].get("path")
        if path and Path(path).is_file():
            try:
                n = sum(1 for _ in open(path, "r", encoding="latin-1"))
                if strat["file_points"].get("has_header", True):
                    n -= 1
                stride = strat["file_points"].get("stride", 1) or 1
                return (int(q * max(0, n) / stride), False)
            except OSError:
                return (None, False)
        return (None, False)
    return (None, False)


def estimate(cfg):
    """Mirror of Simulation._estimate_netcdf_payload_bytes/_estimate_output_timesteps."""
    outputs = cfg.get("outputs", {}) or {}
    ncopt = outputs.get("netcdf", {}) or {}
    coord_b = _DTYPE_BYTES.get(str(ncopt.get("coordinate_dtype", "float32")), 4)
    stat_b = _DTYPE_BYTES.get(str(ncopt.get("status_dtype", "uint8")), 1)
    per_slot = _COORD_FIELDS * coord_b + _STATUS_FIELDS * stat_b

    dur = duration_seconds((cfg.get("time", {}) or {}).get("duration"))
    si = duration_seconds(outputs.get("save_interval", "1H")) or 3600
    if outputs.get("store_tracks", True) is False:
        slots = 1
    elif dur is None or si <= 0:
        slots = None
    else:
        n = int(math.floor(dur / si))
        slots = 1 + n + (1 if n * si < dur - 1e-9 else 0)

    pops, total, exact = [], 0, True
    for pop in (cfg.get("particles", {}) or {}).get("populations", []) or []:
        n, ex = _population_count(pop or {})
        pops.append({"name": (pop or {}).get("name", "?"), "n": n, "exact": ex})
        if n is None:
            exact = False
        else:
            total += n
            exact = exact and ex

    payload = total * slots * per_slot if (slots and total) else None
    threshold = int(ncopt.get("compression_auto_threshold_mb", 1024)) * 1024 * 1024
    compressed = None
    comp = ncopt.get("compression", "auto")
    if payload is not None:
        compressed = bool(comp) if isinstance(comp, bool) else payload >= threshold
    return {"particles": total or None, "particles_exact": exact, "populations": pops,
            "slots": slots, "duration_s": dur, "save_interval_s": si,
            "bytes_per_particle_slot": per_slot, "bytes": payload,
            "compression": compressed}
