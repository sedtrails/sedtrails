"""Seeding tab backend: WYSIWYG preview of a population's seed locations.

Runs the population dict through sedtrails' own PopulationConfig + seeding
strategies (particle_seeder.py), so the preview matches the simulation exactly
— including numpy RNG seeds for the random strategy.
"""

import copy
from pathlib import Path

from .util import resolve_input_path

MAX_POINTS = 50_000


def _resolve_paths(strategy_settings, base):
    """Make pol_file / poly / path entries absolute against the config dir."""
    for key in ("pol_file", "poly", "path"):
        v = strategy_settings.get(key)
        if isinstance(v, str) and v:
            cand = resolve_input_path(v, base)
            if cand.exists():
                strategy_settings[key] = str(cand)


def preview(population, base=None, count_only=False):
    """Seed locations for one population dict. Returns
    {n_locations, n_particles, quantity, points?: [[x,y],...], capped}."""
    from sedtrails.particle_tracer.particle_seeder import PopulationConfig

    pop = copy.deepcopy(population or {})
    pop.setdefault("particle_type", "passive")
    seeding = pop.setdefault("seeding", {})
    seeding.setdefault("quantity", 1)
    if not seeding.get("burial_depth"):
        seeding["burial_depth"] = {"constant": 0}
    strategy = seeding.get("strategy") or {}
    if not strategy:
        raise ValueError("no seeding strategy set for this population")
    name = next(iter(strategy))
    if isinstance(strategy[name], dict):
        _resolve_paths(strategy[name], base)

    config = PopulationConfig(population_config=pop)
    # dispatch exactly like ParticleFactory.create_particles does
    from sedtrails.particle_tracer.particle_seeder import (
        FilePointsStrategy, GridStrategy, PointStrategy, RandomStrategy, TransectStrategy)
    STRATEGY_MAP = {"point": PointStrategy(), "random": RandomStrategy(),
                    "grid": GridStrategy(), "transect": TransectStrategy(),
                    "file_points": FilePointsStrategy()}
    if config.strategy.lower() not in STRATEGY_MAP:
        raise ValueError(f"unknown seeding strategy: {config.strategy}")

    positions = STRATEGY_MAP[config.strategy.lower()].seed(config)
    n_loc = len(positions)
    n_particles = sum(int(q) for q, _, _ in positions)
    out = {"n_locations": n_loc, "n_particles": n_particles,
           "quantity": int(config.quantity), "strategy": config.strategy,
           "capped": n_loc > MAX_POINTS}
    if not count_only:
        pts = positions[:MAX_POINTS]
        out["points"] = [[round(float(x), 2), round(float(y), 2)] for _, x, y in pts]
    return out
