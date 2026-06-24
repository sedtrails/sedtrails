"""Utilities for reading Deltares/Tekal polygon files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def read_tekal_polygons(pol_files: str | Path | Iterable[str | Path] | None) -> list[np.ndarray]:
    """Read one or more Tekal ``.pol`` files as polygon coordinate arrays.

    Parameters
    ----------
    pol_files : str, pathlib.Path, Iterable[str or pathlib.Path], or None
        Tekal polygon file path or iterable of file paths. ``None`` returns an
        empty polygon list.

    Returns
    -------
    list[np.ndarray]
        Polygon coordinate arrays with shape ``(n_vertices, 2)``. Repeated
        closing points are removed.

    Raises
    ------
    FileNotFoundError
        If any input file does not exist.
    ValueError
        If a Tekal block is missing a size line, declares an invalid row count,
        ends before the declared coordinates, or contains a coordinate row with
        fewer than two columns.

    Notes
    -----
    Each polygon block is expected to contain a block name, a size line whose
    first value is the number of coordinate rows, followed by that many rows
    with at least x and y columns. Empty lines and lines starting with ``*`` are
    ignored.
    """

    if pol_files is None:
        return []

    if isinstance(pol_files, (str, Path)):
        paths = [Path(pol_files)]
    else:
        paths = [Path(pol_file) for pol_file in pol_files]

    polygons: list[np.ndarray] = []
    for path in paths:
        polygons.extend(_read_tekal_polygon_file(path))
    return polygons


def _read_tekal_polygon_file(path: Path) -> list[np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f'Tekal polygon file not found: {path}')

    lines = _meaningful_lines(path)
    polygons: list[np.ndarray] = []
    index = 0
    while index < len(lines):
        block_name = lines[index]
        index += 1
        if index >= len(lines):
            raise ValueError(f"Tekal block {block_name!r} in {path} is missing a size line")

        size_tokens = lines[index].replace(',', ' ').split()
        index += 1
        if not size_tokens:
            raise ValueError(f"Tekal block {block_name!r} in {path} has an empty size line")

        try:
            n_rows = int(float(size_tokens[0]))
        except ValueError as exc:
            raise ValueError(f"Tekal block {block_name!r} in {path} has invalid row count {size_tokens[0]!r}") from exc

        if n_rows < 0:
            raise ValueError(f"Tekal block {block_name!r} in {path} has negative row count {n_rows}")
        if index + n_rows > len(lines):
            raise ValueError(f"Tekal block {block_name!r} in {path} declares {n_rows} rows but file ended early")

        coords = []
        for _ in range(n_rows):
            coord_tokens = lines[index].replace(',', ' ').split()
            index += 1
            if len(coord_tokens) < 2:
                raise ValueError(f"Tekal block {block_name!r} in {path} contains a coordinate row with <2 columns")
            coords.append((float(coord_tokens[0]), float(coord_tokens[1])))

        if len(coords) >= 3:
            polygon = np.asarray(coords, dtype=float)
            polygons.append(_drop_repeated_closing_point(polygon))

    return polygons


def _meaningful_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip() and not line.lstrip().startswith('*')
    ]


def _drop_repeated_closing_point(polygon: np.ndarray) -> np.ndarray:
    if polygon.shape[0] > 1 and np.allclose(polygon[0], polygon[-1]):
        return polygon[:-1]
    return polygon
