"""Create a UGRID NetCDF mesh file from a Delft3D FM map file.

The script extracts node coordinates and face-node connectivity from common
Delft3D Flexible Mesh NetCDF conventions and writes a small UGRID-1.0 file.
If the input only contains point coordinates such as ``net_xcc/net_ycc``, it
creates a Delaunay triangular UGRID from those points.

Examples
--------
Create a mesh-only UGRID file:

    python examples/create_ugrid_from_fm_map.py dflowfm_map.nc mesh_ugrid.nc

Also copy face/node data variables from the map file:

    python examples/create_ugrid_from_fm_map.py dflowfm_map.nc mesh_ugrid.nc --include-data
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr
from scipy.spatial import Delaunay


NODE_X_CANDIDATES = (
    'mesh2d_node_x',
    'Mesh2_node_x',
    'Mesh2d_node_x',
    'NetNode_x',
    'net_node_x',
)
NODE_Y_CANDIDATES = (
    'mesh2d_node_y',
    'Mesh2_node_y',
    'Mesh2d_node_y',
    'NetNode_y',
    'net_node_y',
)
FACE_NODE_CANDIDATES = (
    'mesh2d_face_nodes',
    'Mesh2_face_nodes',
    'Mesh2d_face_nodes',
    'NetElemNode',
    'net_elem_node',
    'net_element_node',
)
FACE_X_CANDIDATES = (
    'mesh2d_face_x',
    'Mesh2_face_x',
    'Mesh2d_face_x',
    'FlowElem_xcc',
    'flow_elem_xcc',
    'net_xcc',
)
FACE_Y_CANDIDATES = (
    'mesh2d_face_y',
    'Mesh2_face_y',
    'Mesh2d_face_y',
    'FlowElem_ycc',
    'flow_elem_ycc',
    'net_ycc',
)
POINT_X_CANDIDATES = (
    'net_xcc',
    'mesh2d_face_x',
    'Mesh2_face_x',
    'Mesh2d_face_x',
    'FlowElem_xcc',
    'flow_elem_xcc',
    'mesh2d_node_x',
    'NetNode_x',
)
POINT_Y_CANDIDATES = (
    'net_ycc',
    'mesh2d_face_y',
    'Mesh2_face_y',
    'Mesh2d_face_y',
    'FlowElem_ycc',
    'flow_elem_ycc',
    'mesh2d_node_y',
    'NetNode_y',
)
MESH_TOPOLOGY_CANDIDATES = ('mesh2d', 'Mesh2', 'Mesh2d')
GEOMETRY_VARIABLES = set(
    NODE_X_CANDIDATES
    + NODE_Y_CANDIDATES
    + FACE_NODE_CANDIDATES
    + FACE_X_CANDIDATES
    + FACE_Y_CANDIDATES
    + POINT_X_CANDIDATES
    + POINT_Y_CANDIDATES
    + MESH_TOPOLOGY_CANDIDATES
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_map', type=Path, help='Delft3D FM map NetCDF file')
    parser.add_argument('output_ugrid', type=Path, help='Output UGRID NetCDF file')
    parser.add_argument(
        '--include-data',
        action='store_true',
        help='Copy data variables located on mesh faces or nodes into the UGRID file',
    )
    parser.add_argument(
        '--variables',
        nargs='*',
        default=None,
        help='Optional variable-name allowlist used with --include-data',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Replace output file if it already exists',
    )
    parser.add_argument(
        '--no-point-triangulation',
        action='store_true',
        help='Fail instead of creating a Delaunay UGRID when only point coordinates are present',
    )
    args = parser.parse_args()

    create_ugrid_from_fm_map(
        input_map=args.input_map,
        output_ugrid=args.output_ugrid,
        include_data=args.include_data,
        variables=args.variables,
        overwrite=args.overwrite,
        triangulate_points=not args.no_point_triangulation,
    )


def create_ugrid_from_fm_map(
    input_map: Path,
    output_ugrid: Path,
    include_data: bool = False,
    variables: Iterable[str] | None = None,
    overwrite: bool = False,
    triangulate_points: bool = True,
) -> None:
    """Extract FM mesh geometry and write a UGRID-1.0 NetCDF file."""

    if not input_map.is_file():
        raise FileNotFoundError(f'Input FM map file not found: {input_map}')
    if output_ugrid.exists() and not overwrite:
        raise FileExistsError(f'Output file already exists: {output_ugrid}. Use --overwrite to replace it.')

    with xr.open_dataset(input_map, decode_times=False, decode_timedelta=False) as source:
        mesh = extract_mesh(source, triangulate_points=triangulate_points)
        output = build_ugrid_dataset(source, mesh, include_data=include_data, variables=variables)
        output_ugrid.parent.mkdir(parents=True, exist_ok=True)
        encoding = {
            'mesh2d_face_nodes': {
                '_FillValue': -1,
                'dtype': _netcdf_int_dtype(mesh.face_nodes),
            }
        }
        output.to_netcdf(output_ugrid, encoding=encoding)

    print(
        f'Wrote {output_ugrid} '
        f'({mesh.node_x.size} nodes, {mesh.face_nodes.shape[0]} faces, max {mesh.face_nodes.shape[1]} nodes/face)'
    )


class Mesh:
    """Container for normalized UGRID geometry arrays."""

    def __init__(
        self,
        node_x: np.ndarray,
        node_y: np.ndarray,
        face_nodes: np.ndarray,
        face_x: np.ndarray,
        face_y: np.ndarray,
        source_face_dim: str,
        source_node_dim: str,
        generated_connectivity: bool = False,
    ) -> None:
        self.node_x = node_x
        self.node_y = node_y
        self.face_nodes = face_nodes
        self.face_x = face_x
        self.face_y = face_y
        self.source_face_dim = source_face_dim
        self.source_node_dim = source_node_dim
        self.generated_connectivity = generated_connectivity


def extract_mesh(source: xr.Dataset, triangulate_points: bool = True) -> Mesh:
    """Read and normalize Delft3D FM mesh arrays from a source dataset."""

    node_x_var = _optional_var(source, NODE_X_CANDIDATES)
    node_y_var = _optional_var(source, NODE_Y_CANDIDATES)
    face_nodes_var = _optional_var(source, FACE_NODE_CANDIDATES)
    if node_x_var is None or node_y_var is None or face_nodes_var is None:
        return extract_point_cloud_mesh(source, triangulate_points=triangulate_points)

    node_x = _as_1d_float(node_x_var, 'node x-coordinate')
    node_y = _as_1d_float(node_y_var, 'node y-coordinate')
    if node_x.shape != node_y.shape:
        raise ValueError(f'Node x/y coordinate shapes differ: {node_x.shape} != {node_y.shape}')

    face_nodes = normalize_face_nodes(face_nodes_var, node_count=node_x.size)
    face_x = _optional_1d(source, FACE_X_CANDIDATES)
    face_y = _optional_1d(source, FACE_Y_CANDIDATES)
    if face_x is None or face_y is None or face_x.shape[0] != face_nodes.shape[0] or face_y.shape[0] != face_nodes.shape[0]:
        face_x, face_y = compute_face_centroids(node_x, node_y, face_nodes)

    return Mesh(
        node_x=node_x,
        node_y=node_y,
        face_nodes=face_nodes,
        face_x=np.asarray(face_x, dtype=np.float64),
        face_y=np.asarray(face_y, dtype=np.float64),
        source_face_dim=face_nodes_var.dims[0],
        source_node_dim=node_x_var.dims[0],
    )


def extract_point_cloud_mesh(source: xr.Dataset, triangulate_points: bool = True) -> Mesh:
    """Create a triangular UGRID mesh from point coordinates when connectivity is absent."""

    point_x_var = _required_var(source, POINT_X_CANDIDATES, 'point x-coordinate')
    point_y_var = _required_var(source, POINT_Y_CANDIDATES, 'point y-coordinate')
    point_x = _as_1d_float(point_x_var, 'point x-coordinate')
    point_y = _as_1d_float(point_y_var, 'point y-coordinate')
    if point_x.shape != point_y.shape:
        raise ValueError(f'Point x/y coordinate shapes differ: {point_x.shape} != {point_y.shape}')
    if point_x.size < 3:
        raise ValueError('At least three point coordinates are required to triangulate a UGRID mesh')
    if not triangulate_points:
        raise KeyError(
            'No source face-node connectivity found. Re-run without --no-point-triangulation to create a '
            'Delaunay UGRID from point coordinates.'
        )

    points = np.column_stack((point_x, point_y))
    finite = np.isfinite(points).all(axis=1)
    if not np.all(finite):
        raise ValueError('Point coordinates must be finite for Delaunay triangulation')

    face_nodes = np.asarray(Delaunay(points).simplices, dtype=np.int64)
    return Mesh(
        node_x=point_x,
        node_y=point_y,
        face_nodes=face_nodes,
        face_x=np.nanmean(point_x[face_nodes], axis=1),
        face_y=np.nanmean(point_y[face_nodes], axis=1),
        source_face_dim='__generated_delaunay_faces',
        source_node_dim=point_x_var.dims[0],
        generated_connectivity=True,
    )


def build_ugrid_dataset(
    source: xr.Dataset,
    mesh: Mesh,
    include_data: bool = False,
    variables: Iterable[str] | None = None,
) -> xr.Dataset:
    """Build an xarray Dataset following UGRID-1.0 naming and attributes."""

    allowed = None if variables is None else set(variables)
    data_vars: dict[str, tuple] = {
        'mesh2d': (
            (),
            np.int32(0),
            {
                'cf_role': 'mesh_topology',
                'long_name': 'Topology data of 2D mesh',
                'topology_dimension': 2,
                'node_coordinates': 'mesh2d_node_x mesh2d_node_y',
                'face_node_connectivity': 'mesh2d_face_nodes',
                'face_coordinates': 'mesh2d_face_x mesh2d_face_y',
            },
        ),
        'mesh2d_node_x': (
            ('mesh2d_nNodes',),
            mesh.node_x,
            _coordinate_attrs(_first_existing_attrs(source, NODE_X_CANDIDATES), 'x'),
        ),
        'mesh2d_node_y': (
            ('mesh2d_nNodes',),
            mesh.node_y,
            _coordinate_attrs(_first_existing_attrs(source, NODE_Y_CANDIDATES), 'y'),
        ),
        'mesh2d_face_nodes': (
            ('mesh2d_nFaces', 'mesh2d_nMax_face_nodes'),
            mesh.face_nodes,
            {
                'cf_role': 'face_node_connectivity',
                'long_name': 'Maps every face to its corner nodes',
                'start_index': 0,
            },
        ),
        'mesh2d_face_x': (
            ('mesh2d_nFaces',),
            mesh.face_x,
            _coordinate_attrs(_first_existing_attrs(source, FACE_X_CANDIDATES), 'x'),
        ),
        'mesh2d_face_y': (
            ('mesh2d_nFaces',),
            mesh.face_y,
            _coordinate_attrs(_first_existing_attrs(source, FACE_Y_CANDIDATES), 'y'),
        ),
    }

    if include_data:
        for name, variable in source.data_vars.items():
            if name in GEOMETRY_VARIABLES:
                continue
            if allowed is not None and name not in allowed:
                continue
            copied = _copy_located_variable(name, variable, mesh)
            if copied is not None:
                data_vars[name] = copied

    output = xr.Dataset(data_vars=data_vars, attrs=_global_attrs(source))
    if mesh.generated_connectivity:
        output.attrs['mesh_generation'] = (
            'face_node_connectivity was not present in the source file; '
            'mesh2d_face_nodes was generated with scipy.spatial.Delaunay from point coordinates'
        )
    output = _copy_nonspatial_coordinates(source, output, mesh)
    _copy_crs_variables(source, output)
    return output


def normalize_face_nodes(face_nodes_var: xr.DataArray, node_count: int) -> np.ndarray:
    """Convert source face-node connectivity to zero-based, -1-filled UGRID connectivity."""

    raw = np.asarray(face_nodes_var.values)
    if raw.ndim != 2:
        raise ValueError(f'Face-node connectivity must be 2-D, got shape {raw.shape}')
    if raw.shape[0] <= 8 and raw.shape[1] > 8:
        raw = raw.T

    raw_float = raw.astype(np.float64)
    valid = np.isfinite(raw_float)
    fill_value = _variable_fill_value(face_nodes_var)
    if fill_value is not None and np.isfinite(float(fill_value)):
        valid &= raw_float != float(fill_value)

    start_index = _variable_start_index(face_nodes_var)
    if start_index is None:
        start_index = _infer_start_index(raw_float[valid], node_count, face_nodes_var.name)

    normalized = np.full(raw.shape, -1, dtype=np.int64)
    normalized[valid] = raw_float[valid].astype(np.int64) - int(start_index)
    normalized[normalized < 0] = -1

    valid_normalized = normalized >= 0
    if not np.any(valid_normalized):
        raise ValueError('Face-node connectivity does not contain any valid node references')
    max_index = int(np.max(normalized[valid_normalized]))
    if max_index >= node_count:
        raise ValueError(
            f'Face-node connectivity references node {max_index}, but only {node_count} nodes are available'
        )
    if np.any(np.count_nonzero(valid_normalized, axis=1) < 3):
        raise ValueError('Every UGRID face must reference at least three valid nodes')

    return normalized


def compute_face_centroids(node_x: np.ndarray, node_y: np.ndarray, face_nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute face centroids from node coordinates and normalized face-node connectivity."""

    valid = face_nodes >= 0
    clipped = np.where(valid, face_nodes, 0)
    face_node_x = np.where(valid, node_x[clipped], np.nan)
    face_node_y = np.where(valid, node_y[clipped], np.nan)
    return np.nanmean(face_node_x, axis=1), np.nanmean(face_node_y, axis=1)


def _copy_located_variable(name: str, variable: xr.DataArray, mesh: Mesh) -> tuple | None:
    if not variable.dims:
        return None

    spatial_dims = [
        dim
        for dim in variable.dims
        if dim in {mesh.source_face_dim, mesh.source_node_dim}
        or variable.sizes[dim] in {mesh.face_nodes.shape[0], mesh.node_x.size}
    ]
    if len(spatial_dims) != 1:
        return None

    spatial_dim = spatial_dims[0]
    is_face = spatial_dim == mesh.source_face_dim or variable.sizes[spatial_dim] == mesh.face_nodes.shape[0]
    target_dim = 'mesh2d_nFaces' if is_face else 'mesh2d_nNodes'
    dims = tuple(target_dim if dim == spatial_dim else dim for dim in variable.dims)
    attrs = dict(variable.attrs)
    attrs.setdefault('mesh', 'mesh2d')
    attrs.setdefault('location', 'face' if is_face else 'node')
    attrs.setdefault(
        'coordinates',
        'mesh2d_face_x mesh2d_face_y' if is_face else 'mesh2d_node_x mesh2d_node_y',
    )
    return dims, variable.data, attrs


def _copy_nonspatial_coordinates(source: xr.Dataset, output: xr.Dataset, mesh: Mesh) -> xr.Dataset:
    spatial_dims = {mesh.source_face_dim, mesh.source_node_dim}
    output_dims = set(output.dims)
    for name, coordinate in source.coords.items():
        if name in output or any(dim in spatial_dims for dim in coordinate.dims):
            continue
        if all(dim in output_dims for dim in coordinate.dims):
            output[name] = (coordinate.dims, coordinate.data, dict(coordinate.attrs))
    return output


def _copy_crs_variables(source: xr.Dataset, output: xr.Dataset) -> None:
    for name, variable in source.variables.items():
        attrs = getattr(variable, 'attrs', {}) or {}
        if (
            name in {'crs', 'spatial_ref', 'wgs84', 'projected_coordinate_system'}
            or 'grid_mapping_name' in attrs
            or 'epsg' in attrs
            or 'crs_wkt' in attrs
        ):
            output[name] = (variable.dims, variable.data, dict(attrs))


def _required_var(source: xr.Dataset, candidates: tuple[str, ...], description: str) -> xr.DataArray:
    for name in candidates:
        if name in source:
            return source[name]
    raise KeyError(f'Could not find {description}. Tried: {", ".join(candidates)}')


def _optional_var(source: xr.Dataset, candidates: tuple[str, ...]) -> xr.DataArray | None:
    for name in candidates:
        if name in source:
            return source[name]
    return None


def _optional_1d(source: xr.Dataset, candidates: tuple[str, ...]) -> np.ndarray | None:
    for name in candidates:
        if name in source:
            values = np.asarray(source[name].values, dtype=np.float64).ravel()
            return values
    return None


def _as_1d_float(variable: xr.DataArray, description: str) -> np.ndarray:
    values = np.asarray(variable.values, dtype=np.float64).ravel()
    if values.ndim != 1:
        raise ValueError(f'{description} must be 1-D after flattening, got shape {values.shape}')
    return values


def _variable_start_index(variable: xr.DataArray) -> int | None:
    attrs = getattr(variable, 'attrs', {}) or {}
    if 'start_index' in attrs:
        return int(attrs['start_index'])
    return None


def _variable_fill_value(variable: xr.DataArray) -> float | None:
    attrs = getattr(variable, 'attrs', {}) or {}
    encoding = getattr(variable, 'encoding', {}) or {}
    value = encoding.get('_FillValue', attrs.get('_FillValue', attrs.get('missing_value')))
    if value is None:
        return None
    values = np.asarray(value)
    if values.size == 0:
        return None
    return float(values.ravel()[0])


def _infer_start_index(valid_values: np.ndarray, node_count: int, variable_name: str | None) -> int:
    if variable_name in {'NetElemNode', 'net_elem_node', 'net_element_node'}:
        return 1
    if valid_values.size and np.nanmin(valid_values) >= 1 and np.nanmax(valid_values) <= node_count:
        return 1
    return 0


def _first_existing_attrs(source: xr.Dataset, candidates: tuple[str, ...]) -> dict:
    for name in candidates:
        if name in source:
            return dict(source[name].attrs)
    return {}


def _coordinate_attrs(source_attrs: dict, axis: str) -> dict:
    attrs = dict(source_attrs)
    attrs.setdefault('long_name', f'{axis.upper()} coordinate')
    attrs.setdefault('standard_name', 'projection_x_coordinate' if axis == 'x' else 'projection_y_coordinate')
    attrs.setdefault('units', 'm')
    return attrs


def _global_attrs(source: xr.Dataset) -> dict:
    attrs = dict(source.attrs)
    existing = attrs.get('Conventions', '')
    conventions = [part.strip() for part in existing.split() if part.strip()]
    for convention in ('CF-1.8', 'UGRID-1.0'):
        if convention not in conventions:
            conventions.append(convention)
    attrs['Conventions'] = ' '.join(conventions)
    attrs.setdefault('title', 'UGRID mesh extracted from Delft3D FM map output')
    attrs['source_file'] = str(source.encoding.get('source', ''))
    attrs['history'] = (
        f'{datetime.now(timezone.utc).isoformat()} '
        'created by create_ugrid_from_fm_map.py'
    )
    return attrs


def _netcdf_int_dtype(values: np.ndarray) -> str:
    valid = values >= 0
    if not np.any(valid):
        return 'int32'
    max_value = int(np.max(values[valid]))
    return 'int32' if max_value <= np.iinfo(np.int32).max else 'int64'


if __name__ == '__main__':
    main()
