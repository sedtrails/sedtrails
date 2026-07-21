import numpy as np

from sedtrails.transport_converter.domain_mask import (
    classify_boundary_edges,
    extract_boundary_edges,
    filter_connectivity_by_inner_polygons,
    triangulate_face_connectivity,
)
from sedtrails.transport_converter.tekal import read_tekal_polygons


def test_read_tekal_polygons_accepts_multiple_blocks_and_files(tmp_path):
    """Tekal reader returns every polygon block across all configured files."""

    first = tmp_path / 'islands_a.pol'
    first.write_text(
        """
* comment
island_a
4 2
0 0
1 0
1 1
0 1
island_b
4 2
2 2
3 2
3 3
2 3
""".strip()
    )
    second = tmp_path / 'islands_b.pol'
    second.write_text(
        """
island_c
5 2
4 4
5 4
5 5
4 5
4 4
""".strip()
    )

    polygons = read_tekal_polygons([first, second])

    assert len(polygons) == 3
    np.testing.assert_allclose(polygons[0], np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float))
    np.testing.assert_allclose(polygons[2], np.array([[4, 4], [5, 4], [5, 5], [4, 5]], dtype=float))


def test_filter_connectivity_by_inner_polygons_masks_faces_by_centroid():
    """Faces whose centroids are inside island polygons are removed."""

    node_x = np.array([0.0, 1.0, 0.0, 10.0, 11.0, 10.0])
    node_y = np.array([0.0, 0.0, 1.0, 10.0, 10.0, 11.0])
    connectivity = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    island = np.array([[9.0, 9.0], [12.0, 9.0], [12.0, 12.0], [9.0, 12.0]])

    result = filter_connectivity_by_inner_polygons(node_x, node_y, connectivity, [island])

    assert result.removed_count == 1
    np.testing.assert_array_equal(result.active_mask, np.array([True, False]))
    np.testing.assert_array_equal(result.connectivity, np.array([[0, 1, 2]], dtype=np.int64))


def test_triangulate_face_connectivity_splits_quads_without_reindexing_nodes():
    """Polygonal faces are fan-split while preserving original node indices."""

    connectivity = np.array([[0, 1, 2, 3], [4, 5, 6, -1]], dtype=np.int64)

    triangles = triangulate_face_connectivity(connectivity)

    np.testing.assert_array_equal(triangles, np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]], dtype=np.int64))


def test_extract_boundary_edges_returns_edges_used_once():
    """Internal triangle edges are excluded from the active boundary edge table."""

    connectivity = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

    edges = extract_boundary_edges(connectivity)

    assert {tuple(sorted(edge)) for edge in edges.tolist()} == {(0, 1), (1, 2), (2, 3), (0, 3)}


def test_classify_boundary_edges_uses_open_and_land_override_polygons():
    """Boundary edge midpoints inside override polygons receive the configured class."""

    node_x = np.array([0.0, 1.0, 1.0, 0.0])
    node_y = np.array([0.0, 0.0, 1.0, 1.0])
    connectivity = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    open_polygons = [
        np.array([[0.25, -0.10], [0.75, -0.10], [0.75, 0.10], [0.25, 0.10]]),
        np.array([[0.90, 0.25], [1.10, 0.25], [1.10, 0.75], [0.90, 0.75]]),
    ]
    land_polygons = [np.array([[-0.10, 0.25], [0.10, 0.25], [0.10, 0.75], [-0.10, 0.75]])]

    classification = classify_boundary_edges(
        node_x,
        node_y,
        connectivity,
        {'open': open_polygons, 'land': land_polygons},
        class_files={'open': ['open_a.pol', 'open_b.pol'], 'land': ['land.pol']},
    )

    assert classification.class_counts == {'open': 2, 'land': 1, 'unclassified': 1, 'ambiguous': 0}
    assert classification.polygon_counts == {'open': 2, 'land': 1}
    assert classification.to_metadata()['class_pol_files']['open'] == ['open_a.pol', 'open_b.pol']


def test_classify_boundary_edges_marks_overlapping_override_polygons_as_land():
    """Land wins when an edge is selected by both open and land override polygons."""

    node_x = np.array([0.0, 1.0, 0.0])
    node_y = np.array([0.0, 0.0, 1.0])
    connectivity = np.array([[0, 1, 2]], dtype=np.int64)
    selector = np.array([[0.25, -0.10], [0.75, -0.10], [0.75, 0.10], [0.25, 0.10]])

    classification = classify_boundary_edges(
        node_x,
        node_y,
        connectivity,
        {'open': [selector], 'land': [selector]},
    )

    assert classification.class_counts['land'] == 1
    assert classification.class_counts['ambiguous'] == 0
    assert ('open', 'land') in classification.class_sources
