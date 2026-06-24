from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
import yaml

from sedtrails.application_interfaces.seeding_gui import (
    BathymetryViewData,
    MAP_ZOOM_IN_FACTOR,
    SeedingGuiApp,
    SeedingGuiError,
    add_population_from_existing,
    clip_points_by_elevation,
    default_seeded_config_path,
    generate_grid_points_in_polygon,
    generate_random_points_in_polygon,
    generate_transect_points,
    load_bathymetry_view_data,
    load_config,
    remove_population,
    rename_population,
    save_seeded_config,
    update_config_for_file_points,
    write_points_file,
)
from sedtrails.particle_tracer.particle_seeder import FilePointsStrategy, PopulationConfig


@pytest.fixture
def seeding_gui_app(tmp_path, monkeypatch):
    """Create a headless seeding GUI app with synthetic map data."""

    import matplotlib

    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    config_file = tmp_path / 'config.yaml'
    config_file.write_text(
        yaml.safe_dump(
            {
                'particles': {
                    'populations': [
                        {
                            'name': 'sand_a',
                            'seeding': {'quantity': 1, 'strategy': {'point': {'locations': ['0,0']}}},
                        }
                    ]
                }
            }
        ),
        encoding='utf-8',
    )
    view_data = BathymetryViewData(
        x=np.array([0.0, 1.0, 0.0, 1.0]),
        y=np.array([0.0, 0.0, 1.0, 1.0]),
        values=np.array([-1.0, -0.5, 0.0, 0.5]),
        variable='bedlevel',
        input_file=tmp_path / 'input.nc',
    )
    monkeypatch.setattr(
        'sedtrails.application_interfaces.seeding_gui.load_bathymetry_view_data',
        lambda *args, **kwargs: view_data,
    )
    app = SeedingGuiApp(
        config_path=config_file,
        output_path=tmp_path / 'seeded.yaml',
        points_output_path=tmp_path / 'seeded.points.txt',
        population_name=None,
        format_override=None,
        variable=None,
    )
    app.fig.canvas.draw()
    try:
        yield app
    finally:
        plt.close(app.fig)


def _test_output_dir(name: str) -> Path:
    """Create a workspace-local scratch directory for tests that write files."""

    path = Path('tests') / '_tmp_seeding_gui' / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cleanup_output_dir(path: Path) -> None:
    """Remove a scratch directory and its empty parent when possible."""

    root = path.parent
    shutil.rmtree(path, ignore_errors=True)
    try:
        root.rmdir()
    except OSError:
        pass


def _axis_center_and_span(limits):
    """Return the center and signed span for axis limits."""

    return (limits[0] + limits[1]) * 0.5, limits[1] - limits[0]


def _map_mouse_event(app, *, button=1, xdata=0.5, ydata=0.5, x_fraction=0.5, y_fraction=0.5):
    """Create a minimal Matplotlib mouse event for the app's map axes."""

    bbox = app.ax.bbox
    return SimpleNamespace(
        inaxes=app.ax,
        button=button,
        x=float(bbox.x0 + bbox.width * x_fraction),
        y=float(bbox.y0 + bbox.height * y_fraction),
        xdata=xdata,
        ydata=ydata,
    )


def test_gui_zoom_buttons_change_axis_span_without_changing_center(seeding_gui_app):
    """Zoom controls scale the map around the current view center."""

    app = seeding_gui_app
    initial_xlim = app.ax.get_xlim()
    initial_ylim = app.ax.get_ylim()
    initial_x_center, initial_x_span = _axis_center_and_span(initial_xlim)
    initial_y_center, initial_y_span = _axis_center_and_span(initial_ylim)

    app._zoom_in()
    zoom_xlim = app.ax.get_xlim()
    zoom_ylim = app.ax.get_ylim()

    np.testing.assert_allclose(
        _axis_center_and_span(zoom_xlim),
        (initial_x_center, initial_x_span * MAP_ZOOM_IN_FACTOR),
    )
    np.testing.assert_allclose(
        _axis_center_and_span(zoom_ylim),
        (initial_y_center, initial_y_span * MAP_ZOOM_IN_FACTOR),
    )

    app._zoom_out()

    np.testing.assert_allclose(_axis_center_and_span(app.ax.get_xlim()), _axis_center_and_span(initial_xlim))
    np.testing.assert_allclose(_axis_center_and_span(app.ax.get_ylim()), _axis_center_and_span(initial_ylim))


def test_gui_map_navigation_buttons_do_not_overlap_x_axis_labels(seeding_gui_app):
    """Map navigation controls stay clear of x-axis tick labels and label."""

    app = seeding_gui_app
    app.fig.canvas.draw()
    renderer = app.fig.canvas.get_renderer()
    x_axis_bbox = app.ax.xaxis.get_tightbbox(renderer)

    for button in app._map_buttons:
        assert not button.ax.get_window_extent(renderer).overlaps(x_axis_bbox)


def test_gui_pan_button_toggles_drag_panning_without_creating_seed_points(seeding_gui_app):
    """Pan mode drags the map view and leaves seed selection disabled until toggled off."""

    app = seeding_gui_app
    initial_xlim = app.ax.get_xlim()
    initial_ylim = app.ax.get_ylim()
    press = _map_mouse_event(app)

    app._toggle_pan_mode()
    assert app._pan_button.color == '0.70'
    assert app._pan_button.hovercolor == '0.78'
    np.testing.assert_allclose(app._pan_button.ax.get_facecolor(), (0.70, 0.70, 0.70, 1.0))
    app._on_map_button_press(press)
    motion = SimpleNamespace(
        x=press.x + app.ax.bbox.width * 0.25,
        y=press.y - app.ax.bbox.height * 0.10,
    )
    app._on_map_motion(motion)

    x_delta = 0.25 * (initial_xlim[1] - initial_xlim[0])
    y_delta = -0.10 * (initial_ylim[1] - initial_ylim[0])
    np.testing.assert_allclose(app.ax.get_xlim(), (initial_xlim[0] - x_delta, initial_xlim[1] - x_delta))
    np.testing.assert_allclose(app.ax.get_ylim(), (initial_ylim[0] - y_delta, initial_ylim[1] - y_delta))
    assert app.points == []
    assert 'Pan: on' in app.status_text.get_text()

    app._on_map_button_release(motion)
    assert app._pan_start is None

    app._toggle_pan_mode()
    assert app._pan_button.color == '0.85'
    assert app._pan_button.hovercolor == '0.95'
    np.testing.assert_allclose(app._pan_button.ax.get_facecolor(), (0.85, 0.85, 0.85, 1.0))
    app._on_map_button_press(_map_mouse_event(app, xdata=2.5, ydata=3.5))

    assert app.points == [(2.5, 3.5)]
    assert 'Pan: on' not in app.status_text.get_text()


def test_update_config_for_file_points_updates_only_selected_population():
    """Updating seed output changes only the chosen population strategy."""

    config = {
        'particles': {
            'populations': [
                {
                    'name': 'sand_a',
                    'seeding': {'quantity': 1, 'strategy': {'point': {'locations': ['1,2']}}},
                },
                {
                    'name': 'sand_b',
                    'seeding': {'quantity': 2, 'strategy': {'random': {'bbox': '0,0 1,1', 'nlocations': 3}}},
                },
            ]
        }
    }

    updated = update_config_for_file_points(
        config,
        population_name='sand_b',
        points_path='./generated_points.txt',
    )

    assert updated['particles']['populations'][0]['seeding']['strategy'] == {'point': {'locations': ['1,2']}}
    assert updated['particles']['populations'][1]['seeding']['strategy'] == {
        'file_points': {
            'path': './generated_points.txt',
            'has_header': False,
            'x_col': 0,
            'y_col': 1,
            'deduplicate': False,
            'dropna': False,
            'stride': 1,
        }
    }
    assert config['particles']['populations'][1]['seeding']['strategy'] == {
        'random': {'bbox': '0,0 1,1', 'nlocations': 3}
    }


def test_update_config_for_file_points_rejects_unknown_population():
    """Unknown population names fail with a user-facing setup error."""

    config = {'particles': {'populations': [{'name': 'sand_a', 'seeding': {'strategy': {'point': {}}}}]}}

    with pytest.raises(SeedingGuiError, match='Population'):
        update_config_for_file_points(config, population_name='missing', points_path='./points.txt')


def test_add_rename_remove_population_helpers_copy_and_update_names():
    """Population helper functions copy settings, rename, remove, and leave inputs untouched."""

    config = {
        'particles': {
            'populations': [
                {
                    'name': 'sand_a',
                    'particle_type': 'sand',
                    'characteristics': {'grain_size': 0.1, 'density': 2650.0},
                    'seeding': {'quantity': 1, 'strategy': {'point': {'locations': ['1,2']}}},
                }
            ]
        }
    }

    added, added_name = add_population_from_existing(config, source_population_name='sand_a', new_population_name='sand_b')
    renamed = rename_population(added, old_name='sand_b', new_name='sand_c')
    removed = remove_population(renamed, population_name='sand_a')

    assert added_name == 'sand_b'
    assert get_population_names_for_test(added) == ['sand_a', 'sand_b']
    assert added['particles']['populations'][1]['characteristics'] == config['particles']['populations'][0]['characteristics']
    assert get_population_names_for_test(renamed) == ['sand_a', 'sand_c']
    assert get_population_names_for_test(removed) == ['sand_c']
    assert config['particles']['populations'][0]['name'] == 'sand_a'


def get_population_names_for_test(config):
    """Return population names from a minimal test config."""

    return [population['name'] for population in config['particles']['populations']]


def test_default_seeded_config_path_is_next_to_source_yaml():
    """Default GUI output path is generated beside the source YAML."""

    assert default_seeded_config_path(Path('examples/sedtrails-example-multisource.yaml')) == Path(
        'examples/sedtrails-example-multisource-seeded.yaml'
    )


def test_load_bathymetry_view_data_supports_fm_netcdf(tmp_path, monkeypatch):
    """Bathymetry loading extracts first-time FM arrays using net_xcc/net_ycc."""

    config_file = tmp_path / 'config.yaml'
    input_file = tmp_path / 'input.nc'
    config_file.write_text(
        yaml.safe_dump(
            {
                'general': {'input_model': {'format': 'fm_netcdf'}},
                'inputs': {'data': str(input_file.name)},
            }
        ),
        encoding='utf-8',
    )
    input_file.write_text('', encoding='utf-8')

    dataset = xr.Dataset(
        {
            'net_xcc': (('mesh2d_nFaces',), np.array([0.0, 1.0, 2.0])),
            'net_ycc': (('mesh2d_nFaces',), np.array([3.0, 4.0, 5.0])),
            'bedlevel': (('time', 'mesh2d_nFaces'), np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])),
        }
    )
    monkeypatch.setattr(
        'sedtrails.application_interfaces.seeding_gui._open_netcdf_dataset',
        lambda _: dataset,
    )

    view_data = load_bathymetry_view_data(config_file)

    np.testing.assert_allclose(view_data.x, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(view_data.y, np.array([3.0, 4.0, 5.0]))
    np.testing.assert_allclose(view_data.values, np.array([10.0, 11.0, 12.0]))
    assert view_data.variable == 'bedlevel'


def test_load_bathymetry_view_data_supports_xbeach(tmp_path, monkeypatch):
    """Bathymetry loading extracts first mean-time XBeach arrays using globalx/globaly."""

    config_file = tmp_path / 'config.yaml'
    input_file = tmp_path / 'input.nc'
    config_file.write_text(
        yaml.safe_dump(
            {
                'general': {'input_model': {'format': 'xbeach'}},
                'inputs': {'data': str(input_file.name)},
            }
        ),
        encoding='utf-8',
    )
    input_file.write_text('', encoding='utf-8')

    dataset = xr.Dataset(
        {
            'globalx': (('ny', 'nx'), np.array([[0.0, 1.0], [2.0, 3.0]])),
            'globaly': (('ny', 'nx'), np.array([[10.0, 11.0], [12.0, 13.0]])),
            'zb_mean': (
                ('meantime', 'ny', 'nx'),
                np.array([
                    [[-1.0, -2.0], [-3.0, -4.0]],
                    [[-5.0, -6.0], [-7.0, -8.0]],
                ]),
            ),
        }
    )
    monkeypatch.setattr(
        'sedtrails.application_interfaces.seeding_gui._open_netcdf_dataset',
        lambda _: dataset,
    )

    view_data = load_bathymetry_view_data(config_file)

    np.testing.assert_allclose(view_data.x, np.array([0.0, 1.0, 2.0, 3.0]))
    np.testing.assert_allclose(view_data.y, np.array([10.0, 11.0, 12.0, 13.0]))
    np.testing.assert_allclose(view_data.values, np.array([-1.0, -2.0, -3.0, -4.0]))
    assert view_data.variable == 'zb_mean'


def test_load_bathymetry_view_data_filters_xbeach_cutout_points(tmp_path, monkeypatch):
    """Bathymetry loading omits non-finite XBeach cutout coordinates."""

    config_file = tmp_path / 'config.yaml'
    input_file = tmp_path / 'input.nc'
    config_file.write_text(
        yaml.safe_dump(
            {
                'general': {'input_model': {'format': 'xbeach'}},
                'inputs': {'data': str(input_file.name)},
            }
        ),
        encoding='utf-8',
    )
    input_file.write_text('', encoding='utf-8')

    dataset = xr.Dataset(
        {
            'globalx': (('ny', 'nx'), np.array([[0.0, np.nan, 2.0], [0.0, 1.0, 2.0]])),
            'globaly': (('ny', 'nx'), np.array([[0.0, np.nan, 0.0], [1.0, 1.0, 1.0]])),
            'zb': (('globaltime', 'ny', 'nx'), np.arange(6, dtype=float).reshape(1, 2, 3)),
        }
    )
    monkeypatch.setattr(
        'sedtrails.application_interfaces.seeding_gui._open_netcdf_dataset',
        lambda _: dataset,
    )

    view_data = load_bathymetry_view_data(config_file)

    np.testing.assert_allclose(view_data.x, np.array([0.0, 2.0, 0.0, 1.0, 2.0]))
    np.testing.assert_allclose(view_data.y, np.array([0.0, 0.0, 1.0, 1.0, 1.0]))
    np.testing.assert_allclose(view_data.values, np.array([0.0, 2.0, 3.0, 4.0, 5.0]))
    assert view_data.variable == 'zb'
    assert np.isfinite(view_data.x).all()
    assert np.isfinite(view_data.y).all()
    assert np.isfinite(view_data.values).all()


def test_generate_transect_points_from_endpoint_pairs():
    """Transect generation interpolates k points for every clicked endpoint pair."""

    points = generate_transect_points([(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (0.0, 20.0)], 3)

    assert points == [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (0.0, 10.0), (0.0, 15.0), (0.0, 20.0)]


def test_generate_random_points_in_polygon_is_seeded_and_inside():
    """Random polygon generation is repeatable for a fixed seed and stays inside the polygon."""

    polygon = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]

    first = generate_random_points_in_polygon(polygon, nlocations=5, seed=42)
    second = generate_random_points_in_polygon(polygon, nlocations=5, seed=42)

    assert first == second
    assert len(first) == 5
    assert all(0.0 <= x <= 2.0 and 0.0 <= y <= 2.0 for x, y in first)


def test_generate_random_points_in_skinny_polygon_uses_vectorized_batches():
    """Random polygon generation handles low acceptance-rate polygons."""

    polygon = [(0.0, 0.0), (1000.0, 0.0), (1000.0, 10.0), (0.0, 1.0)]

    points = generate_random_points_in_polygon(polygon, nlocations=50, seed=42)

    assert len(points) == 50
    assert all(0.0 <= x <= 1000.0 and 0.0 <= y <= 10.0 for x, y in points)


def test_generate_grid_points_in_polygon_filters_to_polygon():
    """Grid generation keeps only dx/dy candidate points inside the drawn polygon."""

    polygon = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]

    points = generate_grid_points_in_polygon(polygon, dx=1.0, dy=1.0)

    assert set(points).issubset({(x, y) for x in (0.0, 1.0, 2.0) for y in (0.0, 1.0, 2.0)})
    assert (1.0, 1.0) in points


def test_generate_grid_points_in_polygon_rejects_excessive_candidates():
    """Grid generation fails before allocating excessive candidate arrays."""

    polygon = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]

    with pytest.raises(SeedingGuiError, match='candidate points'):
        generate_grid_points_in_polygon(polygon, dx=1.0, dy=1.0, max_candidates=100)


def test_clip_points_by_elevation_deletes_above_or_below():
    """Elevation clipping removes points above or below a nearest-cell threshold."""

    points = [(0.1, 0.0), (1.1, 0.0), (2.1, 0.0)]
    field_x = np.array([0.0, 1.0, 2.0])
    field_y = np.array([0.0, 0.0, 0.0])
    field_values = np.array([-1.0, 0.0, 2.0])

    assert clip_points_by_elevation(
        points,
        field_x=field_x,
        field_y=field_y,
        field_values=field_values,
        threshold=0.0,
        delete='above',
    ) == [(0.1, 0.0), (1.1, 0.0)]
    assert clip_points_by_elevation(
        points,
        field_x=field_x,
        field_y=field_y,
        field_values=field_values,
        threshold=0.0,
        delete='below',
    ) == [(1.1, 0.0), (2.1, 0.0)]


def test_write_points_file_round_trips_through_file_points_strategy():
    """Point files written by the GUI remain readable by the existing file_points seeder."""

    output_dir = _test_output_dir('roundtrip')
    try:
        points_path = output_dir / 'points.txt'
        write_points_file([(1.5, 2.5), (3.0, 4.25)], points_path)
        population_config = PopulationConfig(
            {
                'particle_type': 'sand',
                'seeding': {
                    'quantity': 2,
                    'burial_depth': {'constant': 0},
                    'strategy': {
                        'file_points': {
                            'path': str(points_path),
                            'has_header': False,
                            'x_col': 0,
                            'y_col': 1,
                            'deduplicate': False,
                            'dropna': False,
                            'stride': 1,
                        }
                    },
                },
            }
        )

        assert FilePointsStrategy().seed(population_config) == [(2, 1.5, 2.5), (2, 3.0, 4.25)]
    finally:
        _cleanup_output_dir(output_dir)


def test_save_seeded_config_writes_valid_copied_yaml():
    """Saving writes a copied YAML config and single point file that validate together."""

    output_dir = _test_output_dir('save')
    try:
        output_config = output_dir / 'validation-case.yaml'
        points_output = output_dir / 'validation-points.txt'

        saved_config, saved_points = save_seeded_config(
            source_config_path='examples/sedtrails-example-multisource.yaml',
            output_config_path=output_config,
            points_output_path=points_output,
            points=[(10.0, 20.0), (30.0, 40.0)],
            population_name='population_1',
        )

        assert saved_config == output_config
        assert saved_points == points_output
        with output_config.open('r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        file_points = config['particles']['populations'][0]['seeding']['strategy']['file_points']
        assert file_points['path'] == './validation-points.txt'
        assert file_points['has_header'] is False
        assert points_output.read_text(encoding='utf-8') == '10 20\n30 40\n'
    finally:
        _cleanup_output_dir(output_dir)


def test_save_seeded_config_writes_multiple_population_point_files():
    """Saving multiple edited populations writes one point file and YAML entry per population."""

    output_dir = _test_output_dir('multi-save')
    try:
        source_config = load_config('examples/sedtrails-example-multisource.yaml')
        source_config, second_name = add_population_from_existing(
            source_config,
            source_population_name='population_1',
            new_population_name='population_2',
        )
        output_config = output_dir / 'validation-case.yaml'
        points_output = output_dir / 'validation-points.txt'

        save_seeded_config(
            source_config_path='examples/sedtrails-example-multisource.yaml',
            output_config_path=output_config,
            points_output_path=points_output,
            points=[],
            population_name='population_1',
            config_data=source_config,
            population_points={
                'population_1': [(10.0, 20.0)],
                second_name: [(30.0, 40.0)],
            },
        )

        with output_config.open('r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        populations = config['particles']['populations']
        first_file = output_dir / 'validation-points.population_1.txt'
        second_file = output_dir / 'validation-points.population_2.txt'
        assert first_file.read_text(encoding='utf-8') == '10 20\n'
        assert second_file.read_text(encoding='utf-8') == '30 40\n'
        assert populations[0]['seeding']['strategy']['file_points']['path'] == './validation-points.population_1.txt'
        assert populations[1]['seeding']['strategy']['file_points']['path'] == './validation-points.population_2.txt'
    finally:
        _cleanup_output_dir(output_dir)
