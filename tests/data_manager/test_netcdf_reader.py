import pytest
from sedtrails.data_manager.netcdf_reader import NetCDFReader


class DummyDataset:
    def __init__(self, path):
        self.path = path


def test_missing_file():
    """Missing file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        NetCDFReader("no_such_file.nc")


def test_invalid_extension(tmp_path):
    """Wrong extension should raise ValueError before trying to open."""
    bad_extension = tmp_path / "data.txt"
    bad_extension.write_text("x")
    with pytest.raises(ValueError):
        NetCDFReader(str(bad_extension))


def test_valid_nc_opens(monkeypatch, tmp_path):
    """Valid .nc path calls xugrid.open_dataset and stores dataset."""
    nc = tmp_path / "ok.nc"
    nc.write_text("placeholder")  # existence + suffix only

    import xugrid as xu
    called = {}

    def fake_open_dataset(path):
        called["path"] = path
        return DummyDataset(path)

    monkeypatch.setattr(xu, "open_dataset", fake_open_dataset)

    reader = NetCDFReader(str(nc))
    assert hasattr(reader, "data")
    assert isinstance(reader.data, DummyDataset)
    assert called["path"] == str(nc)
    assert reader.data.path == str(nc)


@pytest.mark.parametrize("ext", [".nc", ".nc4", ".netcdf"])
def test_supported_extensions(monkeypatch, tmp_path, ext):
    """All supported extensions accepted."""
    f = tmp_path / f"sample{ext}"
    f.write_text("x")

    import xugrid as xu
    monkeypatch.setattr(xu, "open_dataset", lambda path: DummyDataset(str(f)))

    reader = NetCDFReader(str(f))
    assert reader.data.path == str(f)