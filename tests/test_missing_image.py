"""A missing PNG/JPG must exit nonzero, not report an error with exit 0."""

import sys

import pytest

from viewinline.viewinline import main


def _run(argv, capsys):
    sys.argv = argv
    with pytest.raises(SystemExit) as exc:
        main()
    out = capsys.readouterr().out
    return exc.value.code, out


def test_missing_png_exits_nonzero(capsys):
    code, out = _run(["viewinline", "/nonexistent-xyz-440.png"], capsys)
    assert code == 1
    assert "[ERROR]" in out


def test_missing_jpg_exits_nonzero(capsys):
    code, out = _run(["viewinline", "/nonexistent-xyz-440.jpg"], capsys)
    assert code == 1
    assert "[ERROR]" in out


@pytest.mark.parametrize("path", [
    "/nonexistent-xyz-440.tif",
    "/nonexistent-xyz-440.shp",
    "/nonexistent-xyz-440.geojson",
    "/nonexistent-xyz-440.nc",
])
def test_missing_raster_or_vector_exits_nonzero(path, capsys):
    code, out = _run(["viewinline", path], capsys)
    assert code == 1
    assert "[ERROR]" in out