import pandas as pd

from src.io.geometry import (
    read_iirs_geometry,
    geometry_bounds,
    pixel_to_geolocation,
)


def test_geometry_reader(tmp_path):

    csv_path = tmp_path / "geometry.csv"

    df = pd.DataFrame({
        "longitude": [10.0, 10.1, 10.2],
        "latitude": [20.0, 20.1, 20.2],
        "scan": [0, 0, 1],
        "pixel": [0, 1, 0],
    })

    df.to_csv(csv_path, index=False)

    geometry = read_iirs_geometry(csv_path)

    assert len(geometry) == 3

    bounds = geometry_bounds(geometry)

    assert bounds == (
        10.0,
        10.2,
        20.0,
        20.2,
    )


def test_pixel_to_geolocation(tmp_path):

    csv_path = tmp_path / "geometry.csv"

    df = pd.DataFrame({
        "longitude": [10.0],
        "latitude": [20.0],
        "scan": [5],
        "pixel": [10],
    })

    df.to_csv(csv_path, index=False)

    geometry = read_iirs_geometry(csv_path)

    lon, lat = pixel_to_geolocation(
        geometry,
        scan=5,
        pixel=10
    )

    assert lon == 10.0
    assert lat == 20.0
