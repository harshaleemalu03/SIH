from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "longitude",
    "latitude",
    "scan",
    "pixel",
}


def read_iirs_geometry(csv_path):
    """
    Read an IIRS calibrated geometry grid.

    The IIRS SIS specifies the geometry grid as a CSV
    containing longitude, latitude, scan and pixel values.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Geometry file not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    # Normalize column names.
    df.columns = [
        column.strip().lower()
        for column in df.columns
    ]

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required geometry columns: {sorted(missing)}"
        )

    return df


def geometry_bounds(geometry):
    """
    Return geographic bounds of an IIRS geometry grid.

    Returns:
        min_lon, max_lon, min_lat, max_lat
    """

    return (
        float(geometry["longitude"].min()),
        float(geometry["longitude"].max()),
        float(geometry["latitude"].min()),
        float(geometry["latitude"].max()),
    )


def pixel_to_geolocation(
    geometry,
    scan,
    pixel
):
    """
    Find the longitude and latitude corresponding
    to an IIRS scan/pixel coordinate.
    """

    result = geometry[
        (geometry["scan"] == scan)
        & (geometry["pixel"] == pixel)
    ]

    if result.empty:
        raise ValueError(
            f"No geometry found for scan={scan}, pixel={pixel}"
        )

    row = result.iloc[0]

    return (
        float(row["longitude"]),
        float(row["latitude"]),
    )
