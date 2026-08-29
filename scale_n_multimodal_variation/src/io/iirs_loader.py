import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


class IIRSProduct:
    """
    Reader for Chandrayaan-2 IIRS calibrated BSQ products.

    The calibrated IIRS cube is stored as:
        BAND × LINE × SAMPLE

    Data type:
        IEEE754LSBSingle (little-endian float32)
    """

    def __init__(self, xml_path, qub_path):
        self.xml_path = Path(xml_path)
        self.qub_path = Path(qub_path)

        if not self.xml_path.exists():
            raise FileNotFoundError(
                f"XML label not found: {self.xml_path}"
            )

        if not self.qub_path.exists():
            raise FileNotFoundError(
                f"QUBE file not found: {self.qub_path}"
            )

        self.metadata = self._parse_label()

        self.bands = self.metadata["bands"]
        self.lines = self.metadata["lines"]
        self.samples = self.metadata["samples"]

        self.dtype = np.dtype("<f4")

    def _parse_label(self):
        """
        Read cube dimensions and band information from PDS4 XML.
        """

        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        dimensions = {}

        for axis in root.iter():
            tag = axis.tag.split("}")[-1]

            if tag != "Axis_Array":
                continue

            axis_name = None
            elements = None

            for child in axis:
                child_tag = child.tag.split("}")[-1]

                if child_tag == "axis_name":
                    axis_name = child.text.strip()

                elif child_tag == "elements":
                    elements = int(child.text.strip())

            if axis_name and elements:
                dimensions[axis_name.upper()] = elements

        required = ["BAND", "LINE", "SAMPLE"]

        missing = [
            axis for axis in required
            if axis not in dimensions
        ]

        if missing:
            raise ValueError(
                f"Missing dimensions in PDS4 label: {missing}"
            )

        return {
            "bands": dimensions["BAND"],
            "lines": dimensions["LINE"],
            "samples": dimensions["SAMPLE"],
        }

    def read_band(self, band):
        """
        Read one complete spectral band.

        Parameters
        ----------
        band : int
            1-based IIRS band number.

        Returns
        -------
        numpy.ndarray
            Array with shape (LINE, SAMPLE).
        """

        if not 1 <= band <= self.bands:
            raise ValueError(
                f"Band must be between 1 and {self.bands}"
            )

        pixels_per_band = self.lines * self.samples

        offset = (
            (band - 1)
            * pixels_per_band
            * self.dtype.itemsize
        )

        with open(self.qub_path, "rb") as file:

            file.seek(offset)

            data = np.fromfile(
                file,
                dtype=self.dtype,
                count=pixels_per_band
            )

        if data.size != pixels_per_band:
            raise IOError(
                "Unexpected end of QUBE file."
            )

        return data.reshape(
            self.lines,
            self.samples
        )

    def read_window(
        self,
        band,
        line_start,
        line_end,
        sample_start,
        sample_end
    ):
        """
        Read a spatial subset from one spectral band.

        Coordinates are zero-based and end-exclusive.

        Returns
        -------
        numpy.ndarray
            Shape:
            (line_end-line_start,
             sample_end-sample_start)
        """

        if not 1 <= band <= self.bands:
            raise ValueError("Invalid band number.")

        if not (
            0 <= line_start < line_end <= self.lines
        ):
            raise ValueError("Invalid line range.")

        if not (
            0 <= sample_start < sample_end <= self.samples
        ):
            raise ValueError("Invalid sample range.")

        width = sample_end - sample_start

        pixels_per_band = self.lines * self.samples

        band_offset = (
            (band - 1)
            * pixels_per_band
            * self.dtype.itemsize
        )

        result = np.empty(
            (line_end - line_start, width),
            dtype=self.dtype
        )

        with open(self.qub_path, "rb") as file:

            for output_row, line in enumerate(
                range(line_start, line_end)
            ):

                pixel_offset = (
                    line * self.samples
                    + sample_start
                )

                offset = (
                    band_offset
                    + pixel_offset
                    * self.dtype.itemsize
                )

                file.seek(offset)

                row = np.fromfile(
                    file,
                    dtype=self.dtype,
                    count=width
                )

                if row.size != width:
                    raise IOError(
                        "Unexpected end of QUBE file."
                    )

                result[output_row] = row

        return result
