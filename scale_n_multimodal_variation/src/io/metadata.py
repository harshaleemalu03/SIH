import xml.etree.ElementTree as ET
from pathlib import Path


def parse_iirs_metadata(xml_path):
    """
    Parse an IIRS PDS4 XML label and extract
    metadata fields that are useful for registration.
    """

    xml_path = Path(xml_path)

    if not xml_path.exists():
        raise FileNotFoundError(
            f"XML file not found: {xml_path}"
        )

    tree = ET.parse(xml_path)
    root = tree.getroot()

    metadata = {}

    # Extract all elements containing useful scalar values.
    for element in root.iter():

        tag = element.tag.split("}")[-1]

        if element.text is None:
            continue

        value = element.text.strip()

        if not value:
            continue

        if tag in {
            "start_date_time",
            "stop_date_time",
            "description",
            "local_identifier",
            "version_id",
            "name",
            "value",
            "unit",
            "axis_name",
            "elements",
            "data_type",
            "axis_index_order",
            "offset",
            "scaling_factor",
            "unit_id"
        }:
            metadata.setdefault(tag, []).append(value)

    return metadata
