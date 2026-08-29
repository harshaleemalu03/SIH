import numpy as np

from src.io.iirs_loader import IIRSProduct


def create_test_product(tmp_path):

    xml = """
    <Product>
        <Array_3D_Spectrum>

            <Axis_Array>
                <axis_name>BAND</axis_name>
                <elements>3</elements>
            </Axis_Array>

            <Axis_Array>
                <axis_name>LINE</axis_name>
                <elements>4</elements>
            </Axis_Array>

            <Axis_Array>
                <axis_name>SAMPLE</axis_name>
                <elements>5</elements>
            </Axis_Array>

        </Array_3D_Spectrum>
    </Product>
    """

    xml_path = tmp_path / "test.xml"
    qub_path = tmp_path / "test.qub"

    xml_path.write_text(xml)

    data = np.arange(
        3 * 4 * 5,
        dtype="<f4"
    )

    data.tofile(qub_path)

    return xml_path, qub_path


def test_iirs_dimensions(tmp_path):

    xml_path, qub_path = create_test_product(
        tmp_path
    )

    product = IIRSProduct(
        xml_path,
        qub_path
    )

    assert product.bands == 3
    assert product.lines == 4
    assert product.samples == 5


def test_read_band(tmp_path):

    xml_path, qub_path = create_test_product(
        tmp_path
    )

    product = IIRSProduct(
        xml_path,
        qub_path
    )

    band = product.read_band(2)

    assert band.shape == (4, 5)

    assert band[0, 0] == 20


def test_read_window(tmp_path):

    xml_path, qub_path = create_test_product(
        tmp_path
    )

    product = IIRSProduct(
        xml_path,
        qub_path
    )

    window = product.read_window(
        band=2,
        line_start=1,
        line_end=3,
        sample_start=2,
        sample_end=5
    )

    assert window.shape == (2, 3)

    assert window[0, 0] == 27
