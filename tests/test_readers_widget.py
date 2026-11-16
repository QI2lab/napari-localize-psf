"""Integration tests of installed readers, npe2 discovery, and the real Qt dock."""

import numpy as np
import pytest
import tifffile
from napari.utils.transforms import Affine
from npe2 import PluginManifest

from napari_localize_psf.config import LocalizationConfig
from napari_localize_psf.data import layer_calibration, open_images, prepare_preview
from napari_localize_psf.widget import LocalizationWidget

pytestmark = pytest.mark.integration


def test_tiff_reader_and_widget_configuration(viewer, qtbot, tmp_path):
    path = tmp_path / "counts.ome.tif"
    image = np.ones((7, 19, 21), dtype=np.uint16)
    tifffile.imwrite(
        path,
        image,
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeZ": 0.2,
            "PhysicalSizeZUnit": "µm",
            "PhysicalSizeY": 0.1,
            "PhysicalSizeYUnit": "µm",
            "PhysicalSizeX": 0.1,
            "PhysicalSizeXUnit": "µm",
        },
        photometric="minisblack",
    )
    layers = open_images(viewer, str(path))
    assert layers[0].source.reader_plugin == "napari-tiff"
    spacing, _ = layer_calibration(layers[0], "zyx")
    np.testing.assert_allclose(spacing, (0.2, 0.1, 0.1))
    widget = LocalizationWidget(viewer)
    qtbot.addWidget(widget)
    widget.threshold.setValue(0.6)
    widget.blend.setChecked(False)
    config = widget.current_config()
    assert config.input_uri == str(path)
    assert config.intensity_threshold == 0.6 and not config.blend_3d
    exported = tmp_path / "localization.json"
    config.save(exported)
    widget.apply_config(LocalizationConfig.load(exported))
    assert widget.current_config().model_dump(exclude={"provenance"}) == config.model_dump(
        exclude={"provenance"}
    )
    assert prepare_preview(layers[0], config)[0].shape == image.shape
    assert widget.batch_button.text() == "Copy batch command"


@pytest.mark.parametrize("group", ["0", ""])
def test_ome_zarr_reader_multichannel_calibration(viewer, qtbot, tmp_path, group):
    from yaozarrs import DimSpec, v05
    from yaozarrs.write.v05 import write_bioformats2raw

    path = tmp_path / "counts.ome.zarr"
    data = np.ones((2, 7, 19, 21), dtype=np.uint16)
    dims = [DimSpec(name="c", size=2)] + [
        DimSpec(name=name, size=size, scale=scale, unit="micrometer")
        for name, size, scale in zip("zyx", data.shape[1:], (0.2, 0.1, 0.1), strict=True)
    ]
    positions = {"0": (-2, 3, -4), "1": (5, -6, 7)}
    images = {}
    ome_xml = tifffile.OmeXml()
    for name, translation in positions.items():
        multiscale = v05.Multiscale.from_dims(dims)
        multiscale.datasets[0].coordinateTransformations.append(
            v05.TranslationTransformation(translation=[0, *translation])
        )
        images[name] = (v05.Image(multiscales=[multiscale]), [data])
        ome_xml.addimage(
            dtype=data.dtype,
            shape=data.shape,
            storedshape=(14, 1, 1, 19, 21, 1),
            axes="CZYX",
        )
    write_bioformats2raw(path, images, ome_xml=str(ome_xml), writer="tensorstore")
    opened = str(path / group)
    selected_group = group or "1"
    layers = open_images(viewer, opened)
    assert len(layers) == (2 if group else 4)
    assert layers[-1].source.reader_plugin == "napari-ome-zarr"
    spacing, origin = layer_calibration(layers[-1], "zyx")
    np.testing.assert_allclose(spacing, (0.2, 0.1, 0.1))
    np.testing.assert_allclose(origin, positions[selected_group])
    assert layers[-1].metadata["napari_localize_psf"] == {
        "input_uri": str(path / selected_group),
        "source_channel": 1,
    }
    # The same metadata adapter must also cover images opened through File > Open.
    viewer.layers.clear()
    native = viewer.open(opened, plugin="napari-ome-zarr")
    placement = Affine(translate=(10, 20, 30))
    native[-1].affine = placement
    widget = LocalizationWidget(viewer)
    qtbot.addWidget(widget)
    widget.layers.setCurrentIndex(widget.layers.findData(native[-1]))
    config = widget.current_config()
    np.testing.assert_allclose(
        config.origin_zyx_um, np.asarray(positions[selected_group]) + (10, 20, 30)
    )
    np.testing.assert_allclose(native[-1].scale, (0.2, 0.1, 0.1))
    np.testing.assert_allclose(native[-1].translate, positions[selected_group])
    np.testing.assert_allclose(native[-1].affine.affine_matrix, placement.affine_matrix)
    assert config.input_uri == str(path / selected_group)
    assert config.channel == 1


def test_npe2_manifest_instantiates_the_real_widget(viewer, qtbot):
    manifest = PluginManifest.from_distribution("napari-localize-psf")
    assert not manifest.contributions.readers
    assert len(manifest.contributions.widgets) == 1
    dock, widget = viewer.window.add_plugin_dock_widget("napari-localize-psf")
    assert isinstance(widget, LocalizationWidget)
    assert dock.widget() is widget
    assert widget._layer is None


def test_uncalibrated_images_require_manual_confirmation(viewer, qtbot):
    viewer.add_image(np.ones((5, 11, 13)))
    widget = LocalizationWidget(viewer)
    qtbot.addWidget(widget)
    with pytest.raises(ValueError, match="manual calibration"):
        widget.current_config()
    widget.manual.setChecked(True)
    assert widget.current_config().voxel_size_zyx_um == (0.2, 0.1, 0.1)


def test_json_restoration_rejects_a_different_open_dataset(viewer, qtbot):
    viewer.add_image(np.ones((5, 11, 13)), units="um")
    widget = LocalizationWidget(viewer)
    qtbot.addWidget(widget)
    with pytest.raises(ValueError, match="input image"):
        widget.apply_config(LocalizationConfig(input_uri="different.ome.zarr"))


def test_unsplit_source_channel_matches_preview_and_batch(viewer, qtbot):
    viewer.add_image(np.ones((2, 5, 11, 13)), axis_labels=("c", "z", "y", "x"), units="um")
    widget = LocalizationWidget(viewer)
    qtbot.addWidget(widget)
    widget.indices["channel"].setValue(1)
    widget.batch_all.setChecked(False)
    config = widget.current_config()
    assert config.preview.channel == config.channel == 1
    assert widget.indices["source_channel"].value() == 1
