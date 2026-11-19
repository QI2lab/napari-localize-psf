"""Unit/integration checks for install metadata, cold imports and real Qt lifecycle."""

import subprocess
import sys
from importlib.metadata import distribution

import numpy as np
import pytest
from packaging.requirements import Requirement

from napari_localize_psf.widget import LocalizationWidget


@pytest.mark.unit
def test_installed_plugin_metadata_does_not_install_the_host():
    for value in distribution("napari-localize-psf").requires or ():
        requirement = Requirement(value)
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        assert not requirement.name.lower().startswith(("pyqt", "pyside"))
        assert requirement.name.lower() not in {
            "napari",
            "qtpy",
            "npe2",
            "napari-ome-zarr",
            "napari-tiff",
        }
    metadata = distribution("napari-localize-psf").metadata
    assert "Framework :: napari" in metadata.get_all("Classifier", [])


@pytest.mark.unit
def test_cold_import_and_unused_detector_cleanup_do_not_load_cuda_packages():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from napari_localize_psf.widget import LocalizationWidget
from napari_localize_psf.processing import Detector
from napari_localize_psf.runtime import package_versions
Detector().release()
assert package_versions()['napari-localize-psf']
assert not {'torch', 'cupy', 'ufish', 'spotiflow'} & sys.modules.keys()
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.integration
def test_cold_dock_creation_does_not_initialize_the_gpu_stack():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import napari
from napari_localize_psf.widget import LocalizationWidget
viewer = napari.Viewer(show=False)
widget = LocalizationWidget(viewer)
assert not {'torch', 'cupy', 'ufish', 'spotiflow'} & sys.modules.keys()
widget.close()
viewer.close()
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.integration
def test_shutdown_disconnects_real_viewer_and_layer_events(viewer, qtbot):
    image = viewer.add_image(np.ones((5, 11, 13)), units="um")
    widget = LocalizationWidget(viewer)
    qtbot.addWidget(widget)
    widget.shutdown()
    revision = widget._revision
    image.data = np.zeros((5, 11, 13))
    viewer.add_image(np.ones((5, 11, 13)), units="um")
    assert widget._revision == revision
    assert widget._layer is None and widget.layers.count() == 0
    assert widget.detector._detector is None


@pytest.mark.integration
def test_public_dock_removal_cleans_up_the_inner_widget(viewer):
    viewer.add_image(np.ones((5, 11, 13)), units="um")
    _, widget = viewer.window.add_plugin_dock_widget("napari-localize-psf")
    viewer.window.remove_dock_widget(widget)
    assert widget._closing and widget._detached
    assert widget._layer is None
    assert widget._preview is None and widget._detection is None
