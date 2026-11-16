"""Real CUDA integration tests of production U-FISH and bounded localization."""

import numpy as np
import pytest
from qtpy.QtCore import Qt
from simulator import gaussian_image

from napari_localize_psf.config import LocalizationConfig, PreviewSettings
from napari_localize_psf.processing import optical_parameters

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.parametrize("geometry", ["volume", "plane", "xy"])
def test_async_dock_detection_fitting_and_overlay_alignment(viewer, qtbot, geometry):
    from napari_localize_psf.widget import LocalizationWidget

    plane = geometry != "volume"
    image, truth, spacing, origin = gaussian_image(plane)
    # A real time axis exercises selection and crop offsets in display coordinates.
    data = image[0] * 100 if geometry == "xy" else np.stack([image * 100, image * 125])
    source = viewer.add_image(
        data,
        axis_labels=("y", "x") if geometry == "xy" else ("t", "z", "y", "x"),
        scale=spacing[1:] if geometry == "xy" else (1, *spacing),
        translate=origin[1:] if geometry == "xy" else (0, *origin),
        units=("um", "um") if geometry == "xy" else ("s", "um", "um", "um"),
    )
    config = LocalizationConfig(
        fit_model="2d" if plane else "3d",
        voxel_size_zyx_um=spacing,
        origin_zyx_um=origin,
        fit_roi_size_um=(1.8, 1.5, 1.5),
        blend_3d=not plane,
        return_covariance=True,
        preview=PreviewSettings(
            axes="yx" if geometry == "xy" else "tzyx",
            timepoint=0 if geometry == "xy" else 1,
            crop_zyx_px=(0 if plane else 3, 1 if plane else 22, 3, 38, 4, 40),
        ),
    )
    widget = LocalizationWidget(viewer)
    qtbot.addWidget(widget)
    if geometry == "xy":
        assert widget.model.currentText() == "2d"
    widget.apply_config(config)
    widget.run_detection()
    qtbot.waitUntil(lambda: not widget._busy, timeout=60000)
    assert "U-FISH priors" in viewer.layers, widget.status.text()
    assert widget._detection is not None and len(widget._detection[0]) > 0
    enhanced = viewer.layers["U-FISH enhancement"]
    offset = np.array([3, 4] if geometry == "xy" else [1, 0 if plane else 3, 3, 4])
    np.testing.assert_allclose(
        enhanced.data_to_world(np.zeros(source.ndim)), source.data_to_world(offset)
    )
    priors = viewer.layers["U-FISH priors"]
    np.testing.assert_allclose(
        priors.data_to_world(priors.data[0]), source.data_to_world(priors.data[0])
    )
    cached_enhancement = widget._detection[1]
    widget.run_detection()
    qtbot.waitUntil(lambda: not widget._busy, timeout=60000)
    assert widget._detection[1] is cached_enhancement
    assert viewer.layers["U-FISH priors"] is not priors
    assert viewer.layers["U-FISH enhancement"] is not enhanced
    widget.run_fitting()
    qtbot.waitUntil(lambda: not widget._busy, timeout=60000)
    assert "Gaussian fits" in viewer.layers, widget.status.text()
    fitted = viewer.layers["Gaussian fits"]
    assert fitted.features["success"].any()
    assert "covariance_valid" in fitted.features
    nearest = np.asarray(fitted.features[["cz", "cy", "cx"]])
    assert np.min(np.linalg.norm(nearest - truth[[3, 2, 1]], axis=1)) < 0.05
    assert fitted.metadata["configuration"]["fit_model"] == config.fit_model
    if plane:
        np.testing.assert_allclose(fitted.features["cz"], origin[0])
        np.testing.assert_allclose(fitted.features["sz"], 1)
    widget.run_fitting()
    qtbot.waitUntil(lambda: not widget._busy, timeout=60000)
    assert viewer.layers["Gaussian fits"] is not fitted
    assert sum(layer.name == "Gaussian fits" for layer in viewer.layers) == 1


def test_closing_during_real_inference_cancels_publication_and_releases_model(viewer, qtbot):
    from napari_localize_psf.widget import LocalizationWidget

    image, _, spacing, origin = gaussian_image()
    viewer.add_image(image * 100, scale=spacing, translate=origin, units="um")
    widget = LocalizationWidget(viewer)
    qtbot.addWidget(widget)
    widget.run_detection()
    assert widget._busy
    widget._copy_batch_command()
    assert "Finish the current preview" in widget.status.text()
    widget.close()
    qtbot.waitUntil(lambda: not widget._busy, timeout=60000)
    assert widget._closing
    assert widget.detector._detector is None
    assert widget._preview is None and widget._detection is None
    assert "U-FISH priors" not in viewer.layers


@pytest.mark.parametrize("action", ["cancel", "scale", "data", "remove_dock"])
def test_active_preview_discards_cancelled_or_changed_sources(viewer, qtbot, action):
    image, _, spacing, origin = gaussian_image()
    source = viewer.add_image(image * 100, scale=spacing, translate=origin, units="um")
    _, widget = viewer.window.add_plugin_dock_widget("napari-localize-psf")
    widget.run_detection()
    worker = widget._worker
    assert widget._busy
    # Starting another preview must retain the currently owned production worker.
    widget.run_detection()
    assert widget._worker is worker
    if action == "cancel":
        qtbot.mouseClick(widget.cancel_button, Qt.MouseButton.LeftButton)
    elif action == "scale":
        source.scale = (0.3, 0.1, 0.1)
    elif action == "data":
        source.data = image * 125
    else:
        viewer.window.remove_dock_widget(widget)
    qtbot.waitUntil(lambda: not widget._busy, timeout=60000)
    assert widget._preview is None and widget._detection is None
    assert "U-FISH priors" not in viewer.layers
    assert "U-FISH enhancement" not in viewer.layers
    if action in {"scale", "data"}:
        assert "Result discarded" in widget.status.text()
        assert widget.tabs.isEnabled()
    elif action == "remove_dock":
        assert widget._closing and widget._detached
        assert widget.detector._detector is None


def test_source_edits_invalidate_existing_priors_and_refresh_calibration(viewer, qtbot):
    from napari_localize_psf.widget import LocalizationWidget

    image, _, spacing, origin = gaussian_image()
    source = viewer.add_image(image * 100, scale=spacing, translate=origin, units="um")
    widget = LocalizationWidget(viewer)
    qtbot.addWidget(widget)
    widget.run_detection()
    qtbot.waitUntil(lambda: not widget._busy, timeout=60000)
    assert widget._detection is not None
    source.translate = (-3, 4, -5)
    assert widget._preview is None and widget._detection is None
    assert widget.current_config().origin_zyx_um == (-3, 4, -5)
    widget.run_fitting()
    assert "Preview U-FISH priors" in widget.status.text()


def test_optical_controls_map_upstream_values_to_the_correct_cells(viewer, qtbot):
    from napari_localize_psf.widget import LocalizationWidget

    image, _, spacing, _ = gaussian_image()
    viewer.add_image(image, scale=spacing, units="um")
    widget = LocalizationWidget(viewer)
    qtbot.addWidget(widget)
    expected = optical_parameters(widget.current_config())
    widget._apply_optics()
    qtbot.waitUntil(lambda: not widget._busy, timeout=60000)
    for name, axis in (("sxy", 1), ("sz", 0)):
        row = widget.parameter_names.index(name)
        for column, key in enumerate(
            ("lower_sigma_zyx_um", "upper_sigma_zyx_um", "nominal_sigma_zyx_um")
        ):
            assert float(widget.parameters.item(row, column).text()) == expected[key][axis]
