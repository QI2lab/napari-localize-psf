"""Real CUDA integration tests of production U-FISH and bounded localization."""

from concurrent.futures import CancelledError
from threading import Event

import numpy as np
import pytest
from napari.layers import Image
from qtpy.QtCore import Qt
from simulator import gaussian_image

from napari_localize_psf.config import LocalizationConfig, Parameter, PreviewSettings
from napari_localize_psf.data import prepare_preview
from napari_localize_psf.processing import Detector, fit_spots, optical_parameters

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def test_real_detector_cache_threshold_and_blending():
    image, _, spacing, origin = gaussian_image()
    layer = Image(image * 100, scale=spacing, translate=origin, units="um")
    config = LocalizationConfig(
        voxel_size_zyx_um=spacing,
        origin_zyx_um=origin,
        blend_3d=False,
        preview=PreviewSettings(crop_zyx_px=(0, 25, 0, 41, 0, 43)),
    )
    preview = prepare_preview(layer, config)
    detector = Detector()
    first = detector.detect(preview, config)
    assert len(first.points) > 0
    stricter = detector.detect(preview, config.model_copy(update={"intensity_threshold": 0.8}))
    assert stricter.enhanced is first.enhanced
    assert set(map(tuple, stricter.points)) <= set(map(tuple, first.points))
    blended = detector.detect(preview, config.model_copy(update={"blend_3d": True}))
    assert blended.enhanced is not first.enhanced
    assert blended.enhanced.shape == image.shape
    # Reuse the real checkpoint that U-FISH loaded; changing its source must
    # invalidate enhancement even when the selected preview is unchanged.
    weights = str(detector._detector.weight_path)
    local = detector.detect(preview, config.model_copy(update={"ufish_weights": weights}))
    assert local.enhanced is not blended.enhanced
    detector.release()


def test_empty_cancelled_and_infeasible_fits():
    image, truth, spacing, origin = gaussian_image()
    config = LocalizationConfig(
        voxel_size_zyx_um=spacing,
        origin_zyx_um=origin,
        preview=PreviewSettings(crop_zyx_px=(0, 25, 0, 41, 0, 43)),
    )
    preview = prepare_preview(Image(image), config)
    assert fit_spots(preview, np.empty((0, 3)), config)["fit_params"].shape == (0, 7)
    points = ((truth[[3, 2, 1]] - origin) / spacing)[None]
    cancelled = Event()
    cancelled.set()
    with pytest.raises(CancelledError):
        fit_spots(preview, points, config, cancelled)
    parameters = dict(config.fit_parameters, cx=Parameter(lower=0.2, upper=0.4))
    with pytest.raises(ValueError, match="feasible"):
        fit_spots(preview, points, config.model_copy(update={"fit_parameters": parameters}))


def test_multiple_gpu_batches_keep_ids_fixed_parameters_and_bounds():
    image, truth, spacing, origin = gaussian_image()
    config = LocalizationConfig(
        voxel_size_zyx_um=spacing,
        origin_zyx_um=origin,
        roi_batch_size=2,
        fit_roi_size_um=(1.8, 1.5, 1.5),
        preview=PreviewSettings(crop_zyx_px=(0, 25, 0, 41, 0, 43)),
    )
    parameters = dict(config.fit_parameters, bg=Parameter(lower=5, upper=5, initial=5, fixed=True))
    config = config.model_copy(update={"fit_parameters": parameters})
    preview = prepare_preview(Image(image), config)
    point = (truth[[3, 2, 1]] - origin) / spacing
    points = np.tile(point, (5, 1))
    messages = []
    result = fit_spots(preview, points, config, progress=messages.append)
    np.testing.assert_array_equal(result["candidate_ids"], np.arange(5))
    np.testing.assert_array_equal(result["fit_params"][:, 6], np.full(5, 5))
    assert result["success"].all()
    assert len(messages) == 3
    assert "5–5 of 5" in messages[-1]


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
    assert widget._detection is not None and len(widget._detection.points) > 0
    enhanced = viewer.layers["U-FISH enhancement"]
    offset = np.array([3, 4] if geometry == "xy" else [1, 0 if plane else 3, 3, 4])
    np.testing.assert_allclose(
        enhanced.data_to_world(np.zeros(source.ndim)), source.data_to_world(offset)
    )
    priors = viewer.layers["U-FISH priors"]
    np.testing.assert_allclose(
        priors.data_to_world(priors.data[0]), source.data_to_world(priors.data[0])
    )
    cached_enhancement = widget._detection.enhanced
    widget.run_detection()
    qtbot.waitUntil(lambda: not widget._busy, timeout=60000)
    assert widget._detection.enhanced is cached_enhancement
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


@pytest.mark.parametrize(
    "points",
    [np.zeros((1, 2)), np.array([[0, 0, np.nan]]), np.array([[-1, 0, 0]]), np.array([[100, 0, 0]])],
)
def test_invalid_priors_do_not_reach_the_native_fitter(points):
    image, _, spacing, origin = gaussian_image()
    config = LocalizationConfig(
        voxel_size_zyx_um=spacing,
        origin_zyx_um=origin,
        preview=PreviewSettings(crop_zyx_px=(0, 25, 0, 41, 0, 43)),
    )
    preview = prepare_preview(Image(image), config)
    with pytest.raises(ValueError, match="Candidate coordinates"):
        fit_spots(preview, points, config)
