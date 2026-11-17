"""Unit tests of real image layers, physical transforms, and lazy slicing."""

import dask.array as da
import numpy as np
import pytest
from napari.layers import Image

from napari_localize_psf.config import LocalizationConfig, PreviewSettings
from napari_localize_psf.data import (
    layer_calibration,
    layer_coordinates,
    prepare_preview,
    source_coordinates,
)

pytestmark = pytest.mark.unit


def test_source_coordinates_keep_per_point_time_and_channel_indices():
    points = np.array([[1.2, 3.4, 5.6], [7.8, 9.0, 10.1]])
    actual = source_coordinates(
        points, "ctxyz", timepoints=np.array([3, 7]), channels=np.array([1, 2])
    )
    np.testing.assert_allclose(actual, [[1, 3, 5.6, 3.4, 1.2], [2, 7, 10.1, 9, 7.8]])
    assert source_coordinates(np.empty((0, 3)), "tzyx").shape == (0, 4)


def test_nanometer_calibration_and_permuted_cropped_coordinates():
    data = np.arange(2 * 9 * 11 * 7, dtype=np.float32).reshape(2, 9, 11, 7)
    layer = Image(
        data,
        axis_labels=("t", "y", "x", "z"),
        scale=(1, 100, 100, 200),
        translate=(0, 3000, -4000, -2000),
        units=("s", "nm", "nm", "nm"),
    )
    spacing, origin = layer_calibration(layer, "tyxz")
    np.testing.assert_allclose(spacing, (0.2, 0.1, 0.1))
    np.testing.assert_allclose(origin, (-2, 3, -4))
    selection = PreviewSettings(axes="tyxz", timepoint=1, crop_zyx_px=(1, 6, 2, 8, 3, 10))
    config = LocalizationConfig(preview=selection, voxel_size_zyx_um=spacing, origin_zyx_um=origin)
    preview = prepare_preview(layer, config)
    np.testing.assert_array_equal(preview[0], data[1, 2:8, 3:10, 1:6].transpose(2, 0, 1))
    np.testing.assert_allclose(preview[1], (-1.8, 3.2, -3.7))
    np.testing.assert_array_equal(
        layer_coordinates(np.array([[1, 2, 3.0]]), config), [[1, 4, 6, 2]]
    )


def test_lazy_multiscale_uses_highest_resolution_plane():
    array = da.arange(9 * 21 * 23, dtype=np.float32).reshape((9, 21, 23))
    layer = Image([array, array[:, ::2, ::2]], multiscale=True)
    config = LocalizationConfig(
        fit_model="2d",
        preview=PreviewSettings(axes="zyx", z_plane=4, crop_zyx_px=(0, 9, 2, 12, 3, 15)),
    )
    preview = prepare_preview(layer, config)
    assert preview[0].shape == (1, 10, 12)
    np.testing.assert_array_equal(preview[0][0], array[4, 2:12, 3:15].compute())
    np.testing.assert_array_equal(layer_coordinates(np.array([[0, 1, 2.0]]), config), [[4, 3, 5]])


def test_xy_input_and_poisson_validation():
    layer = Image(np.ones((11, 13), dtype=np.float32))
    config = LocalizationConfig(
        fit_model="2d", preview=PreviewSettings(axes="yx", crop_zyx_px=(0, 1, 0, 11, 0, 13))
    )
    assert prepare_preview(layer, config)[0].shape == (1, 11, 13)
    layer.data[0, 0] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        prepare_preview(layer, config)
    with pytest.raises(ValueError, match="Z stack"):
        prepare_preview(layer, config.model_copy(update={"fit_model": "3d"}))


def test_rotated_data_are_not_silently_treated_as_axis_aligned():
    layer = Image(np.ones((11, 13)), rotate=30)
    config = LocalizationConfig(
        fit_model="2d", preview=PreviewSettings(axes="yx", crop_zyx_px=(0, 1, 0, 11, 0, 13))
    )
    with pytest.raises(ValueError, match="resampling"):
        prepare_preview(layer, config)


@pytest.mark.parametrize("case", ["axes", "crop", "selector", "nonfinite", "small", "scale", "rgb"])
def test_invalid_data_selection_fails_explicitly(case):
    data = np.ones((2, 5, 11, 13), dtype=np.float32)
    selection = PreviewSettings(axes="tzyx", crop_zyx_px=(0, 5, 0, 11, 0, 13))
    config = LocalizationConfig(preview=selection)
    layer = Image(data)
    if case == "axes":
        config = config.model_copy(update={"preview": PreviewSettings()})
    elif case == "crop":
        selection = selection.model_copy(update={"crop_zyx_px": (0, 5, 0, 12, 0, 13)})
    elif case == "selector":
        selection = selection.model_copy(update={"timepoint": 2})
    elif case == "nonfinite":
        data[0, 0, 0, 0] = np.nan
    elif case == "small":
        selection = selection.model_copy(update={"crop_zyx_px": (0, 5, 0, 2, 0, 13)})
    elif case == "scale":
        layer.scale = (1, -1, 1, 1)
    elif case == "rgb":
        layer = Image(np.ones((5, 11, 13, 3)), rgb=True)
        selection = selection.model_copy(update={"axes": "zyx"})
    if case != "axes":
        config = config.model_copy(update={"preview": selection})
    with pytest.raises(ValueError):
        prepare_preview(layer, config)
