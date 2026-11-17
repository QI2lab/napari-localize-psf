"""Image overlays from real upstream fits, Parquet exports and Qt workers."""

import numpy as np
import pytest
from localize_psf.batch_fitting import fit_rois
from localize_psf.fit_psf import gaussian3d_psf_model
from localize_psf.localize import get_coords, prepare_rois
from localize_psf.rois import get_centered_rois, get_roi_size
from localize_psf.workflow import read_localizations, save_localizations
from napari.layers import Points
from napari.utils.transforms import Affine
from simulator import gaussian_image

from napari_localize_psf.config import LocalizationConfig
from napari_localize_psf.widget import LocalizationWidget

pytestmark = pytest.mark.integration


@pytest.fixture
def batch_results(tmp_path):
    """Export actual CPU fits of a simulated emitter at several T/C indices.

    This only prepares input files for plugin integration. It neither substitutes
    a fitting backend nor checks the upstream solver's numerical behavior.
    """
    image, truth, spacing, origin = gaussian_image()
    centers_um = np.broadcast_to(truth[[3, 2, 1]], (4, 3)).copy()
    centers_px = (centers_um - origin) / spacing
    rois = get_centered_rois(
        centers_px,
        get_roi_size((1.8, 1.5, 1.5), spacing),
        min_vals=np.zeros(3, int),
        max_vals=np.asarray(image.shape),
    )
    values, coordinates, counts = prepare_rois(
        image, get_coords(image.shape, spacing, origin), rois
    )
    model = gaussian3d_psf_model()
    fits = fit_rois(
        values,
        coordinates,
        counts,
        model.estimate_parameters(values, coordinates, num_preserved_dims=1),
        model=model,
        use_gpu=False,
        return_covariance=False,
    )
    input_path = tmp_path / "source.ome.zarr"
    config = LocalizationConfig(
        input_uri=str(input_path), voxel_size_zyx_um=spacing, origin_zyx_um=origin
    )
    output = tmp_path / "batch"
    save_localizations(
        {
            "metadata": dict(config.model_dump(mode="json"), axes="tczyx", psf_diagnostics=False),
            "parameter_names": model.parameter_names,
            "candidate_centroids_zyx_px": centers_px,
            "candidate_centroids_zyx_um": centers_um,
            "candidate_timepoints": np.array([0, 1, 1, 2]),
            "candidate_channels": np.array([0, 0, 1, 1]),
            "roi_bounds_zyx": rois,
            "fit_results": fits,
            "volume_results": {},
        },
        output,
    )
    return output, image, spacing, origin, input_path


@pytest.mark.parametrize("axes", ["tczyx", "tyxz", "yx"])
def test_batch_overlay_uses_saved_geometry_and_source_transforms(
    viewer, qtbot, batch_results, axes
):
    output, image, spacing, origin, input_path = batch_results
    shape = {"t": 3, "c": 2, **dict(zip("zyx", image.shape, strict=True))}
    scales = {"t": 0.015, "c": 1, **dict(zip("zyx", np.asarray(spacing) * 1000, strict=True))}
    offsets = {"t": 3, "c": 0, **dict(zip("zyx", np.asarray(origin) * 1000, strict=True))}
    source = viewer.add_image(
        np.ones(tuple(shape[a] for a in axes), np.uint8),
        name="counts",
        axis_labels=tuple(axes),
        scale=tuple(scales[a] for a in axes),
        translate=tuple(offsets[a] for a in axes),
        units=tuple("s" if a == "t" else "nm" for a in axes),
        affine=Affine(translate=np.arange(len(axes)) + 5),
        metadata={"napari_localize_psf": {"input_uri": str(input_path), "source_channel": 1}},
    )
    widget = LocalizationWidget(viewer)
    qtbot.addWidget(widget)
    assert "Batch results" in [widget.tabs.tabText(i) for i in range(widget.tabs.count())]
    widget.output.setText(str(output))
    assert widget.results_path.text() == str(output)
    # Stored results ignore the current preview crop and acceptance controls.
    widget.geometry.item(1, 0).setText("9")
    widget.amplitude_min.setText("100000")
    if "t" not in axes:
        widget.results_timepoint.setValue(1)
    widget.load_batch_results()
    qtbot.waitUntil(lambda: not widget._busy, timeout=10000)
    name = "Batch localizations: counts"
    assert name in viewer.layers, widget.status.text()
    overlay = viewer.layers[name]
    features, metadata = read_localizations(
        output, timepoint=None if "t" in axes else 1, channel=None if "c" in axes else 1
    )
    columns = {"t": "frame", "c": "channel", "z": "z", "y": "y", "x": "x"}
    expected = features[[columns[a] for a in axes]].to_numpy()
    np.testing.assert_allclose(overlay.data, expected)
    for actual, point in zip(overlay.data, expected, strict=True):
        np.testing.assert_allclose(overlay.data_to_world(actual), source.data_to_world(point))
    assert overlay.units == source.units
    assert overlay.axis_labels == source.axis_labels
    assert overlay.features.equals(features)
    assert overlay.metadata["configuration"] == metadata
    assert overlay.metadata["batch_results_path"] == str(output / "localizations.parquet")
    assert widget._fit_layer is None and widget.detector._detector is None
    assert widget.results_all_channels.isEnabled() == ("c" in axes)
    assert widget.results_channel.isEnabled() is False
    widget.load_batch_results()
    qtbot.waitUntil(lambda: not widget._busy, timeout=10000)
    assert viewer.layers[name] != overlay
    assert sum(isinstance(layer, Points) for layer in viewer.layers) == 1


def test_batch_scope_selection_and_empty_overlay(viewer, qtbot, batch_results):
    output, image, spacing, origin, _ = batch_results
    viewer.add_image(
        np.broadcast_to(image, (3, 2, *image.shape)),
        axis_labels=tuple("tczyx"),
        scale=(1, 1, *spacing),
        translate=(0, 0, *origin),
        units="um",
    )
    widget = LocalizationWidget(viewer)
    qtbot.addWidget(widget)
    widget.results_path.setText(str(output / "localizations.parquet"))
    widget.results_all_timepoints.setChecked(False)
    widget.results_all_channels.setChecked(False)
    widget.results_timepoint.setValue(1)
    widget.results_channel.setValue(1)
    widget.load_batch_results()
    qtbot.waitUntil(lambda: not widget._busy, timeout=10000)
    overlay = viewer.layers[-1]
    assert len(overlay.data) == 1, widget.status.text()
    assert overlay.features["frame"].tolist() == [1]
    assert overlay.features["channel"].tolist() == [1]
    widget.load_batch_results()
    widget.cancel()
    qtbot.waitUntil(lambda: not widget._busy, timeout=10000)
    assert viewer.layers[-1] == overlay
    widget.load_batch_results()
    widget._layer.translate = (0, 0, 2, 3, 4)
    qtbot.waitUntil(lambda: not widget._busy, timeout=10000)
    assert viewer.layers[-1] == overlay
    widget.results_timepoint.setValue(2)
    widget.results_channel.setValue(0)
    widget.load_batch_results()
    qtbot.waitUntil(lambda: not widget._busy, timeout=10000)
    assert len(viewer.layers[-1].data) == 0
    assert "Showing 0" in widget.results_status.text()
    assert len(viewer.layers[-1].features.columns) > 0


def test_unavailable_and_invalid_results_do_not_publish_overlays(viewer, qtbot, batch_results):
    output, image, _, _, input_path = batch_results
    widget = LocalizationWidget(viewer)
    qtbot.addWidget(widget)
    widget.results_path.setText(str(output))
    widget.load_batch_results()
    assert "Select the matching image" in widget.results_status.text()
    source = viewer.add_image(image, units="um")
    widget.results_path.setText(str(output.parent / "not-produced-yet"))
    widget.load_batch_results()
    assert "not available yet" in widget.results_status.text()
    assert widget._worker is None
    invalid = output / "centroids.parquet"
    # A centroid checkpoint is a real input file, not a final Gaussian fit table.
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"timepoint": [0], "channel": [0], "z_px": [1.0]}), invalid)
    widget.results_path.setText(str(invalid))
    widget.load_batch_results()
    qtbot.waitUntil(lambda: not widget._busy, timeout=10000)
    assert "localize-psf localizations.parquet" in widget.results_status.text()
    source.metadata["napari_localize_psf"] = {"input_uri": str(input_path.parent / "other.zarr")}
    widget.results_path.setText(str(output))
    widget.load_batch_results()
    qtbot.waitUntil(lambda: not widget._busy, timeout=10000)
    assert "matching input image" in widget.results_status.text()
    assert not any(isinstance(layer, Points) for layer in viewer.layers)
    assert widget.detector._detector is None
