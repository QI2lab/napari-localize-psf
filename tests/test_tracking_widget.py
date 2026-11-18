"""Real Qt tracking previews, saved full runs, native overlays, and settings reloads."""

import json

import numpy as np
import pytest
from napari.layers import Tracks
from napari.utils.transforms import Affine

from napari_localize_psf.widget import LocalizationWidget

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("axes", ["tzyx", "tyxz", "tyx"])
def test_widget_tunes_executes_saves_and_reloads_tracks(viewer, qtbot, localizations, axes):
    path, config = localizations
    shape = {"t": 5, "z": 12, "y": 30, "x": 40}
    scales = {
        "t": 0.015,
        **dict(zip("zyx", np.asarray(config.voxel_size_zyx_um) * 1000, strict=True)),
    }
    offsets = {"t": 3, **dict(zip("zyx", np.asarray(config.origin_zyx_um) * 1000, strict=True))}
    source = viewer.add_image(
        np.zeros(tuple(shape[a] for a in axes), dtype=np.uint8),
        name="counts",
        axis_labels=tuple(axes),
        scale=tuple(scales[a] for a in axes),
        translate=tuple(offsets[a] for a in axes),
        units=tuple("s" if a == "t" else "nm" for a in axes),
        affine=Affine(translate=np.arange(len(axes)) + 5),
        metadata={"napari_localize_psf": {"input_uri": config.input_uri, "source_channel": 0}},
    )
    widget = LocalizationWidget(viewer)
    qtbot.addWidget(widget)
    titles = [widget.tabs.tabText(i) for i in range(widget.tabs.count())]
    assert titles.index("Tracking") == titles.index("Batch results") + 1
    assert [widget.tracking_tabs.tabText(i) for i in range(2)] == ["Tune", "Run / results"]
    widget.tracking_path.setText(str(path))
    assert widget.results_path.text() == str(path)
    widget.tracking_all_channels.setChecked(True)
    widget.tracking_memory.setValue(1)
    for box in widget.tracking_ranges.values():
        box.setValue(0.3)
    widget.tracking_frames["first"].setValue(1)
    widget.tracking_frames["last"].setValue(3)
    widget.tracking_preview_button.click()
    assert widget._busy and not widget.tabs.isEnabled()
    qtbot.waitUntil(lambda: not widget._busy, timeout=30000)
    assert "Tracking preview: counts" in viewer.layers, widget.status.text()
    preview = viewer.layers["Tracking preview: counts"]
    assert isinstance(preview, Tracks)
    assert set(preview.features["frame"]) == {1, 2, 3}
    assert not path.with_name("tracks.parquet").exists()
    assert not path.with_name("msd_individual.parquet").exists()
    assert widget.tracking_frame_interval.value() == 0.015
    widget.tracking_run_button.click()
    qtbot.waitUntil(lambda: not widget._busy, timeout=30000)
    assert "Tracks: counts" in viewer.layers, widget.status.text()
    overlay = viewer.layers["Tracks: counts"]
    assert set(overlay.features["frame"]) == {0, 1, 2, 3, 4}
    assert overlay.features["particle"].nunique() == 3
    columns = {"t": "frame", "z": "z", "y": "y", "x": "x"}
    expected = overlay.features[[columns[a] for a in axes]].to_numpy()
    np.testing.assert_allclose(overlay.data[:, 1:], expected)
    for point in expected:
        np.testing.assert_allclose(overlay.data_to_world(point), source.data_to_world(point))
    assert path.with_name("tracks.parquet").is_file()
    saved = json.loads(path.with_name("tracking.json").read_text())
    assert saved["tracking"]["first_frame"] is None and saved["tracking"]["last_frame"] is None
    assert saved["tracking"]["frame_interval_s"] == 0.015
    for name in saved["tracking"]["analysis"]["files"]:
        assert path.with_name(name).is_file()
    original = overlay.data.copy()
    viewer.layers.remove(overlay)
    widget.tracking_memory.setValue(0)
    widget.tracking_frame_interval.setValue(0.5)
    widget.tracking_load_button.click()
    qtbot.waitUntil(lambda: not widget._busy, timeout=10000)
    np.testing.assert_array_equal(viewer.layers["Tracks: counts"].data, original)
    assert widget.tracking_memory.value() == 1
    assert widget.tracking_frame_interval.value() == 0.015
    before = path.with_name("tracks.parquet").read_bytes()
    widget.tracking_min_length.setValue(100)
    widget.tracking_preview_button.click()
    qtbot.waitUntil(lambda: not widget._busy, timeout=10000)
    assert "Tracking preview: counts" not in viewer.layers
    assert "0 tracks" in widget.tracking_preview_status.text()
    assert path.with_name("tracks.parquet").read_bytes() == before
    assert widget.tabs.isEnabled() and not widget.cancel_button.isEnabled()


def test_widget_rejects_unsupported_source_and_bad_frame_range(viewer, qtbot, localizations):
    path, _ = localizations
    viewer.add_image(np.zeros((5, 12, 30, 40)), axis_labels=("t", "z", "y", "x"))
    widget = LocalizationWidget(viewer)
    qtbot.addWidget(widget)
    widget.tracking_path.setText(str(path))
    widget.tracking_frames["first"].setValue(4)
    widget.tracking_frames["last"].setValue(1)
    widget.run_tracking("preview")
    qtbot.waitUntil(lambda: not widget._busy, timeout=10000)
    assert "first tuning frame" in widget.tracking_preview_status.text()
    widget.axes.setText("czyx")
    widget.run_tracking("preview")
    assert "time-first channel image" in widget.tracking_preview_status.text()
    assert not widget._busy
