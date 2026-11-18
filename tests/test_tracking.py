"""Real Trackpy integration with simulated localization coordinates and Parquet IO."""

import json
from concurrent.futures import CancelledError
from threading import Event

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from napari_localize_psf.tracking import link_localizations

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("memory, expected_tracks", [(0, 4), (1, 3)])
def test_real_linking_preserves_units_gaps_channels_and_candidate_ids(
    localizations, memory, expected_tracks
):
    path, _ = localizations
    before = path.read_bytes()
    tracks, metadata = link_localizations(
        path, search_range_um=(0.3, 0.3, 0.3), memory=memory, min_length=2
    )
    assert len(tracks) == 14
    assert tracks["particle"].nunique() == expected_tracks
    assert tracks.groupby("particle")["channel"].nunique().eq(1).all()
    assert set(tracks["candidate_id"]) == set(range(14))
    assert metadata["tracking"]["skipped_nonfinite"] == 1
    np.testing.assert_allclose(tracks["xum"], -2 + 0.1 * tracks["frame"])
    np.testing.assert_allclose(tracks["x"], (tracks["xum"] + 4) / 0.15)
    assert path.read_bytes() == before
    assert not path.with_name("tracks.parquet").exists()


def test_frame_selection_minimum_length_empty_and_saved_results(localizations):
    path, config = localizations
    preview, metadata = link_localizations(
        path,
        search_range_um=(0.3, 0.3, 0.3),
        memory=1,
        min_length=3,
        first_frame=1,
        last_frame=3,
    )
    assert set(preview["frame"]) == {1, 2, 3}
    assert len(preview) == 6 and preview["particle"].nunique() == 2
    assert metadata["tracking"]["first_frame"] == 1
    assert metadata["tracking"]["last_frame"] == 3
    assert not path.with_name("tracking.json").exists()
    tracks, metadata = link_localizations(
        path, search_range_um=(0.3, 0.3, 0.3), memory=1, channel=1, save=True
    )
    pd.testing.assert_frame_equal(pd.read_parquet(path.with_name("tracks.parquet")), tracks)
    assert json.loads(path.with_name("tracking.json").read_text()) == metadata
    assert metadata["origin_zyx_um"] == list(config.origin_zyx_um)
    assert metadata["tracking"]["first_frame"] is None
    assert metadata["tracking"]["channel"] == 1
    empty, _ = link_localizations(
        path, search_range_um=(0.3, 0.3, 0.3), first_frame=10, last_frame=12
    )
    assert empty.empty and empty["particle"].dtype == np.int64


@pytest.mark.parametrize("dimensions, expected_tracks", [("2d", 1), ("3d", 0)])
def test_dimensional_selection_uses_physical_positions(localizations, dimensions, expected_tracks):
    path, _ = localizations
    table = pq.read_table(path).slice(0, 2)
    parameters = np.asarray(table["fit_params"].to_pylist())
    parameters[1, 3] += 1
    table = table.set_column(
        table.schema.get_field_index("fit_params"), "fit_params", pa.array(parameters.tolist())
    )
    pq.write_table(table, path)
    tracks, _ = link_localizations(
        path, search_range_um=(0.2, 0.3, 0.3), dimensions=dimensions, min_length=2
    )
    assert tracks["particle"].nunique() == expected_tracks


def test_cancellation_and_invalid_settings_do_not_write_results(localizations):
    path, _ = localizations
    cancelled = Event()

    def cancel_on_progress(message):
        cancelled.set()

    with pytest.raises(CancelledError):
        link_localizations(
            path,
            search_range_um=(0.3, 0.3, 0.3),
            save=True,
            cancelled=cancelled,
            progress=cancel_on_progress,
        )
    assert not path.with_name("tracks.parquet").exists()
    assert not path.with_name("tracking.json").exists()
    for settings in (
        {"first_frame": 4, "last_frame": 1},
        {"memory": -1},
        {"min_length": 0},
        {"adaptive_stop_um": 0.4},
        {"channel": -1},
        {"input_uri": str(path.parent / "other.ome.zarr")},
    ):
        with pytest.raises(ValueError):
            link_localizations(path, search_range_um=(0.3, 0.3, 0.3), **settings)
