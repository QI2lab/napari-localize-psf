"""Real Trackpy integration with simulated localization coordinates and Parquet IO."""

import json
import re
from concurrent.futures import CancelledError
from threading import Event

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pypdf import PdfReader

from napari_localize_psf.tracking import analyze_tracks, link_localizations, write_analysis_pdf

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


@pytest.mark.parametrize("frame_interval_s", [None, 0.02])
def test_frame_selection_minimum_length_empty_and_saved_results(localizations, frame_interval_s):
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
    assert not path.with_name("tracking_analysis.pdf").exists()
    tracks, metadata = link_localizations(
        path,
        search_range_um=(0.3, 0.3, 0.3),
        memory=1,
        channel=1,
        save=True,
        frame_interval_s=frame_interval_s,
    )
    pd.testing.assert_frame_equal(pd.read_parquet(path.with_name("tracks.parquet")), tracks)
    assert json.loads(path.with_name("tracking.json").read_text()) == metadata
    assert metadata["origin_zyx_um"] == list(config.origin_zyx_um)
    assert metadata["tracking"]["first_frame"] is None
    assert metadata["tracking"]["channel"] == 1
    assert metadata["tracking"]["frame_interval_s"] == frame_interval_s
    analysis = metadata["tracking"]["analysis"]
    assert len(analysis["files"]) == 4
    reader = PdfReader(path.with_name(analysis["report_file"]))
    assert len(reader.pages) == 6
    page_index = 0
    for kind in ("Step-size distribution", "Ensemble MSD", "Individual MSDs"):
        for component in "rzyx" if kind == "Step-size distribution" else "r":
            title = (
                f"{kind}: {component} (channel 1)"
                if kind == "Step-size distribution"
                else f"{kind} (channel 1)"
            )
            text = reader.pages[page_index].extract_text()
            assert text.count(title) == 1
            if kind == "Step-size distribution":
                xlabel = f"{'r' if component == 'r' else 'Δ' + component} (µm)"
                ylabel = "Probability density (µm⁻¹)"
            else:
                xlabel = "Lag (frames)" if frame_interval_s is None else "Lag time (s)"
                ylabel = "MSD (µm²)"
            for label in (title, xlabel, ylabel):
                assert label in text
                text = text.replace(label, "")
            assert re.fullmatch(r"[\s0-9eE+.−-]*", text)
            page_index += 1
    for name in analysis["files"]:
        schema = pq.read_schema(path.with_name(name))
        assert json.loads(schema.metadata[b"napari_localize_psf_tracking"]) == metadata
    steps = pd.read_parquet(path.with_name("step_sizes.parquet"))
    np.testing.assert_array_equal(steps["frame_start"], np.arange(4))
    np.testing.assert_allclose(steps["dx_um"], 0.1)
    np.testing.assert_allclose(steps["r_um"], 0.1)
    individual = pd.read_parquet(path.with_name("msd_individual.parquet"))
    ensemble = pd.read_parquet(path.with_name("msd_ensemble.parquet"))
    for table in (individual, ensemble):
        np.testing.assert_allclose(table["msd_r_um2"], 0.01 * table["lag_frames"] ** 2, atol=1e-12)
        if frame_interval_s is None:
            assert table["lag_time_s"].isna().all()
        else:
            np.testing.assert_allclose(table["lag_time_s"], table["lag_frames"] * frame_interval_s)
    distributions = pd.read_parquet(path.with_name("step_size_distributions.parquet"))
    assert set(distributions["component"]) == {"r", "z", "y", "x"}
    assert distributions.groupby("component")["count"].sum().eq(4).all()
    np.testing.assert_allclose(distributions.groupby("component")["probability"].sum(), 1)
    empty, _ = link_localizations(
        path, search_range_um=(0.3, 0.3, 0.3), first_frame=10, last_frame=12
    )
    assert empty.empty and empty["particle"].dtype == np.int64


def test_analysis_keeps_channels_signed_components_gaps_and_all_frame_lags(localizations):
    path, _ = localizations
    tracks, _ = link_localizations(path, search_range_um=(0.3, 0.3, 0.3), memory=1)
    outputs = analyze_tracks(tracks, frame_interval_s=0.015)
    steps = outputs["step_sizes.parquet"]
    assert len(steps) == 10
    assert set(steps["channel"]) == {0, 1}
    assert steps["frame_end"].sub(steps["frame_start"]).eq(1).all()
    gapped = tracks.loc[(tracks["channel"] == 0) & tracks["yum"].eq(4), "particle"].iloc[0]
    assert steps.loc[steps["particle"] == gapped, "frame_start"].tolist() == [0, 3]
    frames = np.arange(105)
    # Simulate GPU-fit coordinates with float32 storage and a large physical origin.
    velocity = np.array([0.125, -0.0625, 0.03125])
    origin = np.array([6.25, 15.5, 135.5])
    motion = pd.DataFrame(
        {
            "particle": 7,
            "channel": 0,
            "frame": frames,
            **{
                f"{axis}um": (origin[index] + velocity[index] * frames).astype(np.float32)
                for index, axis in enumerate("zyx")
            },
        }
    )
    opposite = motion.assign(
        channel=1,
        **{
            f"{axis}um": (origin[index] - velocity[index] * frames).astype(np.float32)
            for index, axis in enumerate("zyx")
        },
    )
    outputs = analyze_tracks(
        pd.concat([motion, opposite], ignore_index=True), frame_interval_s=0.02
    )
    samples = outputs["step_sizes.parquet"]
    histograms = outputs["step_size_distributions.parquet"]
    shared = histograms.loc[(histograms["channel"] == 0) & (histograms["component"] == "r")]
    for _, histogram in histograms.groupby(["channel", "component"]):
        np.testing.assert_array_equal(histogram["bin_left_um"], shared["bin_left_um"])
        np.testing.assert_array_equal(histogram["bin_right_um"], shared["bin_right_um"])
    report = path.with_name("shared-limits.pdf")
    write_analysis_pdf(outputs, report, frame_interval_s=0.02)
    reader = PdfReader(report)
    assert len(reader.pages) == 12
    reference_ticks = None
    for page_index in [*range(4), *range(6, 10)]:
        text = reader.pages[page_index].extract_text().rsplit("\n", 1)[0]
        ticks = re.findall(r"[−-]?\d+(?:\.\d+)?(?:e[−+-]?\d+)?", text)
        assert ticks
        if reference_ticks is None:
            reference_ticks = ticks
        assert ticks == reference_ticks
    for channel, sign in ((0, 1), (1, -1)):
        selected = samples.loc[samples["channel"] == channel]
        np.testing.assert_allclose(
            selected[["dz_um", "dy_um", "dx_um"]],
            np.tile(sign * velocity, (104, 1)),
        )
        np.testing.assert_allclose(selected["r_um"], np.linalg.norm(velocity))
    for name in ("msd_individual.parquet", "msd_ensemble.parquet"):
        table = outputs[name]
        assert table.groupby("channel")["lag_frames"].max().eq(104).all()
        np.testing.assert_allclose(
            table["msd_r_um2"], np.sum(velocity**2) * table["lag_frames"] ** 2, atol=1e-8
        )
        for axis, speed in zip("zyx", velocity, strict=True):
            np.testing.assert_allclose(
                table[f"msd_{axis}_um2"], speed**2 * table["lag_frames"] ** 2, atol=1e-8
            )
        np.testing.assert_allclose(table["lag_time_s"], 0.02 * table["lag_frames"])


@pytest.mark.parametrize(
    "cancel_stage",
    ["Analyzing", "Saving msd_individual.parquet", "Writing PDF: Step-size distribution: z"],
)
def test_empty_analysis_and_cancelled_analysis_preserve_saved_files(localizations, cancel_stage):
    path, _ = localizations
    tracks, metadata = link_localizations(
        path, search_range_um=(0.3, 0.3, 0.3), memory=1, save=True
    )
    filenames = [
        "tracks.parquet",
        "tracking.json",
        metadata["tracking"]["analysis"]["report_file"],
        *metadata["tracking"]["analysis"]["files"],
    ]
    before = {name: path.with_name(name).read_bytes() for name in filenames}
    cancelled = Event()

    def cancel_analysis(message):
        if message.startswith(cancel_stage):
            cancelled.set()

    with pytest.raises(CancelledError):
        link_localizations(
            path,
            search_range_um=(0.3, 0.3, 0.3),
            memory=1,
            save=True,
            cancelled=cancelled,
            progress=cancel_analysis,
        )
    assert {name: path.with_name(name).read_bytes() for name in filenames} == before
    assert not list(path.parent.glob(".tracking-*"))
    singleton = analyze_tracks(tracks.iloc[:1])
    assert all(table.empty for table in singleton.values())
    empty, metadata = link_localizations(
        path, search_range_um=(0.3, 0.3, 0.3), min_length=100, save=True
    )
    assert empty.empty
    for name in metadata["tracking"]["analysis"]["files"]:
        table = pd.read_parquet(path.with_name(name))
        assert table.empty and len(table.columns) > 0
    assert len(PdfReader(path.with_name("tracking_analysis.pdf")).pages) == 6


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
        {"frame_interval_s": 0},
        {"frame_interval_s": float("nan")},
        {"input_uri": str(path.parent / "other.ome.zarr")},
    ):
        with pytest.raises(ValueError):
            link_localizations(path, search_range_um=(0.3, 0.3, 0.3), **settings)
