"""Trackpy adapters for saved localizations; no detection, fitting, or image IO."""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import CancelledError
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from typing import Any, Literal

import numpy as np
import pandas as pd


def analyze_tracks(
    tracks: pd.DataFrame,
    *,
    frame_interval_s: float | None = None,
    cancelled: Event | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, pd.DataFrame]:
    """Adapt Trackpy motion statistics to four analysis Parquet tables.

    Input rows have unique (channel, particle, frame) identities and finite
    physical ``zum/yum/xum`` positions in µm. Signed components and radial
    magnitude use ZYX, including Z when linking used an XY search. Planar data
    have constant Z. Only adjacent-frame observations contribute to step
    distributions; gaps are never treated as one-frame steps. Marginals use
    one set of NumPy Freedman–Diaconis histogram bins pooled across every
    component and channel, retaining raw step samples too.

    Trackpy computes individual MSD components and its N-weighted ensemble MSD
    independently per channel, through every available frame lag. N is its
    effective independent measurement count, rather than a count of pairs.
    Positions remain in µm (mpp=1); squared displacements are in µm². Missing
    frame calibration leaves lag_time_s null; lag_frames is always available.
    No drift subtraction or diffusion-model fit is applied.
    Physical coordinates are promoted to float64 before Trackpy's arithmetic
    to avoid cancellation errors from float32 fits with large global origins.
    """
    import trackpy as tp

    if frame_interval_s is not None and (
        not np.isfinite(frame_interval_s) or frame_interval_s <= 0
    ):
        raise ValueError("Frame interval must be a positive finite number of seconds")
    required = {"channel", "particle", "frame", "zum", "yum", "xum"}
    if not required <= set(tracks.columns):
        raise ValueError("Track analysis requires channel, particle, frame and physical Z/Y/X")
    if tracks.duplicated(["channel", "particle", "frame"]).any():
        raise ValueError("Each particle must have at most one observation per frame and channel")
    step_types = dict(
        channel="int64",
        particle="int64",
        frame_start="int64",
        frame_end="int64",
        lag_frames="int64",
        lag_time_s="float64",
        dz_um="float64",
        dy_um="float64",
        dx_um="float64",
        r_um="float64",
    )
    msd_types = dict(
        channel="int64",
        lag_frames="int64",
        lag_time_s="float64",
        mean_dz_um="float64",
        mean_dy_um="float64",
        mean_dx_um="float64",
        msd_z_um2="float64",
        msd_y_um2="float64",
        msd_x_um2="float64",
        msd_r_um2="float64",
        n_effective="float64",
    )
    histogram_types = dict(
        channel="int64",
        component="string",
        bin_left_um="float64",
        bin_right_um="float64",
        count="int64",
        n_steps="int64",
        probability="float64",
        density_per_um="float64",
    )
    schemas = {
        "step_sizes.parquet": step_types,
        "step_size_distributions.parquet": histogram_types,
        "msd_individual.parquet": {"particle": "int64", **msd_types},
        "msd_ensemble.parquet": msd_types,
    }
    collected: dict[str, list[pd.DataFrame]] = {name: [] for name in schemas}
    renamed = {
        **{f"<{axis}>": f"mean_d{axis}_um" for axis in "zyx"},
        **{f"<{axis}^2>": f"msd_{axis}_um2" for axis in "zyx"},
        "msd": "msd_r_um2",
        "N": "n_effective",
        "lagt": "lag_time_s",
    }
    fps = 1.0 if frame_interval_s is None else 1.0 / frame_interval_s
    for channel, rows in tracks.groupby("channel", sort=True):
        if cancelled is not None and cancelled.is_set():
            raise CancelledError("Track analysis cancelled")
        if progress is not None:
            progress(f"Analyzing step sizes for channel {channel}")
        positions = (
            rows[["particle", "frame", "zum", "yum", "xum"]]
            .rename(columns={"zum": "z", "yum": "y", "xum": "x"})
            .astype({axis: "float64" for axis in "zyx"})
            .sort_values(["particle", "frame"], kind="stable")
        )
        previous, previous_frame = None, None
        for frame, current in positions.groupby("frame", sort=True):
            if cancelled is not None and cancelled.is_set():
                raise CancelledError("Track analysis cancelled")
            if previous is not None and frame == previous_frame + 1:
                related = tp.relate_frames(
                    pd.concat((previous, current)),
                    previous_frame,
                    frame,
                    pos_columns=["z", "y", "x"],
                ).dropna(subset=["dz", "dy", "dx"])
                steps = (
                    related[["dz", "dy", "dx", "dr"]]
                    .rename(columns={"dz": "dz_um", "dy": "dy_um", "dx": "dx_um", "dr": "r_um"})
                    .reset_index()
                )
                if len(steps):
                    collected["step_sizes.parquet"].append(
                        steps.assign(
                            channel=channel,
                            frame_start=previous_frame,
                            frame_end=frame,
                            lag_frames=1,
                            lag_time_s=frame_interval_s if frame_interval_s is not None else np.nan,
                        )
                    )
            previous, previous_frame = current, frame
        groups = positions.groupby("particle", sort=True)
        eligible = []
        for index, (particle, trajectory) in enumerate(groups, 1):
            if cancelled is not None and cancelled.is_set():
                raise CancelledError("Track analysis cancelled")
            if len(trajectory) < 2:
                continue
            if progress is not None and (index % 100 == 0 or index == len(groups)):
                progress(f"Individual MSD, channel {channel}: track {index} / {len(groups)}")
            result = tp.msd(
                trajectory,
                mpp=1,
                fps=fps,
                max_lagtime=int(trajectory["frame"].max() - trajectory["frame"].min()),
                detail=True,
                pos_columns=["z", "y", "x"],
            ).rename(columns=renamed)
            result.index.name = "lag_frames"
            if frame_interval_s is None:
                result["lag_time_s"] = np.nan
            collected["msd_individual.parquet"].append(
                result.reset_index().assign(channel=channel, particle=particle)
            )
            eligible.append(trajectory)
        if eligible:
            if progress is not None:
                progress(f"Ensemble MSD for channel {channel}")
            if cancelled is not None and cancelled.is_set():
                raise CancelledError("Track analysis cancelled")
            ensemble = tp.emsd(
                pd.concat(eligible),
                mpp=1,
                fps=fps,
                max_lagtime=int(positions["frame"].max() - positions["frame"].min()),
                detail=True,
                pos_columns=["z", "y", "x"],
            ).rename(columns=renamed)
            ensemble.index.name = "lag_frames"
            if frame_interval_s is None:
                ensemble["lag_time_s"] = np.nan
            collected["msd_ensemble.parquet"].append(ensemble.reset_index().assign(channel=channel))
    step_samples = collected["step_sizes.parquet"]
    if step_samples:
        samples = pd.concat(step_samples, ignore_index=True)
        edges = np.histogram_bin_edges(
            samples[["r_um", "dz_um", "dy_um", "dx_um"]].to_numpy().ravel(), bins="fd"
        )
        for channel, steps in samples.groupby("channel", sort=True):
            for component, column in (
                ("r", "r_um"),
                ("z", "dz_um"),
                ("y", "dy_um"),
                ("x", "dx_um"),
            ):
                if cancelled is not None and cancelled.is_set():
                    raise CancelledError("Track analysis cancelled")
                values = steps[column].to_numpy()
                counts, _ = np.histogram(values, bins=edges)
                collected["step_size_distributions.parquet"].append(
                    pd.DataFrame(
                        dict(
                            channel=channel,
                            component=component,
                            bin_left_um=edges[:-1],
                            bin_right_um=edges[1:],
                            count=counts,
                            n_steps=len(values),
                            probability=counts / len(values),
                            density_per_um=counts / (len(values) * np.diff(edges)),
                        )
                    )
                )
    if cancelled is not None and cancelled.is_set():
        raise CancelledError("Track analysis cancelled")
    return {
        name: pd.concat(collected[name], ignore_index=True)[list(types)].astype(types)
        if collected[name]
        else pd.DataFrame({column: pd.Series(dtype=dtype) for column, dtype in types.items()})
        for name, types in schemas.items()
    }


def write_analysis_pdf(
    analyses: dict[str, pd.DataFrame],
    path: str | Path,
    *,
    frame_interval_s: float | None = None,
    cancelled: Event | None = None,
    progress: Callable[[str], None] | None = None,
) -> None:
    """Plot saved analysis values, one titled/labeled axes per PDF page.

    Each channel has six pages: r/Z/Y/X marginals, total ensemble MSD and all
    individual total MSD curves on one axes. All step pages share
    the exported bin edges and identical x/y limits across channels. Step
    probabilities are unitless. Distances use µm, squared displacements µm²,
    and lag times seconds when calibrated.
    Empty results produce labeled empty plots. No legends or narrative text
    are added. Figure/PDF objects are used directly without a GUI backend.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure

    distributions = analyses["step_size_distributions.parquet"]
    individual = analyses["msd_individual.parquet"]
    ensemble = analyses["msd_ensemble.parquet"]
    channels = sorted(
        {int(channel) for table in analyses.values() for channel in table["channel"].unique()}
    )
    x_limits = (
        (0.0, 1.0)
        if distributions.empty
        else (float(distributions["bin_left_um"].min()), float(distributions["bin_right_um"].max()))
    )
    peak = 0.0 if distributions.empty else float(distributions["probability"].max())
    y_limits = (0.0, peak * 1.05 if peak > 0 else 1.0)
    lag_column = "lag_frames" if frame_interval_s is None else "lag_time_s"
    lag_label = "Lag (frames)" if frame_interval_s is None else "Lag time (s)"
    with PdfPages(
        path, metadata={"Title": "Tracking analysis", "Creator": "napari-localize-psf"}
    ) as pdf:
        for channel in channels or [None]:
            suffix = "" if channel is None else f" (channel {channel})"
            for kind, table in (
                ("Step-size distribution", distributions),
                ("Ensemble MSD", ensemble),
                ("Individual MSDs", individual),
            ):
                selected = (
                    table.loc[table["channel"] == channel]
                    if channel is not None
                    else table.iloc[:0]
                )
                for component in "rzyx" if kind == "Step-size distribution" else "r":
                    title = (
                        f"{kind}: {component}{suffix}"
                        if kind == "Step-size distribution"
                        else f"{kind}{suffix}"
                    )
                    if progress is not None:
                        progress(f"Writing PDF: {title}")
                    if cancelled is not None and cancelled.is_set():
                        raise CancelledError("Tracking report cancelled")
                    figure = Figure(figsize=(7.5, 5.5), layout="constrained")
                    try:
                        axis = figure.subplots()
                        if kind == "Step-size distribution":
                            bins = selected.loc[selected["component"] == component].sort_values(
                                "bin_left_um"
                            )
                            if len(bins):
                                edges = np.r_[
                                    bins["bin_left_um"].to_numpy(), bins["bin_right_um"].iloc[-1]
                                ]
                                axis.stairs(
                                    bins["probability"].to_numpy(),
                                    edges,
                                    fill=True,
                                    color="tab:blue",
                                    alpha=0.7,
                                )
                            axis.set(
                                xlim=x_limits,
                                ylim=y_limits,
                                xlabel=f"{'r' if component == 'r' else 'Δ' + component} (µm)",
                                ylabel="Probability per bin",
                            )
                        else:
                            column = f"msd_{component}_um2"
                            if kind == "Individual MSDs" and len(selected):
                                curves = [
                                    trajectory[[lag_column, column]].to_numpy()
                                    for _, trajectory in selected.groupby("particle", sort=True)
                                ]
                                axis.add_collection(
                                    LineCollection(
                                        curves,
                                        colors="tab:blue",
                                        linewidths=0.6,
                                        alpha=0.3,
                                        rasterized=True,
                                    )
                                )
                                axis.scatter(
                                    selected[lag_column],
                                    selected[column],
                                    color="tab:blue",
                                    s=3,
                                    alpha=0.25,
                                    rasterized=True,
                                )
                                axis.autoscale_view()
                            elif len(selected):
                                axis.plot(
                                    selected[lag_column],
                                    selected[column],
                                    color="tab:blue",
                                    linewidth=1.5,
                                    marker=".",
                                    markersize=3,
                                )
                            axis.set(xlabel=lag_label, ylabel="MSD (µm²)")
                        axis.set_title(title)
                        pdf.savefig(figure, dpi=200)
                    finally:
                        figure.clear()


def link_localizations(
    path: str | Path,
    *,
    search_range_um: tuple[float, float, float] = (1.0, 1.0, 1.0),
    memory: int = 2,
    min_length: int = 10,
    dimensions: Literal["2d", "3d"] = "3d",
    first_frame: int | None = None,
    last_frame: int | None = None,
    channel: int | None = None,
    converged_only: bool = True,
    adaptive_stop_um: float | None = None,
    adaptive_step: float = 0.95,
    input_uri: str = "",
    save: bool = False,
    frame_interval_s: float | None = None,
    cancelled: Event | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Link saved fitted centers with Trackpy, independently for each channel.

    Distances are Z/Y/X micrometers; 2D uses only Y/X. Frame limits are inclusive,
    zero-based source indices; None selects all available frames. Parquet filters
    select frames/channels before reading. The upstream converter supplies pixel
    and physical coordinates; saved acceptance masks are preserved. Nonfinite
    centers are excluded and counted. Candidate IDs, when present, are retained.

    Trackpy's iterator receives every frame, including empty ones, so ``memory``
    counts missing image frames. Particle IDs are unique across channels. Minimum
    length counts detections, using Trackpy's filter_stubs. Cancellation is checked
    between frames and before saving. Full runs may atomically replace tracks.parquet
    and tracking.json beside the input, plus automatic step distributions and
    individual/ensemble MSD Parquet analyses and a PDF plot report; tuning runs
    leave ``save=False``.
    ``frame_interval_s`` calibrates the analysis lag times; None keeps frame lags.
    ``adaptive_stop_um`` enables Trackpy's local retries for oversized subnets;
    None disables them. Each retry multiplies every search radius by the unitless
    ``adaptive_step`` (strictly between zero and one). With unequal radii,
    Trackpy rescales the stop so the smallest selected radius reaches this µm
    threshold first. No subnet-size limits or linking algorithms are overridden.
    Return linked rows plus saved localization metadata and tracking settings.
    """
    import pyarrow.parquet as pq
    import trackpy as tp
    from localize_psf.workflow import to_trackpy_dataframe

    from .config import LocalizationConfig

    distances = np.asarray(search_range_um, dtype=float)
    if distances.shape != (3,) or not np.all(np.isfinite(distances) & (distances > 0)):
        raise ValueError("Displacement limits must be three positive finite Z/Y/X distances in µm")
    if memory < 0 or min_length < 1 or dimensions not in {"2d", "3d"}:
        raise ValueError("Use nonnegative memory, positive minimum length, and 2D or 3D tracking")
    if frame_interval_s is not None and (
        not np.isfinite(frame_interval_s) or frame_interval_s <= 0
    ):
        raise ValueError("Frame interval must be a positive finite number of seconds")
    if any(value is not None and value < 0 for value in (first_frame, last_frame, channel)):
        raise ValueError("Frame and channel indices must be nonnegative")
    if first_frame is not None and last_frame is not None and first_frame > last_frame:
        raise ValueError("The first tuning frame must not exceed the last frame")
    selected_distances = tuple(distances if dimensions == "3d" else distances[1:])
    if adaptive_stop_um is not None and (
        not np.isfinite(adaptive_stop_um) or not 0 < adaptive_stop_um < min(selected_distances)
    ):
        raise ValueError("Adaptive stop must be positive and smaller than every displacement limit")
    if not np.isfinite(adaptive_step) or not 0 < adaptive_step < 1:
        raise ValueError(
            "Adaptive step must be a finite factor greater than zero and less than one"
        )
    if cancelled is not None and cancelled.is_set():
        raise CancelledError("Tracking cancelled")
    source = Path(path).expanduser()
    if source.is_dir():
        source /= "localizations.parquet"
    schema = pq.read_schema(source)
    required = {"fit_params", "success", "candidate_timepoints", "candidate_channels"}
    if not required <= set(schema.names) or b"localize_psf" not in (schema.metadata or {}):
        raise ValueError("Select a localize-psf localizations.parquet file")
    metadata = json.loads(schema.metadata[b"localize_psf"])["metadata"]
    LocalizationConfig.model_validate_json(
        json.dumps({key: metadata[key] for key in ("voxel_size_zyx_um", "origin_zyx_um")})
    )
    saved_uri = metadata.get("input_uri", "")
    if input_uri and saved_uri:
        uris = {
            uri.rstrip("/") if "://" in uri else str(Path(uri).resolve())
            for uri in (input_uri, saved_uri)
        }
        if len(uris) != 1:
            raise ValueError("Select the localizations' matching input image in Data")
    filters = [
        (name, operator, value)
        for name, operator, value in (
            ("candidate_timepoints", ">=", first_frame),
            ("candidate_timepoints", "<=", last_frame),
            ("candidate_channels", "=", channel),
            ("success", "=", True if converged_only else None),
            ("filter_pass", "=", True if "filter_pass" in schema.names else None),
        )
        if value is not None
    ]
    # Covariance matrices are unnecessary for linking. Keep the validity flag,
    # identity and physical/pixel positions without reading those large columns.
    columns = required | {"filter_pass", "covariance_valid", "candidate_ids", "roi_truncated"}
    table = pq.read_table(
        source, columns=[name for name in schema.names if name in columns], filters=filters or None
    )
    points = to_trackpy_dataframe(table, converged_only=False)
    points = points.drop(columns=[name for name in points if name.startswith(("size_", "ep_"))])
    if "candidate_ids" in table.column_names:
        points["candidate_id"] = np.asarray(table["candidate_ids"])
    finite = np.isfinite(points[["zum", "yum", "xum"]].to_numpy()).all(axis=1)
    skipped = int((~finite).sum())
    points = points.loc[finite]
    position_columns = ["zum", "yum", "xum"] if dimensions == "3d" else ["yum", "xum"]
    frames, offset = [], 0
    for channel_id, selected in points.groupby("channel", sort=True):
        groups = dict(iter(selected.groupby("frame", sort=True)))
        first, last = int(selected["frame"].min()), int(selected["frame"].max())
        empty = selected.iloc[:0]
        frame_data = (groups.get(frame, empty) for frame in range(first, last + 1))
        next_offset = offset
        for frame, linked in enumerate(
            tp.link_df_iter(
                frame_data,
                search_range=selected_distances,
                pos_columns=position_columns,
                memory=memory,
                adaptive_stop=adaptive_stop_um,
                adaptive_step=adaptive_step,
            ),
            start=first,
        ):
            if cancelled is not None and cancelled.is_set():
                raise CancelledError("Tracking cancelled")
            if len(linked):
                linked["particle"] += offset
                next_offset = max(next_offset, int(linked["particle"].max()) + 1)
                frames.append(linked)
            if progress is not None:
                progress(f"Tracking channel {channel_id}: frame {frame} / {last}")
        offset = next_offset
    tracks = (
        pd.concat(frames, ignore_index=True)
        if frames
        else points.assign(particle=pd.Series(dtype="int64"))
    )
    if len(tracks):
        tracks = tp.filter_stubs(tracks, threshold=min_length).reset_index(drop=True)
    tracks["particle"] = tracks["particle"].astype("int64")
    tracks = tracks.sort_values(["particle", "frame"], kind="stable").reset_index(drop=True)
    settings = dict(
        schema_version=1,
        source=str(source.resolve()),
        search_range_zyx_um=distances.tolist(),
        memory=memory,
        min_length=min_length,
        dimensions=dimensions,
        first_frame=first_frame,
        last_frame=last_frame,
        channel=channel,
        converged_only=converged_only,
        adaptive_stop_um=adaptive_stop_um,
        adaptive_step=adaptive_step,
        skipped_nonfinite=skipped,
        trackpy_version=tp.__version__,
        frame_interval_s=frame_interval_s,
    )
    metadata = {**metadata, "tracking": settings}
    if cancelled is not None and cancelled.is_set():
        raise CancelledError("Tracking cancelled")
    if save:
        import pyarrow as pa

        analyses = analyze_tracks(
            tracks, frame_interval_s=frame_interval_s, cancelled=cancelled, progress=progress
        )
        settings["analysis"] = dict(
            schema_version=1,
            files=list(analyses),
            step_lag_frames=1,
            radial_definition="sqrt(dz_um**2 + dy_um**2 + dx_um**2)",
            component_steps="signed",
            histogram_bins="numpy-fd-shared-components-and-channels",
            report_file="tracking_analysis.pdf",
            step_plot_limits="shared-x-and-y-across-components-and-channels",
            ensemble_scope="per_channel",
            msd_weighting="trackpy-effective-N",
            max_lag_frames="all_available",
            position_unit="micrometer",
            msd_unit="micrometer**2",
            frame_interval_s=frame_interval_s,
            drift_subtracted=False,
            msd_min_observations=2,
            motion_coordinate_dtype="float64",
            report_pages_per_channel=6,
            report_msd_component="r",
        )
        encoded = json.dumps(metadata, indent=2).encode("utf-8")
        with TemporaryDirectory(prefix=".tracking-", dir=source.parent) as staging:
            directory = Path(staging)
            for name, frame in {"tracks.parquet": tracks, **analyses}.items():
                if progress is not None:
                    progress(f"Saving {name}")
                if cancelled is not None and cancelled.is_set():
                    raise CancelledError("Tracking cancelled")
                table = pa.Table.from_pandas(frame, preserve_index=False)
                table = table.replace_schema_metadata(
                    {
                        **(table.schema.metadata or {}),
                        b"napari_localize_psf_tracking": encoded,
                    }
                )
                pq.write_table(table, directory / name)
            write_analysis_pdf(
                analyses,
                directory / settings["analysis"]["report_file"],
                frame_interval_s=frame_interval_s,
                cancelled=cancelled,
                progress=progress,
            )
            (directory / "tracking.json").write_bytes(encoded)
            if cancelled is not None and cancelled.is_set():
                raise CancelledError("Tracking cancelled")
            for staged in directory.iterdir():
                staged.replace(source.with_name(staged.name))
    return tracks, metadata
