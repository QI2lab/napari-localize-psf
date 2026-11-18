"""Trackpy adapters for saved localizations; no detection, fitting, or image IO."""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import CancelledError
from pathlib import Path
from threading import Event
from typing import Any, Literal

import numpy as np
import pandas as pd


def link_localizations(
    path: str | Path,
    *,
    search_range_um: tuple[float, float, float],
    memory: int = 0,
    min_length: int = 2,
    dimensions: Literal["2d", "3d"] = "3d",
    first_frame: int | None = None,
    last_frame: int | None = None,
    channel: int | None = None,
    converged_only: bool = True,
    adaptive_stop_um: float | None = None,
    input_uri: str = "",
    save: bool = False,
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
    and tracking.json beside the input; tuning runs should leave ``save=False``.
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
    if any(value is not None and value < 0 for value in (first_frame, last_frame, channel)):
        raise ValueError("Frame and channel indices must be nonnegative")
    if first_frame is not None and last_frame is not None and first_frame > last_frame:
        raise ValueError("The first tuning frame must not exceed the last frame")
    selected_distances = tuple(distances if dimensions == "3d" else distances[1:])
    if adaptive_stop_um is not None and (
        not np.isfinite(adaptive_stop_um) or not 0 < adaptive_stop_um < min(selected_distances)
    ):
        raise ValueError("Adaptive stop must be positive and smaller than every displacement limit")
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
    columns = required | {"filter_pass", "covariance_valid", "candidate_ids"}
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
        skipped_nonfinite=skipped,
        trackpy_version=tp.__version__,
    )
    metadata = {**metadata, "tracking": settings}
    if cancelled is not None and cancelled.is_set():
        raise CancelledError("Tracking cancelled")
    if save:
        result_path = source.with_name("tracks.parquet")
        settings_path = source.with_name("tracking.json")
        temporary = result_path.with_suffix(".parquet.tmp")
        settings_temporary = settings_path.with_suffix(".json.tmp")
        try:
            tracks.to_parquet(temporary, index=False)
            settings_temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            if cancelled is not None and cancelled.is_set():
                raise CancelledError("Tracking cancelled")
            temporary.replace(result_path)
            settings_temporary.replace(settings_path)
        finally:
            temporary.unlink(missing_ok=True)
            settings_temporary.unlink(missing_ok=True)
    return tracks, metadata
