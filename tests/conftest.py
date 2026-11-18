"""Real napari fixtures; no package, detector, or fitter is replaced in tests."""

from __future__ import annotations

import os
from collections.abc import Iterator

import napari
import pytest
from napari.utils._proxies import PublicOnlyProxy
from qtpy.QtCore import QCoreApplication, QEvent


def pytest_configure(config: pytest.Config) -> None:
    """Keep preferences isolated and enforce napari's production main-thread checks."""
    os.environ["NAPARI_CONFIG"] = ""
    os.environ["NAPARI_ENSURE_PLUGIN_MAIN_THREAD"] = "1"


@pytest.fixture
def viewer(qtbot) -> Iterator[PublicOnlyProxy[napari.Viewer]]:
    # Napari's standard viewer factories mock the console/status checker/app
    # model. Retain their production proxy but construct the real Viewer to
    # honor this repository's stricter no-replacement testing policy.
    instance = napari.Viewer(show=False)
    try:
        yield PublicOnlyProxy(instance)
    finally:
        instance.close()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Enforce the repository's unit/integration-only test contract."""
    for item in items:
        if not any(item.get_closest_marker(name) for name in ("unit", "integration")):
            raise pytest.UsageError(f"Test requires a unit or integration marker: {item.nodeid}")


@pytest.fixture
def localizations(tmp_path):
    """Serialize simulated physical coordinates as a real localization input table.

    No detector or fitting backend is replaced: these coordinates are the input
    to tracking. Two channels overlap physically; one emitter misses frame two.
    Calibration is anisotropic with a nonzero origin.
    """
    import json

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from napari_localize_psf.config import LocalizationConfig

    spacing, origin = (0.23, 0.115, 0.15), (-2.0, 3.0, -4.0)
    rows = [(0, frame, 4.0) for frame in (0, 1, 3, 4)]
    rows += [(0, frame, 5.0) for frame in range(5)]
    rows += [(1, frame, 4.0) for frame in range(5)]
    rows += [(0, 2, 6.0), (0, 1, float("nan")), (0, 3, 7.0)]
    parameters = [[500, -2 + 0.1 * frame, y, -0.4, 0.15, 0.3, 5] for _, frame, y in rows]
    config = LocalizationConfig(
        input_uri=str(tmp_path / "source.ome.zarr"),
        voxel_size_zyx_um=spacing,
        origin_zyx_um=origin,
    )
    table = pa.table(
        {
            "candidate_ids": np.arange(len(rows)),
            "candidate_timepoints": [frame for _, frame, _ in rows],
            "candidate_channels": [channel for channel, _, _ in rows],
            "fit_params": parameters,
            "success": [True] * 14 + [False, True, True],
            "filter_pass": [True] * 16 + [False],
        }
    ).replace_schema_metadata(
        {b"localize_psf": json.dumps({"metadata": config.model_dump(mode="json")}).encode()}
    )
    path = tmp_path / "localizations.parquet"
    pq.write_table(table, path, row_group_size=3)
    return path, config
