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
