"""Record installed package provenance without initializing the GPU stack."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, distribution


def package_versions() -> dict[str, str]:
    """Record installed distribution versions/revisions without importing backends."""
    result: dict[str, str] = {}
    for name in (
        "napari-localize-psf",
        "localize-psf",
        "pygpufit",
        "ufish",
        "spotiflow",
        "torchvision",
        "torch",
        "cupy-cuda12x",
    ):
        try:
            package = distribution(name)
        except PackageNotFoundError:
            continue
        result[name] = package.version
        origin = package.read_text("direct_url.json")
        if origin:
            revision = json.loads(origin).get("vcs_info", {}).get("commit_id")
            if revision:
                result[f"{name}_commit"] = revision
    return result
