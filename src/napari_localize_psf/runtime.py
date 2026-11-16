"""Record installed package provenance without initializing the GPU stack."""

from __future__ import annotations

import json
from importlib.metadata import distribution, version


def package_versions() -> dict[str, str]:
    """Record distribution versions and Git revisions without initializing CUDA."""
    result: dict[str, str] = {}
    for name in (
        "napari-localize-psf",
        "localize-psf",
        "pygpufit",
        "ufish",
        "torch",
        "cupy-cuda12x",
    ):
        result[name] = version(name)
        origin = distribution(name).read_text("direct_url.json")
        if origin:
            revision = json.loads(origin).get("vcs_info", {}).get("commit_id")
            if revision:
                result[f"{name}_commit"] = revision
    return result
