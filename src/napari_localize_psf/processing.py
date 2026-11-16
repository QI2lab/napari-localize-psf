"""Thin configuration adapters around upstream U-FISH, ROI tools, and GPU fitting."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import CancelledError
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from .config import LocalizationConfig
from .data import Preview

if TYPE_CHECKING:
    from ufish.api import UFish


class FitResults(TypedDict):
    """Upstream result fields consumed by this plugin; upstream owns all calculations."""

    fit_params: NDArray[np.floating]
    init_params: NDArray[np.floating]
    success: NDArray[np.bool_]
    candidate_ids: NDArray[np.integer]
    niters: NDArray[np.integer]
    chi_sqrs: NDArray[np.floating]
    bounds: tuple[NDArray[np.floating], NDArray[np.floating]]
    active_bounds: NDArray[np.bool_]
    covariance_valid: NotRequired[NDArray[np.bool_]]
    covariance_backend: NotRequired[str]


@dataclass(frozen=True)
class Detection:
    """U-FISH enhancement and candidate coordinates in crop-local ZYX pixels."""

    enhanced: NDArray[np.float32]
    points: NDArray[np.float64]


class Detector:
    """Cache one production enhancement so threshold tuning only recalls spots."""

    def __init__(self) -> None:
        self._detector: UFish | None = None
        self._key: tuple[int, bool, int, str, str] | None = None
        self._preview: Preview | None = None
        self._enhanced: NDArray[np.float32] | None = None

    def detect(self, preview: Preview, config: LocalizationConfig) -> Detection:
        """Run the supplied CUDA U-FISH weights, reusing enhancement when valid."""
        import_module("localize_psf")  # Let upstream preload the installed CUDA components.
        from ufish.api import UFish

        axes = "yx" if config.fit_model == "2d" else "zyx"
        key = (
            id(preview),
            config.blend_3d,
            config.inference_batch_size,
            config.ufish_weights,
            axes,
        )
        if self._key != key:
            self._detector = UFish(device="cuda")
            weights = config.ufish_weights
            if Path(weights).is_file():
                self._detector.load_weights(weights_path=weights)
            else:
                self._detector.load_weights(weights_file=weights)
            image = preview.image[0] if axes == "yx" else preview.image
            spots, enhanced = self._detector.predict(
                image,
                axes=axes,
                blend_3d=config.blend_3d,
                batch_size=config.inference_batch_size,
                intensity_threshold=config.intensity_threshold,
            )
            self._enhanced = enhanced
            self._preview, self._key = preview, key
        else:
            assert self._detector is not None and self._enhanced is not None
            spots = self._detector.call_spots(
                self._enhanced, intensity_threshold=config.intensity_threshold
            )
        points = spots[[f"axis-{i}" for i in range(len(axes))]].to_numpy(dtype=float)
        if axes == "yx":
            points = np.column_stack((np.zeros(len(points)), points))
        assert self._enhanced is not None
        return Detection(self._enhanced, points)

    def release(self) -> None:
        """Release the inference model before GPUfit; retain no stale cache key."""
        loaded = self._detector is not None
        self._detector = self._enhanced = self._key = self._preview = None
        if loaded:
            import torch

            torch.cuda.empty_cache()


def optical_parameters(config: LocalizationConfig) -> dict[str, tuple[float, float, float]]:
    """Return upstream nominal/lower/upper width estimates in ZYX µm on CUDA."""
    from localize_psf.optical_bounds import gaussian_width_bounds

    result = gaussian_width_bounds(
        config.numerical_aperture,
        config.wavelength_um,
        config.refractive_index,
        config.voxel_size_zyx_um,
        relative_sigma_range=config.relative_sigma_range,
        sampling_floor_pixels=0,
        use_gpu=True,
    )
    return {name: tuple(value.get().tolist()) for name, value in result.items()}


def fit_spots(
    preview: Preview,
    points: NDArray[np.float64],
    config: LocalizationConfig,
    cancelled: Event | None = None,
    progress: Callable[[str], None] | None = None,
) -> FitResults:
    """Apply parameter policy, then use upstream packed, bounded GPUfit batches.

    Center limits/initial values are offsets from each U-FISH prior in µm,
    intersected with that candidate's ROI. 2D uses the existing native Gaussian
    on one plane with cz fixed at its coordinate and sz fixed at 1 µm; its axial
    factor is exactly one. No alternate Gaussian or optimizer is implemented.
    """
    import_module("localize_psf")  # Initialize upstream's CUDA loader before CuPy.
    import cupy as cp
    from localize_psf.batch_fitting import _accumulate_fit_batch, fit_rois
    from localize_psf.fit_psf import gaussian3d_psf_model
    from localize_psf.localize import get_coords, prepare_rois
    from localize_psf.rois import get_centered_rois, get_roi_size

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Candidate coordinates must be finite (N,3) ZYX pixels")
    if np.any(points < 0) or np.any(points >= np.asarray(preview.image.shape)):
        raise ValueError("Candidate coordinates must lie inside the preview")
    model = gaussian3d_psf_model()
    if not len(points):
        empty = np.empty((0, 0))
        return cast(
            FitResults,
            fit_rois(
                empty,
                (empty, empty, empty),
                np.empty(0, int),
                np.empty((0, 7)),
                model=model,
                use_gpu=True,
            ),
        )
    roi_size = np.asarray(get_roi_size(config.fit_roi_size_um, preview.spacing, ensure_odd=True))
    if config.fit_model == "2d":
        if preview.image.shape[0] != 1:
            raise ValueError("2D fitting requires a single XY plane")
        roi_size[0] = 1
    image = cp.asarray(preview.image)
    coords = get_coords(image.shape, preview.spacing, origin=preview.origin, use_gpu=True)
    centers = cp.asarray(preview.origin) + cp.asarray(points) * cp.asarray(preview.spacing)
    rois = get_centered_rois(
        cp.rint(cp.asarray(points)).astype(int), roi_size, [0, 0, 0], image.shape
    )
    complete = None
    try:
        for start in range(0, len(points), config.roi_batch_size):
            if cancelled is not None and cancelled.is_set():
                raise CancelledError("Localization cancelled")
            stop = min(start + config.roi_batch_size, len(points))
            if progress is not None:
                progress(f"Fitting candidates {start + 1}–{stop} of {len(points)}")
            selected = slice(start, stop)
            values, coordinates, sizes = prepare_rois(image, coords, rois[selected])
            initial = model.estimate_parameters(values, coordinates, num_preserved_dims=1)
            initial[:, 1:4] = centers[selected, ::-1]
            limits = [
                model.estimate_bounds(tuple(c[i, : int(count)] for c in coordinates))
                for i, count in enumerate(sizes.get())
            ]
            lower = cp.stack([cp.stack([cp.asarray(v) for v in pair[0]]) for pair in limits])
            upper = cp.stack([cp.stack([cp.asarray(v) for v in pair[1]]) for pair in limits])
            fixed = np.zeros(model.nparams, dtype=bool)
            for i, name in enumerate(model.parameter_names):
                policy = config.fit_parameters[name]
                base = centers[selected, 3 - i] if name in {"cx", "cy", "cz"} else 0
                if policy.lower is not None:
                    lower[:, i] = cp.maximum(lower[:, i], base + policy.lower)
                if policy.upper is not None:
                    upper[:, i] = cp.minimum(upper[:, i], base + policy.upper)
                if policy.initial is not None:
                    initial[:, i] = base + policy.initial
                fixed[i] = policy.fixed
            if config.fit_model == "2d":
                initial[:, 3], initial[:, 5] = centers[selected, 0], 1
                lower[:, 3] = upper[:, 3] = initial[:, 3]
                lower[:, 5] = upper[:, 5] = 1
                fixed[[3, 5]] = True
            truncated = cp.any(
                rois[selected, 1::2] - rois[selected, ::2] != cp.asarray(roi_size), axis=1
            )
            batch = fit_rois(
                values,
                coordinates,
                sizes,
                initial,
                model=model,
                use_gpu=True,
                bounds=(lower.get(), upper.get()),
                fixed_params=fixed,
                estimator=config.estimator,
                noise_model="poisson" if config.estimator == "MLE" else "estimated_gaussian",
                max_number_iterations=config.max_number_iterations,
                tolerance=config.tolerance,
                return_covariance=config.return_covariance,
                quality_flags={"truncated": truncated.get()},
                covariance_batch_size=config.roi_batch_size,
            )
            complete = _accumulate_fit_batch(complete, batch, start, len(points))
            del values, coordinates, initial, lower, upper, batch
            cp.get_default_memory_pool().free_all_blocks()
    finally:
        del image, coords
        cp.get_default_memory_pool().free_all_blocks()
    assert complete is not None
    return cast(FitResults, complete)
