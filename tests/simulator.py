"""Synthetic inputs built with upstream PSF models; never a replacement fitter."""

import numpy as np
from localize_psf.fit_psf import (
    gaussian2d_psf_model,
    gaussian3d_asymmetric_rotated_pixelated,
    gaussian3d_psf_model,
)
from localize_psf.localize import get_coords


def gaussian_image(plane: bool = False) -> tuple[np.ndarray, np.ndarray, tuple, tuple]:
    """Return noiseless counts, exact parameters, spacing and nonzero physical origin."""
    spacing, origin = (0.2, 0.1, 0.1), (-2.0, 3.0, -4.0)
    shape = (1 if plane else 25, 41, 43)
    coords = get_coords(shape, spacing, origin=origin)
    center = np.asarray(origin) + np.array([0 if plane else 12.3, 20.2, 21.4]) * spacing
    truth = np.array([100, *center[::-1], 0.15, 1 if plane else 0.3, 5.0])
    if plane:
        model = gaussian2d_psf_model()
        image = model.model(coords[1:], truth[[0, 1, 2, 4, 6]]).reshape(shape)
    else:
        image = gaussian3d_psf_model().model(coords, truth)
    return np.asarray(image, dtype=np.float32), truth, spacing, origin


def bead_image(tilt_yz_rad: float | None = None) -> tuple[np.ndarray, np.ndarray, tuple, tuple]:
    """Return five separated beads in original ZYX counts with physical Gaussian parameters."""
    shape, spacing, origin = (25, 71, 71), (0.2, 0.1, 0.1), (-2.0, 3.0, -4.0)
    pixels = np.array(
        [[12.3, 15, 15], [13, 15, 55], [11.7, 55, 15], [12.6, 55, 55], [12.2, 35, 35]]
    )
    centers = origin + pixels * spacing
    parameters = np.array(
        [
            [amplitude, *center[::-1], 0.15, 0.3, 5]
            for amplitude, center in zip((100, 110, 120, 130, 400), centers, strict=True)
        ]
    )
    coords, model = get_coords(shape, spacing, origin=origin), gaussian3d_psf_model()
    if tilt_yz_rad is not None:
        parameters = np.column_stack(
            (
                parameters[:, :4],
                parameters[:, 4],
                parameters[:, 4],
                parameters[:, 5:],
                np.full(len(parameters), np.pi / 2),
                np.full(len(parameters), tilt_yz_rad),
                np.full(len(parameters), -np.pi / 2),
            )
        )
        model = gaussian3d_asymmetric_rotated_pixelated()
    background = 6 if tilt_yz_rad is None else 7
    image = 5 + sum(model.model(coords, p) - p[background] for p in parameters)
    return np.asarray(image, dtype=np.float32), parameters, spacing, origin
