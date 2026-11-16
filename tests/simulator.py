"""Synthetic inputs built with upstream PSF models; never a replacement fitter."""

import numpy as np
from localize_psf.fit_psf import gaussian2d_psf_model, gaussian3d_psf_model
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
