"""Reader integration and explicit layer-to-physical preview coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from napari.layers import Image
from napari.utils.transforms import Affine
from numpy.typing import NDArray
from pint import get_application_registry

from .config import LocalizationConfig

if TYPE_CHECKING:
    from napari import Viewer


@dataclass(frozen=True)
class Preview:
    """One cropped ZYX volume; origin is the crop's physical first voxel in µm."""

    image: NDArray[np.float32]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    config: LocalizationConfig

    def layer_coordinates(self, points_zyx_px: NDArray[np.floating]) -> NDArray[np.float64]:
        """Map crop-local ZYX pixels back to the source layer's full axis order."""
        selection = self.config.preview
        offsets = np.asarray(selection.crop_zyx_px[::2], dtype=float)
        if self.config.fit_model == "2d" and "z" in selection.axes:
            offsets[0] = selection.z_plane
        points = np.asarray(points_zyx_px, dtype=float) + offsets
        columns = {
            "t": np.full(len(points), selection.timepoint),
            "c": np.full(len(points), selection.channel),
            **dict(zip("zyx", points.T, strict=True)),
        }
        return np.column_stack([columns[axis] for axis in selection.axes])


def open_images(viewer: Viewer, path: str) -> list[Image]:
    """Delegate file decoding entirely to the installed npe2 reader plugins.

    OME-Zarr paths should identify a single image group. Split channel layers
    retain their reader order; the UI exposes the source batch channel explicitly.
    """
    plugin = "napari-tiff" if path.lower().endswith((".tif", ".tiff")) else "napari-ome-zarr"
    images = [layer for layer in viewer.open(path, plugin=plugin) if isinstance(layer, Image)]
    prepare_image_layers(images, path)
    return images


def prepare_image_layers(images: list[Image], path: str) -> None:
    """Retain reader channel order and stored transforms for native or dock opening."""
    transform = None
    if images and images[0].source.reader_plugin == "napari-ome-zarr":
        from ome_zarr import OMEZarrMultiscale

        # The reader currently omits ordinary image-group translations. Use its
        # upstream metadata models and napari's affine composition to retain the
        # entire stored transform, without implementing another metadata parser.
        multiscale = OMEZarrMultiscale.from_ome_zarr(path)
        axes = multiscale.images[0].axes
        transform = Affine(ndim=len(axes))
        metadata = multiscale.metadata
        for stored in (
            *metadata.datasets[0].coordinateTransformations,
            *(metadata.coordinateTransformations or ()),
        ):
            matrix = np.asarray(stored.as_affine().affine_matrix)
            transform = Affine(linear_matrix=matrix[:, :-1], translate=matrix[:, -1]).compose(
                transform
            )
        transform = transform.set_slice([i for i, axis in enumerate(axes) if axis != "c"])
    for channel, layer in enumerate(images):
        if transform is not None:
            layer.scale = np.ones(layer.ndim)
            layer.translate = np.zeros(layer.ndim)
            layer.affine = transform
        layer.metadata["napari_localize_psf"] = {"input_uri": path, "source_channel": channel}


def layer_calibration(
    layer: Image, axes: str, *, physical: bool = True
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Read diagonal physical transforms with Pint; missing units require manual input.

    Rotated/sheared coordinate frames need an explicit resampling policy and are
    rejected instead of discarding those transforms during localization.
    """
    ndim = len(axes)
    if ndim != layer.ndim:
        raise ValueError("Axis labels must match the selected image dimensions")
    zero = np.asarray(layer.data_to_world(np.zeros(ndim)))
    matrix = np.column_stack(
        [np.asarray(layer.data_to_world(vector)) - zero for vector in np.eye(ndim)]
    )
    if not np.allclose(matrix, np.diag(np.diag(matrix))):
        raise ValueError("Rotate/shear transforms require resampling before localization")
    registry = get_application_registry()
    spacing, origin = [], []
    for axis in "zyx":
        if axis not in axes:
            spacing.append(1.0)
            origin.append(0.0)
            continue
        i = axes.index(axis)
        factor = (
            registry.Quantity(1, str(layer.units[i])).to("micrometer").magnitude if physical else 1
        )
        spacing.append(float(matrix[i, i] * factor))
        origin.append(float(zero[i] * factor))
    if not np.all(np.isfinite(spacing)) or np.any(np.asarray(spacing) <= 0):
        raise ValueError("Voxel sizes must be finite and positive")
    return tuple(spacing), tuple(origin)


def prepare_preview(layer: Image, config: LocalizationConfig) -> Preview:
    """Read only the selected highest-resolution crop and convert it to ZYX.

    Geometry in config is source geometry. Crop offsets are applied exactly once.
    2D means one XY plane, not a maximum-intensity projection.
    """
    settings = config.preview
    layer_calibration(layer, settings.axes, physical=False)
    data = layer.data[0] if layer.multiscale else layer.data
    if layer.rgb or len(settings.axes) != data.ndim:
        raise ValueError("Select a scalar image with an explicit, matching axis order")
    crop = np.asarray(settings.crop_zyx_px).reshape(3, 2).copy()
    if config.fit_model == "2d":
        crop[0] = (settings.z_plane, settings.z_plane + 1) if "z" in settings.axes else (0, 1)
    elif "z" not in settings.axes or data.shape[settings.axes.index("z")] < 2:
        raise ValueError("3D fitting requires a Z stack; select 2D for an XY image")
    indices: list[int | slice] = []
    for axis, size in zip(settings.axes, data.shape, strict=True):
        if axis in "tc":
            index = settings.timepoint if axis == "t" else settings.channel
            if index >= size:
                raise ValueError(f"Selected {axis} index is outside the image")
            indices.append(index)
        else:
            start, stop = crop["zyx".index(axis)]
            if stop > size:
                raise ValueError(f"Preview crop exceeds the {axis} dimension ({size})")
            indices.append(slice(int(start), int(stop)))
    selected = data[tuple(indices)]
    spatial_axes = "".join(a for a in settings.axes if a not in "tc")
    selected = selected.transpose(tuple(spatial_axes.index(a) for a in "zyx" if a in spatial_axes))
    image = np.asarray(selected, dtype=np.float32)
    if "z" not in spatial_axes:
        image = image[None]
    if not np.isfinite(image).all():
        raise ValueError("Preview data must contain only finite counts")
    if config.estimator == "MLE" and np.any(image < 0):
        raise ValueError("Poisson MLE requires nonnegative photon/electron counts")
    if min(image.shape[1:]) < 3:
        raise ValueError("Preview must contain at least three pixels along X and Y")
    origin = np.asarray(config.origin_zyx_um) + crop[:, 0] * config.voxel_size_zyx_um
    return Preview(image, config.voxel_size_zyx_um, tuple(origin), config)
