"""Reader integration and explicit layer-to-physical preview coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from localize_psf.processing import prepare_volume
from napari.layers import Image
from napari.utils.transforms import Affine
from numpy.typing import NDArray
from pint import get_application_registry
from pint.errors import PintError

from .config import LocalizationConfig

if TYPE_CHECKING:
    from napari import Viewer


type Preview = tuple[np.ndarray, tuple[float, float, float], LocalizationConfig]


def source_coordinates(
    points_zyx_px: NDArray[np.floating],
    axes: str,
    *,
    timepoints: int | NDArray[np.integer] = 0,
    channels: int | NDArray[np.integer] = 0,
) -> NDArray[np.float64]:
    """Order full-image ZYX pixels and zero-based T/C indices for a source layer.

    ``axes`` supplies its unique t/c/z/y/x labels. T/C inputs are scalar indices
    or arrays matching the points. Source scale, units and global transforms
    stay on the layer and must be copied to the corresponding points overlay.
    """
    points = np.asarray(points_zyx_px, dtype=float)
    columns = {
        "t": np.broadcast_to(timepoints, (len(points),)),
        "c": np.broadcast_to(channels, (len(points),)),
        **dict(zip("zyx", points.T, strict=True)),
    }
    return np.column_stack([columns[axis] for axis in axes])


def layer_coordinates(
    points_zyx_px: NDArray[np.floating], config: LocalizationConfig
) -> NDArray[np.float64]:
    """Map crop-local ZYX pixels back to the source layer's full axis order."""
    selection = config.preview
    offsets = np.asarray(selection.crop_zyx_px[::2], dtype=float)
    if config.fit_model == "2d" and "z" in selection.axes:
        offsets[0] = selection.z_plane
    points = np.asarray(points_zyx_px, dtype=float) + offsets
    return source_coordinates(
        points, selection.axes, timepoints=selection.timepoint, channels=selection.channel
    )


def open_images(viewer: Viewer, path: str) -> list[Image]:
    """Delegate file decoding entirely to the installed npe2 reader plugins.

    OME-Zarr roots are resolved upstream before delegating each image to the
    reader. Pyramid order comes from OME metadata; previews use its highest
    resolution. Split channels record their image-group URI and source channel.
    """
    plugin = "napari-tiff" if path.lower().endswith((".tif", ".tiff")) else "napari-ome-zarr"
    paths = [path]
    if plugin == "napari-ome-zarr":
        from localize_psf.workflow import ome_zarr_image_paths

        paths = ome_zarr_image_paths(path)
    images = []
    for image_path in paths:
        images.extend(
            layer for layer in viewer.open(image_path, plugin=plugin) if isinstance(layer, Image)
        )
    prepare_image_layers(images, path)
    return images


def prepare_image_layers(images: list[Image], path: str) -> None:
    """Retain reader order and transforms for native or dock opening.

    Metadata records the resolved image-group ``input_uri``, original
    ``container_uri`` and zero-based ``source_channel`` for batch export.
    """
    prepared: list[tuple[str, list[Image], Affine | None]] = [(path, images, None)]
    if images and images[0].source.reader_plugin == "napari-ome-zarr":
        from localize_psf.workflow import ome_zarr_image_paths
        from napari_ome_zarr.utils import count_layers_in_image
        from ome_zarr import OMEZarrMultiscale
        from zarr import open_group

        # The reader currently omits ordinary image-group translations. Use upstream
        # container discovery, channel counts and metadata models to associate
        # each layer with its own image, without implementing another parser.
        prepared = []
        offset = 0
        for image_uri in ome_zarr_image_paths(path):
            if offset >= len(images):
                break
            group = open_group(image_uri, mode="r")
            multiscale = OMEZarrMultiscale.from_ome_zarr(group)
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
            count = count_layers_in_image(multiscale)["image_layers"]
            prepared.append((image_uri, images[offset : offset + count], transform))
            offset += count
    for image_uri, layers, transform in prepared:
        for channel, layer in enumerate(layers):
            if transform is not None:
                layer.scale = transform.scale
                layer.translate = transform.translate
            layer.metadata["napari_localize_psf"] = {
                "input_uri": image_uri,
                "container_uri": path,
                "source_channel": channel,
            }


def layer_frame_interval(layer: Image, axes: str) -> float | None:
    """Return a uniform source-frame interval in seconds, or None if uncalibrated.

    Respect the layer's complete transform and time-axis units. A transform
    mixing spatial coordinates into time cannot supply a uniform frame clock.
    Missing/non-time units and nonpositive intervals retain frame-based lags.
    """
    if len(axes) != layer.ndim or axes.count("t") != 1:
        return None
    index = axes.index("t")
    origin = np.asarray(layer.data_to_world(np.zeros(layer.ndim)))
    time_row = np.asarray(
        [
            np.asarray(layer.data_to_world(vector))[index] - origin[index]
            for vector in np.eye(layer.ndim)
        ]
    )
    if not np.allclose(np.delete(time_row, index), 0):
        return None
    try:
        seconds = float(
            get_application_registry()
            .Quantity(time_row[index], str(layer.units[index]))
            .to("second")
            .magnitude
        )
    except (PintError, ValueError, TypeError):
        return None
    return seconds if np.isfinite(seconds) and seconds > 0 else None


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

    Return (image, crop origin in ZYX µm, captured configuration).
    Geometry in config is source geometry. Crop offsets are applied exactly once.
    2D means one XY plane, not a maximum-intensity projection.
    """
    settings = config.preview
    layer_calibration(layer, settings.axes, physical=False)
    data = layer.data[0] if layer.multiscale else layer.data
    if layer.rgb or len(settings.axes) != data.ndim:
        raise ValueError("Select a scalar image with an explicit, matching axis order")
    image, origin = prepare_volume(data, config)
    return image, origin, config
