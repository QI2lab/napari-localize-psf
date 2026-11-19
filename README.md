# napari-localize-psf

A [napari] plugin for tuning spot detection and Gaussian localization with [localize-psf].

## Installation

From this repository, install into your active napari environment:

```bash
uv pip install . napari-ome-zarr napari-tiff "napari[all]"
```

## Usage

Open **Plugins → QI2lab Localization** in napari.

1. Load an OME-Zarr store or OME-TIFF image and check its calibration. The highest
   resolution is selected automatically.
2. Preview U-FISH detections and adjust the model, threshold, and edge filtering.
3. Preview 2D or 3D Gaussian fits and tune their bounds and acceptance filters.
4. Save the JSON settings and select **Copy batch command** to run localization
   from your terminal.
5. Load completed localizations in **Batch results** to overlay them on the image.

Use **Tracking** to preview trajectories or track the full dataset. Saved results
include tracks, step-size and MSD analyses as Parquet files, and a PDF plot report.

Optionally use **PSF / field** after a 3D fit preview to average accepted PSFs and
plot focus tilt and field curvature.

[napari]: https://napari.org/
[localize-psf]: https://github.com/QI2lab/localize-psf
