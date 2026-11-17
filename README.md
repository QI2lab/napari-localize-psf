# napari-localize-psf

A [napari] plugin for configuring QI2lab spot localization with [localize-psf].
Preview U-FISH detections, tune bounded 2D/3D Gaussian fits with GPUfit, and
export settings for batch processing.

## Installation

Requires [uv], Python ≥3.12, and an NVIDIA GPU with a working driver on Linux
x86-64 or Windows x64.

In your active Python environment, run this from the repository root:

```bash
uv pip install . napari-ome-zarr napari-tiff "napari[all]"
```

This installs napari, the plugin, readers, and localize-psf's `opm-ptv` GPU and
example dependencies into that environment.

## Usage

Open **Plugins → QI2lab Localization**.

1. Open an OME-TIFF or OME-Zarr dataset, or select an existing image layer.
2. Confirm the axes, physical calibration, and preview region. Choose a 2D plane
   or 3D volume.
3. Select a pretrained checkpoint from **U-FISH model**, or type a custom
   weights path into the same dropdown. SimFISH remains the default; the list
   also includes the general all-data model and nine other fine-tuned checkpoints.
   The selected weights are saved in JSON for batch prediction.
   Preview U-FISH priors and adjust `blend_3d` and the intensity threshold.
   Nearby priors use localize-psf's existing `filter_nearby_peaks` keep-one policy,
   prioritizing the strongest peaks within **2 × optical PSF FWHM** axially and
   radially in XY, using the optical settings in **Fits**.
   Final fitted centers use the same distance check to remove duplicate localizations.
   Enable **Exclude centroids near data edges** to omit edge artifacts from
   previews, fits, and batch processing. The margin defaults to two voxels;
   adjust **Edge margin** to widen it. Existing priors update automatically.
4. Set fit bounds, initial values, fixed parameters, and solver options, then
   preview Gaussian fits. Positions and widths use µm; center settings are
   offsets from each prior.
   Use **Fit filters** to select an amplitude range and covariance or convergence
   pass/fail. Filters update the overlay and are saved for batch exports.
5. Save the JSON settings. Choose an output directory and select **Copy batch
   command** to copy a command for your terminal.

Export writes only the JSON settings. Preview and batch execution call the
shared localize-psf workflow. An example configuration is available in
[localization.json](examples/localization.json).

## Contributing

Run unit and integration tests with
`uv run --locked --group test pytest` from a graphical session
with the NVIDIA GPU available.

GitHub Actions runs unit and non-GPU integration tests on Linux and Windows.

[napari]: https://napari.org/
[localize-psf]: https://github.com/QI2lab/localize-psf/tree/opm-ptv
[uv]: https://docs.astral.sh/uv/
