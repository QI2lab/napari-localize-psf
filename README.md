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
   Output/results directories and file dialogs default to the dataset's parent folder.
2. Confirm the axes, physical calibration, and preview region. Choose a 2D plane
   or 3D volume.
3. Select a pretrained checkpoint from **U-FISH model**, or type a custom
   weights path into the same dropdown. SunTag is the default; the list
   also includes the general all-data model and nine other fine-tuned checkpoints.
   The selected weights are saved in JSON for batch prediction.
   Preview U-FISH priors and adjust `blend_3d` and the intensity threshold (default 0.1).
   **GPU median noise suppression** is enabled by default and applies a 3×3×3 spatial
   filter before U-FISH (3×3 in 2D). Each timepoint/channel is filtered independently;
   Gaussian fitting uses the original counts.
   Nearby priors and duplicate fitted centers are merged automatically using
   the optical settings in **Fits**.
   **Exclude centroids near data edges** is enabled by default to omit edge artifacts
   from previews, fits, and batch processing. The margin defaults to one voxel;
   adjust **Edge margin** to widen it. Existing priors update automatically.
4. Set fit bounds, initial values, fixed parameters, and solver options, then
   preview Gaussian fits. Positions and widths use µm; center settings are
   offsets from each prior.
   Use **Fit filters** to select an amplitude range and covariance or convergence
   pass/fail. Filters update the overlay and are saved for batch exports.
5. Save the JSON settings. Choose an output directory and select **Copy batch
   command** to copy a command for your terminal.
6. Once batch fits are available, use **Batch results** to overlay
   `localizations.parquet` on the selected image. Choose all timepoints/channels
   or a single selection, then select **Load / refresh overlay**.

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

## Tracking

After loading `localizations.parquet` in **Batch results**, open **Tracking**.
Select the matching time-series image in **Data**. Reader-opened OME images
already have separate channel layers; tracks use time-first image axes such as
`TZYX` or `TYX` and preserve the image's transforms.

1. In **Tune**, select an inclusive, zero-based first/last frame range. Set the
   maximum displacement along Z/Y/X in micrometres per frame, allowed missing
   frames, and minimum detections per track. Defaults are 1.0 µm per axis,
   2 missing frames, and 10 detections. Choose 2D XY or 3D linking. The
   standard Trackpy linker is used for diffusion, without velocity prediction.
   **Adaptive stop** optionally lets Trackpy reduce the displacement range for
   crowded subnetworks; zero disables it. Saved localization acceptance masks
   remain in effect, and converged-only linking is selected by default.
2. Select one source channel or link all channels independently, then click
   **Preview tracks on selected frames**. Tuning overlays trajectories without
   writing files. Empty frames count toward the allowed missing-frame limit.
3. In **Run / results**, click **Track all frames and save**. This uses the same
   tuning parameters over every saved frame, runs in a background CPU worker,
   and writes tracks, tracking settings and motion-analysis Parquet files beside
   the input. **Cancel** stops between linking or analysis operations. The original
   localizations are preserved.
4. Tracks overlay the full source image with their original frame indices and
   unique particle IDs across channels. Use the native Tracks layer's head/tail
   controls to adjust the displayed trajectory window. **Load saved tracks**
   restores both trajectories and the saved linking parameters.

Full runs automatically save `step_sizes.parquet` (signed Δz/Δy/Δx and the 3D
magnitude r), `step_size_distributions.parquet` (marginal histograms),
`msd_individual.parquet`, and `msd_ensemble.parquet`, separately by channel.
`tracking_analysis.pdf` has six pages per channel: four step-size distributions,
one ensemble MSD plot, and one plot of all individual MSD curves, with titles and axes only.
All step-size distributions share the same bin edges and plot limits across channels.
Steps use adjacent frames; MSD includes all available lags and handles gaps
through Trackpy. Distances use µm and MSD uses µm². **Frame interval (s)** reads
the image calibration and can be overridden; unknown timing retains frame lags
with null seconds. No drift correction is applied.

Linking, motion analysis and trajectory-length filtering call Trackpy directly.
Input coordinate conversion reuses localize-psf; tracking is implemented only in this plugin.
