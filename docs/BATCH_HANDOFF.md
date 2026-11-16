# Implement the localize-psf batch runner

Work on QI2lab/localize-psf's `opm-ptv` branch. The napari plugin is implemented
and tested locally; its existing batch example has not been changed. Make batch
processing reproduce the plugin's exported settings.

1. Read `src/napari_localize_psf/config.py`, `processing.py`, and `data.py` in
   napari-localize-psf. Use `docs/localization.schema.json` and
   `examples/localization.json` as the configuration contract. Move reusable
   configuration and parameter-policy adapters into localize-psf and update the
   plugin to import them; do not maintain duplicate implementations or make the
   batch runner depend on Qt/napari. Preserve legacy metadata JSON compatibility,
   but validate schema version 1 strictly and reject unsupported settings.

2. Package a runner callable as:

   ```bash
   python -m localize_psf.batch INPUT --metadata CONFIG.json --output-dir OUTPUT
   ```

   The plugin saves current JSON and copies this command with its own uv
   interpreter and quoted paths. Users run it in a terminal; the plugin never
   launches or manages a batch process. Make `examples/localize_ufish_prior.py` a thin caller of the
   shared workflow while preserving its current CLI and exports.

3. Accept a single OME-Zarr image group or OME-TIFF. Reuse upstream readers and
   metadata/transform models. Preserve axis order, channel identity, highest
   resolution, units, translations, and explicit exported calibration. Treat
   `preview` as preview/reproduction scope: normal batch execution processes the
   full spatial dataset. `batch_axes=true` selects all times/channels; otherwise
   use the explicit `timepoint` and `channel`. For 2D, process XY planes separately.

4. Forward `ufish_weights`, `blend_3d`, `intensity_threshold`, and
   `inference_batch_size` to the supplied QI2lab U-FISH. Fit original counts.
   Checkpoint identity must include the input geometry, weights, dimensional
   mode, blend setting, threshold, and preprocessing policy. Detector changes
   invalidate centroids; fit-only changes allow reuse.

5. Reuse production `get_coords`, `get_roi_size`, `get_centered_rois`,
   `prepare_rois`, `fit_rois`, bounds validation, accumulation, and covariance.
   Honor `fit_parameters`, ROI size, estimator, tolerance, iteration limit,
   batch size, and covariance selection. Center limits/initial values are µm
   offsets from each prior, intersected with its ROI; null means upstream
   estimate/ROI limit. Validate feasible starts without clipping. For 2D, reuse
   the existing native 3D Gaussian on one plane with `cz` fixed at that plane's
   physical Z and `sz=1 µm`, exactly as tested in the plugin. Preserve all
   candidate rows, IDs, statuses, active bounds, and uncertainty flags.

6. Use only the installed uv CUDA environment and existing package-managed
   GPUfit toolchain. Add annotated code with minimal helpers. Add only unit and
   integration tests using production code; simulated inputs may use upstream
   PSF models. Require a real GPU. Verify preview/batch parity in 2D/3D,
   nonzero origins, T/C selection, bounded/fixed parameters, empty detections,
   checkpoint invalidation, and configuration round trips. Update both locks
   after the shared API is available and rerun both repositories' required tests.
   Keep solver, CUDA kernel, ABI, and installation validation in the upstream
   packages; plugin tests should verify only its own adapters and UI behavior.
