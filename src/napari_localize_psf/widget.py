"""A napari dock for previewing priors, bounded fits, and reproducible JSON settings."""

from __future__ import annotations

import shlex
import sys
from collections.abc import Callable
from concurrent.futures import CancelledError
from functools import partial
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from napari.layers import Image
from napari.qt.threading import create_worker
from napari.utils.notifications import show_error
from napari.utils.transforms import Affine
from qtpy.QtCore import QEvent, QSignalBlocker, Signal, Slot
from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .config import (
    DEFAULT_PSF_ROI_SIZE_PX,
    FitFilters,
    LocalizationConfig,
    Parameter,
    PreviewSettings,
)
from .data import (
    Preview,
    layer_calibration,
    layer_coordinates,
    layer_frame_interval,
    open_images,
    prepare_image_layers,
    prepare_preview,
    source_coordinates,
)
from .processing import Detector, FitResults, fit_filter_mask, fit_spots, optical_parameters
from .runtime import package_versions

if TYPE_CHECKING:
    from napari import Viewer
    from napari.layers import Points

type BatchOverlay = tuple[pd.DataFrame, dict[str, object], str, str, int]
type PSFAnalysis = dict[str, Any]
type JobResult = (
    dict[str, tuple[float, float, float]]
    | tuple[Preview, tuple[np.ndarray, np.ndarray]]
    | tuple[Preview, FitResults, LocalizationConfig]
    | BatchOverlay
    | PSFAnalysis
)

_SOURCE_EVENTS = ("data", "scale", "translate", "affine", "rotate", "shear", "units", "axis_labels")


class LocalizationWidget(QWidget):
    """Tune one preview while keeping detector, fit, and batch scopes explicit."""

    progress = Signal(str)

    def __init__(self, napari_viewer: Viewer) -> None:
        super().__init__()
        self.setMinimumWidth(480)
        self.viewer = napari_viewer
        self.detector = Detector()
        self._preview: Preview | None = None
        self._detection: tuple[np.ndarray, np.ndarray] | None = None
        self._fit_layer: Points | None = None
        self._psf_result: PSFAnalysis | None = None
        self._tilt_yz_rad: float | None = None
        self.tilt_status = QLabel("Tilt calibration runs during the fit preview.", self)
        self.fit_filter_status = QLabel("Run a GPU fit preview to filter its results.", self)
        self.psf_button = QPushButton("Average PSF and fit field curvature", self)
        self.psf_status = QLabel("Run a 3D fit preview to enable PSF analysis.", self)
        self.psf_plots = QVBoxLayout()
        self._layer: Image | None = None
        self._revision = 0
        self._job_revision = 0
        self._cancelled = Event()
        self._worker = None
        self._job_kind = ""
        self._busy = False
        self._closing = False
        self._has_parent = False
        self._detached = False
        self._calibrated = False
        self._source_error: str | None = None
        self._config_path: Path | None = None
        self._directory_source: Path | None = None
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        (
            data_page,
            prior_page,
            fit_page,
            filter_page,
            export_page,
            results_page,
            tracking_page,
            psf_page,
        ) = (QWidget() for _ in range(8))
        for page, title in zip(
            (
                data_page,
                prior_page,
                fit_page,
                filter_page,
                export_page,
                results_page,
                tracking_page,
                psf_page,
            ),
            (
                "Data",
                "U-FISH",
                "Fits",
                "Fit filters",
                "Export / batch",
                "Batch results",
                "Tracking",
                "PSF / field",
            ),
            strict=True,
        ):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.tabs.addTab(scroll, title)

        form = QFormLayout(data_page)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.path = QLineEdit()
        self.path.setPlaceholderText("OME-TIFF file or OME-Zarr store / URL")
        form.addRow("Input", self.path)
        row = QHBoxLayout()
        for text, callback in (
            ("Choose TIFF", self._choose_tiff),
            ("Choose Zarr", self._choose_zarr),
            ("Open", self._open),
        ):
            button = QPushButton(text)
            button.clicked.connect(callback)
            row.addWidget(button)
        form.addRow(row)
        self.layers = QComboBox()
        self.layers.currentIndexChanged.connect(self._select_layer)
        form.addRow("Image layer", self.layers)
        self.axes = QLineEdit("zyx")
        self.axes.editingFinished.connect(self._refresh_geometry)
        form.addRow("Layer axes", self.axes)
        self.model = QComboBox()
        self.model.addItems(["3d", "2d"])
        self.model.currentTextChanged.connect(self._model_changed)
        form.addRow("Gaussian model", self.model)
        self.indices: dict[str, QSpinBox] = {}
        for key, label in (
            ("timepoint", "Preview time"),
            ("channel", "Layer channel"),
            ("z_plane", "2D Z plane"),
            ("source_channel", "Source channel for batch"),
        ):
            box = QSpinBox()
            box.setRange(0, 1_000_000)
            box.valueChanged.connect(self._invalidate_data)
            self.indices[key] = box
            form.addRow(label, box)
        self.geometry = QTableWidget(3, 4)
        self.geometry.setHorizontalHeaderLabels(["Start px", "Stop px", "Voxel µm", "Origin µm"])
        self.geometry.setVerticalHeaderLabels(["Z", "Y", "X"])
        self.geometry.horizontalHeader().setStretchLastSection(True)
        self.geometry.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.geometry.setMaximumHeight(160)
        self.geometry.itemChanged.connect(self._invalidate_data)
        defaults = LocalizationConfig()
        for axis in range(3):
            for column, value in enumerate(
                (
                    0,
                    1,
                    defaults.voxel_size_zyx_um[axis],
                    0,
                )
            ):
                self.geometry.setItem(axis, column, QTableWidgetItem(str(value)))
        form.addRow(QLabel("Preview and calibration"))
        form.addRow(self.geometry)
        self.manual = QCheckBox("Use manual calibration")
        self.manual.setToolTip(
            "Use the voxel sizes and origins above instead of the image calibration"
        )
        self.manual.toggled.connect(self._invalidate_data)
        form.addRow(self.manual)
        self.calibration_status = QLabel("Select an image layer.")
        self.calibration_status.setWordWrap(True)
        form.addRow(self.calibration_status)

        form = QFormLayout(prior_page)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.blend = QCheckBox("Blend enhancement along Z, Y, and X")
        self.blend.setChecked(True)
        self.blend.toggled.connect(self._invalidate_detection)
        form.addRow(self.blend)
        self.median_filter = QCheckBox("GPU median noise suppression before U-FISH")
        self.median_filter.setChecked(defaults.ufish_median_filter)
        self.median_filter.setToolTip(
            "Apply a 3×3×3 spatial median filter (3×3 in 2D) with reflected borders. "
            "Timepoints and channels are processed independently. Applies to preview "
            "and batch detection; Gaussian fits use original image counts."
        )
        self.median_filter.toggled.connect(self._invalidate_detection)
        form.addRow(self.median_filter)
        self.threshold = QDoubleSpinBox()
        self.threshold.setDecimals(5)
        self.threshold.setRange(0, 1e9)
        self.threshold.setValue(defaults.intensity_threshold)
        self.threshold.valueChanged.connect(self._invalidate_detection)
        form.addRow("Intensity threshold", self.threshold)
        self.collapse_priors = QCheckBox("Collapse nearby priors and fits")
        self.collapse_priors.setChecked(defaults.collapse_close_priors)
        self.collapse_priors.setToolTip(
            "Keep the strongest U-FISH enhancement peak within the optical PSF width "
            "axially and radially in XY, using NA, wavelength, "
            "refractive index and voxel calibration. Final fitted centers are also collapsed, "
            "preferring converged fits, valid covariance, then amplitude. Applies to preview "
            "and batch exports."
        )
        self.collapse_priors.toggled.connect(self._prior_policy_changed)
        form.addRow(self.collapse_priors)
        self.exclude_edges = QCheckBox("Exclude centroids near data edges")
        self.exclude_edges.setChecked(defaults.exclude_edge_centroids)
        self.exclude_edges.setToolTip(
            "Exclude a voxel-distance band at image faces and sloping boundaries of "
            "zero-padded data (Y/X only in 2D). Applies to priors, fits, and batch "
            "processing. Changes reuse enhancement and footprint distances."
        )
        self.edge_margin = QSpinBox()
        self.edge_margin.setRange(1, 1_000_000)
        self.edge_margin.setValue(defaults.edge_margin_px)
        self.edge_margin.setEnabled(defaults.exclude_edge_centroids)
        self.exclude_edges.toggled.connect(self.edge_margin.setEnabled)
        self.exclude_edges.toggled.connect(self._prior_policy_changed)
        self.edge_margin.valueChanged.connect(self._prior_policy_changed)
        form.addRow(self.exclude_edges)
        form.addRow("Edge margin (voxels)", self.edge_margin)
        self.weights = QComboBox()
        self.weights.setEditable(True)
        self.weights.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.weights.addItem("v1.0-alldata-ufish_c32.pth")
        self.weights.addItems(
            [
                f"finetune_models/v1.0.1-{name}_model.pth"
                for name in (
                    "simfish",
                    "deepspot",
                    "deepblink",
                    "seqFISH",
                    "MERFISH",
                    "ExSeq",
                    "RCA",
                    "DNAFISH",
                    "suntag",
                    "CISH",
                )
            ]
        )
        self.weights.setCurrentText(defaults.ufish_weights)
        self.weights.setToolTip(
            "Select published U-FISH weights or enter a custom checkpoint path. "
            "The selected value is saved in JSON for preview and batch prediction."
        )
        self.weights.currentTextChanged.connect(self._invalidate_detection)
        form.addRow("U-FISH model", self.weights)
        self.inference_batch = QSpinBox()
        self.inference_batch.setRange(1, 4096)
        self.inference_batch.setValue(defaults.inference_batch_size)
        self.inference_batch.valueChanged.connect(self._invalidate_detection)
        form.addRow("Inference batch size", self.inference_batch)
        note = QLabel(
            "The threshold is U-FISH's local-maxima response cutoff. Fits use the original counts. Threshold changes reuse the cached enhancement."
        )
        note.setWordWrap(True)
        form.addRow(note)
        self.detect_button = QPushButton("Preview U-FISH priors")
        self.detect_button.clicked.connect(self.run_detection)
        form.addRow(self.detect_button)

        form = QFormLayout(fit_page)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.tilt = QCheckBox("Tilted Gaussian with automatic Y–Z calibration")
        self.tilt.setToolTip("Calibrate from isolated 3D spots and save the angle for batch reuse.")
        self.tilt.toggled.connect(self._invalidate_tilt)
        form.addRow(self.tilt)
        self.tilt_status.setWordWrap(True)
        form.addRow(self.tilt_status)
        self.fit_roi: dict[str, QDoubleSpinBox] = {}
        for axis, value in zip("zyx", defaults.fit_roi_size_um, strict=True):
            box = QDoubleSpinBox()
            box.setDecimals(6)
            box.setRange(0.000001, 1_000_000)
            box.setSingleStep(0.1)
            box.setValue(value)
            box.setToolTip(
                "Full fitting-window width centered on each prior. "
                "Converted upstream to an odd number of voxels."
            )
            self.fit_roi[axis] = box
            form.addRow(f"Fit ROI {axis.upper()} (µm)", box)
        self.parameters = QTableWidget(7, 4)
        self.parameters.setHorizontalHeaderLabels(["Lower", "Upper", "Initial", "Fixed"])
        self.parameter_names = tuple(defaults.fit_parameters)
        self.parameters.setVerticalHeaderLabels(list(self.parameter_names))
        self.parameters.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.parameters.setMaximumHeight(260)
        for row_index, name in enumerate(self.parameter_names):
            setting = defaults.fit_parameters[name]
            for column, value in enumerate((setting.lower, setting.upper, setting.initial)):
                self.parameters.setItem(
                    row_index, column, QTableWidgetItem("" if value is None else str(value))
                )
            self.parameters.setCellWidget(row_index, 3, QCheckBox())
        form.addRow(self.parameters)
        note = QLabel(
            "Positions and widths: µm. Center bounds and initial values are offsets from each U-FISH prior, limited to its ROI. Blank values use upstream estimates / ROI limits. For tilted fits, sxy bounds apply to both principal lateral widths. 2D fixes cz and sz."
        )
        note.setWordWrap(True)
        form.addRow(note)
        self.optics: dict[str, QDoubleSpinBox] = {}
        for key, label, value in (
            ("numerical_aperture", "NA", 1.35),
            ("wavelength_um", "Wavelength µm", 0.515),
            ("refractive_index", "Refractive index", 1.4),
            ("sigma_low", "Lower sigma factor", 0.5),
            ("sigma_high", "Upper sigma factor", 2),
        ):
            box = QDoubleSpinBox()
            box.setDecimals(5)
            box.setRange(0.00001, 100)
            box.setValue(value)
            if not key.startswith("sigma_"):
                box.valueChanged.connect(self._invalidate_detection)
                box.editingFinished.connect(self._prior_policy_changed)
            self.optics[key] = box
            form.addRow(label, box)
        self.optics_button = QPushButton("Apply optical width bounds")
        self.optics_button.clicked.connect(self._apply_optics)
        form.addRow(self.optics_button)
        self.estimator = QComboBox()
        self.estimator.addItems(["MLE", "LSE"])
        form.addRow("Estimator", self.estimator)
        self.solver: dict[str, QSpinBox | QDoubleSpinBox] = {}
        for key, label, value in (
            ("max_number_iterations", "Iteration limit", 200),
            ("roi_batch_size", "ROIs per batch", 32),
            ("tolerance", "Tolerance", 1e-6),
        ):
            box = QDoubleSpinBox() if key == "tolerance" else QSpinBox()
            if isinstance(box, QDoubleSpinBox):
                box.setDecimals(10)
                box.setRange(1e-10, 1)
            else:
                box.setRange(1, 1_000_000)
            box.setValue(value)
            self.solver[key] = box
            form.addRow(label, box)
        self.covariance = QCheckBox("Compute covariance and uncertainty diagnostics")
        form.addRow(self.covariance)
        self.fit_button = QPushButton("Preview bounded GPU fits")
        self.fit_button.clicked.connect(self.run_fitting)
        form.addRow(self.fit_button)

        form = QFormLayout(filter_page)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.amplitude_min = QLineEdit()
        self.amplitude_max = QLineEdit()
        for control, label, placeholder in (
            (self.amplitude_min, "Minimum amplitude (counts)", "No minimum"),
            (self.amplitude_max, "Maximum amplitude (counts)", "No maximum"),
        ):
            control.setPlaceholderText(placeholder)
            control.editingFinished.connect(self._apply_fit_filters)
            form.addRow(label, control)
        self.fit_convergence = QComboBox()
        self.fit_covariance = QComboBox()
        for control, label in (
            (self.fit_convergence, "Converged"),
            (self.fit_covariance, "Covariance valid"),
        ):
            control.addItems(["All", "Pass", "Fail"])
            control.currentIndexChanged.connect(self._apply_fit_filters)
            form.addRow(label, control)
        note = QLabel(
            "All selected conditions must pass. Amplitude limits are inclusive and use "
            "original image counts. Filters update existing fits without refitting and "
            "are saved in JSON for batch exports. Covariance filtering enables diagnostics "
            "for the next fit; existing fits without diagnostics must be rerun."
        )
        note.setWordWrap(True)
        form.addRow(note)
        self.fit_filter_status.setWordWrap(True)
        form.addRow(self.fit_filter_status)
        reset_filters = QPushButton("Reset fit filters")
        reset_filters.clicked.connect(self._reset_fit_filters)
        form.addRow(reset_filters)

        form = QFormLayout(export_page)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.batch_all = QCheckBox("Batch all timepoints and channels")
        self.batch_all.setChecked(True)
        form.addRow(self.batch_all)
        self.config_view = QPlainTextEdit()
        self.config_view.setReadOnly(True)
        form.addRow(self.config_view)
        row = QHBoxLayout()
        for text, callback in (
            ("Inspect JSON", self._show_json),
            ("Save JSON", self._save_json),
            ("Load JSON", self._load_json),
            ("Copy JSON", self._copy_json),
        ):
            button = QPushButton(text)
            button.clicked.connect(callback)
            row.addWidget(button)
        form.addRow(row)
        self.output = QLineEdit()
        self.output.setPlaceholderText("Batch output directory")
        form.addRow("Output directory", self.output)
        self.batch_button = QPushButton("Copy batch command")
        self.batch_button.clicked.connect(self._copy_batch_command)
        form.addRow(self.batch_button)
        self.batch_note = QLabel(
            "Save JSON, choose an output directory, then copy the command to run in a terminal. "
            "The command uses this installed environment."
        )
        self.batch_note.setWordWrap(True)
        form.addRow(self.batch_note)
        self.command_view = QPlainTextEdit()
        self.command_view.setReadOnly(True)
        form.addRow(
            "PowerShell command" if sys.platform == "win32" else "Shell command", self.command_view
        )

        form = QFormLayout(results_page)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.results_path = QLineEdit()
        self.results_path.setPlaceholderText("Batch output directory or localizations.parquet")
        form.addRow("Saved results", self.results_path)
        button = QPushButton("Choose results")
        button.clicked.connect(self._choose_results)
        form.addRow(button)
        self.results_all_timepoints = QCheckBox("All timepoints")
        self.results_all_timepoints.setChecked(True)
        self.results_timepoint = QSpinBox()
        self.results_timepoint.setRange(0, 1_000_000)
        form.addRow(self.results_all_timepoints)
        form.addRow("Timepoint", self.results_timepoint)
        self.results_all_channels = QCheckBox("All channels")
        self.results_all_channels.setChecked(True)
        self.results_channel = QSpinBox()
        self.results_channel.setRange(0, 1_000_000)
        form.addRow(self.results_all_channels)
        form.addRow("Channel", self.results_channel)
        note = QLabel(
            "Overlay saved Gaussian fits on the image selected in Data. "
            "Timepoints and channels are zero-based. Saved batch filters are preserved. "
            "Fit parameters and diagnostics are available in the points features table."
        )
        note.setWordWrap(True)
        form.addRow(note)
        self.results_button = QPushButton("Load / refresh overlay")
        self.results_button.clicked.connect(self.load_batch_results)
        form.addRow(self.results_button)
        self.results_status = QLabel()
        self.results_status.setWordWrap(True)
        form.addRow(self.results_status)

        tracking_layout = QVBoxLayout(tracking_page)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        tracking_layout.addLayout(form)
        self.tracking_path = QLineEdit()
        self.tracking_path.setPlaceholderText("Batch output directory or localizations.parquet")
        self.results_path.textChanged.connect(self.tracking_path.setText)
        self.tracking_path.textChanged.connect(self.results_path.setText)
        form.addRow("Localizations", self.tracking_path)
        button = QPushButton("Choose localizations")
        button.clicked.connect(self._choose_results)
        form.addRow(button)
        self.tracking_tabs = QTabWidget()
        tracking_layout.addWidget(self.tracking_tabs)
        tune_page, run_page = QWidget(), QWidget()
        self.tracking_tabs.addTab(tune_page, "Tune")
        self.tracking_tabs.addTab(run_page, "Run / results")
        form = QFormLayout(tune_page)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.tracking_frames: dict[str, QSpinBox] = {}
        for key, label, value in (("first", "First frame", 0), ("last", "Last frame", 49)):
            box = QSpinBox()
            box.setRange(0, 1_000_000)
            box.setValue(value)
            self.tracking_frames[key] = box
            form.addRow(label, box)
        self.tracking_dimensions = QComboBox()
        self.tracking_dimensions.addItems(["3d", "2d"])
        form.addRow("Link positions", self.tracking_dimensions)
        self.tracking_frame_interval = QDoubleSpinBox()
        self.tracking_frame_interval.setDecimals(9)
        self.tracking_frame_interval.setRange(0, 1e9)
        self.tracking_frame_interval.setSpecialValueText("Unknown (frame lags)")
        self.tracking_frame_interval.setToolTip(
            "Read from the image's time calibration when available. Override in seconds "
            "if needed; zero keeps analysis lags in frames without assuming a frame rate."
        )
        form.addRow("Frame interval (s)", self.tracking_frame_interval)
        self.tracking_ranges: dict[str, QDoubleSpinBox] = {}
        for axis in "zyx":
            box = QDoubleSpinBox()
            box.setDecimals(6)
            box.setRange(0.000001, 1_000_000)
            box.setValue(1.0)
            self.tracking_ranges[axis] = box
            form.addRow(f"Max displacement {axis.upper()} (µm/frame)", box)
        self.tracking_dimensions.currentTextChanged.connect(
            lambda value: self.tracking_ranges["z"].setEnabled(value == "3d")
        )
        self.tracking_memory = QSpinBox()
        self.tracking_memory.setRange(0, 1_000_000)
        self.tracking_memory.setValue(2)
        form.addRow("Allowed missing frames", self.tracking_memory)
        self.tracking_min_length = QSpinBox()
        self.tracking_min_length.setRange(1, 1_000_000)
        self.tracking_min_length.setValue(10)
        form.addRow("Minimum detections per track", self.tracking_min_length)
        self.tracking_adaptive_stop = QDoubleSpinBox()
        self.tracking_adaptive_stop.setDecimals(6)
        self.tracking_adaptive_stop.setRange(0, 1_000_000)
        self.tracking_adaptive_stop.setSpecialValueText("Off")
        form.addRow("Adaptive stop (µm)", self.tracking_adaptive_stop)
        self.tracking_converged = QCheckBox("Use converged localizations only")
        self.tracking_converged.setChecked(True)
        form.addRow(self.tracking_converged)
        self.tracking_all_channels = QCheckBox("Link all channels independently")
        self.tracking_channel = QSpinBox()
        self.tracking_channel.setRange(0, 1_000_000)
        self.tracking_all_channels.toggled.connect(self.tracking_channel.setDisabled)
        form.addRow(self.tracking_all_channels)
        form.addRow("Source channel", self.tracking_channel)
        note = QLabel(
            "Tune on an inclusive frame range. Displacements use physical µm and "
            "Trackpy's ellipsoidal search range. Empty frames count toward memory. "
            "Diffusion uses the standard linker without velocity prediction. "
            "Select a time-series channel image in Data (time first, then spatial axes)."
        )
        note.setWordWrap(True)
        form.addRow(note)
        self.tracking_preview_button = QPushButton("Preview tracks on selected frames")
        self.tracking_preview_button.clicked.connect(partial(self.run_tracking, "preview"))
        form.addRow(self.tracking_preview_button)
        self.tracking_preview_status = QLabel()
        self.tracking_preview_status.setWordWrap(True)
        form.addRow(self.tracking_preview_status)
        form = QFormLayout(run_page)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        note = QLabel(
            "Run the Tune settings over every saved frame in the selected channel(s). "
            "Tracking runs here on a background CPU worker. Save tracks.parquet, "
            "tracking.json, step distributions and individual/ensemble MSD Parquet "
            "analyses and tracking_analysis.pdf beside the localizations, then overlay trajectories on "
            "the full image. Use the Tracks layer's head/tail controls to change visibility."
        )
        note.setWordWrap(True)
        form.addRow(note)
        self.tracking_run_button = QPushButton("Track all frames and save")
        self.tracking_run_button.clicked.connect(partial(self.run_tracking, "full"))
        form.addRow(self.tracking_run_button)
        self.tracking_load_button = QPushButton("Load saved tracks")
        self.tracking_load_button.clicked.connect(partial(self.run_tracking, "saved"))
        form.addRow(self.tracking_load_button)
        self.tracking_status = QLabel()
        self.tracking_status.setWordWrap(True)
        form.addRow(self.tracking_status)

        psf_layout = QVBoxLayout(psf_page)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        psf_layout.addLayout(form)
        note = QLabel(
            "Optional calibration from the current 3D fit preview, using successful fits "
            "that pass Fit filters. Uses original counts from the selected frame, channel "
            "and crop. Field fitting requires at least three noncollinear beads."
        )
        note.setWordWrap(True)
        form.addRow(note)
        self.psf_roi: dict[str, QSpinBox] = {}
        for axis, size in zip("zyx", DEFAULT_PSF_ROI_SIZE_PX, strict=True):
            box = QSpinBox()
            box.setRange(1, 1001)
            box.setSingleStep(2)
            box.setValue(size)
            box.setToolTip("Centered averaging window; even values round up to an odd size.")
            box.valueChanged.connect(self._clear_psf_analysis)
            self.psf_roi[axis] = box
            form.addRow(f"PSF window {axis.upper()} (voxels)", box)
        self.psf_button.setEnabled(False)
        self.psf_button.clicked.connect(self.run_psf_analysis)
        form.addRow(self.psf_button)
        self.psf_status.setWordWrap(True)
        form.addRow(self.psf_status)
        psf_layout.addLayout(self.psf_plots)
        psf_layout.addStretch()

        self.status = QLabel("Select data, verify calibration, then preview U-FISH priors.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)
        layout.addWidget(self.cancel_button)
        self.progress.connect(self.status.setText)
        self.output.textChanged.connect(self.results_path.setText)
        self.results_path.textChanged.connect(self._results_available)
        self.results_all_timepoints.toggled.connect(self._results_available)
        self.results_all_channels.toggled.connect(self._results_available)
        self.tabs.currentChanged.connect(self._results_available)
        self.axes.editingFinished.connect(self._results_available)
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)
        self._refresh_layers()
        self._model_changed()
        self.indices["channel"].valueChanged.connect(self._channel_changed)
        self.destroyed.connect(self._cancelled.set)
        self._results_available()

    def _refresh_layers(self, *args: object) -> None:
        """Keep source selection stable when result layers are added."""
        if self._closing:
            return
        if self._fit_layer is not None and self._fit_layer not in self.viewer.layers:
            self._fit_layer = None
            self._clear_psf_analysis()
        images = [
            layer
            for layer in self.viewer.layers
            if isinstance(layer, Image) and not layer.metadata.get("localization_output")
        ]
        if [self.layers.itemData(i) for i in range(self.layers.count())] == images:
            return
        current = self.layers.currentData()
        self.layers.blockSignals(True)
        self.layers.clear()
        for layer in images:
            self.layers.addItem(layer.name, layer)
        self.layers.setCurrentIndex(
            images.index(current) if current in images else (0 if images else -1)
        )
        self.layers.blockSignals(False)
        self._select_layer()

    def _select_layer(self, *args: object) -> None:
        """Use reader geometry, seed paths per dataset, and reject stale source work."""
        if self._closing:
            return
        if self._layer is not None:
            for name in _SOURCE_EVENTS:
                getattr(self._layer.events, name).disconnect(self._source_changed)
        self._layer = self.layers.currentData()
        if self._layer is None:
            self._invalidate_data()
            self._results_available()
            return
        layer = self._layer
        self._source_error = None
        if (
            layer.source.path
            and layer.source.reader_plugin in {"napari-tiff", "napari-ome-zarr"}
            and "napari_localize_psf" not in layer.metadata
        ):
            siblings = [
                image
                for image in self.viewer.layers
                if isinstance(image, Image) and image.source.path == layer.source.path
            ]
            try:
                prepare_image_layers(siblings, layer.source.path)
            except Exception as error:
                self._source_error = f"Could not read OME image calibration: {error}"
        labels = "".join(str(a).lower() for a in layer.axis_labels)
        self.axes.setText(
            labels
            if len(labels) == layer.ndim and set(labels) <= set("tczyx")
            else {2: "yx", 3: "zyx", 4: "tzyx", 5: "tczyx"}.get(layer.ndim, "")
        )
        metadata = layer.metadata.get("napari_localize_psf", {})
        self.path.setText(metadata.get("input_uri", layer.source.path or ""))
        source = metadata.get("container_uri", layer.source.path or metadata.get("input_uri", ""))
        if source and "://" not in source:
            dataset = Path(source).expanduser().absolute()
            # Nested OME-Zarr image groups share the store's parent directory.
            dataset = next(
                (path for path in (dataset, *dataset.parents) if path.suffix.lower() == ".zarr"),
                dataset,
            )
            if (
                dataset.suffix.lower() in {".zarr", ".tif", ".tiff"}
                or layer.source.reader_plugin in {"napari-tiff", "napari-ome-zarr"}
            ) and dataset != self._directory_source:
                self._directory_source = dataset
                self._config_path = None
                directory = str(dataset.parent)
                self.output.setText(directory)
                self.results_path.setText(directory)
        self.indices["source_channel"].setValue(metadata.get("source_channel", 0))
        self.tracking_channel.setValue(metadata.get("source_channel", 0))
        if layer.ndim == 2:
            self.model.setCurrentText("2d")
        for name in _SOURCE_EVENTS:
            getattr(layer.events, name).connect(self._source_changed)
        self._refresh_geometry()
        self._results_available()

    def _source_changed(self, *args: object) -> None:
        """Invalidate cached data and calibration after edits to the source layer."""
        self._refresh_geometry()

    def _refresh_geometry(self) -> None:
        """Read calibration and bound preview selectors to the source shape."""
        self._invalidate_data()
        layer = self._layer
        if layer is None:
            return
        if self._source_error:
            self._calibrated = False
            self.calibration_status.setText(self._source_error)
            return
        axes = self.axes.text().lower().strip()
        self.tracking_frame_interval.setValue(layer_frame_interval(layer, axes) or 0)
        if len(axes) != layer.ndim or len(set(axes)) != len(axes):
            self.calibration_status.setText("Enter one unique axis label per image dimension.")
            self._calibrated = False
            return
        shape = (layer.data[0] if layer.multiscale else layer.data).shape
        for axis, key in (("t", "timepoint"), ("c", "channel"), ("z", "z_plane")):
            self.indices[key].setMaximum(shape[axes.index(axis)] - 1 if axis in axes else 0)
        for i, axis in enumerate("zyx"):
            size = shape[axes.index(axis)] if axis in axes else 1
            self.geometry.item(i, 0).setText("0")
            self.geometry.item(i, 1).setText(str(size))
        try:
            spacing, origin = layer_calibration(layer, axes)
        except Exception as error:
            self._calibrated = False
            self.manual.setChecked(False)
            self.calibration_status.setText(
                f"Calibration unavailable: {error}. Enter voxel sizes/origins and enable manual calibration."
            )
        else:
            self._calibrated = True
            self.manual.setChecked(False)
            for i in range(3):
                self.geometry.item(i, 2).setText(str(spacing[i]))
                self.geometry.item(i, 3).setText(str(origin[i]))
            self.calibration_status.setText(
                "Physical calibration read from the image in µm. Use manual calibration to override it."
            )

    def _model_changed(self, *args: object) -> None:
        """Expose only the free parameters of the selected native plane/volume fit."""
        if not hasattr(self, "parameters"):
            return
        plane = self.model.currentText() == "2d"
        for row in (3, 5):
            self.parameters.setRowHidden(row, plane)
        self.blend.setEnabled(not plane)
        self.fit_roi["z"].setEnabled(not plane)
        self.tilt.setEnabled(not plane)
        if plane:
            self.tilt.setChecked(False)
        self._invalidate_data()

    def _invalidate_tilt(self, *args: object) -> None:
        """Recalibrate after changing the model while retaining valid U-FISH priors."""
        self._tilt_yz_rad = None
        self._fit_layer = None
        self._revision += 1
        self.tilt_status.setText("Tilt calibration runs during the fit preview.")
        self._clear_psf_analysis()

    def _invalidate_data(self, *args: object) -> None:
        """Drop data-derived state while an in-flight result is marked stale."""
        self._preview = None
        self._invalidate_detection()

    def _channel_changed(self, value: int) -> None:
        """An unsplit channel axis directly determines the selected batch channel."""
        if "c" in self.axes.text():
            self.indices["source_channel"].setValue(value)

    def _invalidate_detection(self, *args: object) -> None:
        """Threshold changes keep enhancement reusable but invalidate candidate points."""
        self._revision += 1
        self._detection = None
        self._fit_layer = None
        self._tilt_yz_rad = None
        self.tilt_status.setText("Tilt calibration runs during the fit preview.")
        self.fit_filter_status.setText("Run a GPU fit preview to filter its results.")
        self._clear_psf_analysis()

    def _prior_policy_changed(self, *args: object) -> None:
        """Refresh existing priors through the shared detector's cached enhancement."""
        self._invalidate_detection()
        if self._preview is not None and not self._busy and not self._closing:
            self.run_detection()

    def _fit_filter_settings(self) -> FitFilters:
        """Adapt optional count limits and status selectors to the upstream policy."""
        return FitFilters(
            amplitude_min=float(self.amplitude_min.text())
            if self.amplitude_min.text().strip()
            else None,
            amplitude_max=float(self.amplitude_max.text())
            if self.amplitude_max.text().strip()
            else None,
            convergence=self.fit_convergence.currentText().lower(),
            covariance=self.fit_covariance.currentText().lower(),
        )

    def _apply_fit_filters(self, *args: object) -> None:
        """Apply the shared mask through napari's public per-point visibility API."""
        if self._closing:
            return
        self._clear_psf_analysis()
        try:
            filters = self._fit_filter_settings()
            if filters.covariance != "all":
                self.covariance.setChecked(True)
            layer = self._fit_layer
            if layer is None or layer not in self.viewer.layers:
                self.fit_filter_status.setText("Run a GPU fit preview to filter its results.")
                return
            features = layer.features
            if filters.covariance != "all" and "covariance_valid" not in features:
                layer.shown = np.zeros(len(layer.data), dtype=bool)
                layer.features = features.assign(filter_pass=layer.shown)
                self._clear_psf_analysis()
                self.fit_filter_status.setText(
                    "Covariance diagnostics were not computed. Enable covariance and rerun GPU fits."
                )
                return
            mask = fit_filter_mask(
                features[
                    list(layer.metadata.get("parameter_names", self.parameter_names))
                ].to_numpy(),
                features["success"].to_numpy(),
                features["covariance_valid"].to_numpy() if "covariance_valid" in features else None,
                filters,
                collapse_distance_zyx_um=layer.metadata.get("collapse_distance_zyx_um"),
            )
            layer.shown = mask
            layer.features = features.assign(filter_pass=mask)
            layer.metadata = {
                **layer.metadata,
                "configuration": {
                    **layer.metadata["configuration"],
                    "fit_filters": filters.model_dump(),
                },
            }
            self.fit_filter_status.setText(f"Showing {int(mask.sum())} / {len(mask)} fits.")
        except Exception as error:
            self.fit_filter_status.setText(str(error))
            self._clear_psf_analysis()
            self.psf_button.setEnabled(False)
            return
        self._clear_psf_analysis()

    def _reset_fit_filters(self) -> None:
        """Restore all preview fits and the default batch acceptance policy."""
        with QSignalBlocker(self.fit_convergence), QSignalBlocker(self.fit_covariance):
            self.amplitude_min.clear()
            self.amplitude_max.clear()
            self.fit_convergence.setCurrentIndex(0)
            self.fit_covariance.setCurrentIndex(0)
        self._apply_fit_filters()

    def _clear_psf_analysis(self, *args: object) -> None:
        """Drop stale plots and enable analysis only for accepted 3D preview fits."""
        if self._busy and self._job_kind == "PSF analysis":
            self._revision += 1
        self._psf_result = None
        while self.psf_plots.count():
            widget = self.psf_plots.takeAt(0).widget()
            widget.deleteLater()
        layer = self._fit_layer
        ready = (
            not self._closing
            and self._preview is not None
            and layer is not None
            and layer in self.viewer.layers
            and layer.metadata["configuration"]["fit_model"] == "3d"
            and min(self._preview[0].shape) > 1
            and (layer.features["success"] & layer.features.get("filter_pass", False)).any()
        )
        self.psf_button.setEnabled(bool(ready))
        self.psf_status.setText(
            "Ready to analyze accepted 3D preview fits."
            if ready
            else "Run a 3D fit preview with successful fits passing Fit filters to enable PSF analysis."
        )

    def run_psf_analysis(self) -> None:
        """Analyze captured original ZYX counts and physical fit centers on a worker."""
        if self._busy or self._closing:
            return
        if not self.psf_button.isEnabled():
            self.status.setText(self.psf_status.text())
            return
        preview, layer = self._preview, self._fit_layer
        assert preview is not None and layer is not None
        config = LocalizationConfig.model_validate(layer.metadata["configuration"])
        features = layer.features
        fits = {
            "fit_params": features[
                list(layer.metadata.get("parameter_names", self.parameter_names))
            ].to_numpy(copy=True),
            "success": features["success"].to_numpy(copy=True),
            "filter_pass": features["filter_pass"].to_numpy(copy=True),
        }
        size = tuple(self.psf_roi[axis].value() for axis in "zyx")
        self._clear_psf_analysis()

        def compute() -> PSFAnalysis:
            from localize_psf.workflow import analyze_psf

            return analyze_psf(
                preview[0], fits, config, roi_size_zyx_px=size, origin_zyx_um=preview[1]
            )

        self.psf_status.setText("Averaging PSFs and fitting the focus surface…")
        self._start(compute, "PSF analysis")

    def _show_psf_analysis(self, result: PSFAnalysis) -> None:
        """Embed PSF slices in relative µm and upstream field plots on the GUI thread."""
        import matplotlib.pyplot as plt
        from localize_psf.workflow import plot_calibration
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        volume = result["volume_results"][0, 0]
        psf = volume["average_psf"]
        coords = [coordinate.ravel() for coordinate in volume["average_psf_coords_zyx_um"]]
        spacing = result["metadata"]["voxel_size_zyx_um"]
        figure = Figure(figsize=(9, 3.5), layout="constrained")
        figure.suptitle("Average PSF")
        for axis, (horizontal, vertical, image, title) in zip(
            figure.subplots(1, 3),
            (
                (2, 1, psf[psf.shape[0] // 2], "XY"),
                (2, 0, psf[:, psf.shape[1] // 2], "XZ"),
                (1, 0, psf[:, :, psf.shape[2] // 2], "YZ"),
            ),
            strict=True,
        ):
            extent = [
                bound
                for dimension in (horizontal, vertical)
                for bound in (
                    coords[dimension][0] - spacing[dimension] / 2,
                    coords[dimension][-1] + spacing[dimension] / 2,
                )
            ]
            axis.imshow(image, origin="lower", extent=extent, cmap="gray", vmin=0, vmax=1)
            axis.set(
                title=title, xlabel=f"{'ZYX'[horizontal]} (µm)", ylabel=f"{'ZYX'[vertical]} (µm)"
            )
        figures = [figure, *plot_calibration(result).values()]
        for index, figure in enumerate(figures):
            # Detach upstream pyplot managers before embedding in this dock.
            plt.close(figure)
            figure.set_size_inches(9 if index == 0 else 13, 3.5 if index == 0 else 5)
            canvas = FigureCanvasQTAgg(figure)
            canvas.setMinimumSize(720 if index == 0 else 1100, 320 if index == 0 else 450)
            self.psf_plots.addWidget(NavigationToolbar2QT(canvas, self))
            self.psf_plots.addWidget(canvas)
            canvas.draw_idle()
        self._psf_result = result
        fits, focus = result["fit_results"], volume["focus"]
        count = int(np.sum(fits["success"] & fits.get("filter_pass", True)))
        text = f"Averaged {count} PSFs. "
        text += (
            f"Focus slope {focus['gradient_magnitude']:.4g} µm/µm; "
            f"field curvature {focus['curvature_per_um']:.4g} µm⁻¹ "
            f"({'converged' if focus['curvature_success'] else 'not converged'})."
            if focus["available"]
            else f"Field fit unavailable: {focus['reason']}."
        )
        self.psf_status.setText(text)
        self.status.setText(text)

    def current_config(self) -> LocalizationConfig:
        """Snapshot and validate all controls before starting any worker or export."""
        if self._layer is None:
            raise ValueError("Select an image layer")
        if self._source_error:
            raise ValueError(self._source_error)
        if not self._calibrated and not self.manual.isChecked():
            raise ValueError("Confirm manual calibration or select an image with physical units")
        if not self.manual.isChecked():
            spacing, origin = layer_calibration(self._layer, self.axes.text().strip().lower())
        else:
            spacing = tuple(float(self.geometry.item(i, 2).text()) for i in range(3))
            origin = tuple(float(self.geometry.item(i, 3).text()) for i in range(3))
        parameters = {}
        for row, name in enumerate(self.parameter_names):
            values = [self.parameters.item(row, column).text().strip() for column in range(3)]
            parameters[name] = Parameter(
                **dict(
                    zip(
                        ("lower", "upper", "initial"),
                        (float(v) if v else None for v in values),
                        strict=True,
                    )
                ),
                fixed=self.parameters.cellWidget(row, 3).isChecked(),
            )
        return LocalizationConfig(
            input_uri=self._layer.metadata.get("napari_localize_psf", {}).get(
                "input_uri", self._layer.source.path or ""
            ),
            source_layer=self._layer.name,
            fit_model=self.model.currentText(),
            psf_model="tilted_gaussian" if self.tilt.isChecked() else "gaussian",
            psf_tilt_yz_rad=self._tilt_yz_rad if self.tilt.isChecked() else None,
            blend_3d=self.blend.isChecked(),
            ufish_median_filter=self.median_filter.isChecked(),
            intensity_threshold=self.threshold.value(),
            collapse_close_priors=self.collapse_priors.isChecked(),
            exclude_edge_centroids=self.exclude_edges.isChecked(),
            edge_margin_px=self.edge_margin.value(),
            ufish_weights=self.weights.currentText().strip(),
            inference_batch_size=self.inference_batch.value(),
            fit_parameters=parameters,
            fit_roi_size_um=tuple(self.fit_roi[axis].value() for axis in "zyx"),
            voxel_size_zyx_um=spacing,
            origin_zyx_um=origin,
            estimator=self.estimator.currentText(),
            return_covariance=self.covariance.isChecked(),
            fit_filters=self._fit_filter_settings(),
            **{key: box.value() for key, box in self.solver.items()},
            **{
                key: box.value() for key, box in self.optics.items() if not key.startswith("sigma_")
            },
            relative_sigma_range=(
                self.optics["sigma_low"].value(),
                self.optics["sigma_high"].value(),
            ),
            batch_axes=self.batch_all.isChecked(),
            timepoint=self.indices["timepoint"].value(),
            channel=self.indices["channel"].value()
            if "c" in self.axes.text()
            else self.indices["source_channel"].value(),
            preview=PreviewSettings(
                axes=self.axes.text().strip().lower(),
                **{key: self.indices[key].value() for key in ("timepoint", "channel", "z_plane")},
                crop_zyx_px=tuple(
                    int(self.geometry.item(i, j).text()) for i in range(3) for j in (0, 1)
                ),
            ),
            provenance=package_versions(),
        )

    def _start(self, function: Callable[[], JobResult], kind: str) -> None:
        """Serialize background operations; publish results on the GUI thread."""
        if self._busy or self._closing:
            return
        self._busy, self._job_kind, self._job_revision = True, kind, self._revision
        self._cancelled.clear()
        self.tabs.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status.setText(f"Running {kind}…")
        self._worker = create_worker(function, _start_thread=False, _ignore_errors=True)
        self._worker.returned.connect(self._returned)
        self._worker.errored.connect(self._errored)
        self._worker.finished.connect(self._finished)
        self._worker.start()

    def run_detection(self) -> None:
        """Preview priors using the same captured data/settings as any subsequent fit."""
        try:
            config = self.current_config()
        except Exception as error:
            self._errored(error)
            return
        layer, preview = self._layer, self._preview
        assert layer is not None

        def compute() -> tuple[Preview, tuple[np.ndarray, np.ndarray]]:
            selected = preview if preview is not None else prepare_preview(layer, config)
            return selected, self.detector.detect(selected[0], config)

        self._start(compute, "U-FISH")

    def _choose_results(self) -> None:
        """Choose an existing upstream localization table through a parented dialog."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open batch localizations",
            self.results_path.text(),
            "Localizations (*.parquet)",
        )
        if path:
            self.results_path.setText(path)

    def _results_available(self, *args: object) -> Path | None:
        """Resolve the output file and limit overlay scope to the selected image axes."""
        layer = self._layer
        if layer is not None:
            axes = self.axes.text().strip().lower()
            shape = (layer.data[0] if layer.multiscale else layer.data).shape
            channel = layer.metadata.get("napari_localize_psf", {}).get("source_channel")
            if len(axes) == layer.ndim and "t" in axes:
                for box in self.tracking_frames.values():
                    box.setMaximum(shape[axes.index("t")] - 1)
            if "z" not in axes:
                self.tracking_dimensions.setCurrentText("2d")
            for axis, all_box, selector in (
                ("t", self.results_all_timepoints, self.results_timepoint),
                ("c", self.results_all_channels, self.results_channel),
            ):
                present = len(axes) == layer.ndim and axis in axes
                all_box.setEnabled(present)
                if not present:
                    with QSignalBlocker(all_box):
                        all_box.setChecked(False)
                selector.setMaximum(shape[axes.index(axis)] - 1 if present else 1_000_000)
                selector.setEnabled(not all_box.isChecked())
                if axis == "c" and not present and channel is not None:
                    selector.setValue(channel)
                    selector.setEnabled(False)
        text = self.results_path.text().strip()
        self.results_button.setEnabled(layer is not None and bool(text))
        if not text:
            self.results_status.setText("Choose a batch output directory or localizations.parquet.")
            return None
        path = Path(text).expanduser()
        if path.is_dir():
            path /= "localizations.parquet"
        if not path.is_file():
            self.results_status.setText("Batch localizations are not available yet.")
            return None
        if layer is None:
            self.results_status.setText(
                "Select the matching image in Data to overlay these results."
            )
            return None
        self.results_status.setText(f"Results available for overlay on {layer.name}.")
        return path

    def load_batch_results(self) -> None:
        """Read existing fits on a worker; publication uses the source's pixel transforms.

        Upstream converts physical centers to full-image ZYX pixels using the saved
        geometry. This adapter adds frame/channel indices in source axis order;
        preview crops and current fit/filter settings do not affect saved results.
        """
        path = self._results_available()
        if path is None:
            return
        try:
            layer = self._layer
            assert layer is not None
            axes = PreviewSettings(axes=self.axes.text().strip().lower()).axes
            if layer.rgb or len(axes) != layer.ndim:
                raise ValueError("Select a scalar image with an explicit, matching axis order")
            timepoint = (
                None if self.results_all_timepoints.isChecked() else self.results_timepoint.value()
            )
            channel = (
                None if self.results_all_channels.isChecked() else self.results_channel.value()
            )
            source_uri = layer.metadata.get("napari_localize_psf", {}).get(
                "input_uri", layer.source.path or ""
            )
        except Exception as error:
            self.results_status.setText(str(error))
            self._errored(error)
            return

        def compute() -> BatchOverlay:
            from localize_psf.workflow import read_localizations

            features, metadata = read_localizations(path, timepoint=timepoint, channel=channel)
            saved_uri = metadata.get("input_uri", "")
            if source_uri and saved_uri:
                source = (
                    source_uri.rstrip("/")
                    if "://" in source_uri
                    else str(Path(source_uri).resolve())
                )
                saved = (
                    saved_uri.rstrip("/") if "://" in saved_uri else str(Path(saved_uri).resolve())
                )
                if source != saved:
                    raise ValueError("Select the batch results' matching input image in Data")
            finite = np.isfinite(features[["z", "y", "x"]].to_numpy()).all(axis=1)
            return (
                features.loc[finite].reset_index(drop=True),
                metadata,
                str(path),
                axes,
                int((~finite).sum()),
            )

        self._start(compute, "Batch results")

    def run_tracking(self, scope: str = "preview") -> None:
        """Tune or execute Trackpy on saved fits; publish only on the GUI thread.

        Preview uses inclusive frame limits and never saves. Full execution uses
        all source frames and saves the current tuning settings and linked rows.
        Saved results reload without linking. Native Tracks layers require a
        time-first channel image; reader-opened OME channels already use this layout.
        """
        path = self._results_available()
        if path is None:
            self.tracking_status.setText(self.results_status.text())
            self.tracking_preview_status.setText(self.results_status.text())
            return
        try:
            layer = self._layer
            assert layer is not None
            axes = PreviewSettings(axes=self.axes.text().strip().lower()).axes
            if layer.rgb or len(axes) != layer.ndim or not axes.startswith("t") or "c" in axes:
                raise ValueError(
                    "Select a time-first channel image in Data (T,[Z,]Y,X); "
                    "open OME data through the readers to split channels."
                )
            if scope not in {"preview", "full", "saved"}:
                raise ValueError("Choose tracking preview, full execution, or saved results")
            source_uri = layer.metadata.get("napari_localize_psf", {}).get(
                "input_uri", layer.source.path or ""
            )
            settings = dict(
                search_range_um=tuple(self.tracking_ranges[axis].value() for axis in "zyx"),
                memory=self.tracking_memory.value(),
                min_length=self.tracking_min_length.value(),
                dimensions=self.tracking_dimensions.currentText() if "z" in axes else "2d",
                first_frame=self.tracking_frames["first"].value() if scope == "preview" else None,
                last_frame=self.tracking_frames["last"].value() if scope == "preview" else None,
                channel=None
                if self.tracking_all_channels.isChecked()
                else self.tracking_channel.value(),
                converged_only=self.tracking_converged.isChecked(),
                adaptive_stop_um=self.tracking_adaptive_stop.value() or None,
                input_uri=source_uri,
                save=scope == "full",
                frame_interval_s=self.tracking_frame_interval.value() or None,
            )
        except Exception as error:
            self.tracking_status.setText(str(error))
            self.tracking_preview_status.setText(str(error))
            self._errored(error)
            return

        def compute() -> BatchOverlay:
            if scope == "saved":
                import json

                tracks = pd.read_parquet(path.with_name("tracks.parquet"))
                metadata = json.loads(path.with_name("tracking.json").read_text(encoding="utf-8"))
                if metadata.get("tracking", {}).get("schema_version") != 1:
                    raise ValueError("Unsupported tracking settings version")
                if not {"particle", "frame", "channel", "z", "y", "x"} <= set(tracks.columns):
                    raise ValueError(
                        "Select a saved Trackpy result with particle, frame and positions"
                    )
                saved_uri = metadata.get("input_uri", "")
                if (
                    source_uri
                    and saved_uri
                    and len(
                        {
                            uri.rstrip("/") if "://" in uri else str(Path(uri).resolve())
                            for uri in (source_uri, saved_uri)
                        }
                    )
                    != 1
                ):
                    raise ValueError("Select the tracks' matching input image in Data")
            else:
                from .tracking import link_localizations

                tracks, metadata = link_localizations(
                    path, **settings, cancelled=self._cancelled, progress=self.progress.emit
                )
            return tracks, metadata, str(path), axes, int(metadata["tracking"]["skipped_nonfinite"])

        self._start(compute, f"Tracking {scope}")

    def run_fitting(self) -> None:
        """Fit only candidate priors produced for the current input and detector controls."""
        try:
            config = self.current_config()
            if self._preview is None or self._detection is None:
                raise ValueError("Preview U-FISH priors for the current selection before fitting")
        except Exception as error:
            self._errored(error)
            return
        preview, detection = self._preview, self._detection

        def compute() -> tuple[Preview, FitResults, LocalizationConfig]:
            self.detector.release()
            result = fit_spots(
                preview[0],
                detection[0],
                config,
                self._cancelled,
                self.progress.emit,
                origin_zyx_um=preview[1],
            )
            resolved = config.model_copy(update={"psf_tilt_yz_rad": result.get("psf_tilt_yz_rad")})
            return preview, result, resolved

        if not self._busy:
            self._clear_psf_analysis()
        self._start(compute, "GPUfit")

    @Slot(object)
    def _returned(self, value: object) -> None:
        """Discard cancelled/stale results and add overlays in source pixel coordinates."""
        if self._closing:
            return
        if self._cancelled.is_set() or self._job_revision != self._revision:
            self.status.setText(
                "Result discarded because the job was cancelled or the source changed."
            )
            if self._job_kind == "PSF analysis":
                self.psf_status.setText(self.status.text())
            return
        layer = self._layer
        if self._job_kind == "Optical bounds":
            bounds = cast(dict[str, tuple[float, float, float]], value)
            for name, axis in (("sxy", 1), ("sz", 0)):
                row = self.parameter_names.index(name)
                for column, key in enumerate(
                    ("lower_sigma_zyx_um", "upper_sigma_zyx_um", "nominal_sigma_zyx_um")
                ):
                    self.parameters.item(row, column).setText(str(bounds[key][axis]))
            self.status.setText("Applied upstream optical width estimates and bounds.")
            return
        if layer is None:
            return
        if self._job_kind == "PSF analysis":
            try:
                self._show_psf_analysis(cast(PSFAnalysis, value))
            except Exception as error:
                self._errored(error)
            return
        transforms = {
            name: getattr(layer, name)
            for name in ("scale", "translate", "rotate", "shear", "affine", "axis_labels", "units")
        }
        if self._job_kind == "U-FISH":
            preview, detection = cast(tuple[Preview, tuple[np.ndarray, np.ndarray]], value)
            self._preview, self._detection = preview, detection
            self._fit_layer = None
            self._clear_psf_analysis()
            points, enhancement = detection
            source_config = preview[2]
            self._replace_points(
                "U-FISH priors",
                layer_coordinates(points, source_config),
                transforms,
                pd.DataFrame(),
            )
            enhanced_name = "U-FISH enhancement"
            if enhanced_name in self.viewer.layers:
                self.viewer.layers.remove(enhanced_name)
            # Enhancement is crop-local; express the crop origin in source pixels.
            selection = source_config.preview
            enhanced = enhancement if source_config.fit_model == "3d" else enhancement[None]
            full_shape = [
                1 if a in "tc" else enhanced.shape["zyx".index(a)] for a in selection.axes
            ]
            spatial = "".join(a for a in selection.axes if a in "zyx")
            data = enhanced[0] if "z" not in spatial else enhanced
            data = (
                data.transpose(tuple("zyx".replace("z", "").index(a) for a in spatial))
                if "z" not in spatial
                else data.transpose(tuple("zyx".index(a) for a in spatial))
            )
            offset = layer_coordinates(np.zeros((1, 3)), source_config)[0]
            # Keep voxel calibration and the shared global affine separate.
            # Only the crop's first voxel changes the enhancement translation.
            source_transform = Affine(
                scale=layer.scale,
                translate=layer.translate,
                rotate=layer.rotate,
                shear=layer.shear,
                ndim=layer.ndim,
            )
            self.viewer.add_image(
                data.reshape(full_shape),
                name=enhanced_name,
                **{**transforms, "translate": source_transform(offset)},
                colormap="magenta",
                blending="additive",
                opacity=0.4,
                metadata={"localization_output": True},
            )
            self.status.setText(f"{len(points)} U-FISH priors. Tune threshold or preview GPU fits.")
        elif self._job_kind == "Batch results":
            features, metadata, path, axes, skipped = cast(BatchOverlay, value)
            points = source_coordinates(
                features[["z", "y", "x"]].to_numpy(),
                axes,
                timepoints=features["frame"].to_numpy(),
                channels=features["channel"].to_numpy(),
            )
            overlay = self._replace_points(
                f"Batch localizations: {layer.name}", points, transforms, features
            )
            overlay.metadata = {
                **overlay.metadata,
                "configuration": metadata,
                "batch_results_path": path,
                "source_layer": layer.name,
            }
            text = f"Showing {len(points)} batch localizations on {layer.name}."
            if skipped:
                text += f" Skipped {skipped} nonfinite fitted centers."
            self.results_status.setText(text)
            self.status.setText(text)
        elif self._job_kind.startswith("Tracking "):
            tracks, metadata, path, axes, skipped = cast(BatchOverlay, value)
            if self._job_kind == "Tracking saved":
                settings = metadata["tracking"]
                for axis, distance in zip("zyx", settings["search_range_zyx_um"], strict=True):
                    self.tracking_ranges[axis].setValue(distance)
                self.tracking_dimensions.setCurrentText(settings["dimensions"])
                self.tracking_memory.setValue(settings["memory"])
                self.tracking_min_length.setValue(settings["min_length"])
                self.tracking_converged.setChecked(settings["converged_only"])
                if "frame_interval_s" in settings:
                    self.tracking_frame_interval.setValue(settings["frame_interval_s"] or 0)
                self.tracking_adaptive_stop.setValue(settings["adaptive_stop_um"] or 0)
                self.tracking_all_channels.setChecked(settings["channel"] is None)
                if settings["channel"] is not None:
                    self.tracking_channel.setValue(settings["channel"])
            points = source_coordinates(
                tracks[["z", "y", "x"]].to_numpy(),
                axes,
                timepoints=tracks["frame"].to_numpy(),
                channels=tracks["channel"].to_numpy(),
            )
            name = f"{'Tracking preview' if self._job_kind == 'Tracking preview' else 'Tracks'}: {layer.name}"
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)
            other = f"{'Tracks' if self._job_kind == 'Tracking preview' else 'Tracking preview'}: {layer.name}"
            if other in self.viewer.layers:
                self.viewer.layers[other].visible = False
            if len(tracks):
                span = int(tracks["frame"].max() - tracks["frame"].min() + 1)
                self.viewer.add_tracks(
                    np.column_stack((tracks["particle"].to_numpy(), points)),
                    name=name,
                    **transforms,
                    features=tracks,
                    tail_length=span,
                    head_length=span,
                    metadata={
                        "localization_output": True,
                        "configuration": metadata,
                        "batch_results_path": path,
                        "source_layer": layer.name,
                    },
                )
            text = f"{tracks['particle'].nunique()} tracks, {len(tracks)} linked localizations."
            if skipped:
                text += f" Skipped {skipped} nonfinite centers."
            if self._job_kind == "Tracking full":
                text += f" Saved tracks, analyses and tracking_analysis.pdf in {Path(path).parent}."
            target = (
                self.tracking_preview_status
                if self._job_kind == "Tracking preview"
                else self.tracking_status
            )
            target.setText(text)
            self.status.setText(text)
        elif self._job_kind == "GPUfit":
            preview, result, config = cast(tuple[Preview, FitResults, LocalizationConfig], value)
            parameters = result["fit_params"]
            _, origin, source_config = preview
            points = (parameters[:, (3, 2, 1)] - origin) / source_config.voxel_size_zyx_um
            names = tuple(result.get("parameter_names", self.parameter_names))
            features = pd.DataFrame(parameters, columns=names)
            features["success"] = result["success"]
            features["iterations"] = result["niters"]
            features["objective"] = result["chi_sqrs"]
            features["active_bound"] = np.any(result["active_bounds"], axis=1)
            if "covariance_valid" in result:
                features["covariance_valid"] = result["covariance_valid"]
            if "roi_truncated" in result:
                features["roi_truncated"] = result["roi_truncated"]
            self._tilt_yz_rad = config.psf_tilt_yz_rad
            if self._tilt_yz_rad is not None:
                self.tilt_status.setText(
                    f"Calibrated Y–Z tilt: {np.degrees(self._tilt_yz_rad):.2f}°. Included in JSON export."
                )
            self._fit_layer = self._replace_points(
                "Gaussian fits",
                layer_coordinates(points, source_config),
                transforms,
                features,
                config,
            )
            self._fit_layer.metadata = {
                **self._fit_layer.metadata,
                "collapse_distance_zyx_um": result.get("collapse_distance_zyx_um"),
                "parameter_names": names,
            }
            self._apply_fit_filters()
            self.status.setText(
                f"Converged {int(np.sum(result['success']))} / {len(parameters)} fits; {int(np.sum(features['active_bound']))} with active bounds. Parameters are in the points features table."
            )

    def _replace_points(
        self,
        name: str,
        data: np.ndarray,
        transforms: dict[str, object],
        features: pd.DataFrame,
        config: LocalizationConfig | None = None,
    ) -> Points:
        """Keep named overlays and their inspectable production fit features together."""
        if name in self.viewer.layers:
            self.viewer.layers.remove(name)
        kwargs = {"features": features} if len(features.columns) else {}
        return self.viewer.add_points(
            data,
            name=name,
            size=5,
            face_color="cyan" if name.startswith("U-FISH") else "yellow",
            **transforms,
            **kwargs,
            metadata={
                "localization_output": True,
                "configuration": config.model_dump() if config else {},
            },
        )

    @Slot(object)
    def _errored(self, error: Exception) -> None:
        """Surface validation and worker failures through Qt notifications."""
        if self._closing:
            return
        text = str(error) or type(error).__name__
        self.status.setText(text)
        if self._busy and self._job_kind == "Batch results":
            self.results_status.setText(text)
        if self._busy and self._job_kind.startswith("Tracking "):
            self.tracking_status.setText(text)
            self.tracking_preview_status.setText(text)
        if self._busy and self._job_kind == "PSF analysis":
            self._clear_psf_analysis()
            self.psf_status.setText(text)
        if not isinstance(error, CancelledError):
            show_error(text)

    @Slot()
    def _finished(self) -> None:
        """Restore controls after upstream work returns or raises."""
        self._busy = False
        self._worker = None
        if self._closing:
            self.detector.release()
            if self._detached:
                self.deleteLater()
            return
        self.tabs.setEnabled(True)
        self.cancel_button.setEnabled(False)

    def cancel(self) -> None:
        """Cancel publication and stop between upstream inference or fit batches."""
        self._cancelled.set()
        if self._worker is not None:
            self._worker.quit()
        self.status.setText("Cancel requested; waiting for the current operation.")

    def _apply_optics(self) -> None:
        try:
            config = self.current_config()
        except Exception as error:
            self._errored(error)
            return
        self._start(partial(optical_parameters, config), "Optical bounds")

    def _choose_tiff(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open OME-TIFF",
            str(self._directory_source.parent) if self._directory_source else "",
            "TIFF (*.tif *.tiff)",
        )
        if path:
            self.path.setText(path)
            self._open()

    def _choose_zarr(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Open OME-Zarr store",
            str(self._directory_source.parent) if self._directory_source else "",
        )
        if path:
            self.path.setText(path)
            self._open()

    def _open(self) -> None:
        try:
            images = open_images(self.viewer, self.path.text().strip())
            if not images:
                raise ValueError("The reader returned no image layers")
            self._refresh_layers()
            self.layers.setCurrentIndex(self.layers.findData(images[0]))
            self._select_layer()
        except Exception as error:
            self._errored(error)

    def _show_json(self) -> None:
        try:
            self.config_view.setPlainText(self.current_config().model_dump_json(indent=2))
        except Exception as error:
            self._errored(error)

    def _copy_json(self) -> None:
        try:
            text = self.current_config().model_dump_json(indent=2)
            self.config_view.setPlainText(text)
            QApplication.clipboard().setText(text)
        except Exception as error:
            self._errored(error)

    def _save_json(self) -> None:
        try:
            config = self.current_config()
            suggested = self._config_path or (
                (self._directory_source.parent if self._directory_source else Path.cwd())
                / "localization.json"
            )
            path, _ = QFileDialog.getSaveFileName(
                self, "Save localization settings", str(suggested), "JSON (*.json)"
            )
            if path:
                config.save(path)
                self._config_path = Path(path)
                self.config_view.setPlainText(config.model_dump_json(indent=2))
                self.status.setText(f"Saved {path}. Batch execution requires the updated runner.")
        except Exception as error:
            self._errored(error)

    def _load_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load localization settings",
            str(self._config_path or self._directory_source.parent)
            if self._config_path or self._directory_source
            else "",
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            config = LocalizationConfig.load(path)
            self.apply_config(config)
            self._config_path = Path(path)
        except Exception as error:
            self._errored(error)

    def apply_config(self, config: LocalizationConfig) -> None:
        """Restore a validated configuration without opening or executing its input URI."""
        if self._layer is not None:
            source = self._layer.metadata.get("napari_localize_psf", {}).get(
                "input_uri", self._layer.source.path or ""
            )
            if config.input_uri != source:
                raise ValueError("Open the configuration's input image before loading its settings")
        self.model.setCurrentText(config.fit_model)
        with QSignalBlocker(self.tilt):
            self.tilt.setChecked(config.psf_model == "tilted_gaussian")
        self.axes.setText(config.preview.axes)
        self.path.setText(config.input_uri)
        self.blend.setChecked(config.blend_3d)
        self.median_filter.setChecked(config.ufish_median_filter)
        self.threshold.setValue(config.intensity_threshold)
        with (
            QSignalBlocker(self.exclude_edges),
            QSignalBlocker(self.edge_margin),
            QSignalBlocker(self.collapse_priors),
        ):
            self.exclude_edges.setChecked(config.exclude_edge_centroids)
            self.edge_margin.setValue(config.edge_margin_px)
            self.collapse_priors.setChecked(config.collapse_close_priors)
        self.edge_margin.setEnabled(config.exclude_edge_centroids)
        self.weights.setCurrentText(config.ufish_weights)
        self.inference_batch.setValue(config.inference_batch_size)
        self.manual.setChecked(True)
        for key in ("timepoint", "channel", "z_plane"):
            self.indices[key].setValue(getattr(config.preview, key))
        self.indices["source_channel"].setValue(config.channel)
        for i in range(3):
            for j, value in enumerate(
                (
                    *config.preview.crop_zyx_px[i * 2 : i * 2 + 2],
                    config.voxel_size_zyx_um[i],
                    config.origin_zyx_um[i],
                )
            ):
                self.geometry.item(i, j).setText(str(value))
        for axis, value in zip("zyx", config.fit_roi_size_um, strict=True):
            self.fit_roi[axis].setValue(value)
        for row, name in enumerate(self.parameter_names):
            setting = config.fit_parameters[name]
            for column, value in enumerate((setting.lower, setting.upper, setting.initial)):
                self.parameters.item(row, column).setText("" if value is None else str(value))
            self.parameters.cellWidget(row, 3).setChecked(setting.fixed)
        for key, box in self.solver.items():
            box.setValue(getattr(config, key))
        for key, box in self.optics.items():
            box.setValue(
                config.relative_sigma_range[0 if key == "sigma_low" else 1]
                if key.startswith("sigma_")
                else getattr(config, key)
            )
        self.estimator.setCurrentText(config.estimator)
        self.covariance.setChecked(config.return_covariance)
        with QSignalBlocker(self.fit_convergence), QSignalBlocker(self.fit_covariance):
            self.amplitude_min.setText(
                ""
                if config.fit_filters.amplitude_min is None
                else str(config.fit_filters.amplitude_min)
            )
            self.amplitude_max.setText(
                ""
                if config.fit_filters.amplitude_max is None
                else str(config.fit_filters.amplitude_max)
            )
            self.fit_convergence.setCurrentText(config.fit_filters.convergence.capitalize())
            self.fit_covariance.setCurrentText(config.fit_filters.covariance.capitalize())
        self.batch_all.setChecked(config.batch_axes)
        self._invalidate_data()
        self._tilt_yz_rad = config.psf_tilt_yz_rad
        if self._tilt_yz_rad is not None:
            self.tilt_status.setText(
                f"Calibrated Y–Z tilt: {np.degrees(self._tilt_yz_rad):.2f}°. Included in JSON export."
            )
        self.config_view.setPlainText(config.model_dump_json(indent=2))

    def _copy_batch_command(self) -> None:
        """Save current settings and copy a quoted terminal command without executing it."""
        try:
            if self._busy or self._closing:
                raise ValueError("Finish the current preview before preparing a batch command")
            config = self.current_config()
            if self._config_path is None or not self.output.text().strip() or not config.input_uri:
                raise ValueError("Save JSON and choose an input and output directory first")
            input_uri = config.input_uri
            container = self._layer.metadata.get("napari_localize_psf", {}).get("container_uri")
            if container and container != input_uri:
                from localize_psf.workflow import ome_zarr_image_paths

                if ome_zarr_image_paths(container) == [input_uri]:
                    input_uri = container
            config.save(self._config_path)
            arguments = [
                sys.executable,
                "-m",
                "localize_psf.batch",
                input_uri if "://" in input_uri else str(Path(input_uri).resolve()),
                "--metadata",
                str(self._config_path.resolve()),
                "--output-dir",
                str(Path(self.output.text().strip()).resolve()),
            ]
            command = (
                "& " + " ".join("'" + argument.replace("'", "''") + "'" for argument in arguments)
                if sys.platform == "win32"
                else shlex.join(arguments)
            )
            self.config_view.setPlainText(config.model_dump_json(indent=2))
            self.command_view.setPlainText(command)
            QApplication.clipboard().setText(command)
            self.status.setText("Saved current JSON; copied the batch command for your terminal.")
        except Exception as error:
            self._errored(error)

    def shutdown(self) -> None:
        """Disconnect napari events and release owned resources without racing a worker.

        Upstream inference is cooperative: an active worker keeps its detector
        alive until finished, then releases it without publishing stale results.
        """
        self._closing = True
        self._cancelled.set()
        self._revision += 1
        self.viewer.layers.events.inserted.disconnect(self._refresh_layers)
        self.viewer.layers.events.removed.disconnect(self._refresh_layers)
        if self._layer is not None:
            for name in _SOURCE_EVENTS:
                getattr(self._layer.events, name).disconnect(self._source_changed)
        self._layer = self._preview = self._detection = self._fit_layer = None
        self._clear_psf_analysis()
        self.layers.clear()
        if self._worker is not None:
            self._worker.quit()
        if not self._busy:
            self.detector.release()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Clean up both standalone widgets and widgets explicitly closed by their owner."""
        self.shutdown()
        super().closeEvent(event)

    def event(self, event: QEvent) -> bool:
        """Handle napari removing the inner widget without sending it a close event."""
        if event.type() == QEvent.Type.ParentChange and hasattr(self, "_closing"):
            if self.parent() is not None:
                self._has_parent = True
            elif self._has_parent:
                self._detached = True
                self.shutdown()
                if not self._busy:
                    self.deleteLater()
        return super().event(event)
