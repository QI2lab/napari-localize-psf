"""A napari dock for previewing priors, bounded fits, and reproducible JSON settings."""

from __future__ import annotations

import shlex
import sys
from collections.abc import Callable
from concurrent.futures import CancelledError
from functools import partial
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from napari.layers import Image
from napari.qt.threading import create_worker
from napari.utils.notifications import show_error
from napari.utils.transforms import Affine
from qtpy.QtCore import QEvent, Signal, Slot
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

from .config import LocalizationConfig, Parameter, PreviewSettings
from .data import (
    Preview,
    layer_calibration,
    layer_coordinates,
    open_images,
    prepare_image_layers,
    prepare_preview,
)
from .processing import Detector, FitResults, fit_spots, optical_parameters
from .runtime import package_versions

if TYPE_CHECKING:
    from napari import Viewer

type JobResult = (
    dict[str, tuple[float, float, float]]
    | tuple[Preview, tuple[np.ndarray, np.ndarray]]
    | tuple[Preview, FitResults, LocalizationConfig]
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
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        data_page, prior_page, fit_page, export_page = (QWidget() for _ in range(4))
        for page, title in zip(
            (data_page, prior_page, fit_page, export_page),
            ("Data", "U-FISH", "Fits", "Export / batch"),
            strict=True,
        ):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.tabs.addTab(scroll, title)

        form = QFormLayout(data_page)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.path = QLineEdit()
        self.path.setPlaceholderText("OME-TIFF file or OME-Zarr image-group path / URL")
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
        self.geometry = QTableWidget(3, 5)
        self.geometry.setHorizontalHeaderLabels(
            ["Start px", "Stop px", "Voxel µm", "Origin µm", "Fit ROI µm"]
        )
        self.geometry.setVerticalHeaderLabels(["Z", "Y", "X"])
        self.geometry.horizontalHeader().setStretchLastSection(True)
        self.geometry.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.geometry.setMaximumHeight(160)
        self.geometry.itemChanged.connect(self._geometry_changed)
        defaults = LocalizationConfig()
        for axis in range(3):
            for column, value in enumerate(
                (
                    0,
                    (41, 128, 128)[axis],
                    defaults.voxel_size_zyx_um[axis],
                    0,
                    defaults.fit_roi_size_um[axis],
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
        self.threshold = QDoubleSpinBox()
        self.threshold.setDecimals(5)
        self.threshold.setRange(0, 1e9)
        self.threshold.setValue(0.2)
        self.threshold.valueChanged.connect(self._invalidate_detection)
        form.addRow("Intensity threshold", self.threshold)
        self.weights = QLineEdit(defaults.ufish_weights)
        self.weights.textChanged.connect(self._invalidate_detection)
        form.addRow("Weights file / model name", self.weights)
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
            "Positions and widths: µm. Center bounds and initial values are offsets from each U-FISH prior, limited to its ROI. Blank values use upstream estimates / ROI limits. 2D fits one XY plane; cz and sz are fixed."
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

        self.status = QLabel("Select data, verify calibration, then preview U-FISH priors.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)
        layout.addWidget(self.cancel_button)
        self.progress.connect(self.status.setText)
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)
        self._refresh_layers()
        self._model_changed()
        self.indices["channel"].valueChanged.connect(self._channel_changed)
        self.destroyed.connect(self._cancelled.set)

    def _refresh_layers(self, *args: object) -> None:
        """Keep source selection stable when result layers are added."""
        if self._closing:
            return
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
        """Use reader axes/calibration and observe source edits to reject stale work."""
        if self._closing:
            return
        if self._layer is not None:
            for name in _SOURCE_EVENTS:
                getattr(self._layer.events, name).disconnect(self._source_changed)
        self._layer = self.layers.currentData()
        if self._layer is None:
            self._invalidate_data()
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
        self.indices["source_channel"].setValue(metadata.get("source_channel", 0))
        if layer.ndim == 2:
            self.model.setCurrentText("2d")
        for name in _SOURCE_EVENTS:
            getattr(layer.events, name).connect(self._source_changed)
        self._refresh_geometry()

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
            self.geometry.item(i, 1).setText(str(min(size, (41, 128, 128)[i])))
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
        self._invalidate_data()

    def _invalidate_data(self, *args: object) -> None:
        """Drop data-derived state while an in-flight result is marked stale."""
        self._preview = None
        self._invalidate_detection()

    def _geometry_changed(self, item: QTableWidgetItem) -> None:
        """Fit ROI changes retain existing priors; crop/calibration changes do not."""
        if item.column() != 4:
            self._invalidate_data()

    def _channel_changed(self, value: int) -> None:
        """An unsplit channel axis directly determines the selected batch channel."""
        if "c" in self.axes.text():
            self.indices["source_channel"].setValue(value)

    def _invalidate_detection(self, *args: object) -> None:
        """Threshold changes keep enhancement reusable but invalidate candidate points."""
        self._revision += 1
        self._detection = None

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
            blend_3d=self.blend.isChecked(),
            intensity_threshold=self.threshold.value(),
            ufish_weights=self.weights.text().strip(),
            inference_batch_size=self.inference_batch.value(),
            fit_parameters=parameters,
            fit_roi_size_um=tuple(float(self.geometry.item(i, 4).text()) for i in range(3)),
            voxel_size_zyx_um=spacing,
            origin_zyx_um=origin,
            estimator=self.estimator.currentText(),
            return_covariance=self.covariance.isChecked(),
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
        """Serialize GPU operations; Qt result handling always happens on the GUI thread."""
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
            return preview, result, config

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
        transforms = {
            name: getattr(layer, name)
            for name in ("scale", "translate", "rotate", "shear", "affine", "axis_labels", "units")
        }
        if self._job_kind == "U-FISH":
            preview, detection = cast(tuple[Preview, tuple[np.ndarray, np.ndarray]], value)
            self._preview, self._detection = preview, detection
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
            affine = layer.affine.affine_matrix.copy()
            # Compose the crop translation with the layer's public data/world map.
            crop_transform = Affine(translate=offset, ndim=layer.ndim)
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
                affine=Affine(affine_matrix=affine)
                .compose(source_transform)
                .compose(crop_transform),
                axis_labels=layer.axis_labels,
                units=layer.units,
                colormap="magenta",
                blending="additive",
                opacity=0.4,
                metadata={"localization_output": True},
            )
            self.status.setText(f"{len(points)} U-FISH priors. Tune threshold or preview GPU fits.")
        elif self._job_kind == "GPUfit":
            preview, result, config = cast(tuple[Preview, FitResults, LocalizationConfig], value)
            parameters = result["fit_params"]
            _, origin, source_config = preview
            points = (parameters[:, (3, 2, 1)] - origin) / source_config.voxel_size_zyx_um
            features = pd.DataFrame(parameters, columns=self.parameter_names)
            features["success"] = result["success"]
            features["iterations"] = result["niters"]
            features["objective"] = result["chi_sqrs"]
            features["active_bound"] = np.any(result["active_bounds"], axis=1)
            if "covariance_valid" in result:
                features["covariance_valid"] = result["covariance_valid"]
            self._replace_points(
                "Gaussian fits",
                layer_coordinates(points, source_config),
                transforms,
                features,
                config,
            )
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
    ) -> None:
        """Keep named overlays and their inspectable production fit features together."""
        if name in self.viewer.layers:
            self.viewer.layers.remove(name)
        kwargs = {"features": features} if len(features) else {}
        self.viewer.add_points(
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
        """Surface validation/worker failures without replacing GPU work with CPU work."""
        if self._closing:
            return
        text = str(error) or type(error).__name__
        self.status.setText(text)
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
        self.status.setText("Cancel requested; waiting for the current inference / fit batch.")

    def _apply_optics(self) -> None:
        try:
            config = self.current_config()
        except Exception as error:
            self._errored(error)
            return
        self._start(partial(optical_parameters, config), "Optical bounds")

    def _choose_tiff(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open OME-TIFF", filter="TIFF (*.tif *.tiff)")
        if path:
            self.path.setText(path)
            self._open()

    def _choose_zarr(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open OME-Zarr image group")
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
            path, _ = QFileDialog.getSaveFileName(
                self, "Save localization settings", "localization.json", "JSON (*.json)"
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
            self, "Load localization settings", filter="JSON (*.json)"
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
        self.axes.setText(config.preview.axes)
        self.path.setText(config.input_uri)
        self.blend.setChecked(config.blend_3d)
        self.threshold.setValue(config.intensity_threshold)
        self.weights.setText(config.ufish_weights)
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
                    config.fit_roi_size_um[i],
                )
            ):
                self.geometry.item(i, j).setText(str(value))
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
        self.batch_all.setChecked(config.batch_axes)
        self._invalidate_data()
        self.config_view.setPlainText(config.model_dump_json(indent=2))

    def _copy_batch_command(self) -> None:
        """Save current settings and copy a quoted terminal command without executing it."""
        try:
            if self._busy or self._closing:
                raise ValueError("Finish the current preview before preparing a batch command")
            config = self.current_config()
            if self._config_path is None or not self.output.text().strip() or not config.input_uri:
                raise ValueError("Save JSON and choose an input and output directory first")
            config.save(self._config_path)
            arguments = [
                sys.executable,
                "-m",
                "localize_psf.batch",
                config.input_uri
                if "://" in config.input_uri
                else str(Path(config.input_uri).resolve()),
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
        self._layer = self._preview = self._detection = None
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
