"""Integration of real dialog actions, JSON IO, notifications, and command export."""

import json
import shlex
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QApplication, QDialogButtonBox, QFileDialog, QWidget

from napari_localize_psf.config import LocalizationConfig
from napari_localize_psf.widget import LocalizationWidget

pytestmark = pytest.mark.integration


@pytest.fixture
def select_real_file():
    """Drive real Qt file dialogs with public buttons; bound modal waits to five seconds."""
    old_attribute = QApplication.testAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs, True)
    deadline = QTimer()
    deadline.setSingleShot(True)

    def dismiss() -> None:
        dialog = QApplication.activeModalWidget()
        if isinstance(dialog, QFileDialog):
            dialog.close()

    deadline.timeout.connect(dismiss)

    def select(path: Path | None, parent: QWidget) -> None:
        def accept_selection() -> None:
            dialog = QApplication.activeModalWidget()
            assert isinstance(dialog, QFileDialog)
            assert dialog.parentWidget() is parent
            buttons = dialog.findChild(QDialogButtonBox)
            if path is None:
                buttons.button(QDialogButtonBox.StandardButton.Cancel).click()
                return
            dialog.setDirectory(str(path.parent))
            dialog.selectFile(path.name)
            standard = (
                QDialogButtonBox.StandardButton.Save
                if dialog.acceptMode() == QFileDialog.AcceptMode.AcceptSave
                else QDialogButtonBox.StandardButton.Open
            )
            buttons.button(standard).click()

        deadline.start(5000)
        QTimer.singleShot(0, accept_selection)

    try:
        yield select
    finally:
        deadline.stop()
        deadline.deleteLater()
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs, old_attribute)


@pytest.mark.parametrize(
    "weights",
    [
        "v1.0-alldata-ufish_c32.pth",
        "finetune_models/v1.0.1-MERFISH_model.pth",
        "custom trained weights.pth",
    ],
)
def test_real_json_dialogs_are_parented_and_roundtrip(
    viewer, qtbot, tmp_path, select_real_file, weights
):
    viewer.add_image(np.ones((5, 11, 13)), units="um")
    widget = LocalizationWidget(viewer)
    qtbot.addWidget(widget)
    path = tmp_path / "settings.json"
    assert widget.weights.isEditable() and widget.weights.count() == 11
    assert widget.weights.currentText() == LocalizationConfig().ufish_weights
    widget.weights.setCurrentText(weights)
    widget.exclude_edges.setChecked(True)
    widget.edge_margin.setValue(3)
    widget.collapse_priors.setChecked(False)
    widget.amplitude_min.setText("150")
    widget.amplitude_max.setText("5000")
    widget.fit_covariance.setCurrentText("Pass")
    widget.fit_convergence.setCurrentText("Fail")
    widget._show_json()
    assert json.loads(widget.config_view.toPlainText())["fit_model"] == "3d"
    widget._copy_json()
    assert json.loads(QApplication.clipboard().text())["fit_model"] == "3d"
    select_real_file(path, widget)
    widget._save_json()
    assert LocalizationConfig.load(path).intensity_threshold == 0.2
    assert LocalizationConfig.load(path).ufish_weights == weights
    assert LocalizationConfig.load(path).exclude_edge_centroids
    assert LocalizationConfig.load(path).edge_margin_px == 3
    assert not LocalizationConfig.load(path).collapse_close_priors
    assert LocalizationConfig.load(path).fit_filters == widget.current_config().fit_filters
    assert LocalizationConfig.load(path).return_covariance
    widget.threshold.setValue(0.7)
    widget.weights.setCurrentText(LocalizationConfig().ufish_weights)
    widget.exclude_edges.setChecked(False)
    widget.edge_margin.setValue(1)
    widget.collapse_priors.setChecked(True)
    widget._reset_fit_filters()
    select_real_file(path, widget)
    widget._load_json()
    assert widget.current_config().intensity_threshold == 0.2
    assert widget.weights.currentText() == weights
    assert widget.weights.count() == 11
    assert widget.exclude_edges.isChecked()
    assert widget.edge_margin.value() == 3
    assert not widget.collapse_priors.isChecked()
    assert widget.amplitude_min.text() == "150.0"
    assert widget.amplitude_max.text() == "5000.0"
    assert widget.fit_covariance.currentText() == "Pass"
    assert widget.fit_convergence.currentText() == "Fail"
    assert widget._config_path == path


def test_fit_filter_controls_validate_without_starting_gpu_work(viewer, qtbot):
    viewer.add_image(np.ones((5, 11, 13)), units="um")
    widget = LocalizationWidget(viewer)
    qtbot.addWidget(widget)
    assert "Fit filters" in [widget.tabs.tabText(i) for i in range(widget.tabs.count())]
    assert widget.collapse_priors.isChecked()
    widget.amplitude_min.setText("200")
    widget.amplitude_max.setText("100")
    widget.amplitude_max.editingFinished.emit()
    assert "Minimum amplitude" in widget.fit_filter_status.text()
    with pytest.raises(ValueError, match="Minimum amplitude"):
        widget.current_config()
    widget.amplitude_max.setText("invalid")
    widget.amplitude_max.editingFinished.emit()
    assert "invalid" in widget.fit_filter_status.text()
    widget._reset_fit_filters()
    widget.fit_covariance.setCurrentText("Fail")
    assert widget.covariance.isChecked()
    assert widget.current_config().fit_filters.covariance == "fail"
    widget.covariance.setChecked(False)
    with pytest.raises(ValueError, match="return_covariance"):
        widget.current_config()
    widget._reset_fit_filters()
    assert widget.current_config().fit_filters.amplitude_min is None
    assert widget.current_config().fit_filters.amplitude_max is None
    assert widget.current_config().fit_filters.convergence == "all"
    assert widget.current_config().fit_filters.covariance == "all"
    assert widget._worker is None and not widget._busy


def test_real_tiff_dialog_reads_through_upstream_reader(viewer, qtbot, tmp_path, select_real_file):
    path = tmp_path / "image.ome.tif"
    tifffile.imwrite(
        path,
        np.ones((5, 11, 13), np.uint16),
        ome=True,
        photometric="minisblack",
        metadata={"axes": "ZYX"},
    )
    widget = LocalizationWidget(viewer)
    qtbot.addWidget(widget)
    select_real_file(path, widget)
    widget._choose_tiff()
    assert len(viewer.layers) == 1
    assert viewer.layers[0].source.reader_plugin == "napari-tiff"
    assert Path(widget.path.text()) == path


def test_invalid_actions_show_errors_without_starting_workers(viewer, qtbot):
    widget = LocalizationWidget(viewer)
    qtbot.addWidget(widget)
    widget.run_detection()
    assert "Select an image" in widget.status.text()
    widget._apply_optics()
    assert "Select an image" in widget.status.text()
    viewer.add_image(np.ones((5, 11, 13)), units="um")
    widget.run_fitting()
    assert "Preview U-FISH priors" in widget.status.text()
    widget._copy_batch_command()
    assert "Save JSON" in widget.status.text()
    widget.parameters.item(4, 0).setText("-1")
    widget._show_json()
    assert "physical lower bound" in widget.status.text()
    assert not widget._busy


def test_copy_command_quotes_paths_and_saves_current_settings(
    viewer, qtbot, tmp_path, select_real_file
):
    image_path = tmp_path / "sample's counts & spots.ome.tif"
    tifffile.imwrite(
        image_path,
        np.ones((5, 11, 13), np.uint16),
        ome=True,
        photometric="minisblack",
        metadata={"axes": "ZYX"},
    )
    widget = LocalizationWidget(viewer)
    qtbot.addWidget(widget)
    widget.path.setText(str(image_path))
    widget._open()
    widget.manual.setChecked(True)
    config_path = tmp_path / "settings with spaces.json"
    select_real_file(config_path, widget)
    widget._save_json()
    widget.threshold.setValue(0.65)
    widget.weights.setCurrentIndex(widget.weights.findText("finetune_models/v1.0.1-CISH_model.pth"))
    output = tmp_path / "fit's results & diagnostics"
    widget.output.setText(str(output))
    qtbot.mouseClick(widget.batch_button, Qt.MouseButton.LeftButton)
    command = widget.command_view.toPlainText()
    assert command == QApplication.clipboard().text()
    if sys.platform == "win32":
        assert command.startswith("& '")
        assert "sample''s counts & spots.ome.tif" in command
        assert "fit''s results & diagnostics" in command
    else:
        assert shlex.split(command) == [
            sys.executable,
            "-m",
            "localize_psf.batch",
            str(image_path),
            "--metadata",
            str(config_path),
            "--output-dir",
            str(output),
        ]
    assert LocalizationConfig.load(config_path).intensity_threshold == 0.65
    assert (
        LocalizationConfig.load(config_path).ufish_weights
        == "finetune_models/v1.0.1-CISH_model.pth"
    )
    assert set(tmp_path.iterdir()) == {image_path, config_path}
    assert not output.exists()
    assert not widget._busy
    assert widget.tabs.isEnabled()
    assert not widget.cancel_button.isEnabled()


@pytest.mark.parametrize("action", ["_choose_tiff", "_choose_zarr", "_save_json", "_load_json"])
def test_cancelled_dialog_keeps_current_settings(viewer, qtbot, select_real_file, action):
    viewer.add_image(np.ones((5, 11, 13)), units="um")
    widget = LocalizationWidget(viewer)
    qtbot.addWidget(widget)
    before = widget.current_config()
    select_real_file(None, widget)
    getattr(widget, action)()
    assert widget.current_config() == before
    assert widget._config_path is None
    assert len(viewer.layers) == 1


def test_invalid_json_load_reports_error_and_retains_controls(
    viewer, qtbot, tmp_path, select_real_file
):
    viewer.add_image(np.ones((5, 11, 13)), units="um")
    widget = LocalizationWidget(viewer)
    qtbot.addWidget(widget)
    before = widget.current_config()
    path = tmp_path / "invalid.json"
    path.write_text("{", encoding="utf-8")
    select_real_file(path, widget)
    widget._load_json()
    assert "Invalid JSON" in widget.status.text()
    assert widget.current_config() == before
    assert widget._config_path is None


def test_invalid_axis_labels_prevent_preview_and_export(viewer, qtbot):
    viewer.add_image(np.ones((5, 11, 13)), units="um")
    widget = LocalizationWidget(viewer)
    qtbot.addWidget(widget)
    widget.axes.setText("zzx")
    widget.axes.editingFinished.emit()
    assert "unique axis label" in widget.calibration_status.text()
    widget.run_detection()
    widget._copy_json()
    assert "manual calibration" in widget.status.text()
    assert not widget._busy
