"""Integration of real predictor controls, dimension changes, and shared JSON settings."""

import numpy as np
import pytest

from napari_localize_psf.config import LocalizationConfig
from napari_localize_psf.widget import LocalizationWidget

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("mode", ["2d", "3d"])
def test_predictor_controls_switch_models_and_roundtrip_json(viewer, qtbot, tmp_path, mode):
    viewer.add_image(np.ones((9, 31, 33)), units="um")
    widget = LocalizationWidget(viewer)
    qtbot.addWidget(widget)
    assert "Predict" in [widget.tabs.tabText(i) for i in range(widget.tabs.count())]
    widget.model.setCurrentText(mode)
    widget.weights.setCurrentText("custom-ufish.pth")
    widget.threshold.setValue(2.5)
    widget.predictor.setCurrentText("Spotiflow")
    assert widget.ufish_controls.isHidden()
    assert not widget.spotiflow_controls.isHidden()
    assert widget.detect_button.text() == "Preview Spotiflow priors"
    assert widget.spotiflow_model.findText("smfish_3d" if mode == "3d" else "general") >= 0
    assert widget.spotiflow_model.findText("general" if mode == "3d" else "smfish_3d") == -1
    assert not widget.spotiflow_probability.isEnabled()
    widget.spotiflow_auto_threshold.setChecked(False)
    assert widget.spotiflow_probability.isEnabled()
    widget.spotiflow_probability.setValue(0.35)
    widget.spotiflow_distance.setValue(3)
    widget.spotiflow_subpixel.setChecked(False)
    widget.spotiflow_tile_size.setValue(128)
    config = widget.current_config()
    assert config.predictor == "spotiflow"
    assert config.spotiflow.prob_thresh == 0.35 and config.spotiflow.min_distance == 3
    assert not config.spotiflow.subpix and config.spotiflow.max_tile_size == 128
    path = tmp_path / "prediction.json"
    config.save(path)
    restored = LocalizationConfig.load(path)
    widget.predictor.setCurrentText("U-FISH")
    assert not widget.ufish_controls.isHidden() and widget.spotiflow_controls.isHidden()
    assert widget.weights.currentText() == "custom-ufish.pth" and widget.threshold.value() == 2.5
    widget.apply_config(restored)
    assert widget.current_config() == config
    widget.model.setCurrentText("2d" if mode == "3d" else "3d")
    assert widget.spotiflow_model.findText("general" if mode == "3d" else "smfish_3d") >= 0
    assert widget._worker is None
