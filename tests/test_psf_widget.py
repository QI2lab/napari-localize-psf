"""Optional PSF controls and Qt plotting using real upstream analysis and CPU fits."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from localize_psf.batch_fitting import fit_rois
from localize_psf.fit_psf import gaussian3d_psf_model
from localize_psf.localize import get_coords, prepare_rois
from localize_psf.rois import get_centered_rois
from localize_psf.workflow import analyze_psf
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from simulator import bead_image

from napari_localize_psf.widget import LocalizationWidget

pytestmark = pytest.mark.integration


def test_psf_tab_is_optional_and_requires_completed_3d_fits(viewer, qtbot):
    viewer.add_image(np.ones((5, 11, 13)), units="um")
    widget = LocalizationWidget(viewer)
    qtbot.addWidget(widget)
    assert widget.tabs.tabText(widget.tabs.count() - 1) == "PSF / field"
    assert not widget.psf_button.isEnabled()
    assert tuple(box.value() for box in widget.psf_roi.values()) == (11, 21, 21)
    widget.run_psf_analysis()
    assert "3D fit preview" in widget.status.text()
    widget.model.setCurrentText("2d")
    widget.psf_roi["x"].setValue(15)
    assert not widget.psf_button.isEnabled()
    assert widget._psf_result is None and not widget.psf_plots.count()
    assert widget._worker is None and not widget._busy


@pytest.mark.parametrize("count", [1, 5])
def test_psf_views_embed_real_upstream_analysis_without_changing_image_coordinates(
    viewer, qtbot, count
):
    image, parameters, spacing, origin = bead_image()
    source = viewer.add_image(image, scale=spacing, translate=origin, units="um")
    widget = LocalizationWidget(viewer)
    qtbot.addWidget(widget)
    # These are real CPU fits used only as input to the plotting component.
    # The GPU preview/backend is neither invoked nor replaced in this test.
    coords = get_coords(image.shape, spacing, origin=origin)
    pixels = (parameters[:count, (3, 2, 1)] - origin) / spacing
    rois = get_centered_rois(pixels, (9, 15, 15), min_vals=(0, 0, 0), max_vals=image.shape)
    values, coordinates, sizes = prepare_rois(image, coords, rois)
    fits = fit_rois(values, coordinates, sizes, parameters[:count], model=gaussian3d_psf_model())
    result = analyze_psf(image, fits, widget.current_config())
    figures_before = plt.get_fignums()
    widget._show_psf_analysis(result)
    assert widget._psf_result is result
    assert widget.psf_plots.count() == (2 if count == 1 else 4)
    canvas = widget.psf_plots.itemAt(1).widget()
    assert isinstance(canvas, FigureCanvasQTAgg)
    assert [axis.get_title() for axis in canvas.figure.axes] == ["XY", "XZ", "YZ"]
    assert canvas.figure.axes[0].get_xlabel() == "X (µm)"
    canvas.draw()
    assert plt.get_fignums() == figures_before
    assert len(viewer.layers) == 1 and viewer.layers[0] == source
    np.testing.assert_array_equal(source.scale, spacing)
    assert "Averaged" in widget.psf_status.text()
    if count == 1:
        assert "Field fit unavailable" in widget.psf_status.text()
    widget.psf_roi["x"].setValue(15)
    assert widget._psf_result is None and not widget.psf_plots.count()
