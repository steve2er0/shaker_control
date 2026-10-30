"""Target edits retain sine stop margins and the response plot keeps its scale."""
import json
import math

import numpy as np
import pytest

from control_features import ResponseGuard, ResponseTrip


def prepare_sine(window):
    window.mode_combo.setCurrentIndex(1)
    panel = window.config_tab
    panel.sine_unit_combo.setCurrentIndex(0)
    panel.fields['sine_g_level'].setValue(.5)
    panel.fields['response_max_peak_g'].setValue(.85)
    panel.fields['response_max_rms_g'].setValue(.55)
    panel.fields['response_target_ratio'].setValue(1.5)
    panel.fields['response_limits_enabled'].setChecked(True)
    assert panel.apply_config()
    return panel


def test_target_edit_scales_stops_and_guard_uses_applied_limits(window):
    panel = prepare_sine(window)
    panel.fields['sine_g_level'].setValue(1)
    assert panel.fields['response_max_peak_g'].value() == pytest.approx(1.7)
    assert panel.fields['response_max_rms_g'].value() == pytest.approx(1.1)
    assert window.settings.response_max_peak_g == .85  # Still a draft.
    assert panel.apply_config()
    guard = ResponseGuard(window.settings)
    sample = np.zeros((4, 800))
    sample[0, 400] = 1.6
    guard.check(sample, 0, 0)
    sample[0, 400] = 1.71
    with pytest.raises(ResponseTrip, match='peak limit'):
        guard.check(sample, 0, 0)
    panel.fields['sine_g_level'].setValue(.25)
    assert panel.fields['response_max_peak_g'].value() == pytest.approx(.425)
    assert panel.fields['response_max_rms_g'].value() == pytest.approx(.275)


def test_sine_stop_scaling_converts_peak_and_rms_units_without_drift(window):
    panel = prepare_sine(window)
    panel.sine_unit_combo.setCurrentIndex(1)
    assert panel.fields['response_max_peak_g'].value() == pytest.approx(.85*math.sqrt(2))
    assert panel.fields['response_max_rms_g'].value() == pytest.approx(.55*math.sqrt(2))
    assert panel.apply_config()
    assert window.settings.target_rms_g == .5
    panel.sine_unit_combo.setCurrentIndex(0)
    assert panel.fields['response_max_peak_g'].value() == pytest.approx(.85)
    assert panel.fields['response_max_rms_g'].value() == pytest.approx(.55)


def test_custom_margins_persist_and_fixed_stops_can_reject_higher_target(window):
    panel = prepare_sine(window)
    panel.fields['response_max_peak_g'].setValue(1)
    panel.fields['sine_g_level'].setValue(1)
    assert panel.fields['response_max_peak_g'].value() == pytest.approx(2)
    assert panel.apply_config()
    saved = json.loads(json.dumps(window.shared_config.as_dict()))
    panel.fields['sine_g_level'].setValue(.5)
    assert panel.apply_config()
    window.shared_config.apply_values(saved)
    panel.load_values()
    assert panel.fields['sine_g_level'].value() == 1
    assert panel.fields['response_max_peak_g'].value() == 2
    panel.fields['sine_g_level'].setValue(.5)
    assert panel.fields['response_max_peak_g'].value() == 1
    panel.fields['sine_scale_response_stops'].setChecked(False)
    panel.fields['sine_g_level'].setValue(2)
    assert panel.fields['response_max_peak_g'].value() == 1
    assert not panel.apply_config()
    assert 'above the measured' in panel.validation_label.text()


@pytest.mark.parametrize('target,tolerance', [(.5, 5), (1, 5), (1, 7)])
def test_sine_bounds_and_fixed_y_range_ignore_startup_and_large_response(window, qt_app, target, tolerance):
    panel = prepare_sine(window)
    panel.fields['sine_g_level'].setValue(target)
    panel.fields['tracking_tolerance_percent'].setValue(tolerance)
    assert panel.apply_config()
    view = window.controller_tab
    fraction = tolerance/100
    np.testing.assert_allclose(view.sine_lower_curve.yData, target*(1-fraction))
    np.testing.assert_allclose(view.sine_upper_curve.yData, target*(1+fraction))
    expected = (target*(1-2*fraction), target*(1+2*fraction))
    np.testing.assert_allclose(view.sine_plot.viewRange()[1], expected)
    assert view.sine_plot.getViewBox().state['autoRange'][1] is False
    for index, measured in enumerate((0, target, target*3, target)):
        view.update_metrics({'accel_rms_g': measured/math.sqrt(2), 'accel_peak_g': measured,
            'tone_peak_g': measured, 'level_fraction': 1, 'output_rms_v': .01,
            'output_cf': math.sqrt(2), 'frequency_hz': 100+index, 'target_peak_g': target,
            'sweep_progress': index/4, 'elapsed_s': index, 'plant_gain_g_per_v': 4})
        qt_app.processEvents()
        np.testing.assert_allclose(view.sine_plot.viewRange()[1], expected)
        if measured > expected[1]:
            assert 'ABOVE fixed plot range' in view.detail_label.text()
    # A user auto-range request also cannot move the fixed vertical range.
    view.sine_plot.autoRange()
    qt_app.processEvents()
    np.testing.assert_allclose(view.sine_plot.viewRange()[1], expected)
    assert f'±{tolerance:g}%' in view.sine_range_label.text()
