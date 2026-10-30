import math
from types import SimpleNamespace

import numpy as np
import pytest

from control_features import ResponseGuard, ResponseTrip, SineFeedback, measure_tone
from run_settings import RunSettings, ConfigurationError


def captured(settings_data, **changes):
    settings_data.update(changes)
    return RunSettings.capture(settings_data)


@pytest.mark.parametrize('mode', ['random', 'sine_sweep'])
def test_limits_cover_response_channels_and_negative_peaks(settings_data, mode):
    s = captured(settings_data, test_mode=mode, response_limits_enabled=True,
                 target_psd_points=[(100, .00001), (500, .00001)], sine_g_level=.2,
                 response_max_peak_g=1, response_max_rms_g=.5, response_target_ratio=100)
    data = np.zeros((4, 800))
    data[2, 400] = -1.1
    with pytest.raises(ResponseTrip, match='peak limit') as error:
        ResponseGuard(s).check(data, 0, 0)
    assert error.value.details['channel_raw_peak_g'][2] == 1.1
    data[2] = .8 * np.sin(2 * np.pi * np.arange(800) / 40)
    with pytest.raises(ResponseTrip, match='RMS limit'):
        ResponseGuard(s).check(data, 0, 0)


def test_feedback_loss_is_timed_and_resets_without_drive(settings_data):
    s = captured(settings_data, response_limits_enabled=True,
                 feedback_startup_seconds=0, feedback_loss_seconds=.3)
    guard = ResponseGuard(s)
    data = np.zeros((4, 800))
    for _ in range(20):
        guard.check(data, 0, 0)
    guard.check(data, .1, .5)
    guard.check(data, 0, 0)
    guard.check(data, .1, .5)
    guard.check(data, .1, .5)
    with pytest.raises(ResponseTrip, match='feedback was lost'):
        guard.check(data, .1, .5)


def test_stale_data_and_invalid_input_trip(settings_data):
    s = captured(settings_data, response_limits_enabled=True)
    with pytest.raises(ResponseTrip, match='data-age'):
        ResponseGuard(s).check(np.zeros(800), 0, 0, backlog_seconds=3)
    with pytest.raises(ResponseTrip, match='NaN'):
        ResponseGuard(s).check(np.full(800, np.nan), 0, 0)


def test_tone_fit_separates_phase_dc_and_harmonics():
    fs, f = 8000, 137.25
    t = np.arange(8000) / fs
    values = .5 * np.sin(2*np.pi*f*t + .73) + .08*np.sin(4*np.pi*f*t) + 2
    peak, residual = measure_tone(values, f, fs)
    assert peak == pytest.approx(.5, abs=.0002)
    assert residual == pytest.approx(.08 / math.sqrt(2), rel=.002)


@pytest.mark.parametrize('gain', [4., 40.])
def test_sine_feedback_converges_with_wrong_table_and_bounded_growth(settings_data, gain):
    s = captured(settings_data, sine_closed_loop=True, sim_noise_level=0,
                 sine_settle_seconds=.1, sine_settle_blocks=2)
    control = SineFeedback(s)
    previous = 0
    for _ in range(400):
        peak_v = control.command(100, .4, .5)
        if previous:
            assert peak_v <= previous * (1 + s.sine_correction_rate*s.block_seconds) + 1e-12
        settled = control.observe(100, peak_v*gain, 0, peak_v, 1, False, .5)
        previous = peak_v
        if settled:
            break
    assert settled
    assert peak_v * gain == pytest.approx(.5, rel=.05)
    # A changed gain estimate decreases drive immediately, without an integral.
    control.observe(100, peak_v*gain*2, 0, peak_v, 1, False, .5)
    assert control.command(100, .4, .5) < previous


def test_unresolved_or_clipped_sine_feedback_does_not_increase_drive(settings_data):
    s = captured(settings_data, sim_noise_level=0, sine_closed_loop=True)
    control = SineFeedback(s)
    values = []
    for _ in range(20):
        command = control.command(100, .4, .5)
        values.append(command)
        assert not control.observe(100, .00001, .02, command, 1, False, .5)
    assert max(values) <= s.sine_pilot_vpk
    assert not control.observe(100, .5, 0, .1, 1, True, .5)
    assert not control.gains


@pytest.mark.parametrize('changes', [
    {'response_max_peak_g': 'bad'}, {'feedback_loss_seconds': 0},
    {'sine_settle_blocks': 1.2}, {'tracking_tolerance_percent': float('nan')},
    {'test_mode': 'sine_sweep', 'sine_closed_loop': True, 'sine_start_hz': 1},
    {'calibration_run': True},
])
def test_invalid_new_settings_fail_before_acquisition(settings_data, changes):
    with pytest.raises(ConfigurationError):
        captured(settings_data, **changes)
