"""Closed-loop numerical cases and nominal command association; no real DAQ."""
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from rv_controller import RandomAmplitudeControl, RandomVibrationController


def run_plant(gain, duration=120, delay=1, block_seconds=1., max_volts=1.4):
    controller = RandomAmplitudeControl(1., .1, 2., .9, .06, .001, max_volts)
    commands = deque([(0., 0.)] * delay)
    measurements, levels, integral, voltages = [], [], [], []
    for index in range(int(duration / block_seconds)):
        commands.append((controller.drive_rms, controller.command_level))
        volts, level = commands.popleft()
        plant_gain = gain(index * block_seconds) if callable(gain) else gain
        measured = plant_gain * volts
        controller.observe(measured, volts, level, block_seconds, limited=volts >= max_volts)
        measurements.append(measured)
        levels.append(level)
        integral.append(controller.integral)
        voltages.append(volts)
    return controller, np.array(measurements), np.array(levels), np.array(integral), np.array(voltages)


@pytest.mark.parametrize('gain', [5., 60., 150.])
@pytest.mark.parametrize('block_seconds', [.1, 1.])
def test_unknown_plant_reaches_fixed_target_without_ramp_runaway(gain, block_seconds):
    controller, response, levels, integral, _ = run_plant(gain, block_seconds=block_seconds)
    assert np.max(response) < 1.15
    assert np.mean(response[-int(10 / block_seconds):]) == pytest.approx(1., rel=.03)
    assert np.max(levels) <= 1.
    assert controller.command_level == 1.
    assert np.max(np.abs(integral)) <= .25


def test_limited_actuator_does_not_wind_up_and_recovers_when_target_becomes_reachable():
    controller, response, levels, integral, voltages = run_plant(lambda t: .5 if t < 90 else 5., duration=160)
    assert max(voltages) <= 1.4
    assert max(abs(integral)) <= .25
    assert controller.command_level == 1.
    assert np.mean(response[-10:]) == pytest.approx(1., rel=.03)
    assert voltages[93] < voltages[90] / 2


def test_noise_without_resolved_pilot_never_causes_blind_drive_escalation():
    controller = RandomAmplitudeControl(1., .1, 2., .9, .06, .001, 1.4)
    controller.observe(.002, 0., 0., 1.)
    for _ in range(100):
        controller.observe(.002, controller.drive_rms, controller.command_level, 1.)
    assert controller.drive_rms == .001
    assert controller.gain is None
    assert not controller.feedback_ready
    assert controller.integral == 0


def test_zero_setpoint_and_zero_voltage_cap_remain_zero():
    controller = RandomAmplitudeControl(1., 0., 0., .9, .06, .001, 1.4)
    for _ in range(10):
        controller.observe(0., 0., 0., 1.)
    assert controller.drive_rms == controller.command_level == 0
    muted = RandomAmplitudeControl(1., .1, 2., .9, .06, .001, 0.)
    muted.observe(0., 0., 0., 1.)
    assert muted.drive_rms == 0


def test_zero_acceleration_is_not_replaced_with_synthetic_measurements():
    controller = RandomVibrationController([(100, .001), (500, .001)], 8000, welch_nperseg=512)
    _, psd, metrics = controller.estimate_psd(np.zeros(800))
    assert not np.any(psd)
    assert metrics[0] == metrics[2] == 0


def test_feedback_uses_older_queued_command_not_the_future_write(settings_data):
    import app
    from run_settings import RunSettings
    worker = app.ControllerWorker(RunSettings.capture(settings_data))
    first = app.QueuedDrive(np.zeros(800), 0.)
    next_command = app.QueuedDrive(np.ones(800) * .01, .1)
    worker._pending_commands.append(first)
    written = []
    writer = SimpleNamespace(write_many_sample=lambda data, **kwargs: written.append(data.copy()))
    reader = SimpleNamespace(read_many_sample=lambda data, **kwargs: data.fill(0))
    data, measured_command = worker._exchange_block(writer, reader, next_command, False, 800, 1.)
    assert written[0][0] == .01
    assert measured_command is first
    assert measured_command.level == measured_command.rms == 0
    assert worker._pending_commands[0] is next_command
