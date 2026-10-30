"""Verify waveform dwell and endpoint capture with and without an AO queue."""
from collections import Counter, deque
from types import SimpleNamespace
import time

import numpy as np
import pytest

import app
from run_settings import RunSettings
from sine_sweep import SineAmplitudeRamp, SineOscillator


@pytest.mark.parametrize('initial,corrected', [(.0896077, .0807174), (.0807174, .0896077)])
def test_sine_correction_at_nonzero_phase_has_no_voltage_step(initial, corrected):
    # Reproduce the 20 -> 21.189 Hz run's nonzero-phase amplitude correction.
    fs, frequency = 51200, 21.189261887185907
    oscillator = SineOscillator(fs, frequency)
    oscillator.phase = .8
    ramp = SineAmplitudeRamp(fs)
    ramp.previous_peak = initial
    unit, _ = oscillator.generate(fs)
    waveform = ramp.apply(unit, corrected)
    prior_sample = initial * np.sin(.8 - 2*np.pi*frequency/fs)
    natural_step_bound = max(initial, corrected) * 2*np.pi*frequency/fs
    assert abs(waveform[0] - prior_sample) <= natural_step_bound
    # The amplitude change is distributed across the envelope rather than
    # creating the recorded correction's ~6 mV boundary discontinuity.
    assert np.max(np.abs(np.diff(waveform))) < 2 * natural_step_bound
    np.testing.assert_allclose(waveform[ramp.ramp_samples:], corrected * unit[ramp.ramp_samples:])
    assert np.max(np.abs(waveform)) <= max(initial, corrected)


def test_sine_startup_and_short_blocks_preserve_amplitude_envelope():
    ramp = SineAmplitudeRamp(8000)
    first = ramp.apply(np.ones(80), .1)
    assert first[0] == 0 and first[-1] == pytest.approx(.1)
    assert np.all(np.diff(first) >= 0)
    second = ramp.apply(np.ones(80), .05)
    assert second[0] == pytest.approx(.1) and second[-1] == pytest.approx(.05)
    assert np.all(np.diff(second) <= 0)


def test_low_frequency_feedback_corrections_are_smooth_with_queued_measurements(
        qt_app, settings_data, monkeypatch):
    end = 20 * 2**(1/12)
    settings_data.update(test_mode='sine_sweep', fs=51200, block_seconds=1,
        input_channel_labels=['Control Accel'], response_channels=[],
        sine_start_hz=20, sine_end_hz=end, sine_repeat=False,
        sine_closed_loop=True, sine_g_level=.5, sine_g_level_is_rms=False,
        sine_initial_level=.2, sine_max_level_rate=.2, sine_initial_settle=0,
        sine_step_dwell=.5, sine_octaves_per_min=2.5, sine_points_per_octave=12,
        sine_settle_blocks=3, sine_max_step_seconds=60, ao_volt_limit=1.8,
        response_limits_enabled=True, response_max_peak_g=.85, response_max_rms_g=.55,
        sine_drive_table=[], sine_default_vpk=.02)
    settings = RunSettings.capture(settings_data)
    worker = app.ControllerWorker(settings)
    worker.monotonic_start = time.monotonic()
    transport = deque([app.QueuedDrive(np.zeros(51200), 0)])
    worker._pending_commands.append(transport[0])
    commands, packets = [], []

    def exchange(writer, reader, command, *args):
        commands.append(command)
        transport.append(command)
        applied = transport.popleft()
        gain = 5.58 if applied.frequency == 20 else 6.2
        return applied.waveform * gain, applied

    monkeypatch.setattr(worker, '_exchange_block', exchange)
    monkeypatch.setattr(worker, '_pace_block', lambda *args: None)
    worker.metrics_data_ready.connect(packets.append)
    worker._run_sine_sweep_loop(None, None, True, None, 5.58, 51200, 51200)
    assert worker.terminal_state == 'complete'
    assert packets[-1]['frequency_hz'] == end
    assert packets[-1]['tone_peak_g'] == pytest.approx(.5, rel=.05)
    # The two-second minimum plus identification/settling replaces a 90 s hold.
    assert max(Counter(p['frequency_hz'] for p in packets).values()) < 40
    assert all(p['step_minimum_remaining_s'] <= 1 for p in packets)
    for previous, current in zip(commands, commands[1:]):
        if not current.frequency:  # The final hardware queue drain is silence.
            continue
        amplitude = max(np.max(np.abs(previous.waveform)), np.max(np.abs(current.waveform)))
        slope_bound = amplitude * 2*np.pi*end/settings.fs
        assert abs(current.waveform[0] - previous.waveform[-1]) < 2*slope_bound


@pytest.mark.parametrize('primed', [False, True])
@pytest.mark.parametrize('start,end', [(100, 200), (200, 100), (100, 100)])
@pytest.mark.parametrize('dwell,first_count,next_count', [(.2, 3, 2), (.25, 4, 3)])
def test_sine_dwell_counts_output_blocks_and_captures_final_endpoint(
        qt_app, settings_data, monkeypatch, primed, start, end, dwell, first_count, next_count):
    settings_data.update(test_mode='sine_sweep', sine_start_hz=start, sine_end_hz=end,
        sine_repeat=False, sine_points_per_octave=1, sine_octaves_per_min=300,
        sine_step_dwell=dwell, sine_initial_settle=.1, sine_initial_level=1,
        sine_drive_table=[], sine_default_vpk=.1, sine_g_level=.5, response_channels=[])
    worker = app.ControllerWorker(RunSettings.capture(settings_data))
    worker.monotonic_start = time.monotonic()
    block_samples = 800
    transport = deque()
    if primed:
        silence = app.QueuedDrive(np.zeros(block_samples), 0.)
        worker._pending_commands.append(silence)
        transport.append(silence.waveform)
    writes, recorded, packets = [], [], []

    def write(waveform, **kwargs):
        writes.append(waveform.copy())
        transport.append(waveform.copy())

    def read(data, **kwargs):
        data[:] = transport.popleft() * 4

    def record(ao, ai, level, saturation, frequency, measured, target):
        # The waveform, frequency label, and response must stay associated.
        np.testing.assert_allclose(ai, np.broadcast_to(ao * 4, ai.shape))
        if frequency:
            spectrum = np.abs(np.fft.rfft(ao))
            measured_frequency = np.fft.rfftfreq(len(ao), 1 / 8000)[np.argmax(spectrum)]
            assert measured_frequency == frequency
        recorded.append(frequency)

    worker.metrics_data_ready.connect(packets.append)
    monkeypatch.setattr(worker, '_pace_block', lambda *args: None)
    worker._run_sine_sweep_loop(SimpleNamespace(write_many_sample=write),
        SimpleNamespace(read_many_sample=read), not primed, None, 4., 8000, block_samples,
        SimpleNamespace(log_sine_block=record))

    counts = Counter(packet['frequency_hz'] for packet in packets)
    expected = {start: first_count} if start == end else {start: first_count, end: next_count}
    if primed:
        expected[0.] = 1
    assert counts == expected
    assert recorded == [packet['frequency_hz'] for packet in packets]
    assert packets[-1]['frequency_hz'] == end
    assert worker.terminal_state == 'complete'
    assert np.any(writes[-1]) != primed
    # Hardware queues finish with silence; every nonzero output was measured.
    assert len(transport) == int(primed)
    if primed:
        assert not np.any(transport[0])


def test_stop_on_final_sine_block_remains_stopped(qt_app, settings_data, monkeypatch):
    settings_data.update(test_mode='sine_sweep', sine_start_hz=100, sine_end_hz=100,
        sine_repeat=False, sine_initial_settle=0, sine_step_dwell=.1,
        sine_octaves_per_min=600, sine_points_per_octave=1)
    worker = app.ControllerWorker(RunSettings.capture(settings_data))
    worker.monotonic_start = time.monotonic()
    worker.metrics_data_ready.connect(lambda packet: worker.stop())
    monkeypatch.setattr(worker, '_pace_block', lambda *args: None)
    worker._run_sine_sweep_loop(SimpleNamespace(write_many_sample=lambda *args: None),
        SimpleNamespace(read_many_sample=lambda data, **kwargs: data.fill(0)),
        True, None, 4., 8000, 800)
    assert worker.terminal_state == 'stopped'
