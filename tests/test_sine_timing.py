"""Verify waveform dwell and endpoint capture with and without an AO queue."""
from collections import Counter, deque
from types import SimpleNamespace
import time

import numpy as np
import pytest

import app
from run_settings import RunSettings


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
