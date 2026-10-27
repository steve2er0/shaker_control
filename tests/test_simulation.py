"""Numerical and streaming regressions for block-based simulation kernels."""
import numpy as np
import pytest

import simulation
from simulation import ShakerPlantSimulator, MockAnalogSingleChannelReader


@pytest.mark.parametrize('register_length', [0, 1, 2, 5, 64])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_delay_keeps_legacy_sample_alignment_and_history_across_short_blocks(register_length, dtype):
    plant = ShakerPlantSimulator(51200, noise_level=0, delay_samples=register_length)
    signal = np.arange(1, 112, dtype=dtype)
    actual = []
    start = 0
    for count in [0, 1, 3, 0, 2, 5, 100]:
        block = signal[start:start+count]
        delayed = plant._delay_input(block)
        assert delayed.dtype == dtype
        actual.append(delayed)
        start += count
    delay = max(register_length - 1, 0)
    expected = np.concatenate((np.zeros(delay, dtype=dtype), signal))[:len(signal)]
    np.testing.assert_array_equal(np.concatenate(actual), expected)
    history = np.concatenate((np.zeros(max(register_length, 1)), signal))
    np.testing.assert_array_equal(plant.delay_buffer, history[-max(register_length, 1):][::-1])


def recurrence_reference(signal, b, a):
    """Independent scalar difference equation; deliberately no SciPy filtering."""
    past_inputs = [0.] * (len(b)-1)
    past_outputs = [0.] * (len(a)-1)
    result = []
    for value in signal:
        output = b[0]*value + sum(c*x for c,x in zip(b[1:],past_inputs)) - sum(c*y for c,y in zip(a[1:],past_outputs))
        result.append(output)
        past_inputs = [value] + past_inputs[:-1]
        past_outputs = [output] + past_outputs[:-1]
    return np.asarray(result)


@pytest.mark.parametrize('fs', [2048, 8000, 51200])
@pytest.mark.parametrize('channel', [1, 2, 3])
def test_sdof_stream_matches_original_difference_equation(fs, channel):
    reader = MockAnalogSingleChannelReader(ShakerPlantSimulator(fs, noise_level=0), 4)
    state = reader._sdof_filters[channel]
    rng = np.random.default_rng(143)
    signal = np.r_[1., np.zeros(200), np.full(200, .2), rng.normal(0, .3, 4096)]
    expected = recurrence_reference(signal, state['b'], state['a']) * state['gain']
    blocks = [signal[:1], signal[1:5], signal[5:513], signal[513:513], signal[513:]]
    actual = np.concatenate([reader._apply_sdof(block, state) for block in blocks])
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)


def test_default_rate_four_channel_response_is_independent_of_block_partition():
    signal = np.random.default_rng(83).normal(0, .2, 102400)

    def acquire(partitions):
        writer, reader, _, _ = simulation.create_simulation_system(
            51200, 4., [(150., 15., 3.), (800., 25., 2.5)], 0., 5, .05, 4)
        responses = []
        start = 0
        for count in partitions:
            writer.write_many_sample(signal[start:start+count])
            data = np.empty((4, count))
            reader.read_many_sample(data, count)
            responses.append(data)
            start += count
        assert start == len(signal)
        return np.concatenate(responses, axis=1)

    full = acquire([102400])
    # Keep the first block longer than the delay: the existing zero-response
    # noise guard is intentionally block-dependent, even at noise_level=0.
    fragmented = acquire([17, 1, 2, 5, 684, 50491, 51200])
    assert full.shape == (4, 102400)
    assert np.all(np.isfinite(full))
    np.testing.assert_allclose(fragmented, full, rtol=1e-10, atol=1e-11)


def test_sdof_preserves_single_precision_output():
    reader = MockAnalogSingleChannelReader(ShakerPlantSimulator(51200, noise_level=0), 2)
    state = reader._sdof_filters[1]
    signal = np.random.default_rng(13).normal(0, .2, 1024).astype(np.float32)
    expected = (recurrence_reference(signal, state['b'], state['a']) * state['gain']).astype(np.float32)
    actual = reader._apply_sdof(signal, state)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=1e-10)


@pytest.mark.parametrize('mode', ['random', 'sine_sweep'])
def test_processing_time_is_subtracted_from_simulation_block_wait(settings_data, monkeypatch, mode):
    """A fake processing clock verifies pacing without a flaky speed assertion."""
    import app
    from run_settings import RunSettings
    settings_data.update(test_mode=mode, block_seconds=1)
    worker = app.ControllerWorker(RunSettings.capture(settings_data))
    clock = [0.]
    waits = []

    class StopAfterTwoBlocks:
        def is_set(self):
            return len(waits) >= 2

        def wait(self, seconds):
            waits.append(seconds)
            clock[0] += seconds
            return self.is_set()

    original_read = simulation.MockAnalogSingleChannelReader.read_many_sample

    def read_with_processing_cost(reader, *args, **kwargs):
        original_read(reader, *args, **kwargs)
        clock[0] += .4

    worker.should_stop = StopAfterTwoBlocks()
    monkeypatch.setattr(app.time, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr(simulation.MockAnalogSingleChannelReader, 'read_many_sample', read_with_processing_cost)
    worker.run_controller()
    assert worker.terminal_state == 'stopped'
    assert worker.blocks == 2
    assert waits == pytest.approx([.6, .6])
