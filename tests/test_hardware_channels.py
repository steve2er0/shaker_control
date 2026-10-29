"""Multichannel NI contracts using synthetic readers; never drive real hardware."""
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import app
import workspace_ui
from conftest import wait_until
from run_settings import ConfigurationError, RunSettings, parse_response_channels, format_response_channels


@pytest.mark.parametrize('mode', ['random', 'sine_sweep'])
@pytest.mark.parametrize('fallback', [False, True])
def test_four_hardware_inputs_reach_plots_recording_and_control(window, monkeypatch, tmp_path, mode, fallback):
    tasks, reads, hardware_waits = [], [], []
    worker_init = app.ControllerWorker.__init__
    logger_init, logger_close = app.DataLogger.__init__, app.DataLogger.close

    def track_worker(self, *args, **kwargs):
        worker_init(self, *args, **kwargs)
        original_wait = self.should_stop.wait
        def wait(timeout):
            hardware_waits.append(timeout)
            return original_wait(timeout)
        self.should_stop.wait = wait

    def initialize_recording(self, *args, **kwargs):
        assert tasks and not any(t.started for t in tasks), 'Recording must initialize before hardware starts'
        logger_init(self, *args, **kwargs)

    def finish_recording(self):
        assert not any(t.started for t in tasks), 'Hardware must stop before slow recording finalization'
        logger_close(self)

    monkeypatch.setattr(app.ControllerWorker, '__init__', track_worker)
    monkeypatch.setattr(app.DataLogger, '__init__', initialize_recording)
    monkeypatch.setattr(app.DataLogger, 'close', finish_recording)

    class FixtureDaqError(Exception):
        pass

    class Task:
        def __init__(self):
            self.channels, self.closed, self.started = [], False, False
            self.ai_channels = SimpleNamespace(add_ai_accel_chan=self.accel, add_ai_voltage_chan=self.voltage)
            self.ao_channels = SimpleNamespace(add_ao_voltage_chan=lambda **kwargs: None)
            self.timing = SimpleNamespace(cfg_samp_clk_timing=lambda **kwargs: None)
            self.in_stream = SimpleNamespace(input_buf_size=0, task=self, avail_samp_per_chan=1600)
            self.out_stream = SimpleNamespace(output_buf_size=0)
            tasks.append(self)

        def accel(self, **kwargs):
            self.channels.append(kwargs)
            # Simulate a partially added channel to require a clean task rebuild.
            if fallback and kwargs['physical_channel'] == 'Dev1/ai2':
                raise FixtureDaqError('Acceleration mode unsupported for fixture channel')
            return SimpleNamespace()

        def voltage(self, **kwargs):
            self.channels.append(kwargs)

        def start(self):
            self.started = True

        def stop(self):
            self.started = False

        def close(self):
            self.closed = True

    class Reader:
        def __init__(self, stream):
            assert [c['physical_channel'] for c in stream.task.channels] == [f'Dev1/ai{i}' for i in range(4)]

        def read_many_sample(self, data, number_of_samples_per_channel, timeout):
            assert data.shape == (4, number_of_samples_per_channel)
            reads.append(timeout)
            wave = np.sin(2 * np.pi * 200 * np.arange(number_of_samples_per_channel) / 8000)
            data[:] = np.array([.5, 1., .4 if fallback else 2., 3.])[:, None] * wave

    class Writer:
        def __init__(self, *args, **kwargs):
            pass

        def write_many_sample(self, data, timeout=None):
            pass

    monkeypatch.setattr(app, 'nidaqmx', SimpleNamespace(Task=Task))
    monkeypatch.setattr(app, 'NIDAQMX_AVAILABLE', True)
    monkeypatch.setattr(app, 'DaqError', FixtureDaqError)
    monkeypatch.setattr(app, 'AnalogMultiChannelReader', Reader, raising=False)
    monkeypatch.setattr(app, 'AnalogSingleChannelWriter', Writer, raising=False)
    for name, values in {
        'AcquisitionType': ['CONTINUOUS'], 'TerminalConfiguration': ['PSEUDO_DIFF'],
        'ExcitationSource': ['INTERNAL', 'NONE'], 'AccelUnits': ['G'], 'Coupling': ['AC'],
        'VoltageUnits': ['VOLTS'], 'AccelSensitivityUnits': ['MILLIVOLTS_PER_G'],
        'RegenerationMode': ['DONT_ALLOW_REGENERATION'],
    }.items():
        monkeypatch.setattr(app, name, SimpleNamespace(**{v: v for v in values}), raising=False)

    panel = window.config_tab
    window.backend_combo.setCurrentIndex(0)
    window.mode_combo.setCurrentIndex(1 if mode == 'sine_sweep' else 0)
    panel.fields['response_channels'].setPlainText(
        'Dev1/ai1, Response 1, 100, 2.1\nDev1/ai2, Response 2, 200, 0\nDev1/ai3, Response 3, 100, 2.1')
    panel.fields['data_log_enabled'].setChecked(True)
    panel.fields['data_log_dir'].setText(str(tmp_path))
    panel.fields['sync_ao_clock'].setChecked(False)
    panel.fields['sine_start_hz'].setValue(200)
    panel.fields['sine_end_hz'].setValue(200)
    assert panel.apply_config()
    assert window.settings.num_input_channels == 4
    assert window.start_controller()
    wait_until(lambda: window.realtime_tab.blocks >= 2 or not window.busy)
    assert window.busy, window.last_report
    window.stop_controller()
    wait_until(lambda: not window.busy)

    report = window.last_report
    assert report['outcome'] == 'stopped', report['error']
    assert report['backend'] == 'Hardware'
    assert report['latest_metrics']['accel_rms_g'] == pytest.approx(.5 / np.sqrt(2), rel=.01)
    assert all(t.closed and not t.started for t in tasks)
    assert all(timeout > 0 for timeout in reads)
    assert not hardware_waits, 'Queued hardware data must be drained without simulation pacing delays'
    diagnostics = report['stream_diagnostics']
    assert diagnostics['max_input_backlog_samples'] == 1600
    assert diagnostics['input_buffer_samples_per_channel'] == 128000
    assert diagnostics['max_read_seconds'] >= 0
    assert diagnostics['max_write_seconds'] >= 0
    assert diagnostics['max_record_seconds'] >= 0
    assert diagnostics['priming_samples'] == 800
    assert not diagnostics['regeneration_enabled']
    wave = np.sin(2 * np.pi * 200 * np.arange(800) / 8000)
    expected = np.array([.5, 1., 2., 3.])[:, None] * wave
    np.testing.assert_allclose(window.realtime_tab.buffer[:, -800:], expected, atol=1e-12)
    assert all(curve.xData is not None for curve in window.realtime_tab.psd_curves)
    window.test_browser.choose('channel', 2)
    assert 'Dev1/ai2' in window.config_dock.description.text()
    with h5py.File(Path(report['log_dir']) / 'data.h5') as recorded:
        assert recorded['ai'].shape[1:] == (4, 800)
        np.testing.assert_allclose(recorded['ai'][0], expected, atol=1e-7)
        assert json.loads(recorded.attrs['input_physical_channels']) == [f'Dev1/ai{i}' for i in range(4)]
        assert json.loads(recorded.attrs['input_channel_labels']) == ['Control Accel', 'Response 1', 'Response 2', 'Response 3']
    if fallback:
        assert len(tasks) == 3
        assert 'Dev1/ai2: Voltage input scaled to g' in report['input_mode']
    else:
        assert len(tasks) == 2
        assert tasks[0].channels[2]['sensitivity'] == 200
        assert tasks[0].channels[2]['current_excit_val'] == 0


@pytest.mark.parametrize('changes', [
    {'physical_channel': 'Dev1/ai0'}, {'physical_channel': 'dev1/AI1'},
    {'physical_channel': 'Dev1/ai1:3'}, {'label': 'Control Accel'},
    {'sensitivity_mv_per_g': 0}, {'sensitivity_mv_per_g': float('nan')},
    {'excitation_amps': -1}, {'excitation_amps': float('inf')},
])
def test_bad_response_setup_is_rejected_before_hardware_opens(settings_data, changes):
    settings_data['simulation_mode'] = False
    settings_data['response_channels'] = [dict(physical_channel=f'Dev1/ai{i}', label=f'Response {i}',
                                              sensitivity_mv_per_g=100, excitation_amps=.0021) for i in range(1, 4)]
    settings_data['response_channels'][1].update(changes)
    with pytest.raises(ConfigurationError) as error:
        RunSettings.capture(settings_data)
    assert 'response_channels' in error.value.errors


def test_response_editor_roundtrip_and_legacy_preset(window, tmp_path, monkeypatch):
    channels = [dict(physical_channel='Dev1/ai1', label='Response, fixture',
                     sensitivity_mv_per_g=100, excitation_amps=.0021)]
    restored = parse_response_channels(format_response_channels(channels))
    assert restored[0]['physical_channel'] == channels[0]['physical_channel']
    assert restored[0]['label'] == channels[0]['label']
    assert restored[0]['sensitivity_mv_per_g'] == 100
    assert restored[0]['excitation_amps'] == pytest.approx(.0021)
    source = tmp_path / 'old-preset.json'
    source.write_text(json.dumps({'settings': {'simulation_mode': False}}))
    monkeypatch.setattr(workspace_ui.QFileDialog, 'getOpenFileName', lambda *args: (str(source), 'JSON'))
    window.config_tab.load_preset()
    assert window.settings.num_input_channels == 1
    assert not window.settings.response_channels
    assert window.settings.hardware_input_channels[0]['physical_channel'] == 'Dev1/ai0'


def test_invalid_response_draft_does_not_replace_accepted_setup(window):
    window.backend_combo.setCurrentIndex(0)
    assert window.config_tab.apply_config()
    original_id = window.settings.config_id
    window.config_tab.fields['response_channels'].setPlainText('Dev1/ai1, missing values')
    assert not window.config_tab.apply_config()
    assert window.settings.config_id == original_id
    assert not window.start_button.isEnabled()
