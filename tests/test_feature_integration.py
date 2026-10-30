import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import app
import calibration_wizard
from calibration_wizard import CalibrationWizard
from conftest import wait_until
from run_settings import RunSettings
from run_reports import RunAssessment, render_report_chart


def sine_settings(settings_data, **changes):
    settings_data.update(test_mode='sine_sweep', sine_closed_loop=True, sine_repeat=False,
        sine_start_hz=100, sine_end_hz=200, sine_points_per_octave=1,
        sine_initial_level=1, sine_max_level_rate=1, sine_initial_settle=0,
        sine_octaves_per_min=600, sine_step_dwell=.2,
        sine_settle_seconds=.1, sine_settle_blocks=2, sine_max_step_seconds=30,
        sine_g_level=.5, sine_g_level_is_rms=False, sine_drive_table=[], sine_default_vpk=.4,
        sim_noise_level=.0001, sim_resonances=[], sim_nonlinearity=0,
        response_limits_enabled=True, report_enabled=False)
    settings_data.update(changes)
    return RunSettings.capture(settings_data)


@pytest.mark.parametrize('start,end', [(100, 200), (200, 100)])
@pytest.mark.parametrize('gain', [4., 40.])
def test_closed_loop_worker_tracks_wrong_table_in_both_directions(qt_app, settings_data, monkeypatch, start, end, gain):
    settings = sine_settings(settings_data, sine_start_hz=start, sine_end_hz=end, sim_plant_gain=gain)
    worker = app.ControllerWorker(settings)
    monkeypatch.setattr(worker, '_pace_block', lambda *args: None)
    reports = []
    worker.report_ready.connect(reports.append)
    worker.run_controller()
    report = reports[0]
    assert report['outcome'] == 'complete', report['error']
    assert report['qualification']['status'] == 'pass'
    assert [row['frequency_hz'] for row in report['qualification']['steps']] == [start, end]
    assert all(abs(row['settled_error_percent']) < 5 for row in report['qualification']['steps'])
    assert report['control_algorithm'] == 'sine_gain_feedback_v1'


def test_nonconvergence_cannot_pass_or_produce_calibration(qt_app, settings_data, monkeypatch):
    settings = sine_settings(settings_data, calibration_run=True, sine_max_step_seconds=.5,
                             sine_pilot_vpk=.000001, feedback_min_drive_v=1)
    worker = app.ControllerWorker(settings)
    monkeypatch.setattr(worker, '_pace_block', lambda *args: None)
    reports = []
    worker.report_ready.connect(reports.append)
    worker.run_controller()
    assert reports[0]['outcome'] == 'fault'
    assert 'could not settle' in reports[0]['error']
    assert reports[0]['qualification']['status'] == 'fail'
    assert reports[0]['calibration']['drive_table'] == []


def test_measured_trip_stops_worker_and_saves_triggering_samples(qt_app, settings_data, tmp_path, monkeypatch):
    settings = sine_settings(settings_data, data_log_enabled=True, data_log_dir=str(tmp_path),
                             report_enabled=True, response_max_peak_g=.00001, response_max_rms_g=.000001)
    worker = app.ControllerWorker(settings)
    monkeypatch.setattr(worker, '_pace_block', lambda *args: None)
    reports = []
    worker.report_ready.connect(reports.append)
    worker.run_controller()
    report = reports[0]
    assert report['outcome'] == 'fault' and report['protection_trip']
    assert 'Measured peak limit' in report['error']
    directory = Path(report['log_dir'])
    assert (directory / 'protection-trip-block.npz').is_file()
    assert json.loads((directory / 'test-report.json').read_text())['qualification']['status'] == 'fail'
    assert not worker.is_running
    # Raw HDF5 is finalized even if the first acquired block trips protection.
    with h5py.File(directory / 'data.h5') as data:
        assert data['ai'].shape[0] == report['blocks']


def test_automatic_html_csv_and_chart_include_settled_results(qt_app, settings_data, tmp_path, monkeypatch):
    settings = sine_settings(settings_data, data_log_enabled=True, data_log_dir=str(tmp_path),
                             report_enabled=True, calibration_run=True)
    worker = app.ControllerWorker(settings)
    monkeypatch.setattr(worker, '_pace_block', lambda *args: None)
    reports = []
    worker.report_ready.connect(reports.append)
    worker.run_controller()
    report = reports[0]
    assert report['qualification']['status'] == 'pass', report['error']
    assert report['calibration']['status'] == 'pass'
    assert len(report['calibration']['drive_table']) == 2
    render_report_chart(report)
    directory = Path(report['report_dir'])
    assert (directory / 'tracking.png').stat().st_size > 1000
    assert (directory / 'block-metrics.csv').read_text().count('\n') == report['blocks'] + 1
    assert (directory / 'settled-sine-results.csv').is_file()
    html = (directory / 'test-report.html').read_text(encoding='utf-8')
    assert 'tracking.png' in html and 'Raw transient peak' in html
    assert json.loads((directory / 'calibration.json').read_text())['backend'] == 'Simulation'


def test_random_assessment_requires_duration_and_includes_all_full_target_blocks(settings_data):
    settings_data.update(report_min_seconds=.3, report_settle_seconds=.1)
    s = RunSettings.capture(settings_data)
    assessment = RunAssessment(s)
    for index, level in enumerate([.1, 1, 1, 1]):
        assessment.add({'elapsed_s': index*.1, 'accel_rms_g': s.target_rms_g,
                        'level_fraction': level, 'raw_peak_g': 1})
    assert assessment.finish('stopped', '')['status'] == 'incomplete'
    assessment.add({'elapsed_s': .4, 'accel_rms_g': s.target_rms_g*1.1,
                    'level_fraction': 1, 'raw_peak_g': 1})
    assert assessment.finish('stopped', '')['status'] == 'fail'


def test_calibration_wizard_runs_normal_worker_then_applies_saved_table(window, tmp_path, monkeypatch):
    values = sine_settings(window.shared_config.as_dict(), report_enabled=True, calibration_run=False,
                           data_log_enabled=True, data_log_dir=str(tmp_path))
    window.shared_config.apply_values(values.as_dict())
    window.config_tab.load_values()
    window._settings_applied(values)
    monkeypatch.setattr(app.ControllerWorker, '_pace_block', lambda *args: None)
    monkeypatch.setattr(calibration_wizard, 'default_recording_directory', lambda: tmp_path / 'Runs')
    wizard = CalibrationWizard(window)
    wizard.inputs['start'].setValue(100)
    wizard.inputs['end'].setValue(110)
    wizard.inputs['voltage'].setValue(.3)
    wizard.show()
    assert wizard.start()
    wait_until(lambda: not window.busy, timeout=20)
    assert wizard.result and wizard.result['status'] == 'pass', wizard.status.text()
    wizard.refresh()
    assert wizard.apply_button.isEnabled()
    # Changing the editor after measurement must not falsify the saved coverage.
    wizard.inputs['end'].setValue(200)
    assert wizard.apply_result()
    assert window.settings.sine_end_hz == 110
    assert window.settings.calibration_backend == 'Simulation'
    assert window.settings.calibration_max_hz == 110
    assert not window.settings.calibration_run
    presets = list((tmp_path / 'Presets').glob('*.json'))
    assert len(presets) == 1
    assert json.loads(presets[0].read_text())['settings']['sine_drive_table']
    wizard.close()


def test_wizard_cancellation_never_enables_apply(window):
    wizard = CalibrationWizard(window)
    wizard.running = True
    wizard.finished_run({'outcome': 'stopped', 'error': '', 'calibration': {'status': 'fail', 'drive_table': []}})
    wizard.refresh()
    assert wizard.result is None and not wizard.apply_button.isEnabled()
    wizard.close()


@pytest.mark.parametrize('mode', ['random', 'sine_sweep'])
def test_missing_control_sensor_faults_without_drive_escalation(qt_app, settings_data, monkeypatch, mode):
    s = sine_settings(settings_data, test_mode=mode, feedback_startup_seconds=0,
                       feedback_loss_seconds=.3, feedback_min_drive_v=.0001,
                       sine_max_step_seconds=5)
    original = app.create_simulation_system
    def flat_sensor(*args, **kwargs):
        writer, reader, ai_task, ao_task = original(*args, **kwargs)
        reader.read_many_sample = lambda data, **kwargs: data.fill(0)
        return writer, reader, ai_task, ao_task
    monkeypatch.setattr(app, 'create_simulation_system', flat_sensor)
    worker = app.ControllerWorker(s)
    monkeypatch.setattr(worker, '_pace_block', lambda *args: None)
    reports, packets = [], []
    worker.report_ready.connect(reports.append)
    worker.metrics_data_ready.connect(packets.append)
    worker.run_controller()
    report = reports[0]
    assert report['outcome'] == 'fault'
    assert 'feedback was lost' in report['error']
    assert report['qualification']['status'] == 'fail'
    assert max(packet['output_peak_v'] for packet in packets) < .01


def test_ni_overload_cannot_be_used_as_valid_feedback(settings_data):
    from types import SimpleNamespace
    worker = app.ControllerWorker(sine_settings(settings_data))
    worker.input_stream = SimpleNamespace(overloaded_chans_exist=True)
    with pytest.raises(RuntimeError, match='NI input overload'):
        worker._check_response(np.zeros(800), app.QueuedDrive(np.zeros(800), 0), 0)
    assert worker.protection_trip['input_overload']


def test_partial_sine_sweep_never_passes_assessment(settings_data):
    s = sine_settings(settings_data)
    assessment = RunAssessment(s)
    for index in range(5):
        assessment.add({'frequency_hz': 100, 'elapsed_s': index*.1, 'level_fraction': 1,
            'accel_peak_g': .5, 'tone_peak_g': .5, 'output_peak_v': .125})
    assert assessment.finish('stopped', '')['status'] == 'incomplete'


def test_report_write_failure_invalidates_calibration(qt_app, settings_data, tmp_path, monkeypatch):
    import run_reports
    monkeypatch.setattr(run_reports, 'default_recording_directory', lambda: tmp_path / 'Runs')
    def fail_save(*args):
        raise OSError('Report disk unavailable')
    monkeypatch.setattr(run_reports.ReportFiles, 'save', fail_save)
    worker = app.ControllerWorker(sine_settings(settings_data, calibration_run=True, report_enabled=True))
    monkeypatch.setattr(worker, '_pace_block', lambda *args: None)
    reports = []
    worker.report_ready.connect(reports.append)
    worker.run_controller()
    assert reports[0]['outcome'] == 'fault'
    assert reports[0]['qualification']['status'] == 'fail'
    assert reports[0]['calibration']['drive_table'] == []
    assert 'Report disk unavailable' in reports[0]['error']


def test_default_rate_noisy_sine_honors_pilot_ceiling_and_tracks(qt_app, settings_data, monkeypatch):
    settings = sine_settings(settings_data, fs=51200, block_seconds=1, welch_nperseg=4096,
        sim_noise_level=.02, sine_start_hz=100, sine_end_hz=100, sine_max_step_seconds=30)
    worker = app.ControllerWorker(settings)
    monkeypatch.setattr(worker, '_pace_block', lambda *args: None)
    reports, packets = [], []
    worker.report_ready.connect(reports.append)
    worker.metrics_data_ready.connect(packets.append)
    worker.run_controller()
    assert packets[0]['output_peak_v'] <= settings.sine_pilot_vpk
    assert reports[0]['outcome'] == 'complete', reports[0]['error']
    assert reports[0]['qualification']['status'] == 'pass'


def test_summary_write_failure_cannot_leave_a_passed_calibration(qt_app, settings_data, tmp_path, monkeypatch):
    original = Path.write_text
    def fail_summary(path, *args, **kwargs):
        if path.name == 'run-summary.json':
            raise OSError('Summary disk unavailable')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'write_text', fail_summary)
    worker = app.ControllerWorker(sine_settings(settings_data, calibration_run=True,
        data_log_enabled=True, data_log_dir=str(tmp_path), report_enabled=True))
    monkeypatch.setattr(worker, '_pace_block', lambda *args: None)
    reports = []
    worker.report_ready.connect(reports.append)
    worker.run_controller()
    report = reports[0]
    assert report['outcome'] == 'fault'
    assert report['qualification']['status'] == 'fail'
    assert report['calibration']['drive_table'] == []
    directory = Path(report['report_dir'])
    assert json.loads((directory / 'test-report.json').read_text())['qualification']['status'] == 'fail'
    assert json.loads((directory / 'calibration.json').read_text())['drive_table'] == []
