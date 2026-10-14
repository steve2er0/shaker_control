import json
import numpy as np
import h5py
import pytest
from conftest import wait_until
import app
from run_settings import RunSettings


def test_apply_refreshes_sample_rate_channels_and_blocks_start_for_drafts(window):
    panel = window.config_tab
    panel.fs_spin.setValue(12000)
    assert panel.dirty
    assert not window.start_button.isEnabled()
    assert window.shared_config.fs == 8000
    assert panel.apply_config()
    assert window.shared_config.fs == window.realtime_tab.fs == 12000
    assert window.realtime_tab.num_channels == 4
    assert window.start_button.isEnabled()


def test_invalid_setup_does_not_mutate_applied_settings_or_start(window):
    panel = window.config_tab
    panel.fs_spin.setValue(1000)
    assert not panel.apply_config()
    assert window.settings.fs == window.realtime_tab.fs == 8000
    assert panel.error_labels['target_psd_points'].text()
    assert 'Nyquist' in panel.validation_label.text()
    assert not window.start_controller()
    assert window.controller_thread is None


def test_worker_fault_is_retained_and_thread_is_cleaned_up(window,monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('Synthetic initialization failure')
    monkeypatch.setattr(app,'create_simulation_system',fail)
    assert window.start_controller()
    wait_until(lambda: window.last_report is not None and not window.busy)
    assert window.run_state == 'fault'
    assert 'Synthetic initialization failure' in window.fault_label.text()
    assert window.start_button.isEnabled()
    assert not window.stop_button.isEnabled()
    assert window.last_report['backend'] is None
    assert window.last_report['input_mode'] is None
    assert window.last_report['clock_status'] == 'Not initialized'
    window.dismiss_fault()
    assert window.run_state == 'ready'
    assert not window.fault_message


def test_hardware_package_failure_never_falls_back_to_simulation(window,monkeypatch):
    window.backend_combo.setCurrentIndex(0)
    assert window.config_tab.apply_config()
    monkeypatch.setattr(app,'NIDAQMX_AVAILABLE',False)
    def forbidden_simulation(*args,**kwargs):
        pytest.fail('Hardware failures must not silently simulate')
    monkeypatch.setattr(app,'create_simulation_system',forbidden_simulation)
    assert window.start_controller()
    wait_until(lambda: window.last_report is not None and not window.busy)
    assert window.run_state == 'fault'
    assert window.last_report['requested_backend'] == 'Hardware'
    assert window.last_report['backend'] is None
    assert not window.shared_config.simulation_mode
    window.switch_to_simulation()
    assert window.config_tab.dirty
    assert not window.start_button.isEnabled()


def test_stop_during_startup_and_close_during_run_clean_threads(window):
    assert window.start_controller()
    assert window.stop_controller()
    wait_until(lambda: not window.busy)
    assert window.run_state == 'stopped'
    assert window.start_button.isEnabled()
    assert window.start_controller()
    wait_until(lambda: bool(window.controller_tab.history))
    window.close()
    wait_until(lambda: not window.busy)
    assert window.last_report['outcome'] == 'stopped'


def test_running_snapshot_is_frozen_and_stop_stays_available_in_all_views(window):
    assert window.start_controller()
    wait_until(lambda: bool(window.controller_tab.history))
    snapshot = window.controller_worker.shared_config
    metrics = window.controller_tab.history[-1]
    assert metrics['target_rms_g'] == pytest.approx(snapshot.target_rms_g*metrics['level_fraction'])
    assert not window.config_tab.fs_spin.isEnabled()
    window.config_tab.fs_spin.setValue(12000)  # Programmatic mutation of a disabled draft widget.
    assert not window.config_tab.apply_config()
    assert snapshot.fs == window.shared_config.fs == 8000
    for index in range(4):
        window.show_page(index)
        assert window.stop_button.isEnabled()
    assert window.stop_controller()
    wait_until(lambda: not window.busy)
    assert window.last_report['settings']['fs'] == 8000


def test_sine_completion_resets_mode_curves_and_preserves_results(window):
    panel = window.config_tab
    window.controller_tab.update_psd((np.array([100.,300.]),np.ones(2),np.ones(2),np.ones(2)))
    window.mode_combo.setCurrentIndex(1)
    panel.fields['sine_start_hz'].setValue(100)
    panel.fields['sine_end_hz'].setValue(110)
    panel.fields['sine_repeat'].setChecked(False)
    panel.fields['sine_initial_settle'].setValue(0)
    panel.fields['sine_step_dwell'].setValue(.01)
    panel.fields['sine_octaves_per_min'].setValue(100)
    assert panel.apply_config()
    assert window.controller_tab.psd_target_curve.xData is None
    assert panel.sections['equalizer'][0].isHidden()
    assert window.start_controller()
    wait_until(lambda: window.last_report is not None and not window.busy)
    assert window.run_state == 'complete'
    assert max(window.controller_tab.sine_points) == 110
    assert window.start_button.isEnabled()
    assert len(window.results_tab.reports) == 1
    window.mode_combo.setCurrentIndex(0)
    assert panel.apply_config()
    assert not window.controller_tab.sine_points
    assert not window.controller_tab.history
    assert len(window.results_tab.reports) == 1


def test_equalizer_is_unchanged_by_other_updates_and_legend_does_not_grow(window):
    plot = window.controller_tab
    packet = (np.geomspace(100,500,8),np.ones(8))
    plot.update_eq(packet)
    bar = plot.eq_bargraph
    legend = plot.eq_plot.getPlotItem().legend
    count = len(legend.items)
    for _ in range(5):
        plot.update_psd((np.array([100.,300.]),np.ones(2),np.ones(2),np.ones(2)))
        plot.update_eq(packet)
    assert plot.eq_bargraph is bar
    assert len(legend.items) == count == 2
    plot.update_eq(None)
    assert len(legend.items) == 1


def test_recording_keeps_effective_setup_backend_and_per_block_sine_metadata(settings_data,tmp_path):
    settings_data.update(test_mode='sine_sweep',data_log_enabled=True,data_log_dir=str(tmp_path),calibration_source='simulation fixture')
    settings = RunSettings.capture(settings_data)
    logger = app.DataLogger(settings,'sine_sweep',settings.fs,800,settings.num_input_channels)
    for frequency,level in [(100,.2),(100,.3),(110,.4)]:
        logger.log_sine_block(np.zeros(800),np.zeros((4,800)),level,0,frequency,1,2)
    logger.close()
    with h5py.File(logger.run_dir/'data.h5') as file:
        assert file.attrs['backend'] == 'Simulation'
        assert file.attrs['ai_units'] == 'g'
        assert len(file['metadata/frequency_hz']) == len(file['metadata/timestamp']) == 3
        assert json.loads(file.attrs['settings_json'])['calibration_source'] == 'simulation fixture'
    assert len((logger.run_dir/'sine_summary.csv').read_text().splitlines()) == 3
    assert json.loads((logger.run_dir/'settings.json').read_text())['fs'] == 8000
    other = app.DataLogger(settings,'sine_sweep',settings.fs,800,settings.num_input_channels)
    assert other.run_dir != logger.run_dir
    other.close()


def test_live_recording_summary_is_visible_and_saved(window,tmp_path):
    panel = window.config_tab
    panel.fields['data_log_dir'].setText(str(tmp_path))
    panel.fields['data_log_enabled'].setChecked(True)
    assert panel.apply_config()
    assert window.start_controller()
    wait_until(lambda: bool(window.controller_tab.history))
    window.stop_controller()
    wait_until(lambda: not window.busy)
    report = window.last_report
    directory = tmp_path / report['log_dir'].split('/')[-1]
    assert (directory/'run-summary.json').is_file()
    assert (directory/'settings.json').is_file()
    assert 'Simulation' in window.results_tab.summary_text.toPlainText()
    assert report['config_id'] == json.loads((directory/'run-summary.json').read_text())['config_id']


def test_repeated_runs_finish_with_gui_owned_thread_objects(window,qt_app):
    window.show()
    for _ in range(3):
        assert window.start_controller()
        assert window.controller_worker.thread() == qt_app.thread()
        wait_until(lambda: bool(window.controller_tab.history))
        window.show_page(2)
        window.controller_tab.diagnostics_widget.setCurrentIndex(3)
        window.resize(1200,800)
        assert window.stop_controller()
        wait_until(lambda: not window.busy)
        assert 'review this run in Results' in window.notice_label.text()
        assert window.run_state == 'stopped'
    assert len(window.results_tab.reports) == 3


def test_recording_fault_is_not_suppressed_by_simultaneous_stop(window,tmp_path,monkeypatch):
    panel = window.config_tab
    panel.fields['data_log_dir'].setText(str(tmp_path))
    panel.fields['data_log_enabled'].setChecked(True)
    assert panel.apply_config()
    def fail_recording(*args,**kwargs):
        window.controller_worker.should_stop.set()
        raise RuntimeError('fixture disk write failure')
    monkeypatch.setattr(app.DataLogger,'log_random_block',fail_recording)
    assert window.start_controller()
    wait_until(lambda: window.last_report is not None and not window.busy)
    assert window.run_state == 'fault'
    assert 'fixture disk write failure' in window.last_report['error']
    assert not window.notice_label.isVisible()
