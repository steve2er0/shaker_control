"""Portable paths, UTF-8 imports, and actual recording outside the launch cwd."""
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest
from PySide6.QtCore import QStandardPaths

import app
import app_paths
import config
import workspace_ui
from conftest import wait_until
from run_settings import ConfigurationError, RunSettings


def test_default_recording_uses_redirected_documents_without_creating_it(tmp_path, monkeypatch):
    documents = tmp_path / 'Redirected Documents' / 'André'
    monkeypatch.setattr(QStandardPaths, 'writableLocation', lambda location: str(documents))
    monkeypatch.setattr(config, 'DATA_LOG_DIR', None)
    monkeypatch.chdir(tmp_path)
    settings = app.SharedConfig().snapshot()
    assert Path(settings.data_log_dir) == documents / 'ShakerControl' / 'Runs'
    assert not documents.exists()


def test_documents_fallback_stays_in_user_home(tmp_path, monkeypatch):
    monkeypatch.setattr(QStandardPaths, 'writableLocation', lambda location: '')
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    assert app_paths.default_recording_directory() == tmp_path / 'Documents' / 'ShakerControl' / 'Runs'


def test_relative_recording_paths_are_frozen_before_launch_cwd_changes(settings_data, tmp_path, monkeypatch):
    checkout = tmp_path / 'App checkout'
    checkout.mkdir()
    monkeypatch.setattr(app_paths, 'APP_ROOT', checkout)
    settings_data['data_log_dir'] = 'Existing runs/Δ'
    captured = RunSettings.capture(settings_data)
    monkeypatch.chdir(tmp_path)
    assert Path(captured.data_log_dir) == checkout / 'Existing runs' / 'Δ'
    logger = app.DataLogger(captured, 'random', captured.fs, 800, 4)
    assert not logger.enabled
    assert not (checkout / 'Existing runs').exists()


@pytest.mark.parametrize('value, expected', [
    ('/Users/André/My presets/test.json', 'test.json'),
    (r'C:\Users\André\My presets\test.json', 'test.json'),
    ('D:/Runs/20261026_random/', '20261026_random'),
    ('\\\\server\\Shared runs\\20261026_random\\', '20261026_random'),
    ('', ''),
])
def test_labels_read_paths_from_either_platform(value, expected):
    assert app_paths.display_path_name(value) == expected


def test_foreign_or_drive_relative_recording_path_is_rejected_atomically(settings_data):
    settings_data['data_log_dir'] = r'C:relative\runs' if os.name == 'nt' else r'C:\Users\André\Runs'
    with pytest.raises(ConfigurationError) as error:
        RunSettings.capture(settings_data)
    assert 'data_log_dir' in error.value.errors


def test_native_absolute_and_home_paths_remain_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv('USERPROFILE' if os.name == 'nt' else 'HOME', str(tmp_path))
    assert app_paths.resolve_recording_directory(tmp_path / 'Explicit runs') == tmp_path / 'Explicit runs'
    assert app_paths.resolve_recording_directory('~/My runs') == tmp_path / 'My runs'


def test_imported_utf8_preset_and_recording_survive_unicode_paths_and_cwd_change(window, tmp_path, monkeypatch):
    folder = tmp_path / 'André workspace Δ'
    folder.mkdir()
    source = folder / 'Accélération.json'
    values = window.shared_config.as_dict()
    values.update(data_log_enabled=True, data_log_dir=str(folder / 'Recorded runs'),
                  calibration_source='Banc d’essai — Δ',
                  input_channel_labels=['Contrôle', 'Réponse 1', 'Réponse 2', 'Réponse 3'])
    source.write_text(json.dumps({'schema_version': 1, 'settings': values}, ensure_ascii=False), encoding='utf-8-sig')
    source_bytes = source.read_bytes()
    monkeypatch.setattr(workspace_ui.QFileDialog, 'getOpenFileName', lambda *args: (str(source), 'JSON'))
    window.config_tab.load_preset()
    assert window.settings.calibration_source == values['calibration_source']
    assert window.settings.input_channel_labels[0] == 'Contrôle'
    assert window.setup_name.text() == source.name
    assert source.read_bytes() == source_bytes

    # A shortcut, shell or file dialog must not change the captured destination.
    monkeypatch.chdir(tmp_path)
    assert window.start_controller()
    wait_until(lambda: bool(window.controller_tab.history))
    assert window.stop_controller()
    wait_until(lambda: not window.busy)
    report = window.last_report
    directory = Path(report['log_dir'])
    assert report['outcome'] == 'stopped'
    assert directory.is_absolute() and directory.parent == folder / 'Recorded runs'
    assert window.recording_status.text() == 'Saved · ' + directory.name
    saved = json.loads((directory / 'settings.json').read_text(encoding='utf-8'))
    summary = json.loads((directory / 'run-summary.json').read_text(encoding='utf-8'))
    assert saved['calibration_source'] == values['calibration_source']
    assert saved['data_log_dir'] == str(directory.parent)
    assert summary['log_dir'] == str(directory)
    with h5py.File(directory / 'data.h5') as recording:
        assert recording.attrs['backend'] == 'Simulation'
        assert recording['ai'].shape[0] == report['blocks']
        assert np.all(np.isfinite(recording['ai'][:]))
    assert (directory / 'preview_psd.csv').read_text(encoding='utf-8').startswith('frequency_hz,')


def test_windows_report_labels_keep_full_paths_in_tooltips(window):
    values = window.shared_config.as_dict()
    values['preset_source'] = r'C:\Users\André\Test setups\vibration.json'
    window._settings_applied(RunSettings.capture(values))
    window.active_log_dir = 'D:\\Recorded runs\\20261026_random\\'
    window._sync_shell_state()
    assert window.setup_name.text() == 'vibration.json'
    assert window.setup_name.toolTip() == values['preset_source']
    assert window.recording_status.text() == 'Saved · 20261026_random'
    assert window.recording_status.toolTip() == window.active_log_dir


def test_default_folder_is_used_by_real_logger_from_another_directory(settings_data, tmp_path, monkeypatch):
    destination = tmp_path / 'Documents' / 'ShakerControl' / 'Runs'
    monkeypatch.setattr(app_paths, 'default_recording_directory', lambda: destination)
    monkeypatch.setattr(config, 'DATA_LOG_DIR', None)
    settings_data.update(data_log_enabled=True, data_log_dir=app.SharedConfig().data_log_dir)
    captured = RunSettings.capture(settings_data)
    monkeypatch.chdir(tmp_path)
    logger = app.DataLogger(captured, 'random', captured.fs, 800, 4)
    try:
        assert logger.run_dir.is_absolute() and logger.run_dir.parent == destination
        assert (logger.run_dir / 'settings.json').is_file()
    finally:
        logger.close()
    with h5py.File(logger.run_dir / 'data.h5') as recording:
        assert recording.attrs['schema_version'] == 2


def test_recording_destination_failure_retains_fault(window, tmp_path):
    destination = tmp_path / 'A file is not a recording folder'
    destination.write_text('Occupied', encoding='utf-8')
    panel = window.config_tab
    panel.fields['data_log_dir'].setText(str(destination))
    panel.fields['data_log_enabled'].setChecked(True)
    assert panel.apply_config()
    assert window.start_controller()
    wait_until(lambda: not window.busy and window.last_report is not None)
    assert window.last_report['outcome'] == window.run_state == 'fault'
    assert window.last_report['log_dir'] is None
    assert window.last_report['error']
    assert destination.read_text(encoding='utf-8') == 'Occupied'
