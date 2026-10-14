import numpy as np
import pytest
from PySide6.QtWidgets import QTableWidgetItem
from conftest import wait_until
import accel_monitor as monitor


@pytest.fixture
def monitor_window(qt_app):
    window = monitor.MonitorWindow()
    window.backend_combo.setCurrentIndex(1)
    yield window
    if window._worker is not None:
        window._stop_acquisition()
        wait_until(lambda: window._worker is None)
    window.close()
    qt_app.processEvents()


def test_full_psd_window_is_retained_separately_from_time_view(monitor_window):
    window = monitor_window
    window._psd_spin.setValue(10)
    session = monitor.SimulationSession(window._channel_specs,window._fs,window._block,0,123)
    session.start()
    window._handle_block(session.read_block())
    assert 'collecting full window' in window.psd_effective_label.text()
    for _ in range(int(np.ceil(10*window._fs/window._block))):
        window._handle_block(session.read_block())
    assert window._buffer_fill >= 10*window._fs
    assert 'Effective 10 s' in window.psd_effective_label.text()
    assert '0.1 Hz bins' in window.psd_effective_label.text()
    assert 'collecting' not in window.psd_effective_label.text()
    assert len(window._time_curves[0].xData) == int(window._fs*window._time_window_sec)
    assert window._psd_curves[0].xData[1] == pytest.approx(.1)


def test_monitor_hardware_failure_retains_fault_without_simulation(monitor_window,monkeypatch):
    window = monitor_window
    window.backend_combo.setCurrentIndex(0)
    monkeypatch.setattr(monitor,'NIDAQ_AVAILABLE',False)
    def forbidden(*args,**kwargs):
        pytest.fail('Hardware failures must not select simulation')
    monkeypatch.setattr(monitor,'SimulationSession',forbidden)
    window._start_acquisition()
    wait_until(lambda: window._worker is None)
    assert window._status_label.text() == 'Fault'
    assert window._fault_message
    assert window.current_run_settings['backend'] == 'Hardware'
    assert window._start_button.isEnabled()
    window._handle_worker_status('stopped')
    assert window._status_label.text() == 'Fault'


def test_monitor_stop_close_and_settings_lock(monitor_window):
    window = monitor_window
    window._start_acquisition()
    window._stop_acquisition()  # Stop during startup.
    wait_until(lambda: window._worker is None)
    assert not window._fault_message
    window._start_acquisition()
    wait_until(lambda: window._buffer_fill > 0)
    assert not window.backend_combo.isEnabled()
    with pytest.raises(ValueError,match='Stop acquisition'):
        window.apply_acquisition_setup(window._channel_specs,4000,256)
    dialog = monitor.MonitorSetupDialog(window)
    assert not dialog.table.isEnabled()
    dialog.close()
    window.close()
    wait_until(lambda: window._worker is None)
    assert window._status_label.text() == 'Stopped'
    assert window.current_run_settings['fs'] == 2048


def test_monitor_setup_validation_and_plot_rebuild(monitor_window):
    window = monitor_window
    dialog = monitor.MonitorSetupDialog(window)
    dialog.table.setItem(0,2,QTableWidgetItem('nan'))
    dialog.apply_setup()
    assert dialog.error_label.text()
    assert window._fs == 2048
    dialog.table.setItem(0,2,QTableWidgetItem('100'))
    dialog.add_row(window._channel_specs[0])
    dialog.apply_setup()
    assert 'unique' in dialog.error_label.text()
    dialog.table.removeRow(1)
    dialog.fs_spin.setValue(4000)
    dialog.block_spin.setValue(256)
    dialog.apply_setup()
    assert window._fs == 4000
    assert window._block == 256
    assert window._buffer_capacity == 40000
    assert len(window._time_curves) == 1


def test_monitor_cleanup_error_remains_a_fault(monitor_window,monkeypatch):
    class BrokenClose(monitor.SimulationSession):
        def close(self):
            raise RuntimeError('fixture close failure')
    monkeypatch.setattr(monitor,'SimulationSession',BrokenClose)
    window = monitor_window
    window._start_acquisition()
    wait_until(lambda: window._buffer_fill > 0)
    window._stop_acquisition()
    wait_until(lambda: window._worker is None)
    assert window._status_label.text() == 'Fault'
    assert 'fixture close failure' in window._fault_message
