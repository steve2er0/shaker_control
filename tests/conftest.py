import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import time
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest


@pytest.fixture(scope='session')
def qt_app():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture(autouse=True)
def no_hardware(monkeypatch):
    import app
    import accel_monitor
    def forbidden(*args, **kwargs):
        pytest.fail('Tests must never open real NI-DAQ tasks')
    if app.nidaqmx is not None:
        monkeypatch.setattr(app.nidaqmx, 'Task', forbidden)
    if accel_monitor.nidaqmx is not None:
        monkeypatch.setattr(accel_monitor.nidaqmx, 'Task', forbidden)


def wait_until(predicate, timeout=6):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(20)
    assert predicate(), 'Timed out waiting for Qt workflow state'


@pytest.fixture
def window(qt_app):
    import app
    widget = app.MainWindow()
    widget.backend_combo.setCurrentIndex(1)
    panel = widget.config_tab
    panel.fs_spin.setValue(8000)
    panel.block_seconds_spin.setValue(.1)
    panel.fields['welch_nperseg'].setValue(512)
    panel.fields['eq_num_bands'].setValue(8)
    panel.fields['data_log_enabled'].setChecked(False)
    panel.fields['report_enabled'].setChecked(False)
    panel.fields['sine_closed_loop'].setChecked(False)
    panel.fields['response_limits_enabled'].setChecked(False)
    assert panel.apply_config()
    yield widget
    if widget.busy:
        widget.stop_controller()
        wait_until(lambda: not widget.busy)
    widget.close()
    qt_app.processEvents()


@pytest.fixture
def settings_data():
    from app import SharedConfig
    data = SharedConfig().as_dict()
    data.update(simulation_mode=True, fs=8000, block_seconds=.1,
                welch_nperseg=512, eq_num_bands=8, data_log_enabled=False,
                report_enabled=False, sine_closed_loop=False, response_limits_enabled=False)
    return data
