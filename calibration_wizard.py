"""GUI calibration using the same worker, protections, recording and Stop path."""
import json
import math
from pathlib import Path
import uuid

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QHBoxLayout, QLabel,
                              QLineEdit, QDoubleSpinBox, QPushButton, QPlainTextEdit)
from app_paths import default_recording_directory
from run_settings import RunSettings, ConfigurationError


class CalibrationWizard(QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle('Sine drive calibration')
        self.setWindowModality(Qt.WindowModal)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.resize(690, 700)
        self.running = False
        self.result = None
        self._closing = False
        self.applied = False
        s = window.shared_config.snapshot()
        voltage_default = min(s.ao_volt_limit, max(.03, 1.5 * min(.5, s.sine_g_level) / s.sim_plant_gain)) if s.simulation_mode else min(.03, s.ao_volt_limit)
        layout = QVBoxLayout(self)
        heading = QLabel(f'{s.backend.upper()} calibration · {s.device_ai} · {s.accel_mv_per_g:g} mV/g')
        heading.setWordWrap(True)
        layout.addWidget(heading)
        note = QLabel('A low pilot identifies the response, then bounded feedback approaches the target at each frequency. '
                      'Start uses the selected backend and drives output. Stop remains available throughout. '
                      'A table is available to apply only after every frequency settles successfully.')
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        layout.addLayout(form)
        self.inputs = {}
        for key, label, value in (
            ('equipment', 'Shaker / fixture / load', s.calibration_equipment),
            ('amplifier', 'Amplifier gain / setting', s.calibration_amp_setting),
            ('sensor', 'Control accelerometer ID', s.calibration_sensor_id)):
            field = QLineEdit(value or ('Synthetic plant' if s.simulation_mode else ''))
            form.addRow(label, field)
            self.inputs[key] = field
        for key, label, value, minimum, maximum, suffix in (
            ('start', 'Start frequency', s.sine_start_hz, .1, s.fs / 2 - 1, ' Hz'),
            ('end', 'End frequency', s.sine_end_hz, .1, s.fs / 2 - 1, ' Hz'),
            ('target', 'Calibration level', min(.5, s.sine_g_level), .001, 10, ' g peak'),
            ('voltage', 'Calibration output ceiling', voltage_default, .00001, s.ao_volt_limit, ' V peak'),
            ('spacing', 'Frequency spacing', min(6, s.sine_points_per_octave), 1, 48, ' points/octave')):
            field = QDoubleSpinBox()
            field.setDecimals(6 if key == 'voltage' else 3 if key == 'target' else 2)
            field.setRange(minimum, maximum)
            field.setValue(value)
            field.setSuffix(suffix)
            form.addRow(label, field)
            self.inputs[key] = field
        self.status = QLabel('Enter the equipment details, review the range and output ceiling, then start calibration.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        layout.addWidget(self.preview, 1)
        actions = QHBoxLayout()
        self.start_button = QPushButton(f'Start {s.backend.lower()} calibration')
        self.stop_button = QPushButton('Stop calibration')
        self.apply_button = QPushButton('Apply table and save preset')
        self.close_button = QPushButton('Close')
        for button in (self.start_button, self.stop_button, self.apply_button, self.close_button):
            actions.addWidget(button)
        layout.addLayout(actions)
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(window.stop_controller)
        self.apply_button.clicked.connect(self.apply_result)
        self.close_button.clicked.connect(self.reject)
        window.run_completed.connect(self.finished_run)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(100)
        self.refresh()

    def refresh(self):
        busy = self.window.busy
        self.start_button.setEnabled(not busy and not self.running)
        self.stop_button.setEnabled(self.running and self.window.run_state in ('starting', 'running'))
        self.apply_button.setEnabled(not busy and not self.applied and self.result is not None
                                     and self.result.get('status') == 'pass')
        for field in self.inputs.values():
            field.setEnabled(not busy)
        if self._closing and not busy:
            self.restore_setup()
            super().reject()

    def start(self):
        if self.window.busy:
            return False
        self.result = None
        self.applied = False
        values = self.window.shared_config.as_dict()
        equipment, amplifier, sensor = (self.inputs[key].text().strip() for key in ('equipment', 'amplifier', 'sensor'))
        if not all((equipment, amplifier, sensor)):
            self.status.setText('Enter the shaker/load, amplifier setting, and sensor ID so the calibration can be identified.')
            return False
        target = self.inputs['target'].value()
        limit = self.inputs['voltage'].value()
        values.update(test_mode='sine_sweep', sine_closed_loop=True, calibration_run=True,
            sine_start_hz=self.inputs['start'].value(), sine_end_hz=self.inputs['end'].value(),
            sine_g_level=target, sine_g_level_is_rms=False, sine_repeat=False,
            sine_points_per_octave=self.inputs['spacing'].value(), sine_step_dwell=max(2 * values['block_seconds'], 1.),
            sine_octaves_per_min=10., sine_initial_level=.25, sine_max_level_rate=.2,
            sine_initial_settle=2., sine_drive_table=[], sine_drive_scale=1., sine_default_vpk=.4,
            sine_max_step_seconds=max(45., 6 * values['block_seconds']),
            ao_volt_limit=limit, max_rms_volts=min(values['max_rms_volts'] or limit, limit / math.sqrt(2)),
            response_limits_enabled=True, response_target_ratio=1.5,
            report_enabled=True, calibration_equipment=equipment,
            calibration_amp_setting=amplifier, calibration_sensor_id=sensor,
            calibration_source='Calibration in progress; no validated table yet.')
        try:
            snapshot = RunSettings.capture(values)
        except ConfigurationError as error:
            self.status.setText(str(error))
            return False
        if not self.window.start_controller(snapshot):
            self.status.setText('Calibration could not start. Apply/discard the main setup draft and stop any active run.')
            return False
        self.running = True
        self.preview.clear()
        self.status.setText('Calibration running. Waiting for the first settled frequency…')
        self.window.controller_worker.metrics_data_ready.connect(self.progress)
        self.refresh()
        return True

    def progress(self, packet):
        if self.running:
            self.status.setText(f'{packet.get("frequency_hz", 0):.4g} Hz · fundamental {packet.get("tone_peak_g", 0):.4g} g peak · '
                                f'output {packet["output_peak_v"]:.5g} V peak · '
                                + ('settled' if packet.get('settled') else 'identifying / settling'))

    def finished_run(self, report):
        if not self.running:
            return
        self.running = False
        self.result = report.get('calibration')
        if self.result and self.result['status'] == 'pass':
            lines = ['Frequency (Hz), V peak per 1 g peak']
            lines += [f'{f:.7g}, {v:.9g}' for f, v in self.result['drive_table']]
            self.preview.setPlainText('\n'.join(lines))
            self.status.setText('Every frequency passed. Review the table, then apply it and save the setup preset. Output is stopped.')
        else:
            self.result = None
            self.status.setText('No calibration table was accepted. ' + (report.get('error') or 'Calibration was stopped before completion.'))
        self.refresh()

    def apply_result(self):
        if self.window.busy or not self.result or self.result['status'] != 'pass' or self.applied:
            return False
        result = self.result
        values = self.window.shared_config.as_dict()
        backend = 'Simulation' if values['simulation_mode'] else 'Hardware'
        if (backend != result['backend'] or values['accel_mv_per_g'] != result['sensitivity_mv_per_g']
                or values['device_ai'] != result['input_channel'] or values['device_ao'] != result['output_channel']):
            self.status.setText('Backend or sensor sensitivity changed; this table cannot be applied to that setup.')
            return False
        values.update(test_mode='sine_sweep', sine_closed_loop=True, calibration_run=False,
            sine_drive_table=result['drive_table'], sine_drive_scale=1.,
            sine_start_hz=result['start_hz'], sine_end_hz=result['end_hz'],
            sine_g_level=result['level_peak_g'], sine_g_level_is_rms=False, sine_repeat=False,
            ao_volt_limit=result['output_peak_limit_v'], max_rms_volts=result['output_rms_limit_v'],
            calibration_equipment=result['equipment'], calibration_amp_setting=result['amplifier_setting'],
            calibration_sensor_id=result['sensor_id'], calibration_min_hz=result['minimum_hz'],
            calibration_max_hz=result['maximum_hz'], calibration_level_peak_g=result['level_peak_g'],
            calibration_backend=result['backend'], calibration_source=(f'{result["backend"]} · {result["started_at"]} · '
                f'{result["equipment"]} · amp {result["amplifier_setting"]} · sensor {result["sensor_id"]} · setup {result["config_id"]}'))
        try:
            snapshot = RunSettings.capture(values)
            directory = default_recording_directory().parent / 'Presets'
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f'sine-calibration-{result["config_id"]}-{uuid.uuid4().hex[:8]}.json'
            saved = snapshot.as_dict()
            saved.pop('preset_source', None)
            with path.open('x', encoding='utf-8') as handle:
                handle.write(json.dumps({'schema_version': 1, 'settings': saved}, indent=2, allow_nan=False) + '\n')
            values['preset_source'] = str(path)
            self.window.shared_config.apply_values(values)
            self.window.config_tab.load_values()
            self.window._settings_applied(self.window.shared_config.snapshot())
        except (OSError, ValueError) as error:
            self.status.setText('Calibration could not be saved/applied: ' + str(error))
            return False
        self.applied = True
        self.status.setText(f'Calibration applied. Preset saved: {path}')
        self.refresh()
        return True

    def reject(self):
        if self.running or self.window.busy:
            self._closing = True
            self.window.stop_controller()
            self.status.setText('Stopping calibration; waiting for acquisition and recording cleanup…')
        else:
            self.restore_setup()
            super().reject()

    def restore_setup(self):
        if not self.applied and not self.window.busy:
            self.window.config_tab.load_values()
            self.window._settings_applied(self.window.shared_config.snapshot())

    def closeEvent(self, event):
        if self.running or self.window.busy:
            self.reject()
            event.ignore()
        else:
            self.restore_setup()
            event.accept()
