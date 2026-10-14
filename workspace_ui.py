"""Native Qt workspace with validated setup and an authoritative run lifecycle."""
from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from scipy.signal import welch
from PySide6.QtCore import Qt, Signal, QUrl, QSignalBlocker
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox,
    QLineEdit, QPlainTextEdit, QScrollArea, QToolButton, QDockWidget,
    QStackedWidget, QTabWidget, QFileDialog, QListWidget, QApplication,
)
from run_settings import RunSettings, ConfigurationError, validate_settings, parse_points, format_points


COLORS = ['#287bb5', '#9264ad', '#298a6b', '#b97327', '#bd5673', '#577b85']
BUSY_STATES = {'starting', 'running', 'stopping'}


def configure_plot(widget):
    palette = QApplication.palette()
    text = palette.text().color()
    widget.setBackground(palette.base().color())
    for name in ('left', 'bottom'):
        axis = widget.getAxis(name)
        axis.setPen(text)
        axis.setTextPen(text)
        axis.enableAutoSIPrefix(False)
    widget.showGrid(x=True, y=True, alpha=0.15)


def set_curve(curve, x, y, band=None):
    if x is None or y is None:
        curve.clear()
        return
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape:
        curve.clear()
        return
    mask = np.isfinite(x) & np.isfinite(y)
    if band:
        mask &= (x >= band[0]) & (x <= band[1])
    curve.setData(x[mask], y[mask])


def frequency_ticks(plot, low, high):
    """Keep a narrow log-frequency range legible, including its endpoints."""
    ticks = [(math.log10(value), f'{value:.4g}') for value in np.geomspace(low, high, 5)]
    plot.getAxis('bottom').setTicks([ticks])


class SettingsPanel(QWidget):
    applied = Signal(object)
    dirty_changed = Signal(bool)

    def __init__(self, shared_config):
        super().__init__()
        self.shared_config = shared_config
        self._loading = True
        self.locked = False
        self.dirty = False
        self.validation_errors = {}
        self.fields, self.error_labels, self.sections = {}, {}, {}
        self._edit_widgets = []
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        content = QWidget()
        self.form_layout = QVBoxLayout(content)
        self.form_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_area.setWidget(content)
        outer.addWidget(self.scroll_area)

        self._section('target', 'Random target · Hz, g²/Hz', True)
        self._text('target_psd_points', 'Frequency, PSD pairs', 'target', multiline=True)
        self._section('sine', 'Sine sweep', True)
        self._spin('sine_start_hz', 'Start frequency', 'sine', .1, 20000, ' Hz')
        self._spin('sine_end_hz', 'End frequency', 'sine', .1, 20000, ' Hz')
        self._spin('sine_g_level', 'Target acceleration', 'sine', .001, 30, '', 3)
        self.sine_unit_combo = QComboBox()
        self.sine_unit_combo.setAccessibleName('Sine amplitude unit')
        self.sine_unit_combo.addItems(['g peak', 'g RMS'])
        self.sections['sine'][2].addRow('Amplitude unit', self.sine_unit_combo)
        self.sine_unit_combo.currentIndexChanged.connect(self._changed)
        self._edit_widgets.append(self.sine_unit_combo)
        self._spin('sine_octaves_per_min', 'Sweep rate', 'sine', .01, 1000, ' octave/min', 3)
        self._spin('sine_initial_level', 'Ramp start', 'sine', 0, 100, ' %', 1)
        self._spin('sine_max_level_rate', 'Ramp rate limit', 'sine', 0, 500, ' %/s', 1)
        self._spin('sine_initial_settle', 'Initial settling', 'sine', 0, 30, ' s')
        self._check('sine_repeat', 'Repeat sweep', 'sine')
        self._text('calibration_source', 'Calibration source', 'sine')
        self._section('drive', 'Sine drive table · V peak per 1 g peak')
        self._text('sine_drive_table', 'Frequency, V peak pairs', 'drive', multiline=True)
        self._spin('sine_default_vpk', 'Default drive', 'drive', 0, 10, ' V peak/g')
        self._spin('sine_drive_scale', 'Drive scale', 'drive', 0, 100, ' ×')
        self._spin('sine_points_per_octave', 'Frequency spacing', 'drive', 1, 100, ' points/octave')
        self._spin('sine_step_dwell', 'Minimum dwell', 'drive', .01, 60, ' s')

        self._section('acquisition', 'Acquisition', True)
        self._spin('fs', 'Sample rate', 'acquisition', 1000, 100000, ' Hz')
        self._spin('block_seconds', 'Processing block', 'acquisition', .01, 5, ' s', 3)
        self._spin('buf_seconds', 'DAQ buffer', 'acquisition', .01, 10, ' s', 3)
        self._spin('welch_nperseg', 'PSD segment', 'acquisition', 2, 8192, ' samples', integer=True)
        self.resolution_label = QLabel()
        self.resolution_label.setWordWrap(True)
        self.sections['acquisition'][2].addRow(self.resolution_label)
        self._section('hardware', 'Hardware channels')
        self._text('device_ai', 'Control input', 'hardware')
        self._text('device_ao', 'Drive output', 'hardware')
        self._spin('accel_mv_per_g', 'Sensitivity', 'hardware', .001, 10000, ' mV/g', 3)
        self._spin('accel_excitation_amps', 'IEPE current', 'hardware', 0, 100, ' mA', 3)
        self._check('sync_ao_clock', 'Share AI clock with AO', 'hardware')
        self._note('hardware', 'One hardware control input. Additional response channels are available in the standalone monitor.')
        self._section('simulation', 'Simulation model')
        self._text('input_channel_labels', 'Channel labels, comma separated', 'simulation')
        self._spin('sim_plant_gain', 'Plant gain', 'simulation', .001, 100, ' g/V', 3)
        self._spin('sim_noise_level', 'Measurement noise', 'simulation', 0, 10, ' g RMS', 4)
        self._note('simulation', 'Synthetic plant with configured resonances and delay; these parameters do not calibrate hardware.')

        self._section('limits', 'Voltage limits', True)
        self._spin('ao_volt_limit', 'Absolute peak limit', 'limits', .001, 10, ' V', 3)
        self._spin('max_rms_volts', 'RMS limit', 'limits', 0, 10, ' V RMS', 3)
        self._spin('max_crest_factor', 'Crest-factor limit', 'limits', 1, 10, ' peak/RMS', 2)
        self._section('control', 'Random ramp and PI tuning')
        self._spin('initial_level_fraction', 'Ramp start', 'control', 0, 200, ' %', 1)
        self._spin('max_level_fraction_rate', 'Ramp rate limit', 'control', 0, 200, ' %/s', 1)
        self._spin('kp', 'Proportional gain', 'control', 0, 10, '', 3)
        self._spin('ki', 'Integral gain', 'control', 0, 5, '', 3)
        self._note('control', 'Ramp % scales acceleration amplitude; 10% requests 1% of the full target PSD.')
        self._section('equalizer', 'Random equalizer tuning')
        self._spin('eq_num_bands', 'Bands', 'equalizer', 1, 64, '', integer=True)
        self._spin('eq_adapt_rate', 'Adaptation rate', 'equalizer', 0, 1, '', 3)
        self._spin('eq_smooth_factor', 'Smoothing factor', 'equalizer', 0, 1, '', 3)
        self._section('recording', 'Recording')
        self._check('data_log_enabled', 'Record raw blocks and summaries', 'recording')
        self._text('data_log_dir', 'Output directory', 'recording')
        self.form_layout.addStretch()

        self.validation_label = QLabel()
        self.validation_label.setWordWrap(True)
        outer.addWidget(self.validation_label)
        self.apply_button = QPushButton('Apply setup')
        self.apply_button.setDefault(True)
        self.apply_button.clicked.connect(self.apply_config)
        outer.addWidget(self.apply_button)
        actions = QHBoxLayout()
        self.load_button, self.save_button = QPushButton('Load preset…'), QPushButton('Save preset…')
        self.load_button.clicked.connect(self.load_preset)
        self.save_button.clicked.connect(self.save_preset)
        actions.addWidget(self.load_button)
        actions.addWidget(self.save_button)
        outer.addLayout(actions)
        # Backend and test selection live in the always-visible run bar.
        self.test_mode = shared_config.test_mode
        self.simulation_mode_check = QCheckBox()
        self.simulation_mode_check.setChecked(shared_config.simulation_mode)
        self.simulation_mode_check.toggled.connect(self._changed)
        self.simulation_mode_check.hide()
        self.load_values()
        self._loading = False

    def _section(self, key, title, expanded=False):
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 5)
        toggle = QToolButton()
        toggle.setText(title)
        toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        toggle.setCheckable(True)
        toggle.setChecked(expanded)
        toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        toggle.setSizePolicy(toggle.sizePolicy().horizontalPolicy(), toggle.sizePolicy().verticalPolicy())
        body = QWidget()
        form = QFormLayout(body)
        form.setContentsMargins(8, 3, 0, 3)
        form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        body.setVisible(expanded)
        toggle.toggled.connect(lambda checked: (body.setVisible(checked), toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)))
        layout.addWidget(toggle)
        layout.addWidget(body)
        self.form_layout.addWidget(wrapper)
        self.sections[key] = (wrapper, toggle, form, body)

    def _field(self, key, label, widget, section):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        error = QLabel()
        error.setWordWrap(True)
        error.setStyleSheet('color: #bd493c;')
        error.hide()
        layout.addWidget(widget)
        layout.addWidget(error)
        self.sections[section][2].addRow(label, container)
        self.fields[key], self.error_labels[key] = widget, error
        widget.setAccessibleName(label or widget.text())
        self._edit_widgets.append(widget)
        return widget

    def _spin(self, key, label, section, low, high, suffix='', decimals=2, integer=False):
        widget = QSpinBox() if integer else QDoubleSpinBox()
        if not integer:
            widget.setDecimals(decimals)
        widget.setRange(low, high)
        widget.setSuffix(suffix)
        widget.valueChanged.connect(self._changed)
        self._field(key, label, widget, section)
        aliases = {'fs':'fs_spin', 'welch_nperseg':'welch_nperseg_spin', 'block_seconds':'block_seconds_spin', 'initial_level_fraction':'initial_level_spin', 'sine_initial_level':'sine_initial_level_spin'}
        setattr(self, aliases.get(key, key + '_spin'), widget)

    def _text(self, key, label, section, multiline=False):
        widget = QPlainTextEdit() if multiline else QLineEdit()
        if multiline:
            widget.setFixedHeight(120)
            widget.setPlaceholderText('frequency, value')
        widget.textChanged.connect(self._changed)
        self._field(key, label, widget, section)

    def _check(self, key, label, section):
        widget = QCheckBox(label)
        widget.toggled.connect(self._changed)
        self._field(key, '', widget, section)

    def _note(self, section, text):
        label = QLabel(text)
        label.setWordWrap(True)
        self.sections[section][2].addRow(label)

    def load_values(self):
        previous = self._loading
        self._loading = True
        values = self.shared_config.as_dict()
        self.test_mode = values['test_mode']
        self.simulation_mode_check.setChecked(values['simulation_mode'])
        for key, widget in self.fields.items():
            value = values[key]
            if key in ('target_psd_points', 'sine_drive_table'):
                widget.setPlainText(format_points(value))
            elif key == 'input_channel_labels':
                widget.setText(', '.join(value))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            else:
                if key in ('initial_level_fraction','max_level_fraction_rate','sine_initial_level','sine_max_level_rate'):
                    value *= 100
                elif key == 'accel_excitation_amps':
                    value *= 1000
                # A valid imported value must remain the value shown by the
                # editor, even when it exceeds the suggested initial range.
                if isinstance(widget, QDoubleSpinBox):
                    decimal_text = f'{value:.9f}'.rstrip('0')
                    precision = len(decimal_text.split('.')[1]) if '.' in decimal_text else 0
                    widget.setDecimals(max(widget.decimals(), precision))
                widget.setRange(min(widget.minimum(), value), max(widget.maximum(), value))
                widget.setValue(value)
        self.sine_unit_combo.setCurrentIndex(1 if values['sine_g_level_is_rms'] else 0)
        self._loading = previous
        self.dirty = False
        self._update_sections()
        self._update_resolution()
        self.dirty_changed.emit(False)

    def _changed(self, *args):
        if self._loading:
            return
        self.dirty = True
        self._update_sections()
        self._update_resolution()
        self.dirty_changed.emit(True)

    def _update_sections(self):
        sine = self.test_mode == 'sine_sweep'
        for key in ('target', 'control', 'equalizer'):
            self.sections[key][0].setVisible(not sine)
        for key in ('sine', 'drive'):
            self.sections[key][0].setVisible(sine)
        self.sections['hardware'][0].setVisible(not self.simulation_mode_check.isChecked())
        self.sections['simulation'][0].setVisible(self.simulation_mode_check.isChecked())
        self.fields['sine_g_level'].setSuffix(' g RMS' if self.sine_unit_combo.currentIndex() else ' g peak')

    def _update_resolution(self):
        fs = self.fields['fs'].value()
        segment = self.fields['welch_nperseg'].value()
        self.resolution_label.setText(f'PSD bin spacing {fs / segment:.3g} Hz · segment duration {segment / fs:.3g} s')

    def set_mode(self, mode):
        self.test_mode = mode
        self._changed()

    def set_backend(self, simulation):
        self.simulation_mode_check.setChecked(simulation)
        self._update_sections()

    def collect_values(self):
        values = self.shared_config.as_dict()
        parse_errors = {}
        for key, widget in self.fields.items():
            if key in ('target_psd_points', 'sine_drive_table'):
                try:
                    values[key] = parse_points(widget.toPlainText())
                except ValueError as error:
                    parse_errors[key] = str(error)
            elif key == 'input_channel_labels':
                values[key] = [label.strip() for label in widget.text().split(',') if label.strip()]
            elif isinstance(widget, QLineEdit):
                values[key] = widget.text().strip()
            elif isinstance(widget, QCheckBox):
                values[key] = widget.isChecked()
            else:
                value = widget.value()
                if key in ('initial_level_fraction','max_level_fraction_rate','sine_initial_level','sine_max_level_rate'):
                    value /= 100
                elif key == 'accel_excitation_amps':
                    value /= 1000
                values[key] = value
        values['test_mode'] = self.test_mode
        values['simulation_mode'] = self.simulation_mode_check.isChecked()
        values['sine_g_level_is_rms'] = self.sine_unit_combo.currentIndex() == 1
        return values, parse_errors

    def show_errors(self, errors):
        self.validation_errors = dict(errors)
        for key, label in self.error_labels.items():
            label.setText(errors.get(key, ''))
            label.setVisible(key in errors)
            self.fields[key].setAccessibleDescription(errors.get(key, ''))
        for section, keys in {'target':['target_psd_points'], 'drive':['sine_drive_table'], 'acquisition':['fs','block_seconds','buf_seconds','welch_nperseg'], 'limits':['ao_volt_limit','max_rms_volts','max_crest_factor']}.items():
            if any(key in errors for key in keys):
                self.sections[section][1].setChecked(True)
        self.validation_label.setText('\n'.join(dict.fromkeys(errors.values())) if errors else ('Run settings are locked.' if self.locked else 'Setup applied · ready to start.'))
        self.validation_label.setStyleSheet('color: #bd493c;' if errors else '')

    def apply_config(self):
        if self.locked:
            return False
        values, errors = self.collect_values()
        errors.update(validate_settings(values))
        self.show_errors(errors)
        if errors:
            self.dirty_changed.emit(True)
            return False
        try:
            self.shared_config.apply_values(values)
        except ConfigurationError as error:
            self.show_errors(error.errors)
            return False
        self.dirty = False
        self.applied.emit(self.shared_config.snapshot())
        self.dirty_changed.emit(False)
        return True

    def set_run_locked(self, locked):
        self.locked = locked
        for widget in self._edit_widgets:
            widget.setEnabled(not locked)
        self.apply_button.setEnabled(not locked)
        self.load_button.setEnabled(not locked)
        self.save_button.setEnabled(not locked)
        if locked or not self.validation_errors:
            self.validation_label.setText('Run settings are locked.' if locked else ('Draft changes · apply before starting.' if self.dirty else 'Setup applied · ready to start.'))

    def load_preset(self):
        if self.locked:
            return
        filename, _ = QFileDialog.getOpenFileName(self, 'Load setup preset', '', 'JSON settings (*.json)')
        if not filename:
            return
        try:
            data = json.loads(Path(filename).read_text())
            if not isinstance(data, dict) or data.get('schema_version', 1) != 1:
                raise ValueError('Use a supported version 1 setup object.')
            values = self.shared_config.as_dict()
            source = data.get('settings', data)
            if not isinstance(source, dict):
                raise ValueError('Preset settings must be an object.')
            values.update({key: value for key, value in source.items() if key in values})
            values['preset_source'] = str(Path(filename).resolve())
            RunSettings.capture(values)
            self.shared_config.apply_values(values)
            self.show_errors({})
            self.load_values()
            self.applied.emit(self.shared_config.snapshot())
        except (ValueError, TypeError, OSError) as error:
            self.validation_label.setText(f'Preset not loaded: {error}')

    def save_preset(self):
        if not self.apply_config():
            return
        filename, _ = QFileDialog.getSaveFileName(self, 'Save setup preset', 'test-setup.json', 'JSON settings (*.json)')
        if filename:
            try:
                Path(filename).write_text(json.dumps({'schema_version':1, 'settings': self.shared_config.as_dict()}, indent=2, allow_nan=False) + '\n')
                self.validation_label.setText(f'Preset saved: {filename}')
            except OSError as error:
                self.validation_label.setText(f'Preset could not be saved: {error}')


class RunView(QWidget):
    def __init__(self, settings):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        values = QHBoxLayout()
        self.value_labels = {}
        self.value_titles = {}
        for key, title in [('accel','Band acceleration'),('level','Ramp level'),('clip','AO clipping'),('frequency','Frequency')]:
            box = QWidget()
            column = QVBoxLayout(box)
            column.setContentsMargins(0, 0, 0, 0)
            heading, value = QLabel(title), QLabel('—')
            value.setStyleSheet('font-size: 20px; font-weight: 500;')
            column.addWidget(heading)
            column.addWidget(value)
            values.addWidget(box)
            self.value_labels[key], self.value_titles[key] = value, heading
        layout.addLayout(values)
        self.detail_label = QLabel('No acquired samples yet.')
        self.detail_label.setWordWrap(True)
        layout.addWidget(self.detail_label)
        self.plot_stack = QStackedWidget()
        self.psd_plot = pg.PlotWidget(title='Control response PSD')
        self.sine_plot = pg.PlotWidget(title='Control acceleration vs frequency')
        for plot in (self.psd_plot, self.sine_plot):
            configure_plot(plot)
            plot.addLegend(brush=pg.mkBrush(QApplication.palette().base().color()))
            self.plot_stack.addWidget(plot)
        self.psd_plot.setLogMode(x=True, y=True)
        self.psd_plot.setLabel('left', 'PSD', units='g²/Hz')
        self.psd_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.psd_measured_curve = self.psd_plot.plot(pen=pg.mkPen(COLORS[0], width=2), name='Control · averaged')
        neutral = QApplication.palette().text().color()
        target_pen = pg.mkPen(neutral, width=2, style=Qt.DashLine)
        ramp_pen = pg.mkPen(COLORS[3], width=1, style=Qt.DotLine)
        self.psd_target_curve = self.psd_plot.plot(pen=target_pen, name='Full target')
        self.psd_ramp_curve = self.psd_plot.plot(pen=ramp_pen, name='Ramp target')
        self.sine_plot.setLogMode(x=True)
        self.sine_plot.setLabel('left', 'Acceleration estimate', units='g peak')
        self.sine_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.sine_measured_curve = self.sine_plot.plot(pen=pg.mkPen(COLORS[0], width=2), symbol='o', symbolSize=5, symbolBrush=COLORS[0], name='Measured · max at frequency')
        self.sine_target_curve = self.sine_plot.plot(pen=target_pen, name='Full target')
        self.sine_ramp_curve = self.sine_plot.plot(pen=ramp_pen, symbol='t', symbolSize=5, name='Ramp target')
        layout.addWidget(self.plot_stack, 1)
        self.note_label = QLabel()
        self.note_label.setWordWrap(True)
        layout.addWidget(self.note_label)

        self.diagnostics_widget = QTabWidget()
        self.metrics_plot = pg.PlotWidget(title='Acceleration trend')
        self.percent_plot = pg.PlotWidget(title='Ramp and command clipping')
        self.plant_gain_plot = pg.PlotWidget(title='Estimated plant gain')
        self.eq_plot = pg.PlotWidget(title='Equalizer gains')
        for plot, title, quantity, unit in [
            (self.metrics_plot, 'Acceleration', 'Acceleration', 'g RMS'),
            (self.percent_plot, 'Ramp / clipping', 'Percentage', '%'),
            (self.plant_gain_plot, 'Plant gain', 'Estimated gain', 'g/V'),
            (self.eq_plot, 'Equalizer', 'Amplitude gain', '×'),
        ]:
            configure_plot(plot)
            plot.setLabel('left', quantity, units=unit)
            plot.setLabel('bottom', 'Elapsed run time', units='s')
            plot.addLegend(brush=pg.mkBrush(QApplication.palette().base().color()))
            self.diagnostics_widget.addTab(plot, title)
        self.eq_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.rms_curve = self.metrics_plot.plot(pen=pg.mkPen(COLORS[0], width=2), name='Measured')
        self.level_curve = self.percent_plot.plot(pen=pg.mkPen(COLORS[0], width=2), name='Ramp level')
        self.sat_curve = self.percent_plot.plot(pen=pg.mkPen(COLORS[3], width=2), name='Command clipping')
        self.plant_gain_curve = self.plant_gain_plot.plot(pen=pg.mkPen(COLORS[0], width=2), name='Estimated gain')
        self.eq_unity_curve = self.eq_plot.plot(pen=pg.mkPen(neutral, style=Qt.DashLine), name='Unity gain')
        self.eq_bargraph = None
        self.history = deque(maxlen=600)
        self.sine_points = {}
        self.latest_psd = None
        self.configure_settings(settings)

    def configure_settings(self, settings):
        self.settings = settings
        self.history.clear()
        self.sine_points.clear()
        self.latest_psd = None
        for curve in (self.psd_measured_curve,self.psd_target_curve,self.psd_ramp_curve,self.sine_measured_curve,self.sine_target_curve,self.sine_ramp_curve,self.rms_curve,self.level_curve,self.sat_curve,self.plant_gain_curve):
            curve.clear()
        self.update_eq(None)
        sine = settings.test_mode == 'sine_sweep'
        self.plot_stack.setCurrentIndex(1 if sine else 0)
        self.value_titles['accel'].setText('Acceleration · g peak estimate' if sine else 'Control band · g RMS')
        self.value_titles['frequency'].parentWidget().setVisible(sine)
        self.metrics_plot.setLabel('left', 'Acceleration estimate' if sine else 'Band acceleration', units='g peak' if sine else 'g RMS')
        self.diagnostics_widget.setTabVisible(3, not sine)
        for label in self.value_labels.values():
            label.setText('—')
        self.detail_label.setText('No acquired samples yet.')
        if sine:
            self.band = sorted([settings.sine_start_hz, settings.sine_end_hz])
            if self.band[0] == self.band[1]:
                self.band = [self.band[0] / 1.1, self.band[1] * 1.1]
            target = settings.target_rms_g * math.sqrt(2)
            self.sine_target_curve.setData(self.band, [target, target])
            self.sine_plot.setXRange(math.log10(self.band[0]), math.log10(self.band[1]), padding=.03)
            frequency_ticks(self.sine_plot, *self.band)
            source = settings.calibration_source or 'Calibration source not recorded'
            drive = 'Drive-table amplitude' if settings.sine_drive_table else 'Default drive amplitude'
            self.note_label.setText(f'{drive} · {source}. Peak estimate = √2 × block standard deviation; noise and harmonics affect it.')
        else:
            self.band = [settings.target_psd_points[0][0], settings.target_psd_points[-1][0]]
            f = np.geomspace(*self.band, 256)
            knots = np.asarray(settings.target_psd_points)
            target = 10 ** np.interp(np.log10(f), np.log10(knots[:, 0]), np.log10(knots[:, 1]))
            self.psd_target_curve.setData(f, target)
            self.psd_plot.setXRange(math.log10(self.band[0]), math.log10(self.band[1]), padding=.03)
            frequency_ticks(self.psd_plot, *self.band)
            self.note_label.setText('10% ramp requests 1% of the full PSD. Random commands have a 0.05 V RMS floor before limiting. Band RMS excludes frequencies outside the target band.')
        self._last_level = settings.sine_initial_level if sine else settings.initial_level_fraction

    def update_psd(self, packet):
        if self.settings.test_mode != 'random':
            self.psd_measured_curve.clear()
            self.psd_target_curve.clear()
            self.psd_ramp_curve.clear()
            return
        f, measured, target, averaged = packet
        self.latest_psd = (f, target)
        set_curve(self.psd_measured_curve, f, averaged if averaged is not None else measured, self.band)
        set_curve(self.psd_target_curve, f, target, self.band)
        set_curve(self.psd_ramp_curve, f, target * self._last_level ** 2 if target is not None else None, self.band)

    def update_metrics(self, metrics):
        self.history.append(dict(metrics))
        sine = self.settings.test_mode == 'sine_sweep'
        acceleration = metrics['accel_peak_g'] if sine else metrics['accel_rms_g']
        self.value_labels['accel'].setText(f'{acceleration:.3f}')
        self.value_labels['level'].setText(f'{100 * metrics["level_fraction"]:.1f}%')
        self.value_labels['clip'].setText(f'{100 * metrics.get("clipping_fraction", 0):.2f}%')
        self._last_level = metrics['level_fraction']
        voltage_note = f'Command {metrics["output_rms_v"]:.3f} V RMS · crest factor {metrics["output_cf"]:.2f}'
        if metrics.get('limiter_active') or metrics.get('clipping_fraction', 0):
            voltage_note += ' · command limiting active'
        self.detail_label.setText(f'Elapsed {metrics["elapsed_s"]:.1f} s · {voltage_note}')
        if sine:
            frequency = metrics['frequency_hz']
            self.value_labels['frequency'].setText(f'{frequency:.1f} Hz')
            old = self.sine_points.get(frequency, (0, 0))
            self.sine_points[frequency] = (max(old[0], metrics['accel_peak_g']), max(old[1], metrics['target_peak_g']))
            frequencies = sorted(self.sine_points)
            self.sine_measured_curve.setData(frequencies, [self.sine_points[f][0] for f in frequencies])
            self.sine_ramp_curve.setData(frequencies, [self.sine_points[f][1] for f in frequencies])
            self.detail_label.setText(f'Frequency progress {100 * metrics["sweep_progress"]:.1f}% · ' + self.detail_label.text())
        elif self.latest_psd is not None:
            f, target = self.latest_psd
            set_curve(self.psd_ramp_curve, f, target * self._last_level ** 2 if target is not None else None, self.band)
        times = [item['elapsed_s'] for item in self.history]
        self.rms_curve.setData(times, [item['accel_peak_g'] if sine else item['accel_rms_g'] for item in self.history])
        self.level_curve.setData(times, [item['level_fraction'] * 100 for item in self.history])
        self.sat_curve.setData(times, [item.get('clipping_fraction',0) * 100 for item in self.history])
        self.plant_gain_curve.setData(times, [item['plant_gain_g_per_v'] for item in self.history])

    def update_eq(self, packet):
        if packet is None or self.settings.test_mode != 'random':
            if self.eq_bargraph is not None:
                self.eq_plot.getPlotItem().legend.removeItem('Band gains')
                self.eq_plot.removeItem(self.eq_bargraph)
                self.eq_bargraph = None
            self.eq_unity_curve.clear()
            return
        centers, gains = packet
        edges = np.geomspace(self.settings.target_psd_points[0][0], self.settings.target_psd_points[-1][0], len(gains) + 1)
        if self.eq_bargraph is None:
            self.eq_bargraph = pg.BarGraphItem(x=(edges[:-1]+edges[1:])/2, height=gains, width=np.diff(edges)*.9, brush=COLORS[0])
            self.eq_plot.addItem(self.eq_bargraph)
            self.eq_plot.getPlotItem().legend.addItem(self.eq_bargraph, 'Band gains')
        else:
            self.eq_bargraph.setOpts(height=gains)
        self.eq_plot.setXRange(edges[0], edges[-1], padding=.02)
        self.eq_unity_curve.setData([edges[0],edges[-1]], [1,1])


class ChannelView(QWidget):
    def __init__(self, settings):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        controls = QHBoxLayout()
        controls.addWidget(QLabel('Channel'))
        self.channel_combo = QComboBox()
        self.channel_combo.setAccessibleName('Displayed channel')
        self.channel_combo.currentIndexChanged.connect(self._filter_channels)
        controls.addWidget(self.channel_combo)
        self.info_label = QLabel()
        controls.addWidget(self.info_label)
        controls.addStretch()
        layout.addLayout(controls)
        self.time_plot = pg.PlotWidget(title='Acquired acceleration')
        self.time_plot.setLabel('left','Acceleration', units='g')
        self.time_plot.setLabel('bottom','Acquired sample time', units='s')
        self.psd_plot = pg.PlotWidget(title='Channel PSD')
        self.psd_plot.setLabel('left','PSD',units='g²/Hz')
        self.psd_plot.setLabel('bottom','Frequency',units='Hz')
        self.psd_plot.setLogMode(x=True,y=True)
        for plot in (self.time_plot,self.psd_plot):
            configure_plot(plot)
            plot.addLegend(brush=pg.mkBrush(QApplication.palette().base().color()))
            layout.addWidget(plot,1)
        self.is_streaming = False
        self.configure_settings(settings)

    def configure_settings(self, settings):
        self.settings = settings
        self.fs = settings.fs
        self.num_channels = settings.num_input_channels
        self.channel_labels = list(settings.input_channel_labels)
        self.samples = 0
        self.blocks = 0
        self.buffer = np.empty((self.num_channels, 0))
        self.psd_averaged = [None] * self.num_channels
        for plot in (self.time_plot,self.psd_plot):
            plot.clear()
            plot.getPlotItem().legend.clear()
        with QSignalBlocker(self.channel_combo):
            self.channel_combo.clear()
            self.channel_combo.addItems(['All channels'] + self.channel_labels)
        self.time_curves, self.psd_curves = [], []
        for index, label in enumerate(self.channel_labels):
            color = COLORS[index % len(COLORS)]
            curve = self.time_plot.plot(pen=pg.mkPen(color, width=1.5),name=label)
            curve.setDownsampling(auto=True, method='peak')
            curve.setClipToView(True)
            self.time_curves.append(curve)
            self.psd_curves.append(self.psd_plot.plot(pen=pg.mkPen(color,width=1.5),name=label))
        self.target_curve = self.psd_plot.plot(pen=pg.mkPen(QApplication.palette().text().color(),style=Qt.DashLine),name='Control full target')
        self.info_label.setText(f'{self.fs:g} Hz · {self.num_channels} channel(s) · PSD bin spacing {self.fs/settings.welch_nperseg:g} Hz')
        low, high = self.fs/settings.welch_nperseg, self.fs/2
        if low >= high:
            low = high/10
        self.psd_plot.setXRange(math.log10(low),math.log10(high),padding=.01)
        frequency_ticks(self.psd_plot,low,high)

    def _filter_channels(self):
        selected = self.channel_combo.currentIndex() - 1
        for index, (time_curve, psd_curve) in enumerate(zip(self.time_curves,self.psd_curves)):
            time_curve.setVisible(selected < 0 or selected == index)
            psd_curve.setVisible(selected < 0 or selected == index)
        if hasattr(self,'target_curve'):
            self.target_curve.setVisible(self.settings.test_mode == 'random' and selected in (-1,0))

    def start_streaming(self):
        self.is_streaming = True

    def stop_streaming(self):
        self.is_streaming = False

    def add_data_point(self, block):
        if not self.is_streaming:
            return
        block = np.atleast_2d(np.asarray(block))
        if block.shape[0] != self.num_channels:
            raise ValueError('Acquisition channel count does not match run settings')
        self.samples += block.shape[1]
        self.blocks += 1
        capacity = max(1, int(self.fs * 2))
        self.buffer = np.concatenate((self.buffer,block),axis=1)[:, -capacity:]
        times = (self.samples - self.buffer.shape[1] + np.arange(self.buffer.shape[1])) / self.fs
        for index, curve in enumerate(self.time_curves):
            curve.setData(times,self.buffer[index])
        self.time_plot.setXRange(max(0,self.samples/self.fs-2),self.samples/self.fs,padding=0)
        if self.blocks == 1 or self.blocks % max(1,self.settings.realtime_psd_update_stride) == 0:
            for index in range(1,self.num_channels):
                frequencies, psd = welch(block[index],fs=self.fs,nperseg=self.settings.welch_nperseg)
                previous = self.psd_averaged[index]
                psd = psd if previous is None else .1*psd+.9*previous
                self.psd_averaged[index] = psd
                mask = (frequencies > 0) & (psd > 0)
                self.psd_curves[index].setData(frequencies[mask],psd[mask])

    def update_psd_data(self, packet):
        frequencies, measured, target, averaged = packet
        psd = averaged if averaged is not None else measured
        mask = (frequencies > 0) & (psd > 0) & np.isfinite(psd)
        self.psd_curves[0].setData(frequencies[mask],psd[mask])
        if self.settings.test_mode == 'random' and target is not None:
            set_curve(self.target_curve, frequencies,target)
        else:
            self.target_curve.clear()
        self._filter_channels()


class ResultsView(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.summary = QLabel('Run results will retain the effective backend and settings.')
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.run_list = QListWidget()
        self.run_list.setMaximumHeight(150)
        self.run_list.currentRowChanged.connect(self.show_report)
        layout.addWidget(self.run_list)
        self.tabs = QTabWidget()
        self.summary_text = QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        self.tabs.addTab(self.summary_text,'Run summary')
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.tabs.addTab(self.details,'Captured settings / metadata')
        layout.addWidget(self.tabs,1)
        self.open_button = QPushButton('Open recording folder')
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self.open_folder)
        layout.addWidget(self.open_button)
        self.reports = []

    def add_report(self, report):
        self.reports.append(report)
        mode = 'Sine sweep' if report['settings']['test_mode'] == 'sine_sweep' else 'Random vibration'
        self.run_list.addItem(f'{report["started_at"]} · {report.get("backend") or report["requested_backend"]} · {mode} · {report["outcome"].title()}')
        self.run_list.setCurrentRow(len(self.reports)-1)

    def show_report(self, index):
        if not 0 <= index < len(self.reports):
            return
        report = self.reports[index]
        backend = report['backend'] or f'{report["requested_backend"]} requested · not initialized'
        self.summary.setText(f'{report["outcome"].title()} · {backend} · {report["blocks"]} blocks · {report["duration_s"]:.1f} s · setup {report["config_id"]}')
        settings = RunSettings.capture(report['settings'])
        metrics = report.get('latest_metrics') or {}
        measured = (f'{metrics["accel_peak_g"]:.4g} g peak estimate' if 'accel_peak_g' in metrics else f'{metrics["accel_rms_g"]:.4g} g RMS in the control band' if 'accel_rms_g' in metrics else 'No acquired samples')
        lines = [f'Outcome: {report["outcome"].title()}',f'Backend: {backend}',
            f'Target: {settings.target_summary}',f'Measured (last block): {measured}',
            f'Channels: {", ".join(settings.input_channel_labels)}',
            f'Acquisition: {settings.fs:g} Hz, {settings.block_seconds:g} s blocks',
            f'Input: {report.get("input_mode") or "Not initialized"}',
            f'Clock routing: {report.get("clock_status", "Not recorded")}',
            f'Command limits: ±{settings.ao_volt_limit:g} V, {settings.max_rms_volts:g} V RMS, crest factor {settings.max_crest_factor:g}',
            f'Calibration source: {settings.calibration_source or "Not recorded"}',
            f'Preset source: {settings.preset_source or "Session setup / config.py defaults"}',
            f'Recording: {report["log_dir"] or "Disabled or not initialized"}']
        if report['error']:
            lines.insert(1,f'Fault: {report["error"]}')
        self.summary_text.setPlainText('\n\n'.join(lines))
        self.details.setPlainText(json.dumps(report,indent=2,allow_nan=False))
        self.open_button.setEnabled(bool(report['log_dir']))

    def open_folder(self):
        index = self.run_list.currentRow()
        if 0 <= index < len(self.reports) and self.reports[index]['log_dir']:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.reports[index]['log_dir']).resolve())))


class WorkspaceWindow(QMainWindow):
    def __init__(self, shared_config, worker_factory):
        super().__init__()
        self.setWindowTitle('Shaker Control')
        self.resize(1200,800)
        self.shared_config, self.worker_factory = shared_config, worker_factory
        self.settings = shared_config.snapshot()
        self.controller_thread = None
        self.controller_worker = None
        self.run_state = 'ready'
        self.actual_backend = None
        self.active_log_dir = None
        self.input_mode = None
        self.clock_status = None
        self.last_report = None
        self.fault_message = ''
        self._close_pending = False
        pg.setConfigOptions(antialias=True,foreground=QApplication.palette().text().color(),background=QApplication.palette().base().color())
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(18,14,18,14)
        title_row = QHBoxLayout()
        title = QLabel('Shaker Control')
        title.setStyleSheet('font-size: 20px; font-weight: 500;')
        title_row.addWidget(title)
        title_row.addStretch()
        self.backend_label = QLabel()
        title_row.addWidget(self.backend_label)
        self.status_label = QLabel('Ready')
        self.status_label.setAccessibleName('Run status')
        self.status_label.setStyleSheet('font-weight: 600;')
        title_row.addWidget(self.status_label)
        layout.addLayout(title_row)
        run_bar = QHBoxLayout()
        run_bar.addWidget(QLabel('Test'))
        self.mode_combo = QComboBox()
        self.mode_combo.setAccessibleName('Test mode')
        self.mode_combo.addItem('Random vibration','random')
        self.mode_combo.addItem('Sine sweep','sine_sweep')
        self.mode_combo.setCurrentIndex(1 if self.settings.test_mode == 'sine_sweep' else 0)
        run_bar.addWidget(self.mode_combo)
        run_bar.addWidget(QLabel('Backend'))
        self.backend_combo = QComboBox()
        self.backend_combo.setAccessibleName('Acquisition backend')
        self.backend_combo.addItem('Hardware',False)
        self.backend_combo.addItem('Simulation',True)
        self.backend_combo.setCurrentIndex(1 if self.settings.simulation_mode else 0)
        run_bar.addWidget(self.backend_combo)
        run_bar.addStretch()
        self.setup_button = QPushButton('Set up test')
        self.setup_button.setCheckable(True)
        run_bar.addWidget(self.setup_button)
        self.start_button = QPushButton()
        self.stop_button = QPushButton('Stop')
        self.stop_button.setStyleSheet('QPushButton:enabled { color: #bd493c; }')
        run_bar.addWidget(self.start_button)
        run_bar.addWidget(self.stop_button)
        layout.addLayout(run_bar)
        self.setup_summary = QLabel()
        self.setup_summary.setWordWrap(True)
        layout.addWidget(self.setup_summary)
        self.notice_label = QLabel()
        self.notice_label.setWordWrap(True)
        self.notice_label.hide()
        layout.addWidget(self.notice_label)
        self.fault_banner = QWidget()
        fault_layout = QHBoxLayout(self.fault_banner)
        fault_layout.setContentsMargins(0,5,0,5)
        self.fault_label = QLabel()
        self.fault_label.setWordWrap(True)
        self.fault_label.setStyleSheet('color: #bd493c;')
        fault_layout.addWidget(self.fault_label,1)
        self.switch_simulation_button = QPushButton('Switch to simulation')
        self.switch_simulation_button.clicked.connect(self.switch_to_simulation)
        fault_layout.addWidget(self.switch_simulation_button)
        self.dismiss_fault_button = QPushButton('Dismiss fault')
        self.dismiss_fault_button.clicked.connect(self.dismiss_fault)
        fault_layout.addWidget(self.dismiss_fault_button)
        layout.addWidget(self.fault_banner)
        self.fault_banner.hide()

        self.controller_tab = RunView(self.settings)
        self.realtime_tab = ChannelView(self.settings)
        self.results_tab = ResultsView()
        self.pages = QStackedWidget()
        nav = QHBoxLayout()
        self.nav_buttons = []
        for index, (label, page) in enumerate([
            ('Run',self.controller_tab),('Channels',self.realtime_tab),
            ('Diagnostics',self.controller_tab.diagnostics_widget),('Results',self.results_tab),
        ]):
            self.pages.addWidget(page)
            button = QPushButton(label)
            button.setCheckable(True)
            button.clicked.connect(lambda checked, index=index: self.show_page(index))
            self.nav_buttons.append(button)
            nav.addWidget(button)
        nav.addStretch()
        layout.addLayout(nav)
        layout.addWidget(self.pages,1)
        self.show_page(0)

        self.config_tab = SettingsPanel(shared_config)
        self.config_tab.test_mode_combo = self.mode_combo
        self.config_dock = QDockWidget('Test setup',self)
        self.config_dock.setObjectName('test-setup-dock')
        self.config_dock.setAllowedAreas(Qt.RightDockWidgetArea|Qt.LeftDockWidgetArea)
        self.config_dock.setFeatures(QDockWidget.DockWidgetClosable)
        self.config_dock.setWidget(self.config_tab)
        self.config_dock.setMinimumWidth(340)
        self.addDockWidget(Qt.RightDockWidgetArea,self.config_dock)
        self.config_dock.hide()
        self.config_dock.visibilityChanged.connect(self._dock_visible)
        self.setup_button.toggled.connect(self.config_dock.setVisible)
        self.config_tab.applied.connect(self._settings_applied)
        self.config_tab.dirty_changed.connect(self._draft_changed)
        self.mode_combo.currentIndexChanged.connect(lambda: self.config_tab.set_mode(self.mode_combo.currentData()))
        self.backend_combo.currentIndexChanged.connect(lambda: self.config_tab.set_backend(self.backend_combo.currentData()))
        self.start_button.clicked.connect(self.start_controller)
        self.stop_button.clicked.connect(self.stop_controller)
        self._refresh_controls()

    @property
    def busy(self):
        return self.controller_thread is not None or self.run_state in BUSY_STATES

    def _dock_visible(self, visible):
        with QSignalBlocker(self.setup_button):
            self.setup_button.setChecked(visible)

    def show_page(self, index):
        self.pages.setCurrentIndex(index)
        for position, button in enumerate(self.nav_buttons):
            button.setChecked(position == index)

    def _settings_applied(self, settings):
        if self.busy:
            return
        self.settings = settings
        self.actual_backend = None
        self.active_log_dir = None
        self.input_mode = None
        self.clock_status = None
        self.controller_tab.configure_settings(settings)
        self.realtime_tab.configure_settings(settings)
        with QSignalBlocker(self.mode_combo), QSignalBlocker(self.backend_combo):
            self.mode_combo.setCurrentIndex(1 if settings.test_mode == 'sine_sweep' else 0)
            self.backend_combo.setCurrentIndex(1 if settings.simulation_mode else 0)
        if not self.fault_message:
            self.run_state = 'ready'
        self._refresh_controls()

    def _draft_changed(self, dirty):
        self._refresh_controls()

    def _refresh_controls(self):
        busy = self.busy
        dirty = self.config_tab.dirty if hasattr(self,'config_tab') else False
        self.start_button.setText('Start simulation' if self.settings.simulation_mode else 'Start hardware')
        self.start_button.setEnabled(not busy and not dirty)
        self.stop_button.setEnabled(self.run_state in ('starting','running'))
        self.mode_combo.setEnabled(not busy)
        self.backend_combo.setEnabled(not busy)
        if hasattr(self,'config_tab'):
            self.config_tab.set_run_locked(busy)
        names = {'ready':'Ready','starting':'Starting…','running':'Running','stopping':'Stopping…','stopped':'Stopped','complete':'Complete','fault':'Fault'}
        self.status_label.setText('Draft changes · apply setup' if dirty and not busy and not self.fault_message else names[self.run_state])
        backend = self.actual_backend or f'{self.settings.backend} requested · not initialized'
        if dirty and not busy:
            backend = 'Draft backend · apply setup'
        self.backend_label.setText(backend)
        channel = f'{self.settings.num_input_channels} simulated channel(s)' if self.settings.simulation_mode else self.settings.device_ai
        recording = (f'Recording {"to" if busy else "saved in"} {self.active_log_dir}' if self.active_log_dir else
                     ('Recording requested · not started' if self.settings.data_log_enabled else 'Recording disabled'))
        self.setup_summary.setText(f'Applied setup: {self.settings.target_summary}  |  {channel}  |  {self.settings.fs:g} Hz · {self.settings.block_seconds:g} s blocks · PSD {self.settings.fs/self.settings.welch_nperseg:g} Hz bins  |  {recording}')
        self.fault_banner.setVisible(bool(self.fault_message))
        self.fault_label.setText(self.fault_message)
        self.dismiss_fault_button.setEnabled(not busy)
        self.switch_simulation_button.setVisible(bool(self.fault_message) and not self.settings.simulation_mode)
        self.switch_simulation_button.setEnabled(not busy)

    def start_controller(self):
        if self.busy or self.config_tab.dirty:
            self.config_dock.show()
            return False
        try:
            snapshot = self.shared_config.snapshot()
        except ConfigurationError as error:
            self.config_dock.show()
            self.config_tab.show_errors(error.errors)
            return False
        self.settings = snapshot
        self.fault_message = ''
        self.actual_backend = None
        self.active_log_dir = None
        self.input_mode = None
        self.clock_status = None
        self.notice_label.hide()
        self.controller_tab.configure_settings(snapshot)
        self.realtime_tab.configure_settings(snapshot)
        self.realtime_tab.start_streaming()
        self.controller_worker = self.worker_factory(snapshot)
        self.controller_thread = self.controller_worker
        worker, thread = self.controller_worker, self.controller_thread
        worker.state_changed.connect(self._worker_state)
        worker.status_message.connect(self._worker_notice)
        worker.psd_data_ready.connect(self.controller_tab.update_psd)
        worker.psd_data_ready.connect(self.realtime_tab.update_psd_data)
        worker.metrics_data_ready.connect(self.controller_tab.update_metrics)
        worker.eq_data_ready.connect(self.controller_tab.update_eq)
        worker.realtime_data_ready.connect(self.realtime_tab.add_data_point)
        worker.report_ready.connect(self._worker_report)
        thread.finished.connect(self._thread_finished)
        self.run_state = 'starting'
        self._refresh_controls()
        self.show_page(0)
        thread.start()
        return True

    def stop_controller(self):
        if self.controller_worker is None or self.run_state not in ('starting','running'):
            return False
        self.run_state = 'stopping'
        self.controller_worker.stop()
        self.notice_label.setText('Stop requested · waiting for acquisition and recording to finish.')
        self.notice_label.show()
        self._refresh_controls()
        return True

    def _worker_state(self, event):
        if event['backend']:
            self.actual_backend = event['backend']
        self.active_log_dir = event.get('log_dir')
        self.input_mode = event.get('input_mode')
        self.clock_status = event.get('clock_status')
        state = event['state']
        if self.run_state == 'stopping' and state in ('starting','running'):
            return
        self.run_state = state
        if state == 'fault':
            self.fault_message = event['message']
        self._refresh_controls()

    def _worker_notice(self, message):
        if self.fault_message or message in ('Run stopped','Sweep complete'):
            return
        details = ' · '.join(filter(None, (self.input_mode, self.clock_status)))
        self.notice_label.setText(f'{message} · {details}' if details else message)
        self.notice_label.show()

    def _worker_report(self, report):
        self.last_report = report
        self.results_tab.add_report(report)
        self.realtime_tab.stop_streaming()
        self.run_state = report['outcome']
        if report['error']:
            self.fault_message = report['error']
            self.notice_label.hide()
        else:
            self.notice_label.setText(('Sweep complete' if report['outcome'] == 'complete' else 'Run stopped') + ' · review this run in Results.')
            self.notice_label.show()
        self._refresh_controls()

    def _thread_finished(self):
        # finished is emitted before the native thread's final teardown. Keep
        # both Python wrappers alive until that teardown has completed.
        thread = self.controller_thread
        if thread is not None:
            thread.wait()
        self.controller_worker = None
        self.controller_thread = None
        if thread is not None:
            thread.deleteLater()
        self._refresh_controls()
        if self._close_pending:
            self.close()

    def dismiss_fault(self):
        if not self.busy:
            self.fault_message = ''
            self.run_state = 'ready'
            self.notice_label.hide()
            self._refresh_controls()

    def switch_to_simulation(self):
        if self.busy:
            return
        self.backend_combo.setCurrentIndex(1)
        self.config_dock.show()
        self.notice_label.setText('Simulation selected in the draft. Apply setup before starting.')
        self.notice_label.show()

    def closeEvent(self, event):
        if self.busy:
            self._close_pending = True
            self.stop_controller()
            event.ignore()
        else:
            event.accept()
