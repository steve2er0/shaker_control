"""Native Qt workspace with validated setup and an authoritative run lifecycle."""
from __future__ import annotations

import json
import math
from html import escape
from collections import deque
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from scipy.signal import welch
from PySide6.QtCore import Qt, Signal, QUrl, QSignalBlocker, QTimer
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox,
    QLineEdit, QPlainTextEdit, QScrollArea, QToolButton,
    QStackedWidget, QTabWidget, QFileDialog, QListWidget, QApplication,
)
from run_settings import (RunSettings, ConfigurationError, validate_settings, parse_points, format_points,
                          parse_response_channels, format_response_channels)
from workspace_shell import (
    FloatingWorkspaceMixin, workspace_colors,
)
from run_reports import render_report_chart
from channel_display import ChannelMailbox, peak_envelope


COLORS = ['#287bb5', '#9264ad', '#298a6b', '#b97327', '#bd5673', '#577b85']
BUSY_STATES = {'starting', 'running', 'stopping'}


class ReadableAxisItem(pg.AxisItem):
    def logTickStrings(self, values, scale, spacing):
        labels = super().logTickStrings(values, scale, spacing)
        mantissas = 10 ** np.mod(np.asarray(values, dtype=float), 1)
        keep = np.any(np.isclose(mantissas[:, None], [1, 2, 5, 10]), axis=1)
        # Retain the standard narrow-range labels when filtering would leave
        # too little scale information; otherwise use familiar 1, 2, 5 ticks.
        if np.count_nonzero(keep) >= 2:
            return [label if visible else '' for label, visible in zip(labels, keep)]
        return labels


def create_plot(**kwargs):
    kwargs.setdefault('axisItems', {'left': ReadableAxisItem(orientation='left')})
    return pg.PlotWidget(**kwargs)


def configure_plot(widget):
    colors = workspace_colors()
    text = colors['muted']
    widget.setBackground(colors['paper'])
    for name in ('left', 'bottom'):
        axis = widget.getAxis(name)
        axis.setPen(text)
        axis.setTextPen(text)
        axis.enableAutoSIPrefix(False)
        if name == 'left':
            # Minor logarithmic labels can overlap across a wide PSD range.
            axis.setStyle(maxTextLevel=0)
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
    discarded = Signal()
    context_changed = Signal(str)
    calibration_requested = Signal()

    def __init__(self, shared_config):
        super().__init__()
        self.shared_config = shared_config
        self._loading = True
        self.locked = False
        self.dirty = False
        self.validation_errors = {}
        self.context = 'all'
        self.field_sections = {}
        self.fields, self.error_labels, self.sections = {}, {}, {}
        self._edit_widgets = []
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        self.context_summary = QLabel()
        self.context_summary.setObjectName('mutedText')
        self.context_summary.setWordWrap(True)
        outer.addWidget(self.context_summary)
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
        self._section('drive', 'Sine drive calibration')
        self.calibration_button = QPushButton('Calibration wizard…')
        self.calibration_button.clicked.connect(self.calibration_requested)
        self.sections['drive'][2].addRow(self.calibration_button)
        self._edit_widgets.append(self.calibration_button)
        self._note('drive', 'Table units: V peak per 1 g peak.')
        self._text('calibration_source', 'Calibration source', 'drive')
        self._text('sine_drive_table', 'Frequency, V peak pairs', 'drive', multiline=True)
        self._spin('sine_default_vpk', 'Default drive', 'drive', 0, 10, ' V peak/g')
        self._spin('sine_drive_scale', 'Drive scale', 'drive', 0, 100, ' ×')
        self._spin('sine_points_per_octave', 'Frequency spacing', 'drive', 1, 100, ' points/octave')
        self._spin('sine_step_dwell', 'Minimum dwell', 'drive', .01, 60, ' s')
        self._text('calibration_equipment', 'Shaker / fixture / load', 'drive')
        self._text('calibration_amp_setting', 'Amplifier setting', 'drive')
        self._text('calibration_sensor_id', 'Control sensor ID', 'drive')
        self.calibration_range_label = QLabel()
        self.calibration_range_label.setWordWrap(True)
        self.sections['drive'][2].addRow(self.calibration_range_label)
        self._section('sine_feedback', 'Sine amplitude feedback')
        self._check('sine_closed_loop', 'Enable sine amplitude feedback', 'sine_feedback')
        self._spin('sine_pilot_vpk', 'Initial pilot ceiling', 'sine_feedback', .000001, 1, ' V peak', 6)
        self._spin('sine_correction_rate', 'Maximum drive increase', 'sine_feedback', .001, 5, ' fraction/s', 3)
        self._spin('sine_settle_seconds', 'Settling before assessment', 'sine_feedback', 0, 120, ' s')
        self._spin('sine_settle_blocks', 'Consecutive tracking blocks', 'sine_feedback', 1, 100, '', integer=True)
        self._spin('sine_max_step_seconds', 'Maximum settling time', 'sine_feedback', 1, 600, ' s')
        self._note('sine_feedback', 'Feedback fits the fundamental tone. Dwell extends until full-target tracking settles; failure to settle faults the run.')

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
        self._spin('accel_mv_per_g', 'Control sensitivity', 'hardware', .001, 10000, ' mV/g', 3)
        self._spin('accel_excitation_amps', 'Control IEPE current', 'hardware', 0, 100, ' mA', 3)
        self._text('response_channels', 'Response inputs · channel, label, mV/g, mA', 'hardware', multiline=True)
        self.fields['response_channels'].setPlaceholderText('Dev1/ai1, Response 1, 100, 2.1')
        self._check('sync_ao_clock', 'Share AI clock with AO', 'hardware')
        self._note('hardware', 'The control input drives feedback. Response inputs are displayed and recorded in row order. '
                   'Enter one response per line; remove a line to disable that input. Use each sensor’s sensitivity; 0 mA selects external power.')
        self._section('simulation', 'Simulation model')
        self._text('input_channel_labels', 'Channel labels, comma separated', 'simulation')
        self._spin('sim_plant_gain', 'Plant gain', 'simulation', .001, 100, ' g/V', 3)
        self._spin('sim_noise_level', 'Measurement noise', 'simulation', 0, 10, ' g RMS', 4)
        self._note('simulation', 'Synthetic plant with configured resonances and delay; these parameters do not calibrate hardware.')

        self._section('limits', 'Voltage limits', True)
        self._spin('ao_volt_limit', 'Absolute peak limit', 'limits', .001, 10, ' V', 3)
        self._spin('max_rms_volts', 'RMS limit', 'limits', 0, 10, ' V RMS', 3)
        self._spin('max_crest_factor', 'Crest-factor limit', 'limits', 1, 10, ' peak/RMS', 2)
        self._section('response', 'Measured-response protection', True)
        self._check('response_limits_enabled', 'Enable measured-response stops', 'response')
        for key, label, maximum, unit, precision in (
            ('response_max_rms_g', 'All-input broadband RMS limit', 1000, ' g RMS', 3),
            ('response_max_peak_g', 'All-input raw peak limit', 10000, ' g peak', 3),
            ('response_target_ratio', 'Control RMS / full target limit', 100, ' ×', 2),
            ('response_max_backlog_seconds', 'Maximum unread data age', 60, ' s', 3),
            ('feedback_min_rms_g', 'Minimum control feedback', 10, ' g RMS', 6),
            ('feedback_min_drive_v', 'Driven-output threshold', 10, ' V RMS', 6),
            ('feedback_loss_seconds', 'Feedback-loss timeout', 60, ' s', 3),
            ('feedback_startup_seconds', 'Feedback startup allowance', 60, ' s', 3)):
            self._spin(key, label, 'response', 0 if key == 'feedback_startup_seconds' else .000001, maximum, unit, precision)
        self._note('response', 'Checks run on each acquired block. Response limits cover every enabled input; feedback loss uses the control input. Stop latency includes acquisition block time and driver cleanup.')
        self._section('control', 'Random ramp and PI tuning')
        self._spin('initial_level_fraction', 'Ramp start', 'control', 0, 200, ' %', 1)
        self._spin('max_level_fraction_rate', 'Ramp rate limit', 'control', 0, 200, ' %/s', 1)
        self._spin('kp', 'Proportional gain', 'control', 0, 10, '', 3)
        self._spin('ki', 'Integral gain', 'control', 0, 5, '', 3)
        self._spin('random_initial_rms_volts', 'Initial identification drive', 'control', .000001, 1, ' V RMS', 6)
        self._note('control', 'Ramp % scales acceleration amplitude; 10% requests 1% of the full target PSD.')
        self._note('control', 'PI trims normalized RMS error (integral gain per second). The ramp advances after tracking; '
                   'the initial drive identifies plant gain. Output is held if the sensor cannot resolve that response.')
        self._section('equalizer', 'Random equalizer tuning')
        self._spin('eq_num_bands', 'Bands', 'equalizer', 1, 64, '', integer=True)
        self._spin('eq_adapt_rate', 'Adaptation rate', 'equalizer', 0, 1, '', 3)
        self._spin('eq_smooth_factor', 'Smoothing factor', 'equalizer', 0, 1, '', 3)
        self._section('recording', 'Recording')
        self._check('data_log_enabled', 'Record raw blocks and summaries', 'recording')
        self._text('data_log_dir', 'Output directory', 'recording')
        self._check('report_enabled', 'Save HTML / CSV test reports', 'recording')
        self._spin('tracking_tolerance_percent', 'Tracking tolerance', 'recording', .01, 50, ' %', 2)
        self._spin('report_min_seconds', 'Minimum random assessment time', 'recording', .01, 3600, ' s')
        self._spin('report_settle_seconds', 'Random settling exclusion', 'recording', 0, 120, ' s')
        self.form_layout.addStretch()

        self.validation_label = QLabel()
        self.validation_label.setWordWrap(True)
        outer.addWidget(self.validation_label)
        self.apply_button = QPushButton('Apply setup')
        self.apply_button.setObjectName('applyButton')
        self.apply_button.setDefault(True)
        self.apply_button.clicked.connect(self.apply_config)
        apply_row = QHBoxLayout()
        apply_row.addWidget(self.apply_button, 1)
        self.discard_button = QPushButton('Discard draft')
        self.discard_button.clicked.connect(self.discard_changes)
        apply_row.addWidget(self.discard_button)
        outer.addLayout(apply_row)
        actions = QHBoxLayout()
        self.load_button, self.save_button = QPushButton('Load preset…'), QPushButton('Save preset…')
        self.load_button.clicked.connect(self.load_preset)
        self.save_button.clicked.connect(self.save_preset)
        actions.addWidget(self.load_button)
        actions.addWidget(self.save_button)
        outer.addLayout(actions)
        # Preset file actions live in the ribbon. Retain these public widgets
        # for callers of the standalone settings editor without duplicating UI.
        self.load_button.hide()
        self.save_button.hide()
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
        heading = QLabel(label)
        heading.setWordWrap(True)
        self.sections[section][2].addRow(heading, container)
        self.fields[key], self.error_labels[key] = widget, error
        self.field_sections[key] = section
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
            elif key == 'response_channels':
                widget.setPlainText(format_response_channels(value))
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
        low, high = values['calibration_min_hz'], values['calibration_max_hz']
        coverage = (f'Tested calibration: {low:g}–{high:g} Hz at {values["calibration_level_peak_g"]:g} g peak · {values["calibration_backend"]}'
                    if high > 0 else 'Calibration frequency/level coverage has not been recorded.')
        self.calibration_range_label.setText(coverage)
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
        contexts = {
            'target': {'target', 'sine', 'sine_feedback'}, 'calibration': {'drive', 'sine_feedback'},
            'limits': {'limits', 'response', 'control'}, 'equalizer': {'equalizer'},
            'acquisition': {'acquisition'}, 'channel': {'hardware', 'simulation'},
            'simulation': {'simulation'}, 'recording': {'recording'},
        }
        selected = contexts.get(self.context, set(self.sections))
        for key, (wrapper, toggle, form, body) in self.sections.items():
            applicable = not ((key in ('target', 'control', 'equalizer') and sine)
                             or (key in ('sine', 'sine_feedback') and not sine)
                             or (key == 'hardware' and self.simulation_mode_check.isChecked())
                             or (key == 'simulation' and not self.simulation_mode_check.isChecked()))
            # A drive table can be reviewed before selecting sine mode.
            if key == 'drive' and self.context == 'all' and not sine:
                applicable = False
            wrapper.setVisible(applicable and key in selected)
        self.fields['sine_g_level'].setSuffix(' g RMS' if self.sine_unit_combo.currentIndex() else ' g peak')
        accepted = self.shared_config.snapshot()
        self.context_summary.setText(f'Applied setup · {accepted.target_summary}')

    def set_context(self, context):
        """Switch the inspector without replacing widgets or losing a draft."""
        self.context = context
        self._update_sections()
        if context != 'all':
            for wrapper, toggle, form, body in self.sections.values():
                if not wrapper.isHidden():
                    toggle.setChecked(True)
        self.scroll_area.verticalScrollBar().setValue(0)
        self.context_changed.emit(context)

    def discard_changes(self):
        if self.locked:
            return
        self.load_values()
        self.show_errors({})
        self.discarded.emit()

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
            elif key == 'response_channels':
                try:
                    values[key] = parse_response_channels(widget.toPlainText())
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
        if any(self.sections[self.field_sections[key]][0].isHidden()
               for key in errors if key in self.field_sections):
            self.set_context('all')
        for key, label in self.error_labels.items():
            label.setText(errors.get(key, ''))
            label.setVisible(key in errors)
            self.fields[key].setAccessibleDescription(errors.get(key, ''))
        for section, keys in {'target':['target_psd_points'], 'drive':['sine_drive_table'], 'acquisition':['fs','block_seconds','buf_seconds','welch_nperseg'], 'limits':['ao_volt_limit','max_rms_volts','max_crest_factor'], 'hardware':['device_ai','device_ao','accel_mv_per_g','accel_excitation_amps','response_channels']}.items():
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
        self.discard_button.setEnabled(not locked and self.dirty)
        if locked or not self.validation_errors:
            self.validation_label.setText('Run settings are locked.' if locked else ('Draft changes · apply before starting.' if self.dirty else 'Setup applied · ready to start.'))

    def load_preset(self):
        if self.locked:
            return
        filename, _ = QFileDialog.getOpenFileName(self, 'Load setup preset', '', 'JSON settings (*.json)')
        if not filename:
            return
        try:
            data = json.loads(Path(filename).read_text(encoding='utf-8-sig'))
            if not isinstance(data, dict) or data.get('schema_version', 1) != 1:
                raise ValueError('Use a supported version 1 setup object.')
            values = self.shared_config.as_dict()
            source = data.get('settings', data)
            if not isinstance(source, dict):
                raise ValueError('Preset settings must be an object.')
            values.update({key: value for key, value in source.items() if key in values})
            # Earlier presets selected a single hardware input. Do not enable
            # newly added inputs when loading their original setup.
            values['response_channels'] = source.get('response_channels', [])
            values['sine_closed_loop'] = source.get('sine_closed_loop', False)
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
                Path(filename).write_text(json.dumps({'schema_version':1, 'settings': self.shared_config.as_dict()}, indent=2, allow_nan=False) + '\n', encoding='utf-8')
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
        for key, title in [('accel','Band acceleration'),('target','Full target'),('level','Ramp level'),('clip','AO clipping'),('frequency','Frequency')]:
            box = QWidget()
            column = QVBoxLayout(box)
            column.setContentsMargins(0, 0, 0, 0)
            heading, value = QLabel(title), QLabel('—')
            heading.setObjectName('metricTitle')
            heading.setWordWrap(True)
            value.setObjectName('metricValue')
            column.addWidget(heading)
            column.addWidget(value)
            values.addWidget(box)
            self.value_labels[key], self.value_titles[key] = value, heading
        layout.addLayout(values)
        self.detail_label = QLabel('No acquired samples yet.')
        self.detail_label.setObjectName('mutedText')
        self.detail_label.setWordWrap(True)
        layout.addWidget(self.detail_label)
        self.plot_stack = QStackedWidget()
        self.psd_plot = create_plot(title='Control response PSD')
        self.sine_plot = create_plot(title='Control acceleration vs frequency')
        for plot in (self.psd_plot, self.sine_plot):
            configure_plot(plot)
            plot.addLegend(brush=pg.mkBrush(workspace_colors()['paper']))
            self.plot_stack.addWidget(plot)
        self.psd_plot.setLogMode(x=True, y=True)
        self.psd_plot.setLabel('left', 'PSD', units='g²/Hz')
        self.psd_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.psd_measured_curve = self.psd_plot.plot(pen=pg.mkPen(COLORS[0], width=2), name='Control · averaged')
        self.psd_current_curve = self.psd_plot.plot(pen=pg.mkPen(COLORS[2], width=1), name='Control · current block')
        neutral = QApplication.palette().text().color()
        target_pen = pg.mkPen(neutral, width=2, style=Qt.DashLine)
        ramp_pen = pg.mkPen(COLORS[3], width=1, style=Qt.DotLine)
        self.psd_target_curve = self.psd_plot.plot(pen=target_pen, name='Full target')
        self.psd_ramp_curve = self.psd_plot.plot(pen=ramp_pen, name='Ramp target')
        self.sine_plot.setLogMode(x=True)
        self.sine_plot.setLabel('left', 'Acceleration estimate', units='g peak')
        self.sine_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.sine_measured_curve = self.sine_plot.plot(pen=pg.mkPen(COLORS[0], width=2), symbol='o', symbolSize=5, symbolBrush=COLORS[0], name='Fundamental · latest at frequency')
        self.sine_target_curve = self.sine_plot.plot(pen=target_pen, name='Full target')
        self.sine_ramp_curve = self.sine_plot.plot(pen=ramp_pen, symbol='t', symbolSize=5, name='Ramp target')
        layout.addWidget(self.plot_stack, 1)
        # Keep the legend outside the data area: a large low-frequency sine
        # response must not disappear behind the legend's opaque background.
        self.legend_stack = QStackedWidget()
        for plot, curves in [
            (self.psd_plot, (self.psd_current_curve, self.psd_measured_curve, self.psd_target_curve, self.psd_ramp_curve)),
            (self.sine_plot, (self.sine_measured_curve, self.sine_target_curve, self.sine_ramp_curve)),
        ]:
            labels = []
            for curve in curves:
                pen = pg.mkPen(curve.opts['pen'])
                mark = '····' if pen.style() == Qt.DotLine else '— —' if pen.style() == Qt.DashLine else '━━'
                labels.append(f'<span style="color:{pen.color().name()}">{mark}</span>&nbsp;{escape(curve.name())}')
            legend = QLabel('&nbsp;&nbsp;&nbsp; '.join(labels))
            legend.setObjectName('mutedText')
            legend.setWordWrap(True)
            legend.setContentsMargins(6, 2, 6, 2)
            self.legend_stack.addWidget(legend)
            plot.getPlotItem().legend.hide()
        layout.addWidget(self.legend_stack)
        self.note_label = QLabel()
        self.note_label.setObjectName('mutedText')
        self.note_label.setWordWrap(True)
        layout.addWidget(self.note_label)

        self.diagnostics_widget = QTabWidget()
        self.metrics_plot = create_plot(title='Acceleration trend')
        self.percent_plot = create_plot(title='Ramp and command clipping')
        self.plant_gain_plot = create_plot(title='Estimated plant gain')
        self.eq_plot = create_plot(title='Equalizer gains')
        for plot, title, quantity, unit in [
            (self.metrics_plot, 'Acceleration', 'Acceleration', 'g RMS'),
            (self.percent_plot, 'Ramp / clipping', 'Percentage', '%'),
            (self.plant_gain_plot, 'Plant gain', 'Estimated gain', 'g/V'),
            (self.eq_plot, 'Equalizer', 'Amplitude gain', '×'),
        ]:
            configure_plot(plot)
            plot.setLabel('left', quantity, units=unit)
            plot.setLabel('bottom', 'Elapsed run time', units='s')
            plot.addLegend(brush=pg.mkBrush(workspace_colors()['paper']))
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
        for curve in (self.psd_current_curve,self.psd_measured_curve,self.psd_target_curve,self.psd_ramp_curve,self.sine_measured_curve,self.sine_target_curve,self.sine_ramp_curve,self.rms_curve,self.level_curve,self.sat_curve,self.plant_gain_curve):
            curve.clear()
        self.update_eq(None)
        sine = settings.test_mode == 'sine_sweep'
        self.plot_stack.setCurrentIndex(1 if sine else 0)
        self.legend_stack.setCurrentIndex(1 if sine else 0)
        self.value_titles['accel'].setText('Fundamental · g peak' if sine else 'Control band · g RMS')
        self.value_titles['target'].setText('Full target · g peak' if sine else 'Full target · g RMS')
        self.value_titles['frequency'].parentWidget().setVisible(sine)
        self.metrics_plot.setLabel('left', 'Fundamental amplitude' if sine else 'Band acceleration', units='g peak' if sine else 'g RMS')
        self.diagnostics_widget.setTabVisible(3, not sine)
        for label in self.value_labels.values():
            label.setText('—')
        self.value_labels['target'].setText(f'{settings.target_rms_g * (math.sqrt(2) if sine else 1):.3f}')
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
            mode = 'Closed-loop fundamental amplitude' if settings.sine_closed_loop else 'Feedforward drive table'
            coverage = ''
            if settings.calibration_max_hz and (self.band[0] < settings.calibration_min_hz or self.band[1] > settings.calibration_max_hz
                    or settings.calibration_backend != settings.backend
                    or not math.isclose(target, settings.calibration_level_peak_g, rel_tol=.01)):
                coverage = ' · Requested setup is outside the recorded calibration range/level/backend'
            self.note_label.setText(f'{mode} · {source}{coverage}. Fitted tone is separate from broadband RMS and raw sample peak.')
        else:
            self.band = [settings.target_psd_points[0][0], settings.target_psd_points[-1][0]]
            f = np.geomspace(*self.band, 256)
            knots = np.asarray(settings.target_psd_points)
            target = 10 ** np.interp(np.log10(f), np.log10(knots[:, 0]), np.log10(knots[:, 1]))
            self.psd_target_curve.setData(f, target)
            self.psd_plot.setXRange(math.log10(self.band[0]), math.log10(self.band[1]), padding=.03)
            frequency_ticks(self.psd_plot, *self.band)
            self.note_label.setText('10% ramp requests 1% of the full PSD. Current block is unaveraged; averaging has a 2 s time constant. '
                                   'The ramp advances after tracking. Band RMS excludes frequencies outside the target band.')
        self._last_level = settings.sine_initial_level if sine else settings.initial_level_fraction

    def update_psd(self, packet):
        if self.settings.test_mode != 'random':
            self.psd_current_curve.clear()
            self.psd_measured_curve.clear()
            self.psd_target_curve.clear()
            self.psd_ramp_curve.clear()
            return
        f, measured, target, averaged = packet
        self.latest_psd = (f, target)
        set_curve(self.psd_current_curve, f, measured, self.band)
        set_curve(self.psd_measured_curve, f, averaged if averaged is not None else measured, self.band)
        set_curve(self.psd_target_curve, f, target, self.band)
        set_curve(self.psd_ramp_curve, f, target * self._last_level ** 2 if target is not None else None, self.band)

    def update_metrics(self, metrics):
        self.history.append(dict(metrics))
        sine = self.settings.test_mode == 'sine_sweep'
        acceleration = metrics.get('tone_peak_g', metrics['accel_peak_g']) if sine else metrics['accel_rms_g']
        self.value_labels['accel'].setText(f'{acceleration:.3f}')
        self.value_labels['level'].setText(f'{100 * metrics["level_fraction"]:.1f}%')
        self.value_labels['clip'].setText(f'{100 * metrics.get("clipping_fraction", 0):.2f}%')
        self._last_level = metrics['level_fraction']
        voltage_note = f'Command {metrics["output_rms_v"]:.3f} V RMS · crest factor {metrics["output_cf"]:.2f}'
        if metrics.get('limiter_active') or metrics.get('clipping_fraction', 0):
            voltage_note += ' · command limiting active'
        if not sine and not metrics.get('control_feedback_ready', True):
            voltage_note += ' · identifying response / holding pilot'
        if metrics.get('input_backlog_seconds', 0) > self.settings.block_seconds:
            voltage_note += f' · unread input backlog {metrics["input_backlog_seconds"]:.2f} s'
        self.detail_label.setText(f'Elapsed {metrics["elapsed_s"]:.1f} s · {voltage_note}')
        if sine:
            frequency = metrics['frequency_hz']
            self.value_labels['frequency'].setText(f'{frequency:.1f} Hz')
            if frequency > 0:
                old = self.sine_points.get(frequency, (0, 0))
                self.sine_points[frequency] = (acceleration, max(old[1], metrics['target_peak_g']))
            frequencies = sorted(self.sine_points)
            self.sine_measured_curve.setData(frequencies, [self.sine_points[f][0] for f in frequencies])
            self.sine_ramp_curve.setData(frequencies, [self.sine_points[f][1] for f in frequencies])
            self.detail_label.setText(f'Frequency progress {100 * metrics["sweep_progress"]:.1f}% · ' + self.detail_label.text())
            if self.settings.sine_closed_loop:
                self.detail_label.setText(self.detail_label.text() + (' · tracking settled' if metrics.get('settled') else ' · settling / identifying tone'))
        elif self.latest_psd is not None:
            f, target = self.latest_psd
            set_curve(self.psd_ramp_curve, f, target * self._last_level ** 2 if target is not None else None, self.band)
        times = [item['elapsed_s'] for item in self.history]
        self.rms_curve.setData(times, [item.get('tone_peak_g', item['accel_peak_g']) if sine else item['accel_rms_g'] for item in self.history])
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
        self.time_plot = create_plot(title='Acquired acceleration')
        self.time_plot.setLabel('left','Acceleration', units='g')
        self.time_plot.setLabel('bottom','Acquired sample time', units='s')
        self.psd_plot = create_plot(title='Channel PSD')
        self.psd_plot.setLabel('left','PSD',units='g²/Hz')
        self.psd_plot.setLabel('bottom','Frequency',units='Hz')
        self.psd_plot.setLogMode(x=True,y=True)
        for plot in (self.time_plot,self.psd_plot):
            configure_plot(plot)
            plot.addLegend(brush=pg.mkBrush(workspace_colors()['paper']))
            layout.addWidget(plot,1)
        self.is_streaming = False
        self.display_timer = QTimer(self)
        self.display_timer.setInterval(100)
        self.display_timer.timeout.connect(self.refresh_display)
        self.configure_settings(settings)

    def configure_settings(self, settings):
        self.settings = settings
        self.fs = settings.fs
        self.num_channels = settings.num_input_channels
        self.channel_labels = list(settings.input_channel_labels)
        self.samples = 0
        self.blocks = 0
        self.buffer = np.empty((self.num_channels, 0))
        self.mailbox = ChannelMailbox(max(1, int(self.fs * 2)), self.num_channels)
        self._display_dirty = True
        self._control_psd = None
        self._last_block = None
        self._response_psd_dirty = False
        self._last_response_blocks = 0
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
            curve = self.time_plot.plot(pen=pg.mkPen(color, width=1.5),name=label, antialias=False)
            self.time_curves.append(curve)
            self.psd_curves.append(self.psd_plot.plot(pen=pg.mkPen(color,width=1.5),name=label))
        self.target_curve = self.psd_plot.plot(pen=pg.mkPen(QApplication.palette().text().color(),style=Qt.DashLine),name='Control full target')
        self.info_label.setText(f'{self.fs:g} Hz · {self.num_channels} channel(s) · PSD bin spacing {self.fs/settings.welch_nperseg:g} Hz')
        low, high = self.fs/settings.welch_nperseg, self.fs/2
        if low >= high:
            low = high/10
        self.psd_plot.setXRange(math.log10(low),math.log10(high),padding=.01)
        frequency_ticks(self.psd_plot,low,high)
        self._filter_channels()

    def _filter_channels(self):
        selected = self.channel_combo.currentIndex() - 1
        for index, (time_curve, psd_curve) in enumerate(zip(self.time_curves,self.psd_curves)):
            time_curve.setVisible(selected < 0 or selected == index)
            psd_curve.setVisible(selected < 0 or selected == index)
        if hasattr(self,'target_curve'):
            self.target_curve.setVisible(self.settings.test_mode == 'random' and selected in (-1,0))
        self._display_dirty = True
        self._response_psd_dirty = True
        if self.isVisible():
            self.refresh_display()

    def start_streaming(self):
        self.is_streaming = True
        self.display_timer.start()

    def stop_streaming(self):
        self.is_streaming = False
        self.display_timer.stop()
        # Acquisition has stopped: flush final samples once, including hidden
        # plots, so results remain available when the user opens Channels later.
        self.refresh_display(force=True)

    def queue_data_point(self, block):
        # Producer entry point: no Qt widget access or drawing here.
        if self.is_streaming:
            self.mailbox.submit(block)

    def queue_psd_data(self, packet):
        if self.is_streaming:
            self.mailbox.submit_psd(packet)

    def refresh_display(self, force=False):
        blocks, samples, count, dropped, psd, error = self.mailbox.drain()
        if error:
            self.info_label.setText(error)
        if blocks:
            self._ingest_blocks(blocks, samples, count, dropped)
        if psd is not None:
            self.update_psd_data(psd)
        if not self._display_dirty or (not force and not self.isVisible()):
            return
        self._display_dirty = False
        selected = self.channel_combo.currentIndex() - 1
        times = (self.samples - self.buffer.shape[1] + np.arange(self.buffer.shape[1])) / self.fs
        for index, curve in enumerate(self.time_curves):
            if force or selected < 0 or selected == index:
                x, y = peak_envelope(times, self.buffer[index])
                curve.setData(x, y)
        if self.samples:
            self.time_plot.setXRange(max(0, self.samples/self.fs-2), self.samples/self.fs, padding=0)
        if self._control_psd is not None:
            frequencies, measured, target, averaged = self._control_psd
            psd_values = averaged if averaged is not None else measured
            mask = (frequencies > 0) & (psd_values > 0) & np.isfinite(psd_values)
            if force or selected in (-1, 0):
                self.psd_curves[0].setData(frequencies[mask], psd_values[mask])
            if self.settings.test_mode == 'random' and target is not None:
                set_curve(self.target_curve, frequencies, target)
            else:
                self.target_curve.clear()
        if self._last_block is not None and (self._response_psd_dirty or force):
            for index in range(1, self.num_channels):
                if not force and selected not in (-1, index):
                    continue
                frequencies, psd_values = welch(self._last_block[index], fs=self.fs, nperseg=self.settings.welch_nperseg)
                previous = self.psd_averaged[index]
                alpha = 1. - math.exp(-self.settings.block_seconds / 2.)
                psd_values = psd_values if previous is None else alpha*psd_values+(1-alpha)*previous
                self.psd_averaged[index] = psd_values
                mask = (frequencies > 0) & (psd_values > 0)
                self.psd_curves[index].setData(frequencies[mask], psd_values[mask])
            self._last_response_blocks = self.blocks
            self._response_psd_dirty = False

    def showEvent(self, event):
        super().showEvent(event)
        self._display_dirty = True
        self.refresh_display()

    def add_data_point(self, block):
        if not self.is_streaming:
            return
        block = np.atleast_2d(np.asarray(block))
        if block.ndim != 2 or block.shape[0] != self.num_channels:
            raise ValueError('Acquisition channel count does not match run settings')
        self._ingest_blocks([block], block.shape[1], 1, False)

    def _ingest_blocks(self, blocks, samples, count, dropped):
        self.samples += samples
        self.blocks += count
        capacity = max(1, int(self.fs * 2))
        previous = self.buffer[:, :0] if dropped else self.buffer
        self.buffer = np.concatenate([previous, *blocks], axis=1)[:, -capacity:]
        self._last_block = blocks[-1][:, -min(blocks[-1].shape[1], int(self.fs * self.settings.block_seconds)):]
        self._display_dirty = True
        self._response_psd_dirty |= self.blocks == count or self.blocks - self._last_response_blocks >= max(1, self.settings.realtime_psd_update_stride)

    def update_psd_data(self, packet):
        self._control_psd = packet
        self._display_dirty = True


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
        self.report_button = QPushButton('Open HTML test report')
        self.report_button.setEnabled(False)
        self.report_button.clicked.connect(self.open_report)
        layout.addWidget(self.report_button)
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
        # A historical setup may violate newly introduced start-time checks.
        # Display its captured values without re-validating them for a new run.
        settings = RunSettings(report['settings'])
        metrics = report.get('latest_metrics') or {}
        measured = (f'{metrics["tone_peak_g"]:.4g} g peak fundamental' if 'tone_peak_g' in metrics else
                    f'{metrics["accel_peak_g"]:.4g} g peak estimate' if 'accel_peak_g' in metrics else
                    f'{metrics["accel_rms_g"]:.4g} g RMS in the control band' if 'accel_rms_g' in metrics else 'No acquired samples')
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
        if 'qualification' in report:
            q = report['qualification']
            lines.insert(1, f'Tracking assessment: {q["status"].upper()} · ±{q["tolerance_percent"]:g}% tolerance')
            lines.append(q['definition'])
            lines.append(f'Largest acquired sample: {q["maximum_raw_sample_peak_g"]:.5g} g peak')
            lines.append(f'Test report: {report.get("report_dir") or "Automatic report export disabled"}')
        if report.get('protection_trip'):
            lines.append('Measured-response stop: ' + report['protection_trip']['reason'])
        if report.get('report_plot_error'):
            lines.append('Report plot unavailable: ' + report['report_plot_error'])
        diagnostics = report.get('stream_diagnostics', {})
        if 'input_buffer_samples_per_channel' in diagnostics:
            capacity = diagnostics['input_buffer_samples_per_channel'] / settings.fs
            backlog = diagnostics.get('max_input_backlog_samples', 0) / settings.fs
            lines.append(f'Input buffer: {capacity:g} s per channel · largest unread backlog {backlog:.3g} s')
        stages = [(label, diagnostics[key]) for label, key in (
            ('read', 'max_read_seconds'), ('write', 'max_write_seconds'),
            ('record', 'max_record_seconds'), ('block loop', 'max_loop_seconds')) if key in diagnostics]
        if stages:
            lines.append('Longest operations: ' + ' · '.join(f'{label} {duration:.3g} s' for label, duration in stages))
        self.summary_text.setPlainText('\n\n'.join(lines))
        self.details.setPlainText(json.dumps(report,indent=2,allow_nan=False))
        self.open_button.setEnabled(bool(report['log_dir']))
        self.report_button.setEnabled(bool(report.get('report_dir')))

    def open_report(self):
        index = self.run_list.currentRow()
        if 0 <= index < len(self.reports) and self.reports[index].get('report_dir'):
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.reports[index]['report_dir']) / 'test-report.html')))

    def open_folder(self):
        index = self.run_list.currentRow()
        if 0 <= index < len(self.reports) and self.reports[index]['log_dir']:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.reports[index]['log_dir']).resolve())))


class WorkspaceWindow(FloatingWorkspaceMixin, QMainWindow):
    run_completed = Signal(object)
    def __init__(self, shared_config, worker_factory):
        super().__init__()
        self.setWindowTitle('Shaker Control')
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
        self.controller_tab = RunView(self.settings)
        self.realtime_tab = ChannelView(self.settings)
        self.results_tab = ResultsView()
        self.config_tab = SettingsPanel(shared_config)
        self.config_tab.calibration_requested.connect(self.open_calibration)
        self._build_floating_workspace()
        self._refresh_controls()

    @property
    def busy(self):
        return self.controller_thread is not None or self.run_state in BUSY_STATES

    def _dock_visible(self, visible):
        with QSignalBlocker(self.setup_button):
            self.setup_button.setChecked(visible)
        QTimer.singleShot(0, self._reflow_shell)

    def show_page(self, index):
        self.pages.setCurrentIndex(index)
        for position, button in enumerate(self.nav_buttons):
            button.setChecked(position == index)
        if hasattr(self, 'view_title'):
            self._sync_view_header(index)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'config_dock'):
            QTimer.singleShot(0, self._reflow_shell)

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
        self._rebuild_browser()
        self._record_shell_event(f'Setup applied · {settings.target_summary} · {settings.backend} requested')
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
        channel = (f'{self.settings.num_input_channels} simulated channel(s)' if self.settings.simulation_mode else
                   ', '.join(spec['physical_channel'] for spec in self.settings.hardware_input_channels))
        recording = (f'Recording {"to" if busy else "saved in"} {self.active_log_dir}' if self.active_log_dir else
                     ('Recording requested · not started' if self.settings.data_log_enabled else 'Recording disabled'))
        self.setup_summary.setText(f'Applied setup: {self.settings.target_summary}  |  {channel}  |  {self.settings.fs:g} Hz · {self.settings.block_seconds:g} s blocks · PSD {self.settings.fs/self.settings.welch_nperseg:g} Hz bins  |  {recording}')
        self.fault_banner.setVisible(bool(self.fault_message))
        self.fault_label.setText(self.fault_message)
        self.dismiss_fault_button.setEnabled(not busy)
        self.switch_simulation_button.setVisible(bool(self.fault_message) and not self.settings.simulation_mode)
        self.switch_simulation_button.setEnabled(not busy)
        self._sync_shell_state()

    def start_controller(self, snapshot=None):
        if self.busy or self.config_tab.dirty:
            if not self.busy:
                self.inspect_settings('all')
            return False
        try:
            snapshot = snapshot if isinstance(snapshot, RunSettings) else self.shared_config.snapshot()
        except ConfigurationError as error:
            self.inspect_settings('all')
            self.config_tab.show_errors(error.errors)
            return False
        self.settings = snapshot
        with QSignalBlocker(self.mode_combo), QSignalBlocker(self.backend_combo):
            self.mode_combo.setCurrentIndex(1 if snapshot.test_mode == 'sine_sweep' else 0)
            self.backend_combo.setCurrentIndex(1 if snapshot.simulation_mode else 0)
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
        worker.channel_mailbox = self.realtime_tab.mailbox
        worker.metrics_data_ready.connect(self.controller_tab.update_metrics)
        worker.eq_data_ready.connect(self.controller_tab.update_eq)
        worker.report_ready.connect(self._worker_report)
        thread.finished.connect(self._thread_finished)
        self.run_state = 'starting'
        self._record_shell_event(f'Start requested · {snapshot.backend} · setup {snapshot.config_id}')
        self._refresh_controls()
        self.show_page(0)
        thread.start()
        return True

    def stop_controller(self):
        if self.controller_worker is None or self.run_state not in ('starting','running'):
            return False
        self.run_state = 'stopping'
        self._record_shell_event('Stop requested · waiting for acquisition and recording cleanup')
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
        self._record_shell_event(f'{state.title()} · {event.get("message") or self.actual_backend or "Backend not initialized"}')
        self._refresh_controls()

    def _worker_notice(self, message):
        if self.fault_message or message in ('Run stopped','Sweep complete'):
            return
        details = ' · '.join(filter(None, (self.input_mode, self.clock_status)))
        self.notice_label.setText(f'{message} · {details}' if details else message)
        self.notice_label.show()

    def _worker_report(self, report):
        try:
            render_report_chart(report)
        except Exception as error:
            report['report_plot_error'] = str(error)
        self.last_report = report
        self.results_tab.add_report(report)
        self._rebuild_browser()
        self.realtime_tab.stop_streaming()
        self.run_state = report['outcome']
        if report['error']:
            self.fault_message = report['error']
            self.notice_label.hide()
        else:
            self.notice_label.setText(('Sweep complete' if report['outcome'] == 'complete' else 'Run stopped') + ' · review this run in Results.')
            self.notice_label.show()
        self._refresh_controls()
        self.run_completed.emit(report)

    def open_calibration(self):
        if self.busy or self.config_tab.dirty:
            self.config_tab.validation_label.setText('Apply or discard the draft before calibration; calibration requires an idle controller.')
            return
        from calibration_wizard import CalibrationWizard
        self.calibration_dialog = CalibrationWizard(self)
        self.calibration_dialog.show()

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
        self.inspect_settings('channel')
        self.notice_label.setText('Simulation selected in the draft. Apply setup before starting.')
        self.notice_label.show()

    def closeEvent(self, event):
        if self.busy:
            self._close_pending = True
            self.stop_controller()
            event.ignore()
        else:
            event.accept()
