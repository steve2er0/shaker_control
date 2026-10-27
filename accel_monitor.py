#!/usr/bin/env python3
"""Standalone accelerometer monitoring application."""

from __future__ import annotations

import math
import sys
from threading import Event
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QSpinBox, QTableWidget, QTableWidgetItem,
)
from scipy.signal import welch

import monitor_config as cfg
from workspace_ui import configure_plot, create_plot, COLORS
from workspace_shell import workspace_colors

try:  # Optional hardware support
    import nidaqmx
    from nidaqmx.constants import (
        AcquisitionType,
        AccelSensitivityUnits,
        AccelUnits,
        Coupling,
        ExcitationSource,
        OverwriteMode,
        TerminalConfiguration,
        VoltageUnits,
    )
    from nidaqmx.errors import DaqError
    from nidaqmx.stream_readers import AnalogMultiChannelReader, AnalogSingleChannelReader

    NIDAQ_AVAILABLE = True
except ImportError:  # pragma: no cover - hardware optional
    nidaqmx = None
    AcquisitionType = None
    AccelSensitivityUnits = None
    AccelUnits = None
    Coupling = None
    ExcitationSource = None
    TerminalConfiguration = None
    VoltageUnits = None
    OverwriteMode = None
    AnalogMultiChannelReader = None
    AnalogSingleChannelReader = None
    DaqError = None
    NIDAQ_AVAILABLE = False


@dataclass(frozen=True)
class ChannelSpec:
    physical_channel: str
    label: str
    sensitivity_mV_per_g: float
    voltage_range: float
    use_accel_channel: bool
    excitation_current_amps: float
    terminal_config: str


def load_channel_specs(entries: Iterable[dict]) -> List[ChannelSpec]:
    specs: List[ChannelSpec] = []
    for idx, entry in enumerate(entries):
        phys = str(entry.get("physical_channel", f"Dev1/ai{idx}"))
        label = str(entry.get("label", phys))
        sensitivity = float(entry.get("sensitivity_mV_per_g", 100.0))
        if sensitivity <= 0:
            raise ValueError(f"Channel {label} has invalid sensitivity: {sensitivity}")
        voltage_range = float(max(entry.get("voltage_range", 5.0), 0.1))
        use_accel = bool(entry.get("use_accel_channel", True))
        excitation_current = float(max(entry.get("excitation_current_amps", 0.0), 0.0))
        terminal_cfg = str(entry.get("terminal_config", "DEFAULT")).strip().upper()
        specs.append(
            ChannelSpec(
                physical_channel=phys,
                label=label,
                sensitivity_mV_per_g=sensitivity,
                voltage_range=voltage_range,
                use_accel_channel=use_accel,
                excitation_current_amps=excitation_current,
                terminal_config=terminal_cfg,
            )
        )
    if not specs:
        raise ValueError("No channels configured in monitor_config.CHANNELS")
    return specs


class BaseAcquisitionSession:
    """Minimal interface for the acquisition backends."""

    description: str = ""

    def start(self) -> None:
        raise NotImplementedError

    def read_block(self) -> np.ndarray:
        raise NotImplementedError

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


def _resolve_terminal_config(name: str) -> Optional[TerminalConfiguration]:
    if TerminalConfiguration is None:
        return None

    cleaned = (name or "").strip().upper()
    default = getattr(TerminalConfiguration, 'DEFAULT', None)

    def _get(*candidates):
        for candidate in candidates:
            if candidate and hasattr(TerminalConfiguration, candidate):
                return getattr(TerminalConfiguration, candidate)
        return default

    mapping = {
        'DEFAULT': _get('DEFAULT'),
        'DIFF': _get('DIFFERENTIAL', 'DIFF'),
        'DIFFERENTIAL': _get('DIFFERENTIAL', 'DIFF'),
        'NRSE': _get('NRSE'),
        'PSEUDO_DIFF': _get('PSEUDO_DIFF', 'PSEUDODIFFERENTIAL'),
        'PSEUDODIFF': _get('PSEUDO_DIFF', 'PSEUDODIFFERENTIAL'),
        'RSE': _get('RSE'),
    }

    return mapping.get(cleaned, default)


class NIDaqSession(BaseAcquisitionSession):
    def __init__(self, channels: Sequence[ChannelSpec], fs: float, block_samples: int) -> None:
        if not NIDAQ_AVAILABLE:
            raise RuntimeError("nidaqmx package not available; enable simulation mode or install drivers")
        self._channels = list(channels)
        self._fs = float(fs)
        self._block = int(block_samples)
        self._task: Optional[nidaqmx.Task] = None
        self._reader: Optional[object] = None
        self._use_single_reader = len(self._channels) == 1 and AnalogSingleChannelReader is not None
        self._scales: Optional[np.ndarray] = None
        self._stopping = False
        self.description = "NI-DAQ"  # refined during start

    def start(self) -> None:
        num_channels = len(self._channels)
        task = nidaqmx.Task()
        self._task = task
        scales = np.ones(num_channels, dtype=np.float64)
        mode_tags: List[str] = []

        try:
            for idx, spec in enumerate(self._channels):
                term_cfg = _resolve_terminal_config(spec.terminal_config)
                volts_per_g = spec.sensitivity_mV_per_g / 1000.0
                g_range_guess = max(spec.voltage_range / max(volts_per_g, 1e-9), 0.5)
                excitation_source = ExcitationSource.INTERNAL if spec.excitation_current_amps > 0 else ExcitationSource.NONE

                added = False
                mode_tag = "voltage"
                if spec.use_accel_channel:
                    try:
                        accel_chan = task.ai_channels.add_ai_accel_chan(
                            physical_channel=spec.physical_channel,
                            name_to_assign_to_channel=spec.label,
                            terminal_config=term_cfg,
                            min_val=-g_range_guess,
                            max_val=g_range_guess,
                            units=AccelUnits.G,
                            sensitivity=spec.sensitivity_mV_per_g,
                            sensitivity_units=AccelSensitivityUnits.MILLIVOLTS_PER_G,
                            current_excit_source=excitation_source,
                            current_excit_val=spec.excitation_current_amps,
                        )
                        try:
                            accel_chan.ai_coupling = Coupling.AC
                        except Exception:
                            pass
                        scales[idx] = 1.0
                        mode_tag = "accel"
                        added = True
                    except Exception as accel_err:
                        print(f"Accelerometer channel setup failed for {spec.physical_channel}: {accel_err}")

                if not added:
                    voltage_range = max(spec.voltage_range, 0.5)
                    task.ai_channels.add_ai_voltage_chan(
                        physical_channel=spec.physical_channel,
                        name_to_assign_to_channel=spec.label,
                        terminal_config=term_cfg,
                        min_val=-voltage_range,
                        max_val=voltage_range,
                        units=VoltageUnits.VOLTS,
                    )
                    scales[idx] = 1.0 / max(volts_per_g, 1e-9)
                mode_tags.append(mode_tag)

            samp_per_chan = max(self._block * 4, self._block)
            task.timing.cfg_samp_clk_timing(
                rate=self._fs,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=samp_per_chan,
            )

            min_host_buffer = int(max(self._fs * 0.5, self._block * 32, samp_per_chan * 4))
            try:
                current_size = task.in_stream.input_buf_size
                desired_buf = max(min_host_buffer, current_size)
                task.in_stream.input_buf_size = desired_buf
            except Exception:
                pass

            if OverwriteMode is not None:
                try:
                    task.in_stream.overwrite_mode = OverwriteMode.OVERWRITE_UNREAD_SAMPLES
                except Exception:
                    pass

            self._use_single_reader = len(self._channels) == 1 and AnalogSingleChannelReader is not None
            if self._use_single_reader:
                reader = AnalogSingleChannelReader(task.in_stream)
            else:
                reader = AnalogMultiChannelReader(task.in_stream)
            task.start()

            self._reader = reader
            self._scales = scales
            self._stopping = False
            self.description = "NI-DAQ (" + ", ".join(mode_tags) + ")"
        except Exception:
            self._reader = None
            self._scales = None
            self._stopping = False
            try:
                task.stop()
            except Exception:
                pass
            try:
                task.close()
            except Exception:
                pass
            self._task = None
            raise

    def read_block(self) -> np.ndarray:
        if self._reader is None or self._scales is None:
            raise RuntimeError("Session not started")
        timeout = max(self._block / max(self._fs, 1e-9) * 1.5, 0.1)
        reader = self._reader
        try:
            if self._use_single_reader and AnalogSingleChannelReader is not None:
                buf_single = np.empty(self._block, dtype=np.float64)
                reader.read_many_sample(  # type: ignore[attr-defined]
                    buf_single,
                    number_of_samples_per_channel=self._block,
                    timeout=timeout,
                )
                buf = buf_single[np.newaxis, :]
            else:
                buf_multi = np.empty((len(self._channels), self._block), dtype=np.float64)
                reader.read_many_sample(  # type: ignore[attr-defined]
                    buf_multi,
                    number_of_samples_per_channel=self._block,
                    timeout=timeout,
                )
                buf = buf_multi
        except DaqError as err:
            if self._stopping:
                return np.empty((len(self._channels), 0), dtype=np.float64)
            raise
        return buf * self._scales[:, np.newaxis]

    def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.stop()

    def close(self) -> None:
        if self._task is not None:
            task = self._task
            self._task = None
            task.close()


class SimulationSession(BaseAcquisitionSession):
    def __init__(
        self,
        channels: Sequence[ChannelSpec],
        fs: float,
        block_samples: int,
        noise_level: float,
        seed: Optional[int],
    ) -> None:
        self._num_channels = len(channels)
        self._fs = float(fs)
        self._block = int(block_samples)
        self._noise = float(noise_level)
        self._dt = 1.0 / max(self._fs, 1e-9)
        self._rng = np.random.default_rng(seed)
        self._phase = self._rng.uniform(0, 2 * math.pi, size=self._num_channels)
        self._base_freqs = np.linspace(12.0, 60.0, self._num_channels)
        self._t0 = 0.0
        self.description = "Simulation"
        self.read_interval_seconds = self._block / self._fs

    def start(self) -> None:
        self._t0 = 0.0

    def read_block(self) -> np.ndarray:
        times = self._t0 + np.arange(self._block, dtype=np.float64) * self._dt
        data = np.empty((self._num_channels, self._block), dtype=np.float64)
        for idx in range(self._num_channels):
            base = 0.6 * np.sin(2 * math.pi * self._base_freqs[idx] * times + self._phase[idx])
            overtone = 0.2 * np.sin(2 * math.pi * (self._base_freqs[idx] * 3.5) * times + self._phase[idx] / 2)
            noise = self._rng.normal(scale=self._noise, size=self._block)
            data[idx, :] = base + overtone + noise
        self._t0 += self._block * self._dt
        return data


class AcquisitionWorker(QThread):
    data_ready = Signal(object)
    status = Signal(str)
    mode_info = Signal(str)
    error = Signal(str)

    def __init__(self, session_factory: Callable[[], BaseAcquisitionSession]) -> None:
        super().__init__()
        self._session_factory = session_factory
        self._session: Optional[BaseAcquisitionSession] = None
        self._stop_event = Event()

    def run(self) -> None:
        try:
            if self._stop_event.is_set():
                return
            print('AcquisitionWorker: creating session...', flush=True)
            self._session = self._session_factory()
            print('AcquisitionWorker: starting session...', flush=True)
            if self._stop_event.is_set():
                return
            self._session.start()
            description = getattr(self._session, 'description', '')
            if description:
                self.mode_info.emit(description)
                print(f'AcquisitionWorker: session started ({description})', flush=True)
            else:
                print('AcquisitionWorker: session started', flush=True)
            self.status.emit('running')
            while not self._stop_event.is_set():
                block = self._session.read_block()
                if block.size == 0:
                    continue
                self.data_ready.emit(block)
                self._stop_event.wait(getattr(self._session, 'read_interval_seconds', 0.0))
        except Exception as exc:  # pragma: no cover - runtime error channel
            print('AcquisitionWorker: exception during run:', exc, flush=True)
            import traceback as _tb  # local import to avoid global dependency if optional
            _tb.print_exc()
            message = str(exc).strip()
            if not message:
                message = exc.__class__.__name__
            cancelled_read = (self._stop_event.is_set() and DaqError is not None
                              and isinstance(exc, DaqError) and exc.error_code in (-200284, -200474))
            if not cancelled_read:
                self.error.emit(message)
        finally:
            cleanup_errors = []
            if self._session is not None:
                for operation in (self._session.stop, self._session.close):
                    try:
                        operation()
                    except Exception as exc:
                        cleanup_errors.append(str(exc))
            if cleanup_errors:
                self.error.emit('Acquisition cleanup failed: ' + '; '.join(cleanup_errors))
            self.status.emit('stopped')

    def stop(self) -> None:
        # Hardware reads already have finite timeouts. Do not call the driver on
        # the GUI thread; acquisition finalization owns stop/close.
        self._stop_event.set()




class MonitorSetupDialog(QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.setWindowTitle('Channels and acquisition')
        self.resize(1000,480)
        self.window = window
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.fs_spin = QDoubleSpinBox()
        self.fs_spin.setAccessibleName('Sample rate')
        self.fs_spin.setRange(100,100000)
        self.fs_spin.setValue(window._fs)
        self.fs_spin.setSuffix(' Hz')
        self.block_spin = QSpinBox()
        self.block_spin.setAccessibleName('Acquisition block length')
        self.block_spin.setRange(2,65536)
        self.block_spin.setValue(window._block)
        self.block_spin.setSuffix(' samples')
        form.addRow('Sample rate',self.fs_spin)
        form.addRow('Acquisition block',self.block_spin)
        layout.addLayout(form)
        self.table = QTableWidget(0,7)
        self.table.setHorizontalHeaderLabels(['Physical input','Label','Sensitivity [mV/g]','IEPE [mA]','Voltage range [±V]','Input type','Terminal'])
        for spec in window._channel_specs:
            self.add_row(spec)
        layout.addWidget(self.table,1)
        row_actions = QHBoxLayout()
        add, remove = QPushButton('Add channel'), QPushButton('Remove selected channel')
        add.clicked.connect(lambda: self.add_row())
        remove.clicked.connect(lambda: self.table.removeRow(self.table.currentRow()) if self.table.currentRow() >= 0 else None)
        row_actions.addWidget(add)
        row_actions.addWidget(remove)
        row_actions.addStretch()
        layout.addLayout(row_actions)
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet('color: #bd493c;')
        layout.addWidget(self.error_label)
        buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Close)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply_setup)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        if window._worker is not None:
            self.table.setEnabled(False)
            self.fs_spin.setEnabled(False)
            self.block_spin.setEnabled(False)
            add.setEnabled(False)
            remove.setEnabled(False)
            buttons.button(QDialogButtonBox.Apply).setEnabled(False)
            self.error_label.setText('Acquisition settings are locked for this run. Analysis window controls remain available.')

    def add_row(self, spec=None):
        row = self.table.rowCount()
        self.table.insertRow(row)
        values = [spec.physical_channel, spec.label, str(spec.sensitivity_mV_per_g), str(spec.excitation_current_amps*1000),str(spec.voltage_range)] if spec else ['', '', '', '0', '']
        for column,value in enumerate(values):
            self.table.setItem(row,column,QTableWidgetItem(value))
        mode = QComboBox()
        mode.addItems(['Acceleration','Voltage'])
        mode.setCurrentIndex(0 if spec is None or spec.use_accel_channel else 1)
        self.table.setCellWidget(row,5,mode)
        terminal = QComboBox()
        terminal.addItems(['DEFAULT','PSEUDO_DIFF','DIFF','RSE','NRSE'])
        if spec:
            terminal.setCurrentText(spec.terminal_config)
        self.table.setCellWidget(row,6,terminal)
        self.table.resizeColumnsToContents()

    def apply_setup(self):
        entries = []
        try:
            for row in range(self.table.rowCount()):
                values = [self.table.item(row,column).text().strip() for column in range(5)]
                sensitivity, current, voltage = map(float,values[2:])
                if not values[0] or not values[1] or not all(math.isfinite(value) for value in (sensitivity,current,voltage)) or sensitivity <= 0 or current < 0 or voltage <= 0:
                    raise ValueError('Each channel needs a physical input, label, positive sensitivity/range, and nonnegative IEPE current.')
                entries.append({'physical_channel':values[0],'label':values[1], 'sensitivity_mV_per_g':sensitivity,
                    'excitation_current_amps':current/1000, 'voltage_range':voltage,
                    'use_accel_channel':self.table.cellWidget(row,5).currentIndex()==0,
                    'terminal_config':self.table.cellWidget(row,6).currentText()})
            if not entries or len({entry['physical_channel'] for entry in entries}) != len(entries) or len({entry['label'] for entry in entries}) != len(entries):
                raise ValueError('Use at least one channel with unique physical inputs and labels.')
            self.window.apply_acquisition_setup(load_channel_specs(entries),self.fs_spin.value(),self.block_spin.value())
            self.accept()
        except (ValueError,TypeError) as error:
            self.error_label.setText(str(error))


class MonitorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Accelerometer Monitor')
        self.resize(1100,800)
        pg.setConfigOptions(antialias=True,foreground=QApplication.palette().text().color(),background=QApplication.palette().base().color())
        self._channel_specs = load_channel_specs(cfg.CHANNELS)
        self._fs = float(cfg.SAMPLE_RATE_HZ)
        self._block = int(cfg.BLOCK_SIZE)
        self._time_window_sec = float(cfg.TIME_WINDOW_SECONDS)
        self._psd_window_sec = float(cfg.DEFAULT_PSD_WINDOW_SECONDS)
        self._worker = None
        self._fault_message = ''
        self._close_pending = False
        self.current_run_settings = None
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(18,14,18,14)
        title = QLabel('Accelerometer Monitor · acquisition only')
        title.setStyleSheet('font-size: 20px; font-weight: 500;')
        layout.addWidget(title)
        controls = QHBoxLayout()
        controls.addWidget(QLabel('Backend'))
        self.backend_combo = QComboBox()
        self.backend_combo.setAccessibleName('Acquisition backend')
        self.backend_combo.addItem('Hardware',False)
        self.backend_combo.addItem('Simulation',True)
        self.backend_combo.setCurrentIndex(1 if cfg.SIMULATION_MODE else 0)
        controls.addWidget(self.backend_combo)
        self._mode_label = QLabel('Not initialized')
        controls.addWidget(self._mode_label)
        controls.addStretch()
        self.setup_button = QPushButton('Channels / acquisition…')
        self.setup_button.clicked.connect(self.show_setup)
        controls.addWidget(self.setup_button)
        self._status_label = QLabel('Ready')
        controls.addWidget(self._status_label)
        self._start_button, self._stop_button = QPushButton('Start acquisition'), QPushButton('Stop')
        self._stop_button.setStyleSheet('QPushButton:enabled { color: #bd493c; }')
        self._stop_button.setEnabled(False)
        self._start_button.clicked.connect(self._start_acquisition)
        self._stop_button.clicked.connect(self._stop_acquisition)
        controls.addWidget(self._start_button)
        controls.addWidget(self._stop_button)
        layout.addLayout(controls)
        self.fault_label = QLabel()
        self.fault_label.setWordWrap(True)
        self.fault_label.setStyleSheet('color: #bd493c;')
        self.fault_label.hide()
        layout.addWidget(self.fault_label)
        self.acquisition_label = QLabel()
        layout.addWidget(self.acquisition_label)
        self._time_plot = create_plot(title='Acquired acceleration')
        self._time_plot.setLabel('left','Acceleration',units='g')
        self._time_plot.setLabel('bottom','Time relative to latest sample',units='s')
        self._psd_plot = create_plot(title='Acceleration PSD')
        self._psd_plot.setLabel('left','PSD',units='g²/Hz')
        self._psd_plot.setLabel('bottom','Frequency',units='Hz')
        self._psd_plot.setLogMode(y=True)
        if cfg.TIME_PLOT_Y_RANGE:
            self._time_plot.setYRange(*cfg.TIME_PLOT_Y_RANGE)
        if cfg.PSD_Y_BOUNDS:
            low, high = cfg.PSD_Y_BOUNDS
            if 0 < low < high:
                self._psd_plot.setYRange(math.log10(low),math.log10(high))
        for plot in (self._time_plot,self._psd_plot):
            configure_plot(plot)
            plot.addLegend(brush=pg.mkBrush(workspace_colors()['paper']))
            layout.addWidget(plot,1)
        analysis = QHBoxLayout()
        analysis.addWidget(QLabel('PSD segment duration'))
        self._psd_spin = QDoubleSpinBox()
        self._psd_spin.setAccessibleName('PSD segment duration')
        self._psd_spin.setDecimals(3)
        self._psd_spin.setRange(*cfg.PSD_WINDOW_BOUNDS)
        self._psd_spin.setSuffix(' s')
        self._psd_spin.setValue(self._psd_window_sec)
        self._psd_spin.valueChanged.connect(self._on_psd_window_change)
        analysis.addWidget(self._psd_spin)
        self.psd_effective_label = QLabel('Waiting for samples')
        analysis.addWidget(self.psd_effective_label)
        analysis.addStretch()
        layout.addLayout(analysis)
        self._rebuild_channels()

    def _rebuild_channels(self):
        self._psd_window_samples = max(int(self._psd_window_sec*self._fs),32)
        self._buffer_capacity = max(int(max(self._time_window_sec,cfg.PSD_WINDOW_BOUNDS[1])*self._fs),self._block)
        self._buffer = np.zeros((len(self._channel_specs),self._buffer_capacity))
        self._buffer_fill = 0
        for plot in (self._time_plot,self._psd_plot):
            plot.clear()
            plot.getPlotItem().legend.clear()
        self._time_curves, self._psd_curves = [], []
        for index,spec in enumerate(self._channel_specs):
            color = COLORS[index % len(COLORS)]
            curve = self._time_plot.plot(pen=pg.mkPen(color,width=1.5),name=spec.label)
            curve.setDownsampling(auto=True,method='peak')
            curve.setClipToView(True)
            self._time_curves.append(curve)
            self._psd_curves.append(self._psd_plot.plot(pen=pg.mkPen(color,width=1.5),name=spec.label))
        self._time_plot.setXRange(-self._time_window_sec,0,padding=0)
        self.acquisition_label.setText(f'{self._fs:g} Hz · {self._block} samples/block · {len(self._channel_specs)} channel(s) · time view {self._time_window_sec:g} s')
        self.psd_effective_label.setText('Waiting for samples')

    def show_setup(self):
        dialog = MonitorSetupDialog(self)
        dialog.exec()

    def apply_acquisition_setup(self, specs, fs, block):
        if self._worker is not None:
            raise ValueError('Stop acquisition before changing channels or sample rate.')
        if not math.isfinite(fs) or fs <= 0 or block < 2:
            raise ValueError('Use a positive sample rate and at least two samples per block.')
        self._channel_specs, self._fs, self._block = list(specs),float(fs),int(block)
        self._rebuild_channels()

    def _start_acquisition(self):
        if self._worker is not None:
            return
        self._fault_message = ''
        self.fault_label.hide()
        self._reset_buffers()
        self._start_button.setEnabled(False)
        self._stop_button.setEnabled(True)
        self.backend_combo.setEnabled(False)
        self._status_label.setText('Starting…')
        simulation = bool(self.backend_combo.currentData())
        channels, fs, block = tuple(self._channel_specs),self._fs,self._block
        self._mode_label.setText(('Simulation' if simulation else 'Hardware') + ' requested · not initialized')
        self.current_run_settings = {'backend':'Simulation' if simulation else 'Hardware','fs':fs,'block_samples':block,
            'channels':[spec.__dict__.copy() for spec in channels]}
        def factory():
            if simulation:
                return SimulationSession(channels,fs,block,cfg.SIM_NOISE_LEVEL,cfg.SIM_SEED)
            return NIDaqSession(channels,fs,block)
        self._worker = AcquisitionWorker(factory)
        self._worker.data_ready.connect(self._handle_block)
        self._worker.status.connect(self._handle_worker_status)
        self._worker.error.connect(self._handle_worker_error)
        self._worker.mode_info.connect(self._handle_mode_info)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _stop_acquisition(self):
        if self._worker is None:
            return
        self._status_label.setText('Stopping…')
        self._stop_button.setEnabled(False)
        self._worker.stop()

    def _reset_buffers(self):
        self._buffer.fill(0)
        self._buffer_fill = 0
        for curve in self._time_curves + self._psd_curves:
            curve.clear()
        self.psd_effective_label.setText('Waiting for samples')

    def _handle_block(self, block):
        block = np.atleast_2d(np.asarray(block))
        if block.shape[0] != len(self._channel_specs) or not np.all(np.isfinite(block)):
            self._handle_worker_error('Acquisition returned invalid channel data')
            if self._worker:
                self._worker.stop()
            return
        self._append_to_buffer(block)
        self._update_time_plot()
        self._update_psd_plot()

    def _append_to_buffer(self, block):
        count = block.shape[1]
        capacity = self._buffer_capacity
        if count >= capacity:
            self._buffer[:] = block[:, -capacity:]
            self._buffer_fill = capacity
        elif self._buffer_fill + count <= capacity:
            self._buffer[:,self._buffer_fill:self._buffer_fill+count] = block
            self._buffer_fill += count
        else:
            existing = self._buffer[:,:self._buffer_fill]
            combined = np.concatenate((existing,block),axis=1)[:, -capacity:]
            self._buffer[:,:combined.shape[1]] = combined
            self._buffer_fill = combined.shape[1]

    def _buffer_view(self):
        return self._buffer[:,:self._buffer_fill] if self._buffer_fill else None

    def _update_time_plot(self):
        data = self._buffer_view()
        if data is None:
            return
        data = data[:, -max(1,int(self._time_window_sec*self._fs)):]
        times = (np.arange(data.shape[1])-data.shape[1]+1)/self._fs
        for curve,series in zip(self._time_curves,data):
            curve.setData(times,series)

    def _update_psd_plot(self):
        data = self._buffer_view()
        if data is None:
            return
        samples = min(data.shape[1],self._psd_window_samples)
        if samples < 32:
            self.psd_effective_label.setText('Collecting samples…')
            return
        overlap = int(samples*min(max(cfg.PSD_OVERLAP,0),.95))
        for curve,series in zip(self._psd_curves,data[:,-samples:]):
            f,psd = welch(series,fs=self._fs,nperseg=samples,noverlap=overlap)
            curve.setData(f,psd)
        effective = samples/self._fs
        self.psd_effective_label.setText(f'Effective {effective:.3g} s · {self._fs/samples:.3g} Hz bins' + (' · collecting full window' if samples < self._psd_window_samples else ''))

    def _handle_worker_status(self, state):
        if self._fault_message:
            self._status_label.setText('Fault')
        elif state == 'running':
            if self._status_label.text() != 'Stopping…':
                self._status_label.setText('Running')
        else:
            self._status_label.setText('Stopped')
        self._stop_button.setEnabled(state == 'running' and not self._fault_message and self._status_label.text() != 'Stopping…')

    def _handle_worker_error(self, message):
        self._fault_message = message
        self.fault_label.setText(message)
        self.fault_label.show()
        self._status_label.setText('Fault')
        self._stop_button.setEnabled(False)

    def _handle_mode_info(self, description):
        self._mode_label.setText(description)

    def _on_worker_finished(self):
        worker = self._worker
        if worker:
            worker.wait()
            worker.deleteLater()
        self._worker = None
        self._start_button.setEnabled(True)
        self._stop_button.setEnabled(False)
        self.backend_combo.setEnabled(True)
        self._status_label.setText('Fault' if self._fault_message else 'Stopped')
        if self._close_pending:
            self.close()

    def _on_psd_window_change(self, value):
        self._psd_window_sec = float(value)
        self._psd_window_samples = max(32,int(self._fs*value))
        self._update_psd_plot()

    def closeEvent(self,event):
        if self._worker is not None:
            self._close_pending = True
            self._stop_acquisition()
            event.ignore()
        else:
            event.accept()

def main() -> None:
    app = QApplication(sys.argv)
    window = MonitorWindow()
    window.resize(1000, 800)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
