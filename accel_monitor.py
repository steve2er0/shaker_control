#!/usr/bin/env python3
"""Standalone accelerometer monitoring application."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)
from scipy.signal import welch

import monitor_config as cfg

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
            try:
                self._task.stop()
            except Exception:
                pass

    def close(self) -> None:
        if self._task is not None:
            try:
                self._task.close()
            except Exception:
                pass
            self._task = None


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
        self._should_stop = False

    def run(self) -> None:
        try:
            print('AcquisitionWorker: creating session...', flush=True)
            self._session = self._session_factory()
            print('AcquisitionWorker: starting session...', flush=True)
            self._session.start()
            description = getattr(self._session, 'description', '')
            if description:
                self.mode_info.emit(description)
                print(f'AcquisitionWorker: session started ({description})', flush=True)
            else:
                print('AcquisitionWorker: session started', flush=True)
            self.status.emit('running')
            while not self._should_stop:
                block = self._session.read_block()
                if block.size == 0:
                    continue
                self.data_ready.emit(block)
        except Exception as exc:  # pragma: no cover - runtime error channel
            print('AcquisitionWorker: exception during run:', exc, flush=True)
            import traceback as _tb  # local import to avoid global dependency if optional
            _tb.print_exc()
            message = str(exc).strip()
            if not message:
                message = exc.__class__.__name__
            self.error.emit(message)
        finally:
            if self._session is not None:
                try:
                    self._session.stop()
                except Exception:
                    pass
                try:
                    self._session.close()
                except Exception:
                    pass
            self.status.emit('stopped')

    def stop(self) -> None:
        print('AcquisitionWorker: stop() called', flush=True)
        self._should_stop = True
        if self._session is not None:
            try:
                self._session.stop()
                print('AcquisitionWorker: session.stop() issued', flush=True)
            except Exception:
                print('AcquisitionWorker: error during session.stop()', flush=True)
                pass


class MonitorWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Accelerometer Monitor")

        pg.setConfigOptions(antialias=True)
        self._channel_specs = load_channel_specs(cfg.CHANNELS)
        self._fs = float(cfg.SAMPLE_RATE_HZ)
        self._block = int(cfg.BLOCK_SIZE)
        self._time_window_sec = float(cfg.TIME_WINDOW_SECONDS)
        self._psd_window_sec = float(cfg.DEFAULT_PSD_WINDOW_SECONDS)
        self._psd_window_samples = max(int(self._psd_window_sec * self._fs), 32)

        self._buffer_capacity = max(int(self._time_window_sec * self._fs), self._block)
        self._buffer = np.zeros((len(self._channel_specs), self._buffer_capacity), dtype=np.float64)
        self._buffer_fill = 0

        self._worker: Optional[AcquisitionWorker] = None

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._time_plot = pg.PlotWidget()
        self._time_plot.setLabel("left", "Acceleration", units="g")
        self._time_plot.setLabel("bottom", "Time", units="s")
        self._time_plot.showGrid(x=True, y=True, alpha=0.3)
        self._time_plot.addLegend(offset=(10, 10))
        if cfg.TIME_PLOT_Y_RANGE:
            self._time_plot.setYRange(*cfg.TIME_PLOT_Y_RANGE)
        else:
            self._time_plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        self._time_plot.setXRange(-self._time_window_sec, 0.0)

        self._psd_plot = pg.PlotWidget()
        self._psd_plot.setLabel("left", "PSD", units="g^2/Hz")
        self._psd_plot.setLabel("bottom", "Frequency", units="Hz")
        self._psd_plot.showGrid(x=True, y=True, alpha=0.3)
        self._psd_plot.addLegend(offset=(10, 10))
        self._psd_plot.setLogMode(y=True)
        if cfg.PSD_Y_BOUNDS:
            self._psd_plot.setYRange(*cfg.PSD_Y_BOUNDS)

        layout.addWidget(self._time_plot)
        layout.addWidget(self._psd_plot)

        controls = QHBoxLayout()
        controls.setSpacing(12)

        self._start_button = QPushButton("Start")
        self._start_button.clicked.connect(self._start_acquisition)
        controls.addWidget(self._start_button, 0, Qt.AlignLeft)

        self._stop_button = QPushButton("Stop")
        self._stop_button.setEnabled(False)
        self._stop_button.clicked.connect(self._stop_acquisition)
        controls.addWidget(self._stop_button, 0, Qt.AlignLeft)

        controls.addWidget(QLabel("PSD window (s):"))
        self._psd_spin = QDoubleSpinBox()
        self._psd_spin.setDecimals(3)
        self._psd_spin.setRange(*cfg.PSD_WINDOW_BOUNDS)
        self._psd_spin.setSingleStep(0.1)
        self._psd_spin.setValue(self._psd_window_sec)
        self._psd_spin.valueChanged.connect(self._on_psd_window_change)
        controls.addWidget(self._psd_spin, 0, Qt.AlignLeft)
        controls.addStretch(1)

        layout.addLayout(controls)

        self.setCentralWidget(central)
        status = QStatusBar(self)
        self._status_label = QLabel("Idle")
        self._mode_label = QLabel("Mode: -")
        status.addWidget(self._status_label)
        status.addPermanentWidget(self._mode_label)
        self.setStatusBar(status)

        palette = cfg.COLOR_PALETTE
        self._time_curves = []
        self._psd_curves = []
        for idx, spec in enumerate(self._channel_specs):
            color = palette[idx % len(palette)]
            pen = pg.mkPen(color=color, width=1.4)
            self._time_curves.append(self._time_plot.plot(pen=pen, name=spec.label))
            self._psd_curves.append(self._psd_plot.plot(pen=pen, name=spec.label))

    def _start_acquisition(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        self._reset_buffers()
        self._start_button.setEnabled(False)
        self._stop_button.setEnabled(False)
        self._status_label.setText("Starting...")

        def factory() -> BaseAcquisitionSession:
            if cfg.SIMULATION_MODE or not NIDAQ_AVAILABLE:
                return SimulationSession(
                    channels=self._channel_specs,
                    fs=self._fs,
                    block_samples=self._block,
                    noise_level=cfg.SIM_NOISE_LEVEL,
                    seed=cfg.SIM_SEED,
                )
            return NIDaqSession(self._channel_specs, self._fs, self._block)

        self._worker = AcquisitionWorker(factory)
        self._worker.data_ready.connect(self._handle_block)
        self._worker.status.connect(self._handle_worker_status)
        self._worker.error.connect(self._handle_worker_error)
        self._worker.mode_info.connect(self._handle_mode_info)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _stop_acquisition(self) -> None:
        worker = self._worker
        if not worker:
            return
        self._status_label.setText("Stopping...")
        self._stop_button.setEnabled(False)
        worker.stop()

    def _reset_buffers(self) -> None:
        self._buffer.fill(0.0)
        self._buffer_fill = 0
        for curve in self._time_curves:
            curve.clear()
        for curve in self._psd_curves:
            curve.clear()

    def _handle_block(self, block: np.ndarray) -> None:
        block = np.asarray(block, dtype=np.float64)
        if block.ndim == 1:
            block = block[np.newaxis, :]
        self._append_to_buffer(block)
        self._update_time_plot()
        self._update_psd_plot()

    def _append_to_buffer(self, block: np.ndarray) -> None:
        num_new = block.shape[1]
        cap = self._buffer.shape[1]
        if num_new >= cap:
            self._buffer[:, :] = block[:, -cap:]
            self._buffer_fill = cap
            return
        if self._buffer_fill < cap:
            available = cap - self._buffer_fill
            take = min(available, num_new)
            if take > 0:
                self._buffer[:, self._buffer_fill : self._buffer_fill + take] = block[:, :take]
                self._buffer_fill += take
            if take == num_new:
                return
            remaining = num_new - take
            self._buffer[:, :-remaining] = self._buffer[:, remaining:]
            self._buffer[:, -remaining:] = block[:, take:]
            self._buffer_fill = cap
            return
        self._buffer[:, :-num_new] = self._buffer[:, num_new:]
        self._buffer[:, -num_new:] = block[:, :num_new]

    def _buffer_view(self) -> Optional[np.ndarray]:
        if self._buffer_fill == 0:
            return None
        if self._buffer_fill < self._buffer.shape[1]:
            return self._buffer[:, : self._buffer_fill]
        return self._buffer

    def _update_time_plot(self) -> None:
        data = self._buffer_view()
        if data is None:
            return
        samples = data.shape[1]
        time_axis = np.linspace(-samples / self._fs, 0, samples, endpoint=False)
        for curve, series in zip(self._time_curves, data):
            curve.setData(time_axis, series)

    def _update_psd_plot(self) -> None:
        data = self._buffer_view()
        if data is None:
            return
        psd_samples = min(data.shape[1], self._psd_window_samples)
        if psd_samples < 32:
            return
        segment = data[:, -psd_samples:]
        nperseg = max(min(psd_samples, int(self._psd_window_sec * self._fs)), 32)
        noverlap = int(nperseg * min(max(cfg.PSD_OVERLAP, 0.0), 0.95))
        freqs = None
        for curve, series in zip(self._psd_curves, segment):
            f, pxx = welch(series, fs=self._fs, nperseg=nperseg, noverlap=noverlap)
            if freqs is None:
                freqs = f
            curve.setData(freqs, pxx)

    def _handle_worker_status(self, state: str) -> None:
        print(f'GUI: worker status -> {state}', flush=True)
        if state == "running":
            self._status_label.setText("Running")
            self._start_button.setEnabled(False)
            self._stop_button.setEnabled(True)
        elif state == "stopped":
            self._status_label.setText("Stopped")
            self._start_button.setEnabled(True)
            self._stop_button.setEnabled(False)

    def _handle_worker_error(self, message: str) -> None:
        self._status_label.setText(f"Error: {message}")
        self._start_button.setEnabled(True)
        self._stop_button.setEnabled(False)

    def _handle_mode_info(self, description: str) -> None:
        self._mode_label.setText(f"Mode: {description}")

    def _on_worker_finished(self) -> None:
        print('GUI: worker finished', flush=True)
        self._worker = None
        self._start_button.setEnabled(True)
        self._stop_button.setEnabled(False)
        self._status_label.setText("Stopped")
    def _on_psd_window_change(self, value: float) -> None:
        self._psd_window_sec = float(value)
        self._psd_window_samples = max(int(self._psd_window_sec * self._fs), 32)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._worker:
            self._worker.stop()
            self._worker.wait(1500)
        super().closeEvent(event)

def main() -> None:
    app = QApplication(sys.argv)
    window = MonitorWindow()
    window.resize(1000, 800)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
