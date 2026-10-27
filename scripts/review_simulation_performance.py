#!/usr/bin/env python3
"""Measure default-rate Qt simulation, recording and Stop without NI hardware.

Uses QApplication.exec(), rather than a busy Python/QTest polling loop, so the
worker and GUI share the GIL as they do in the desktop application.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

import app
import simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('random', 'sine_sweep', 'both'), default='both')
    parser.add_argument('--blocks', type=int, default=6)
    parser.add_argument('--check-budget', action='store_true', help='Require 1 s block cadence within 25%% and Stop cleanup within 0.5 s')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/simulation-performance/gui')
    args = parser.parse_args()
    if args.blocks < 2:
        parser.error('Use at least two blocks to measure cadence.')
    args.output.mkdir(parents=True, exist_ok=True)
    hardware_attempts = []

    def forbidden_task(*unused_args, **unused_kwargs):
        hardware_attempts.append(True)
        raise RuntimeError('Performance review must never open a real NI task')

    if app.nidaqmx is not None:
        app.nidaqmx.Task = forbidden_task
    app.NIDAQMX_AVAILABLE = False
    qt = QApplication([])
    window = app.MainWindow()
    modes = ['random', 'sine_sweep'] if args.mode == 'both' else [args.mode]
    cases, errors = [], []
    state = {}
    original_read = simulation.MockAnalogSingleChannelReader.read_many_sample

    def timed_read(reader, *call_args, **call_kwargs):
        start = time.perf_counter()
        try:
            return original_read(reader, *call_args, **call_kwargs)
        finally:
            state.setdefault('read_times', []).append(time.perf_counter() - start)

    simulation.MockAnalogSingleChannelReader.read_many_sample = timed_read

    def heartbeat():
        if 'heartbeat_gaps' not in state:
            return
        now = time.perf_counter()
        if 'heartbeat' in state:
            state['heartbeat_gaps'].append(now - state['heartbeat'])
        state['heartbeat'] = now

    def stop():
        if 'stop_requested' not in state:
            state['stop_requested'] = time.perf_counter()
            window.stop_controller()

    def metrics(packet):
        state['received'].append(time.perf_counter())
        state['metrics'].append(dict(packet))
        if len(state['received']) >= args.blocks:
            stop()

    def finish_case():
        try:
            report = window.last_report
            assert report and report['outcome'] == 'stopped', report
            assert report['backend'] == 'Simulation', report
            assert report['blocks'] >= args.blocks, report
            assert report['settings']['fs'] == 51200
            assert report['settings']['block_seconds'] == 1
            assert report['settings']['num_input_channels'] == 4
            intervals = np.diff(state['received']).tolist()
            log_dir = Path(report['log_dir'])
            with h5py.File(log_dir / 'data.h5') as recording:
                ao = recording['ao'][:]
                ai = recording['ai'][:]
                assert ao.shape == (report['blocks'], 51200)
                assert ai.shape == (report['blocks'], 4, 51200)
                assert np.all(np.isfinite(ai)) and np.all(np.isfinite(ao))
                assert np.max(np.abs(ao)) <= report['settings']['ao_volt_limit'] + 1e-6
                assert len(recording['metadata/timestamp']) == report['blocks']
                if state['mode'] == 'sine_sweep':
                    assert len(recording['metadata/frequency_hz']) == report['blocks']
            assert (log_dir / 'run-summary.json').is_file()
            result = {
                'mode': state['mode'], 'fs_hz': 51200, 'block_samples': 51200,
                'channels': 4, 'blocks': report['blocks'],
                'first_block_wall_s': state['received'][0] - state['start'],
                'median_block_interval_s': statistics.median(intervals),
                'max_block_interval_s': max(intervals),
                'median_mock_read_s': statistics.median(state['read_times']),
                'max_mock_read_s': max(state['read_times']),
                'max_gui_heartbeat_gap_s': max(state['heartbeat_gaps'], default=0),
                'stop_to_finished_s': time.perf_counter() - state['stop_requested'],
                'outcome': report['outcome'], 'recording_checked': True,
            }
            if args.check_budget:
                assert .75 <= result['median_block_interval_s'] <= 1.25, result
                assert result['max_block_interval_s'] <= 1.5, result
                assert result['max_mock_read_s'] < .5, result
                assert result['stop_to_finished_s'] < .5, result
            cases.append(result)
            print(json.dumps(result, indent=2), flush=True)
        except Exception as error:
            errors.append(str(error))
            modes.clear()
        # Let the browser's deferred layout/deletions settle before capturing
        # the retained report. This delay is outside the measured Stop time.
        mode = state.get('mode', 'review')

        def capture_and_continue():
            window.grab().save(str(args.output / (mode + '.png')))
            if modes:
                QTimer.singleShot(100, start_case)
            else:
                window.close()

        QTimer.singleShot(80, capture_and_continue)

    def start_case():
        mode = modes.pop(0)
        state.clear()
        state.update(mode=mode, received=[], metrics=[], read_times=[], heartbeat_gaps=[])
        window.backend_combo.setCurrentIndex(1)
        window.mode_combo.setCurrentIndex(1 if mode == 'sine_sweep' else 0)
        window.config_tab.fields['data_log_dir'].setText(str(args.output / 'recordings'))
        window.config_tab.fields['data_log_enabled'].setChecked(True)
        if not window.config_tab.apply_config():
            errors.append(window.config_tab.validation_label.text())
            window.close()
            return
        window.config_dock.hide()
        state['start'] = time.perf_counter()
        if not window.start_controller():
            errors.append('Simulation start was rejected')
            window.close()
            return
        window.controller_worker.metrics_data_ready.connect(metrics)
        window.controller_worker.finished.connect(finish_case)

    def timeout():
        errors.append('Performance review exceeded 90 s')
        modes.clear()
        if window.busy:
            stop()
        else:
            window.close()

    pulse = QTimer()
    pulse.setInterval(50)
    pulse.timeout.connect(heartbeat)
    pulse.start()
    watchdog = QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(timeout)
    watchdog.start(90000)
    window.show()
    QTimer.singleShot(100, start_case)
    qt.exec()
    result = {'method': 'Qt event loop, live plots, default-rate simulation, real HDF5 recording',
              'hardware_task_attempts': len(hardware_attempts), 'cases': cases, 'errors': errors}
    (args.output / 'validation.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return 1 if errors or hardware_attempts or not cases else 0


if __name__ == '__main__':
    raise SystemExit(main())
