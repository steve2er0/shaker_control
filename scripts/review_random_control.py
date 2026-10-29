#!/usr/bin/env python3
"""Review sample-time random control in accelerated simulation, with no DAQ tasks.

This is a numerical tracking check, not a hardware stability or wall-time test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import app
from run_settings import RunSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blocks', type=int, default=160)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/random-control-review/numerical-validation.json')
    args = parser.parse_args()
    if args.blocks < 40:
        parser.error('Use at least 40 blocks to include startup and tracking.')
    attempts = []

    def forbidden(*positional, **keywords):
        attempts.append(True)
        raise AssertionError('Numerical review must never open a hardware task')

    if app.nidaqmx is not None:
        app.nidaqmx.Task = forbidden
    cases = []
    for name, gain, profile in (
        ('default_profile_gain_4', 4., None),
        ('default_profile_gain_40', 40., None),
        ('broadband_profile_gain_40', 40., [(100., .01), (300., .04), (1000., .04), (2000., .01)]),
    ):
        np.random.seed(20261029)
        values = app.SharedConfig().as_dict()
        values.update(simulation_mode=True, data_log_enabled=False, sim_plant_gain=gain)
        if profile is not None:
            values['target_psd_points'] = profile
        worker = app.ControllerWorker(RunSettings.capture(values))
        packets = []
        publish = worker._publish_metrics

        def receive(packet):
            publish(packet)
            packets.append(dict(packet))
            if len(packets) >= args.blocks:
                worker.stop()

        worker._publish_metrics = receive
        worker.should_stop.wait = lambda seconds: False
        worker.run_controller()
        assert worker.terminal_state == 'stopped', worker.error_message
        assert worker.blocks == args.blocks
        ratios = np.array([m['accel_rms_g'] / worker.shared_config.target_rms_g for m in packets])
        case = {'case': name, 'simulation_plant_gain_g_per_v': gain, 'sample_seconds': args.blocks * worker.shared_config.block_seconds,
                'blocks': worker.blocks, 'target_g_rms': worker.shared_config.target_rms_g,
                'max_target_ratio': float(np.max(ratios)), 'last_20_mean_target_ratio': float(np.mean(ratios[-20:])),
                'last_reference_level': packets[-1]['level_fraction'],
                'first_95pct_block': next((i + 1 for i, ratio in enumerate(ratios) if ratio >= .95), None),
                'last_integral': packets[-1]['control_integral'], 'settings': worker.shared_config.as_dict()}
        assert case['max_target_ratio'] < 1.15, case
        assert abs(case['last_20_mean_target_ratio'] - 1.) < .05, case
        assert case['last_reference_level'] == 1., case
        cases.append(case)
        print(json.dumps({key: value for key, value in case.items() if key != 'settings'}))
    assert not attempts
    result = {'method': 'Accelerated sample-time simulation; software waits omitted; actual control code and synthetic plant',
              'limitations': 'Not a hardware stability, synchronization, calibration, or wall-clock timing qualification.',
              'seed': 20261029, 'hardware_task_attempts': len(attempts), 'cases': cases,
              'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ('app.py', 'rv_controller.py', 'simulation.py', 'run_settings.py')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
