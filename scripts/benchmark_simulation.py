#!/usr/bin/env python3
"""Benchmark four-channel mock acquisition and optional local Git-reference parity.

Example: .venv/bin/python scripts/benchmark_simulation.py --reference-ref 5105dbc
The reference is trusted local repository code; no NI bindings or Qt are loaded.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import config
import simulation


def measure(module, inputs):
    np.random.seed(20261026)
    with contextlib.redirect_stdout(io.StringIO()):
        writer, reader, _, _ = module.create_simulation_system(
            config.FS, config.SIM_PLANT_GAIN, config.SIM_RESONANCES,
            config.SIM_NOISE_LEVEL, config.SIM_DELAY_SAMPLES,
            config.SIM_NONLINEARITY, len(config.INPUT_CHANNEL_LABELS))
    timings, responses = [], []
    for block in inputs:
        data = np.empty((len(config.INPUT_CHANNEL_LABELS), len(block)))
        start = time.perf_counter()
        writer.write_many_sample(block)
        reader.read_many_sample(data, len(block))
        timings.append(time.perf_counter() - start)
        responses.append(data)
    return timings, np.asarray(responses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-ref', help='Optional trusted local Git ref containing the previous simulation.py')
    parser.add_argument('--blocks', type=int, default=5)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/simulation-performance/benchmark.json')
    args = parser.parse_args()
    if args.blocks < 1:
        parser.error('Use at least one block.')
    count = int(config.FS * config.BLOCK_SECONDS)
    t = np.arange(count) / config.FS
    inputs = [.15 * np.sin(2 * np.pi * 150 * t) + .05 * np.sin(2 * np.pi * 800 * t)
              + np.random.default_rng(i).normal(0, .02, count) for i in range(args.blocks)]
    current_times, actual = measure(simulation, inputs)
    result = {'fs_hz': config.FS, 'block_seconds': config.BLOCK_SECONDS,
              'samples_per_block': count, 'channels': len(config.INPUT_CHANNEL_LABELS),
              'blocks': args.blocks, 'seed': 20261026,
              'current_block_wall_s': current_times,
              'current_median_s': statistics.median(current_times),
              'simulation_sha256': hashlib.sha256((ROOT / 'simulation.py').read_bytes()).hexdigest()}
    if args.reference_ref:
        source = subprocess.check_output(['git', 'show', args.reference_ref + ':simulation.py'], cwd=ROOT, text=True)
        previous = types.ModuleType('simulation_reference')
        exec(compile(source, '<trusted Git reference simulation.py>', 'exec'), previous.__dict__)
        previous_times, expected = measure(previous, inputs)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)
        result.update(reference_ref=args.reference_ref,
                      reference_sha256=hashlib.sha256(source.encode()).hexdigest(),
                      reference_block_wall_s=previous_times,
                      reference_median_s=statistics.median(previous_times),
                      median_speedup=statistics.median(previous_times) / statistics.median(current_times),
                      max_abs_difference_g=float(np.max(np.abs(actual - expected))),
                      parity_rtol=1e-10, parity_atol_g=1e-11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
