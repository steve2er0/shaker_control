# Shaker Control

Python desktop tools for random vibration control, stepped sine sweeps, and
accelerometer monitoring. The GUIs use PySide6 and PyQtGraph; signal processing
uses NumPy and SciPy. Acquisition can use a synthetic plant or NI-DAQ hardware.

The repository is a development application. Numerical limiter tests do not
establish shaker performance, hardware stop behavior, or production qualification.

## Install

Use a virtual environment. The pinned setup was validated with Python 3.9.6 on
macOS ARM64. Use Python 3.9 to reproduce that setup; newer interpreter/platform
combinations may require updated dependency pins and their own checks.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` installs the GUI, numerical tools, and HDF5 logging support.
Simulation does not require `nidaqmx`. For hardware support:

```bash
python -m pip install -r requirements-hardware.txt
```

This installs the Python bindings. A compatible NI-DAQmx driver, supported device,
and configured channels are also required; pip does not install that driver.

## Run the controller

```bash
python app.py
```

The application opens three tabs: **Configuration**, **Controller**, and
**Real-Time Data**. Before starting a simulation, select **Enable Simulation
Mode** and click **Apply Configuration**. Alternatively, set
`SIMULATION_MODE = True` in `config.py` before launching.

The checked-in configuration selects **hardware mode**. If `nidaqmx` is absent,
or hardware initialization fails, the controller falls back to simulation and
prints the mode to the console. A simulated run is not a hardware measurement.

Choose **Random Vibration** or **Sine Sweep**, apply settings, then use **Start
Controller** and **Stop Controller**. Stop before changing settings. GUI edits
update the in-memory configuration for that session; they do not save `config.py`.
DAQ channel names, accelerometer sensitivity, the random PSD profile, and the
sine drive table are configured in `config.py`.

### Random vibration

The controller generates band-limited noise, shapes it through logarithmically
spaced equalizer bands, estimates response PSD using Welch's method, and adjusts
the overall level with PI control. It also estimates plant gain from measured
acceleration versus commanded voltage. Equalizer adaptation is gated by level.

The current default PSD points, in Hz and g²/Hz, are:

```python
TARGET_PSD_POINTS = [
    (100.0, 0.001),
    (300.0, 0.004),
    (350.0, 0.004),
    (500.0, 0.001),
]
```

Log-log interpolation and integration give approximately **0.999 g RMS** over
100–500 Hz. The level fraction scales acceleration amplitude, so a fraction of
0.1 requests 0.01 times the target PSD. Actual output also depends on the minimum
drive floor, limiters, and plant response; random mode enforces a 0.05 V RMS
minimum command before limiting.

### Sine sweep and calibration

Select **Sine Sweep** in the main GUI. It uses logarithmically spaced frequency
steps and a phase-continuous oscillator. Dwell time respects the configured
octaves-per-minute rate, minimum dwell, and processing block duration. The first
step has an additional initial settling dwell (3 seconds by default).

Amplitude is derived from `SINE_SWEEP_DRIVE_TABLE`, interpreted as peak volts
required per 1 g-peak, multiplied by the target acceleration and level fraction.
Measured acceleration is displayed and logged; it does not close an amplitude
feedback loop. The drive table must match the actual shaker, fixture, and load.
Displayed g-peak is estimated as sqrt(2) times block standard deviation, assuming
a sinusoidal response; noise and harmonics affect this estimate. The plot and
CSV summary retain the largest estimate during each frequency dwell.

The calibration CLI iteratively adjusts voltage at each frequency to produce a
target acceleration. For an explicit simulation example:

```bash
python record_1g_sine_sweep.py --simulation \
    --frequencies 100 300 500 --target-g 1 \
    --outfile 1g_sine_sweep_config.txt
```

Use `--help` for frequency spacing, tolerances, iteration counts, and voltage
limits. `--hardware` explicitly selects hardware and fails if it is unavailable.
Copy the generated frequency/voltage pairs into `SINE_SWEEP_DRIVE_TABLE` after
reviewing their provenance. The GUI table assumes a 1 g-peak calibration;
normalize tables generated for another target before using them.

`sine_sweep.py` contains reusable generation/interpolation utilities and is not
a standalone application.

## Run the accelerometer monitor

Set `SIMULATION_MODE = True` in `monitor_config.py` for synthetic traces, then:

```bash
python accel_monitor.py
```

This separate application displays multichannel acceleration in g and PSD in
g²/Hz, with an adjustable PSD window. Its default hardware configuration has
one channel (`Dev1/ai0`) at 2,048 Hz, with 512-sample blocks. Add entries to
`CHANNELS` in `monitor_config.py` for additional inputs; configure each channel's
label, sensitivity in mV/g, IEPE excitation, and terminal configuration there.
Use **Start** and **Stop** to control acquisition. Missing Python hardware bindings
select simulation; hardware initialization errors are reported by the worker.

## Current controller defaults

| Setting | Value |
|---|---|
| Test mode | Random vibration |
| Acquisition backend | Hardware (`SIMULATION_MODE = False`) |
| Sample rate / processing block / buffer | 51,200 Hz / 1 s / 4 s |
| Welch segment length | 4,096 samples (12.5 Hz bin spacing) |
| Equalizer | 36 bands |
| Initial random level fraction | 0.1 |
| PI gains | Kp = 0.9, Ki = 0.06 |
| AO absolute voltage clip | ±1.8 V |
| RMS voltage limit / crest-factor limit | 1.4 V / 5 |
| Sine sweep | 20–2,000 Hz, 3 g-peak, 1 octave/min, repeating |
| Initial sine settling dwell | 3 s extra at the first step |
| Logging | Enabled, under `data_logs/` |
| Hardware AI / AO | `Dev1/ai0` / `Dev1/ao0` |
| Sensitivity / IEPE excitation | 100 mV/g / 2.1 mA |

The main controller currently acquires one hardware control accelerometer.
Additional response channels are simulated; the standalone monitor supports
multiple hardware inputs. AI/AO clock sharing is disabled by default.

## Limiters and recorded data

Voltage blocks pass through RMS limiting, crest-factor limiting, and then a
separate absolute voltage clip in the controller. The limiter RMS definition is
`sqrt(mean(x**2))`, including any DC component. Crest factor is
`max(abs(x)) / RMS(x)`; empty and all-zero blocks use zero for both metrics.

Crest limiting compresses individual peaks above a soft knee, then remeasures
output RMS and adjusts the clipping threshold if necessary. It never increases
sample magnitudes. If a sparse waveform cannot satisfy the ratio by attenuation,
it returns a zero block: scaling a single impulse cannot change its crest factor.
Invalid waveforms or limiter settings raise `ValueError`. Reported output RMS
and crest factor describe the final waveform after both limiters, before the
controller's separate absolute voltage clip. Peak compression can alter PSD and
introduce harmonics; it does not guarantee acceleration tracking.

Each logged GUI run creates a timestamped directory containing:

- `data.h5`: AO voltage and AI acceleration blocks stored as float32, plus sample
  rate and per-block timestamps, level fraction, and saturation fraction.
- `preview_psd.csv`: random-run PSD of the last recorded block, per channel.
- `sine_summary.csv`: sine-run frequency, measured g-peak, and target g-peak.

CSV summaries are exported when the logger closes. They are previews, not a
whole-run PSD qualification report. Logs and generated calibration tables are
ignored by Git.

## Checks and remaining limitations

Install the test dependencies and run the numerical regression suite:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m pip check
```

Tests cover measured output crest factor, sparse impulses, unchanged acceptable
signals, extreme amplitudes, invalid inputs, RMS limiting including DC, and final
combined-limiter statistics. They do not open DAQ tasks or operate a shaker.

The PI controller currently has no anti-windup mechanism. Random mode reports
saturation without automatic level backoff; sine mode has saturation backoff.
Hardware control I/O uses unbounded waits, and stopping/closing the GUI has
bounded thread waits; stop behavior still needs hardware validation. Simulation
includes prescribed resonances, noise, delay, and nonlinearity, rather than a
validated model of a particular shaker. Tracking accuracy, stability, and physical
protection require separate evidence for the configured equipment and load.

## Repository map

| File | Responsibility |
|---|---|
| `app.py` | Controller GUI, acquisition worker, and run logger |
| `config.py` | Controller, hardware, simulation, target, and sweep settings |
| `rv_controller.py` | Random control, equalization, noise generation, and limiters |
| `simulation.py` | Synthetic plant and mock DAQ |
| `sine_sweep.py` | Sweep generators, oscillator, and drive-table interpolation |
| `record_1g_sine_sweep.py` | Command-line drive calibration |
| `accel_monitor.py` / `monitor_config.py` | Standalone multichannel monitor |
| `tests/test_limiters.py` | Numerical limiter regressions |
| `requirements*.txt` | Runtime, test, and optional hardware dependency pins |

License: MIT; see `LICENSE`.
