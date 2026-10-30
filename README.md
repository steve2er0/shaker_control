# Shaker Control

Python desktop tools for random vibration control, stepped sine sweeps, and
accelerometer monitoring. The GUIs use PySide6 and PyQtGraph; signal processing
uses NumPy and SciPy. Acquisition can use a synthetic plant or NI-DAQ hardware.

The repository is a development application. Numerical limiter tests do not
establish shaker performance, hardware stop behavior, or production qualification.

## Install

The supported setup targets **64-bit CPython 3.13**, **Windows 11 x64**, and
**macOS**. Runtime dependencies are pinned in `requirements.txt`. Use a separate
environment for this baseline; an existing Python 3.9 `.venv` can stay in place,
but the updated requirements must not be installed into it. Virtual environments
are local to one OS and computer; recreate them after cloning on another machine.

### Windows 11 x64

Install [64-bit Python 3.13](https://www.python.org/downloads/windows/) with the
Python launcher, then clone or extract this repository into a writable folder.
Double-click **setup-windows.cmd**. It creates `.venv-py313`, installs prebuilt
dependencies, and checks their consistency. No activation or administrator
terminal is needed. Then open:

- **start-controller.cmd** for Shaker Control.
- **start-monitor.cmd** for the accelerometer monitor.

The launchers find the checkout independently of the current working directory.
They keep console diagnostics and write a log under
`%LOCALAPPDATA%\ShakerControl\Logs`. A failed launch leaves the console open.
Setup pauses so its result remains visible.

Optional NI bindings and test dependencies can be installed from Command Prompt:

```bat
setup-windows.cmd -Hardware
setup-windows.cmd -Development
```

These options can be combined. The PowerShell scripts in `scripts/` also work
directly; `setup-windows.ps1 -PythonExecutable "C:\path\to\python.exe"` selects a
specific 64-bit Python 3.13 interpreter. The `.cmd` entry points set script
execution policy only for their child PowerShell process.

### macOS

Install Python 3.13, then create the new environment without changing an existing
`.venv`:

```bash
python3.13 -m venv .venv-py313
source .venv-py313/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

The remaining `python` commands assume this environment is active. In Windows
Command Prompt, run `.venv-py313\Scripts\activate.bat`, or replace `python` with
`.venv-py313\Scripts\python.exe`. The Windows launchers work without activation.

`requirements.txt` installs the GUI, numerical tools, and HDF5 logging support.
Simulation does not require `nidaqmx`. For hardware support:

```bash
python -m pip install -r requirements-hardware.txt
```

This installs the Python bindings. A compatible NI-DAQmx driver, supported device,
and configured channels are also required; pip does not install that driver.
Windows hardware operation still requires validation with the actual device.

## Run the controller

```bash
python app.py
```

The floating workspace uses Acoustic Workbench's layout: a slim app bar, grouped
task ribbon, collapsible **Test browser**, large response plot, and contextual
**Inspector**. The task ribbon offers **Test**, **Acquisition**, **Control**,
**Results**, and **View**. The plot's compact view selector switches between
**Response**, **Channels**, **Diagnostics**, and **Results**. Test mode, backend,
run state, **Start**, and **Stop** stay available in every view.

1. Choose **Random vibration** or **Sine sweep** and **Hardware** or **Simulation**.
2. Select **Target** in the ribbon or test browser to edit the PSD or sine
   profile. Use **Acquisition → Clock** for sample rate and block settings,
   **Channels / Backend setup** for input scaling or simulation parameters, and
   **Control** for voltage limits, PI/ramp tuning, equalizer, or drive calibration.
   **Inspector ▾ → All test settings** opens the full form. Only the selected
   context, test mode, and backend controls are shown. Sine acceleration explicitly
   uses g peak or g RMS; ramp controls use acceleration amplitude percent.
3. Click **Apply setup**. Invalid combinations show field errors. Unapplied draft
   changes disable Start. Apply and **Discard draft** remain visible below the
   scrolling form. Switching inspector context preserves the draft; Discard
   restores the accepted settings and mode/backend selectors.
4. Click **Start simulation** or **Start hardware**. The workspace shows Starting
   until the selected backend and requested recording initialize successfully.
5. Use **Stop** from any view, or allow a nonrepeating sweep to reach Complete.
   **Results** retains the outcome, captured setup, actual backend, last-block
   metrics, clock/input mode, calibration source, and recording folder.

The checked-in defaults request **hardware**. Missing bindings or initialization
errors produce a retained **Fault**; they never select simulation automatically.
The fault banner offers an explicit switch to simulation, followed by Apply and
Start. A simulated run is not a hardware measurement. Dismiss clears the banner;
starting another run clears the previous fault while Results retains its report.

Each run captures an immutable, validated setup. Acquisition settings, mode,
backend, and preset controls are locked until its worker finishes cleanup. Stop
also works during Starting. Stopping and closing wait asynchronously for finite
I/O and recording finalization; an active thread is not discarded on a timer.

**Test → Preset → Load preset / Save preset** use JSON setup files. Loading validates before
changing accepted settings and records the source path without rewriting the
imported file. GUI edits persist for the session or in a saved preset; they do not
rewrite `config.py`, which supplies launch defaults. Targets, sine drive tables,
calibration provenance, channels, sensitivity, and recording path are editable in
the setup panel. Optional advanced simulation and control defaults remain in
`config.py` and are included in captured settings.

Validation requires ordered positive PSD points below 95% of Nyquist for random
equalizer filters, sine frequencies below Nyquist, a PSD segment that fits one
processing block, and a DAQ buffer at least one block long. Random PSD estimation
must place at least one bin in the target band.
The setup summary and command scaling share the captured piecewise PSD integral.
Zero PI gains are allowed; random ramp starts support 0–200% and sine starts
0–100%, matching their
engines. The controller and channel viewer use the same accepted sample rate and
effective channel count. Applying settings or starting a new run clears live
curves and histories; earlier reports remain in Results for the session.

Channel display uses a bounded mailbox and a 100 ms render timer. Hidden channel
plots do not redraw; visible time traces use at most 4,096 peak-preserving points
per channel instead of painting every acquired sample. The full two-second sample
buffer and raw recording retain the original values. Delayed GUI frames discard
obsolete display work without changing acquisition, control, or recording data.
Response-channel PSD previews are computed when displayed, with a final refresh
after Stop. Historical reports remain readable even when newer setup checks would
reject their settings for a new run.

When measured-response stops are enabled, setup validation rejects a target above
the configured RMS/peak limits. Switching from a low-level sine profile to a
higher random target requires reviewing those limits or lowering the target;
limits are not raised automatically.

With the amplifier enabled, the updated Channels view was verified on the NI
USB-4431 at **0.5 g-peak and 200 Hz** for 96 seconds. Channels was visible at all
882 display checks, and time traces remained bounded to 4,096 points. The 85-second
settled portion averaged **0.50011 g peak**, ranging from **0.49971 to 0.50299 g
peak** (worst error 0.60%). There were no acquisition faults, underflows, output
limiting, or protection trips. Maximum unread backlog was 805 samples (15.7 ms),
and the largest observed GUI heartbeat interval was 183 ms. See the
[hardware report](docs/hardware-validation/2026-10-30-channels-sine/test-report.html),
[time-tracking chart](docs/hardware-validation/2026-10-30-channels-sine/time-tracking.png),
and [verification summary](docs/hardware-validation/2026-10-30-channels-sine/validation.json).

The browser lists the applied target, limits, calibration source, configured
channels, sample clock, and completed session runs. Selecting a channel shows its
time/PSD view and input settings; selecting a run shows that run's captured report.
Historic reports do not reload old sample plots. **Run history** opens the current
session's setup/start/stop/fault events; physical actions have no undo operation.
**View → Fit plots** fits the current plots. Closing the inspector restores the
plot space. In narrower windows, opening the inspector temporarily hides the
browser; closing it restores the browser if it was enabled. The ribbon reflows
its run controls onto another row when needed. The status strip retains acquisition
resolution and recording state. The interface follows the launch palette for light
or dark appearance and opens at 1,360 × 860 (minimum 900 × 620).

### Random vibration

The controller generates band-limited noise, shapes it through logarithmically
spaced equalizer bands, and estimates response PSD using Welch's method. Random
control starts with a zero baseline and a low-level identification drive (1 mV RMS
by default, editable under **Control**). It holds that pilot if the sensor cannot
resolve its response above the baseline; initial silence never causes an assumed
plant gain to decay and drive to escalate blindly.

Once a response is identified, inverse-gain feedforward and a bounded PI trim on
normalized RMS error adjust voltage. The acceleration ramp is a separate setpoint,
advancing toward 100% only while tracking within 15% of its current level. Ramp
growth respects both the requested rate and 20% of the current amplitude per
second; voltage increases are capped at 50% per second. The integral uses sample
duration, is bounded, and freezes when further correction would push into voltage
or rate limits. PI gains now refer to normalized RMS trim, with integral gain in
1/s, rather than absolute PSD density. The equalizer uses the ramp-scaled target,
with bounded band adjustments, and pauses during identification or command limiting.
The default normalized proportional gain is 0.3; existing presets retain their
saved gains. Gain selection still depends on the configured plant and block delay.

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
0.1 requests 0.01 times the target PSD. There is no forced 0.05 V RMS drive floor;
small corrections and zero commands remain possible. Actual response still depends
on voltage limits, sensor calibration, plant dynamics, and clock/filter delay.

Hardware output regeneration is disabled. One zero block primes the output, and
queued waveform order associates each response with its nominal applied command,
instead of the newest command waiting to play. Missed output deadlines fault
rather than repeating an old drive waveform. This software association does not
establish physical AI/AO synchronization or remove device filter delay.

### Sine sweep and calibration

Select **Sine Sweep** in the main GUI. It uses logarithmically spaced frequency
steps and a phase-continuous oscillator. Dwell time respects the configured
octaves-per-minute rate, minimum dwell, and processing block duration. The first
step has an additional initial settling dwell (3 seconds by default).
Dwell rounds up to whole processing blocks. Hardware scheduling counts the
blocks queued to AO and captures the final endpoint before completion; the
initial zero priming block does not add a block to every frequency dwell.

New sessions enable **Enable sine amplitude feedback**. The table supplies a
feedforward estimate in V peak per 1 g peak; a low pilot identifies measured
gain before drive rises toward target. Gain updates use the confirmed applied
command and a phase-independent fundamental fit. Unresolved or clipped feedback
cannot accumulate correction. Upward voltage changes are bounded by the configured
fraction per second and voltage limits. The ramp advances after tracking, and a
frequency holds until the configured consecutive full-target blocks settle within
tolerance. Failure to settle before the deadline faults and stops the run.

The live sine value/curve shows the fitted fundamental, with noise/harmonics
reported separately as residual RMS. Broadband sqrt(2) × standard deviation and
raw sample peaks remain available in recordings and reports. Both sweep directions
include the exact endpoint; final queued measurements are captured before stopping.
Disabling feedback selects the original feedforward behavior. Legacy imported
presets without the feedback flag retain that behavior.

Open **Control → Calibration → Calibration wizard** while stopped. Review the
backend, frequency range, target in g peak, and output ceiling; identify the
shaker/fixture/load, amplifier setting, and sensor. Start explicitly runs the
selected hardware or simulation backend through the normal worker and protection
checks. Stop/Close cancels through normal cleanup. Only a fully completed,
in-tolerance calibration enables **Apply table and save preset**. The table is
normalized to V peak per 1 g peak regardless of calibration target. Applying saves
a unique preset under Documents/ShakerControl/Presets with backend, date, sensor
sensitivity, equipment, amplitude, and frequency coverage. Simulation provenance
is retained; changing backend/sensitivity/channels prevents applying that result.

On October 29, 2026, actual NI USB-4431 sweeps from 100–500 Hz and back completed
at **0.5 g-peak**, using a newly measured table for the connected setup. Each
direction recorded 29 frequencies in 64 one-second blocks, including priming and
initial settling. The last block at every frequency was within **0.5%** of target,
with no reported faults or output limiting. See the
[results chart](docs/hardware-validation/2026-10-29-sine-sweeps/sine-sweeps.png),
[measurements and validation](docs/hardware-validation/2026-10-29-sine-sweeps/validation.json),
and [tested setup preset](docs/hardware-validation/2026-10-29-sine-sweeps/validated-setup-preset.json).
These earlier sweeps used feedforward control; they do not validate the newly
added feedback/protection/wizard features on hardware. That preset applies to
this shaker/amplifier/fixture/sensor setup; other levels
and frequencies were not validated by these sweeps.

The updated closed-loop controller was subsequently tested on the same NI USB-4431
setup at **0.5 g-peak, 100–500 Hz**, with only ai0 enabled at 10 mV/g and 2.1 mA
IEPE. Both directions completed all 29 frequencies and passed the configured
**±5% tolerance over each final three-block settling window**. The worst individual
reading in those windows was 4.83% from target; the final block at each frequency
was within 0.53%. The largest fitted-tone transition peak was 0.52962 g peak,
and the largest absolute acquired sample was 0.66271 g. There were no acquisition
faults, limiting blocks, or protection trips. The upward/downward runs recorded
130/127 one-second blocks, with automatic reports and raw waveforms saved.
See the [upward report](docs/hardware-validation/2026-10-29-sine-feedback/ascending/test-report.html),
[downward report](docs/hardware-validation/2026-10-29-sine-feedback/descending/test-report.html),
and [waveform verification and settings](docs/hardware-validation/2026-10-29-sine-feedback/validation.json).
Response protections were enabled, and the driver accepted idle-zero output and
input-overload checks. No deliberate protection trip was injected, and physical
stop latency/output voltage and the calibration wizard were not validated by
these normal sweeps.

The calibration CLI iteratively adjusts voltage at each frequency to produce a
target acceleration. For an explicit simulation example:

```bash
python record_1g_sine_sweep.py --simulation \
    --frequencies 100 300 500 --target-g 1 \
    --outfile 1g_sine_sweep_config.txt
```

Use `--help` for frequency spacing, tolerances, iteration counts, and voltage
limits. `--hardware` explicitly selects hardware and fails if it is unavailable.
Paste the generated frequency/voltage pairs into the GUI drive table (or set
`SINE_SWEEP_DRIVE_TABLE`) after reviewing their provenance. Record the calibration
source in setup or `SINE_SWEEP_CALIBRATION_SOURCE`. The table assumes a 1 g-peak
calibration; normalize tables generated for another target before using them.

`sine_sweep.py` contains reusable generation/interpolation utilities and is not
a standalone application.

### Measured-response protection

**Control → Control → Measured-response protection** configures broadband RMS
and absolute raw-sample peak limits for every enabled sensor, a control-RMS limit
relative to the full target, maximum unread-data age, and a driven-output feedback
loss timeout. Startup allowance and minimum feedback/drive thresholds are explicit.
Random feedback-loss detection uses control-band RMS; sine uses fitted-tone RMS.
NI input overload flags are checked when supported; unsupported detection is
reported in stream diagnostics. Nonfinite input always faults.

Threshold violations fault the worker and stop/close output and input tasks before
recording finalization. With raw recording enabled, `protection-trip-block.npz`
preserves the triggering acceleration and associated command. These checks occur
once per acquired block: block/read timeout and driver cleanup affect stop latency.
The driver is asked to use zero-voltage idle output where supported; support is
recorded in diagnostics and physical stop behavior still requires device validation.
Choose the limits for the actual equipment; this software is not an independent
hardware interlock. Defaults are 10 g broadband RMS, 30 g raw peak, twice the full
control RMS target, 2 seconds unread data, and 3 seconds unresolved driven feedback
after a 3-second startup allowance.

### Automatic tracking reports

Enable automatic reports under **Recording** and set tracking tolerance and the
minimum random assessment time. Every run receives a separate tracking status:
**PASS**, **FAIL**, or **INCOMPLETE**. A stopped/partial finite sine sweep cannot
pass. Sine assessment evaluates every block in the last configured settling window
at each frequency; transient maxima are separate. Random assessment excludes ramp
and initial settling, then evaluates all full-target blocks and requires the
configured minimum duration. Any acquisition/protection fault prevents a pass.

Reports include `test-report.html`, `test-report.json`, `block-metrics.csv`, and
`settled-sine-results.csv` for sine. The desktop automatically adds `tracking.png`
with tolerance bands after acquisition stops; headless worker use still exports
HTML/CSV without requiring a GUI chart. Open the report from **Results → Open HTML
test report**. Reports share the raw recording folder when enabled; otherwise they
go to Documents/ShakerControl/Reports. Tracking assessment does not establish
sample-level AI/AO synchronization or whole-run PSD qualification.

Reproduce the simulation-only feature review with
`python scripts/review_control_features.py`. It prohibits real NI tasks and
omits simulation pacing waits. The saved evidence distinguishes sample seconds
from wall seconds; it is not a hardware stability or physical stop test.
Saved review artifacts include the [validation summary](docs/control-features/validation.json),
[calibration wizard](docs/control-features/calibration-wizard.png), and a
[sample automatic report](docs/control-features/sample-report/test-report.html).

## Run the accelerometer monitor

```bash
python accel_monitor.py
```

This separate application displays multichannel acceleration in g and PSD in
g²/Hz using the same channel colors and plot styling as the controller. Select
**Simulation** explicitly for synthetic traces, then **Start acquisition**. Its
default hardware configuration has
one channel (`Dev1/ai0`) at 2,048 Hz, with 512-sample blocks. Add entries to
**Channels / acquisition** while stopped, or set launch defaults in
`monitor_config.py`. The dialog edits physical inputs, labels, sensitivity in
mV/g, IEPE current in mA, input type, terminal configuration, sample rate, and block
size. Acquisition settings are locked during a run; analysis duration stays
editable. Backend errors retain Fault until another start, with no simulation
fallback. Stop and window close finish acquisition asynchronously.

The time view shows the latest two seconds by default. A separate history buffer
retains the maximum selectable PSD duration (ten seconds by default). The PSD
control reports the **effective** segment duration and frequency-bin spacing,
including a collecting state until enough samples arrive. A ten-second selection
therefore reaches ten seconds of data rather than remaining limited to the time
view. Each refresh uses the latest selected segment; this is a live estimate, not
a whole-run qualification report. The monitor does not command shaker output or
record run files.

## Current controller defaults

| Setting | Value |
|---|---|
| Test mode | Random vibration |
| Acquisition backend | Hardware (`SIMULATION_MODE = False`) |
| Sample rate / processing block / buffer | 51,200 Hz / 1 s / 4 s |
| Welch segment length | 4,096 samples (12.5 Hz bin spacing) |
| Equalizer | 36 bands |
| Initial random level fraction | 0.1 |
| PI gains | Kp = 0.3, Ki = 0.06 |
| AO absolute voltage clip | ±1.8 V |
| RMS voltage limit / crest-factor limit | 1.4 V / 5 |
| Sine sweep | 20–2,000 Hz, 3 g-peak, 1 octave/min, repeating |
| Initial sine settling dwell | 3 s extra at the first step |
| Logging | Enabled, under the OS Documents folder / `ShakerControl/Runs` |
| Hardware AI / AO | `Dev1/ai0` / `Dev1/ao0` |
| Sensitivity / IEPE excitation | 100 mV/g / 2.1 mA |

The controller starts with `Dev1/ai0` as the control accelerometer. Enable
`Dev1/ai1`–`Dev1/ai3` as response inputs when their sensors are connected; the
response list is empty by default. Enabled inputs share one AI task and
sample clock. Only the first input supplies random-control feedback and sine
control metrics; the response inputs appear in **Channels** and are recorded in
the same HDF5 blocks. Sine summaries describe the control input; all response
waveforms remain available in `data.h5`. AI/AO clock sharing is disabled by default.

Use **Acquisition → Channels / Backend setup** to edit the control input and its
sensitivity/current. Each response row contains `physical channel, label,
sensitivity in mV/g, IEPE current in mA`; delete a row to disable that input.
For example, `Dev1/ai2, Response 2, 200, 0` selects a 200 mV/g sensor with external
power. The control-input launch defaults use 100 mV/g and 2.1 mA; match these
to the connected sensor before starting. `RESPONSE_CHANNELS` in `config.py`
supplies these defaults, and presets retain each channel's settings. Older
presets without response inputs load with their original single control input.

Channel order, physical names, labels, sensitivities, and excitation settings are
captured in recording settings; HDF5 also includes ordered channel-label and
physical-channel attributes. If an acceleration input needs voltage fallback,
the app scales that channel independently and reports its mode (without IEPE).

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

The Response view separates the full PSD target, ramp-adjusted target, unaveraged
current block, and averaged control response. Averaging has a two-second time
constant based on sample duration; the current-block curve helps distinguish a
new response from a lingering average. Channels uses consistent colors for time and PSD traces.
Diagnostics separates acceleration, ramp/clipping percentage, plant gain in g/V,
and random equalizer amplitude gains into unit-specific plots. Histories use
elapsed seconds. **AO clipping** counts samples exceeding the peak voltage limit
before the final clip; it differs from the near-limit fraction used for sine
backoff. Voltage and crest metrics shown in the workspace describe the command
after the absolute clip, associated by nominal queue order with the acquired block.
They are software command values, not independently measured output voltages.

Each logged GUI run creates a unique timestamped directory containing:

- `data.h5`: AO voltage and AI acceleration blocks stored as float32, plus sample
  rate and per-block timestamps, applied level fraction, and near-limit fraction.
  Schema version 2 attributes include backend, units, setup ID, settings JSON,
  input mode, and clock routing. Sine frequency and acceleration metadata now have
  one row per acquired block, matching the timestamp array.
- `settings.json`: the frozen effective setup, including target, channel labels,
  calibration/preset source, backend selection, and recording settings.
- `run-summary.json`: final outcome/error, effective backend, duration, blocks,
  last metrics, setup ID, recording path, and stream timing diagnostics. Hardware
  diagnostics include effective input/output buffer capacities, largest observed
  unread input backlog, and longest read, write, recording, and block-loop times.
- `preview_psd.csv`: random-run PSD of the last recorded block, per channel.
- `sine_summary.csv`: sine-run frequency, measured g-peak, and target g-peak.

CSV summaries and the final run report are written when the logger closes.
Hardware priming blocks retain their raw samples; sine priming metadata uses
frequency zero and is excluded from sweep-point summaries. Recording metadata
identifies the random-control algorithm and the limits of nominal AO/AI association.
Requested recording failures fault the run instead of silently disabling logging.
CSV summaries are previews, not a
whole-run PSD qualification report. Logs and generated calibration tables are
ignored by Git.

The default recording root follows the system's Documents location, including
Windows folder redirection, with `ShakerControl/Runs` appended. The effective
absolute path is captured in presets, settings and run reports. No recording
directory is created until a recording starts. Change the folder in setup or set
`DATA_LOG_DIR` in `config.py` to an explicit path; `None` selects the system default.
Relative paths resolve against the checkout, independently of the launch
directory. Setting `DATA_LOG_DIR = "data_logs"` retains the earlier checkout-local
destination. Existing recordings are not moved or deleted.

Recording destinations in presets belong to the computer that saved them. When
moving a preset between computers, select an appropriate local recording folder.
Foreign Windows paths on macOS and incomplete drive paths on Windows are rejected
before a run. Imported presets accept UTF-8 with or without a byte-order mark;
new JSON and CSV exports use UTF-8. NI channel names retain their `/` syntax.

## Checks and remaining limitations

Install the test dependencies and run the regression suite:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m pip check
```

Tests cover numerical limiting, immutable setup validation, presets, GUI draft
and run states, startup failure/retry, explicit backend handling, Stop/close
cleanup, plot resets and equalizer legends, exact sine endpoints, recording
metadata, and monitor PSD retention. Qt workflow tests run offscreen with real
simulation workers and prohibit opening real NI-DAQ tasks. No extra Qt test
plugin is required beyond the pinned PySide6 and pytest dependencies.

The [desktop compatibility workflow](.github/workflows/desktop.yml) runs on push,
pull requests and manual dispatch with Python 3.13 on Windows and macOS. It checks
the Windows setup/launcher scripts, the regression suite, default-rate simulation
with actual HDF5 recording, and optional NI binding installation/import. Test
reports, timing JSON and screenshots are retained as workflow artifacts. Hosted
runner timings are recorded without enforcing desktop timing budgets. The Windows
runner is a server image; actual Windows 11 display scaling and hardware behavior
still need the [desktop acceptance checks](docs/windows-compatibility.md).

Random control uses conditional-integration anti-windup and pauses its ramp when
limiting or tracking error prevents an increase; sine mode has saturation backoff.
Controller block I/O uses finite timeouts and Stop remains pending until cleanup
finishes; driver calls, output behavior, and stop latency still need hardware
validation. NI acceleration-channel fallback to voltage scaling is reported in
the run. Failed requests for shared AI/AO clocks fault the run; selecting
independent clocks requires an explicit setup change. Reported routing does not
prove synchronized or calibrated acquisition. Simulation
includes prescribed resonances, noise, delay, and nonlinearity, rather than a
validated model of a particular shaker. Tracking accuracy, stability, and physical
protection require separate evidence for the configured equipment and load.

Reproduce the accelerated random-control review with
`python scripts/review_random_control.py`. It exercises unknown-gain synthetic
plants and checks tracking, overshoot, and bounded correction. This uses sample
time with software waits omitted; it is not a wall-time or hardware stability test.

### Five-minute hardware hold

On October 29, 2026, the NI USB-4431 completed a 300-second full-target random
vibration hold after 21 seconds of startup and ramp. Only the control
accelerometer on `Dev1/ai0` was connected, with 10 mV/g sensitivity and 2.1 mA
IEPE excitation. The 100–500 Hz profile targeted **3.15927 g RMS**, using
`Kp = 0.3` and `Ki = 0.06`.

The hold averaged **3.16082 g RMS** (+0.049%), ranging from **3.06767 to
3.24770 g RMS**. All 300 one-second measurements stayed within **±2.90%** of
target, passing the ±5% RMS criterion. The run stopped after the hold with no
reported faults. These results apply to this equipment setup and control-band
RMS measurement; independent AI/AO clocks were used.

Saved evidence: [chart](docs/hardware-validation/2026-10-29-random-hold/five-minute-hold.png),
[settings, statistics, and recording hashes](docs/hardware-validation/2026-10-29-random-hold/validation.json),
[per-block measurements](docs/hardware-validation/2026-10-29-random-hold/per-block-metrics.csv),
and [final-minute averaged PSD](docs/hardware-validation/2026-10-29-random-hold/final-minute-psd.csv).
The measurements include startup; `in_hold` marks blocks 22–321 and
`hold_sample_end_s` counts the 300-second hold. The PSD export includes endpoint
bins; `interior_bin` marks the bins used for the reported interior PSD statistics.
Raw waveforms remain in the local recording directory identified in the evidence.

### Simulation performance

The simulator evaluates its delay history and response-channel IIR filters by
block, carrying filter state between blocks. Filter coefficients, gains, noise,
nonlinearity, and the existing delay-register convention are preserved. Random
and sine simulation workers subtract processing time from their interruptible
block wait. Hardware reads pace acquisition using the device's sample clock,
without an additional software sleep, so queued blocks can be drained after a
temporary delay. Recording initializes before hardware starts, and hardware stops
before recording finalization. Input overflow faults retain the NI error and
identify the run as incomplete; acquired samples are never silently skipped to
hide a backlog.

The [default-rate performance review](docs/simulation-performance/validation.json)
checks 51,200 Hz, one-second blocks, four channels, live Qt plots, and HDF5 recording.
On the reviewed macOS ARM64 environment, mock acquisition improved from about
0.38 s to 0.009 s per block. The GUI delivered one-second blocks about every
1.00–1.01 s, with Stop and recording cleanup around 0.03 s. These are local
simulation measurements, not hardware or hard real-time guarantees. The sine
review covers the first six blocks of its initial dwell, not a full default sweep.

Reproduce the block benchmark and compare seeded responses with the prior
implementation, then run the GUI timing/recording check:

```bash
python scripts/benchmark_simulation.py --reference-ref 5105dbc
python scripts/review_simulation_performance.py --blocks 6 --check-budget
```

The earlier floating-workspace captures used 8 kHz settings. Their preliminary
39.5-second result used a polling harness and is superseded by the normal Qt event
loop measurements above; that harness's timing is not the desktop application's
block cadence. Response tracking/calibration limitations remain unchanged.

## Repository map

| File | Responsibility |
|---|---|
| `app.py` | Desktop entry point, acquisition worker, and run logger |
| `workspace_ui.py` | Run workspace, setup, channels, diagnostics, and results |
| `workspace_shell.py` | Floating ribbon/browser/inspector layout, navigation, icons, and appearance |
| `run_settings.py` | Immutable setup capture, validation, and target summaries |
| `config.py` | Controller, hardware, simulation, target, and sweep settings |
| `app_paths.py` | System recording defaults, absolute destinations and portable path labels |
| `setup-windows.cmd` / `start-*.cmd` | Windows setup and desktop launchers |
| `.github/workflows/desktop.yml` | Windows/macOS regression and simulation CI |
| `rv_controller.py` | Random control, equalization, noise generation, and limiters |
| `simulation.py` | Synthetic plant and mock DAQ |
| `sine_sweep.py` | Sweep generators, oscillator, and drive-table interpolation |
| `record_1g_sine_sweep.py` | Command-line drive calibration |
| `accel_monitor.py` / `monitor_config.py` | Standalone multichannel monitor |
| `tests/` | Numerical, setup, controller/monitor workflow, preset, and recording regressions |
| `requirements*.txt` | Runtime, test, and optional hardware dependency pins |

## UI audit and review evidence

The implemented floating layout is shown in [Response](docs/floating-workspace/02-random-running.png),
[target inspector](docs/floating-workspace/03-target-inspector.png),
[sine completion](docs/floating-workspace/05-sine-complete.png),
[compact fault view](docs/floating-workspace/07-compact-fault.png), and
[dark appearance](docs/floating-workspace/08-dark.png).
[Floating workspace validation](docs/floating-workspace/validation.json) records
simulation runs, HDF5 checks, source hashes, test results, and review limits.
These captures are offscreen Qt. The native macOS window and target-inspector
navigation were also inspected; no physical hardware was exercised.

The [design audit](docs/design-audit/audit-report.html) and
[interactive proposal](docs/design-audit/ui-flow-proposal.html) describe the
baseline at commit `3aa27a9`. The implemented workspace is captured in
[Run](docs/ui-workspace/03-random-run.png),
[setup](docs/ui-workspace/02-setup.png), and
[monitor](docs/ui-workspace/08-monitor.png) screenshots.
[Validation details](docs/ui-workspace/validation.json) retain the checked source
hashes, regression results, and limits. Captures use offscreen Qt and simulation;
they do not establish hardware behavior. Raw local captures remain in ignored
`output/`; portable review copies are kept in `docs/`.

License: MIT; see `LICENSE`.
