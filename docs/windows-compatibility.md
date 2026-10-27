# Windows compatibility milestone

The source installation targets Windows 11 x64 and CPython 3.13. The same pinned
runtime is used on macOS. This milestone provides source setup and launchers,
portable recording storage and path labels, UTF-8 preset/data text, and a
Windows/macOS CI workflow. Executable packaging and device qualification are
separate follow-up work.

The [local validation record](windows-compatibility/validation.json) records
147 passing tests in each macOS environment (the new Python 3.13 baseline and the
existing Python 3.9 environment), dependency checks, Windows binary dependency
resolution and source hashes. The Python 3.13 random/sine simulation review passed
its cadence and Stop budgets with recording enabled. These are macOS execution
results; Windows execution and the first hosted CI run remain pending.

## Automated checks

The `Desktop compatibility` workflow installs on fresh Windows and macOS runners
and checks:

- Windows setup in Windows PowerShell 5.1, followed by both launcher import checks
  from a different working directory.
- Dependency consistency and simulation availability without NI Python bindings.
- Numerical, setup/preset, Qt workflow, acquisition lifecycle and recording tests.
- Unicode filenames/content, BOM-bearing UTF-8 presets, redirected Documents,
  relative and explicit recording roots, portable labels and recording failures.
- Random and sine simulation at 51,200 Hz, one-second blocks and four channels,
  with live offscreen Qt plots and real HDF5 recording and cleanup.
- Optional NI Python binding installation/import without creating a device task.

JUnit XML, timing JSON, screenshots and simulation recordings are uploaded from
`output/ci`. Timing values on shared runners are diagnostic; run the performance
review with `--check-budget` on the intended workstation for timing acceptance.
All automated acquisition uses simulation or interface doubles.

GitHub's Windows runner is a Windows Server image. Passing that workflow does not
by itself establish Windows 11 desktop appearance or device behavior. A workflow
file in a local checkout is configured coverage; its first successful GitHub run
must still be confirmed after publication.

## Windows 11 desktop acceptance

On the intended Windows 11 x64 computer:

1. Install 64-bit Python 3.13, clone the checkout and run `setup-windows.cmd`
   without NI bindings. Start the controller and monitor through their launchers.
2. Repeat from a checkout folder containing spaces and non-ASCII characters.
   Start from another working directory and confirm startup logs are written to
   `%LOCALAPPDATA%\ShakerControl\Logs`.
3. Check light/dark appearance, readable labels and visible Start/Stop/Apply
   controls at 100%, 125%, 150% and 200% display scaling. Exercise the ribbon,
   browser, inspector and minimum window size.
4. Load/save a UTF-8 preset, choose simulation explicitly, and record random and
   sine runs. Confirm the displayed folder and resulting HDF5/JSON/CSV files.
   Test both the default Documents destination and an explicit writable folder.
5. Confirm an unwritable destination produces a retained Fault. Check Stop,
   retry after failure and closing during acquisition.
6. Run `.venv-py313\Scripts\python.exe -m pip install -r requirements-dev.txt`,
   then `.venv-py313\Scripts\python.exe -m pytest -q`. Run
   `.venv-py313\Scripts\python.exe scripts\review_simulation_performance.py --blocks 6 --check-budget`
   and inspect `output/simulation-performance/gui/validation.json`.

Do not copy a macOS virtual environment to Windows. Existing `data_logs/` files
are left in place; setting `DATA_LOG_DIR = "data_logs"` selects that old root.
The new default is the system Documents folder plus `ShakerControl/Runs`.
Absolute recording paths in imported presets may need changing for this computer.

## NI hardware follow-up

Install the NI-DAQmx driver matched to the exact Windows version and device,
then install optional Python bindings with `setup-windows.cmd -Hardware`.
Confirm channels, IEPE/sensitivity, terminal configuration, supported sample rates,
clock routing and output range for that device before acquisition. Hardware
acceptance must measure scaling, timing, recording integrity and actual output
behavior during Stop, faults and window close. This milestone makes no changes
to device configuration or output control and provides no physical qualification.
