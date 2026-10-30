"""Accelerated simulation-only review; real NI tasks are explicitly prohibited."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import app
from calibration_wizard import CalibrationWizard
from run_reports import render_report_chart
from run_settings import RunSettings
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=ROOT / 'output/control-features-review')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)


def forbidden(*args, **kwargs):
    raise AssertionError('This review must never open NI hardware tasks')


if app.nidaqmx is not None:
    app.nidaqmx.Task = forbidden
qt = app.QApplication.instance() or app.QApplication([])
font_path = Path(os.environ.get('WINDIR', '')) / 'Fonts/segoeui.ttf'
if font_path.is_file():
    index = QFontDatabase.addApplicationFont(str(font_path))
    qt.setFont(QFont(QFontDatabase.applicationFontFamilies(index)[0], 10))
base = app.SharedConfig().as_dict()
base.update(simulation_mode=True, test_mode='sine_sweep', fs=8000., block_seconds=.1,
    buf_seconds=2., welch_nperseg=512, eq_num_bands=8, input_channel_labels=['Control Accel'],
    response_channels=[], sim_noise_level=.0001, sim_resonances=[], sim_nonlinearity=0,
    sine_g_level=.5, sine_g_level_is_rms=False, sine_closed_loop=True, sine_repeat=False,
    sine_start_hz=100., sine_end_hz=500., sine_points_per_octave=3.,
    sine_initial_level=.25, sine_max_level_rate=1., sine_initial_settle=.2,
    sine_octaves_per_min=120., sine_step_dwell=.3, sine_settle_seconds=.2,
    sine_settle_blocks=3, sine_max_step_seconds=30., sine_drive_table=[], sine_default_vpk=.4,
    response_limits_enabled=True, data_log_enabled=True, data_log_dir=str(args.output / 'runs'),
    report_enabled=True, calibration_equipment='Synthetic constant-gain plant',
    calibration_amp_setting='Simulation gain parameter', calibration_sensor_id='Synthetic control channel',
    calibration_source='Simulation-only feature review; not a hardware calibration.')
results = []
last_report = None
for gain, start, end, calibration in ((4., 100., 500., True), (40., 100., 500., False), (4., 500., 100., False)):
    settings = RunSettings.capture({**base, 'sim_plant_gain': gain, 'sine_start_hz': start,
        'sine_end_hz': end, 'calibration_run': calibration})
    worker = app.ControllerWorker(settings)
    worker._pace_block = lambda *args: None
    reports = []
    worker.report_ready.connect(reports.append)
    worker.run_controller()
    report = reports[0]
    assert report['outcome'] == 'complete', report['error']
    assert report['qualification']['status'] == 'pass', report['qualification']
    if calibration:
        assert report['calibration']['status'] == 'pass'
    render_report_chart(report)
    rows = report['qualification']['steps']
    result = {'gain_g_per_v': gain, 'start_hz': start, 'end_hz': end,
        'blocks': report['blocks'], 'sample_seconds': report['blocks'] * settings.block_seconds,
        'wall_seconds': report['duration_s'], 'status': report['qualification']['status'],
        'frequencies': len(rows), 'worst_settled_window_error_percent': max(row['worst_settled_error_percent'] for row in rows),
        'calibration_status': report.get('calibration', {}).get('status'), 'report_dir': report['report_dir']}
    results.append(result)
    last_report = report
    print(json.dumps(result), flush=True)

shared = app.SharedConfig()
shared.apply_values(base)
window = app.WorkspaceWindow(shared, app.ControllerWorker)
window.resize(1360, 900)
window.show()
window.inspect_settings('limits')
QTest.qWait(100)
window.grab().save(str(args.output / 'response-protection.png'))
window.inspect_settings('calibration')
QTest.qWait(100)
window.grab().save(str(args.output / 'drive-calibration.png'))
window.config_tab.sections['drive'][1].setChecked(False)
window.config_tab.scroll_area.verticalScrollBar().setValue(0)
QTest.qWait(100)
window.grab().save(str(args.output / 'sine-feedback.png'))
window.results_tab.add_report(last_report)
window.show_run_report()
QTest.qWait(100)
window.grab().save(str(args.output / 'automatic-report.png'))
wizard = CalibrationWizard(window)
wizard.show()
QTest.qWait(100)
wizard.grab().save(str(args.output / 'calibration-wizard.png'))
wizard.close()
window.close()
summary = {'simulation_only': True, 'real_hardware_tasks_prohibited': True,
    'pacing': 'Processing waits omitted; sample seconds differ from wall seconds.', 'cases': results}
(args.output / 'validation.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
print('Feature simulation review passed.', flush=True)
