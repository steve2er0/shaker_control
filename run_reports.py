"""Automatic tracking assessment and portable HTML/CSV run reports."""
from collections import deque
import csv
from datetime import datetime
from html import escape
import json
import math
from pathlib import Path
import uuid

import numpy as np
from app_paths import default_recording_directory
from sine_sweep import LogSweepStepper


FIELDS = ('block_index', 'elapsed_s', 'frequency_hz', 'level_fraction', 'accel_rms_g',
    'target_rms_g', 'accel_peak_g', 'tone_peak_g', 'target_peak_g', 'raw_peak_g',
    'broadband_rms_g', 'tone_residual_rms_g', 'output_rms_v', 'output_peak_v',
    'limiter_active', 'input_backlog_seconds', 'feedback_ready', 'settled')


class RunAssessment:
    def __init__(self, settings):
        self.settings = settings
        self.points = deque(maxlen=2000)
        self.steps = deque(maxlen=10000)
        self.current = None
        self.random_seconds = 0.
        self.full_seconds = 0.
        self.random_bad = 0
        self.random_error_sum = 0.
        self.random_count = 0
        self.random_worst = 0.
        self.peak = 0.
        self.limiting_blocks = 0
        self.total_steps = 0

    def _finish_step(self):
        if not self.current:
            return
        s, row = self.settings, self.current
        samples = list(row.pop('tail'))
        ready = (len(samples) >= s.sine_settle_blocks and row['seconds'] >= s.sine_settle_seconds
                 and all(p['level_fraction'] >= .999999 and not p.get('limiter_active')
                         and p.get('feedback_ready', True) for p in samples))
        errors = [100 * (p.get('tone_peak_g', p['accel_peak_g']) /
                         (s.target_rms_g * math.sqrt(2)) - 1) for p in samples]
        passed = ready and all(abs(error) <= s.tracking_tolerance_percent for error in errors)
        row.update(status='pass' if passed else 'fail',
            settled_peak_g=float(np.mean([p.get('tone_peak_g', p['accel_peak_g']) for p in samples])),
            settled_error_percent=float(np.mean(errors)),
            worst_settled_error_percent=max(map(abs, errors)),
            target_peak_g=s.target_rms_g * math.sqrt(2),
            volts_peak_per_g=float(np.median([p['output_peak_v'] / max(p.get('tone_peak_g', p['accel_peak_g']), 1e-12)
                                             for p in samples])))
        self.steps.append(row)
        self.total_steps += 1
        self.current = None

    def add(self, packet):
        s = self.settings
        self.peak = max(self.peak, packet.get('raw_peak_g', 0))
        self.limiting_blocks += int(bool(packet.get('limiter_active')))
        if s.test_mode == 'sine_sweep':
            freq = packet['frequency_hz']
            if freq <= 0:
                return
            if self.current is None or self.current['frequency_hz'] != freq:
                self._finish_step()
                self.current = {'frequency_hz': freq, 'seconds': 0., 'transient_peak_g': 0.,
                                'maximum_sine_equivalent_peak_g': 0., 'tail': deque(maxlen=s.sine_settle_blocks)}
            row = self.current
            row['seconds'] += s.block_seconds
            row['transient_peak_g'] = max(row['transient_peak_g'], packet.get('raw_peak_g', 0))
            row['maximum_sine_equivalent_peak_g'] = max(row['maximum_sine_equivalent_peak_g'], packet['accel_peak_g'])
            row['tail'].append(dict(packet))
        else:
            self.points.append([packet['elapsed_s'], packet['accel_rms_g'], s.target_rms_g])
            if abs(packet['level_fraction'] - 1) <= .000001:
                self.full_seconds += s.block_seconds
                if self.full_seconds > s.report_settle_seconds:
                    error = 100 * (packet['accel_rms_g'] / s.target_rms_g - 1)
                    self.random_seconds += s.block_seconds
                    self.random_count += 1
                    self.random_error_sum += error
                    self.random_worst = max(self.random_worst, abs(error))
                    self.random_bad += int(abs(error) > s.tracking_tolerance_percent or packet.get('limiter_active', False))
            else:
                self.full_seconds = 0.

    def finish(self, outcome, error):
        self._finish_step()
        s = self.settings
        result = {'tolerance_percent': s.tracking_tolerance_percent, 'maximum_raw_sample_peak_g': self.peak,
                  'limiting_blocks': self.limiting_blocks}
        if s.test_mode == 'sine_sweep':
            expected = []
            stepper = LogSweepStepper(s.sine_start_hz, s.sine_end_hz, s.sine_points_per_octave)
            while True:
                expected.append(stepper.current_frequency())
                if not stepper.advance():
                    break
            rows = list(self.steps)
            complete = (outcome == 'complete' and len(rows) == len(expected)
                        and self.total_steps == len(rows)
                        and all(math.isclose(row['frequency_hz'], f, rel_tol=1e-9) for row, f in zip(rows, expected)))
            status = 'pass' if complete and all(row['status'] == 'pass' for row in rows) else 'fail' if complete else 'incomplete'
            result.update(steps=rows, expected_frequency_count=len(expected), observed_step_count=self.total_steps,
                chart={'x_label': 'Frequency (Hz)', 'y_label': 'Fitted fundamental (g peak)',
                       'points': [[r['frequency_hz'], r['settled_peak_g'], r['target_peak_g']] for r in rows]})
        else:
            status = ('incomplete' if self.random_seconds < s.report_min_seconds else
                      'fail' if self.random_bad else 'pass')
            result.update(qualified_seconds=self.random_seconds, outside_tolerance_blocks=self.random_bad,
                mean_error_percent=self.random_error_sum / self.random_count if self.random_count else None,
                worst_error_percent=self.random_worst if self.random_count else None,
                chart={'x_label': 'Elapsed time (s); last 2000 blocks', 'y_label': 'Control-band RMS (g)', 'points': list(self.points)})
        if error or outcome == 'fault':
            status = 'fail'
        result['status'] = status
        result['definition'] = ('Tracking assessment against configured tolerance, separate from run completion. '
            'Sine uses the last settling-window blocks at each dwell; transient raw peaks are reported separately. '
            'Random excludes startup/ramp and the initial settling interval; all subsequent full-target blocks are evaluated.')
        return result


class ReportFiles:
    def __init__(self, settings, recording_dir=None):
        self.directory = None
        self.handle = None
        if not settings.report_enabled:
            return
        self.directory = Path(recording_dir) if recording_dir else default_recording_directory().parent / 'Reports' / (
            datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8])
        self.directory.mkdir(parents=True, exist_ok=True)
        self.handle = (self.directory / 'block-metrics.csv').open('w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.handle, fieldnames=FIELDS, extrasaction='ignore')
        self.writer.writeheader()

    def add(self, packet):
        if self.handle:
            self.writer.writerow(packet)

    def close(self):
        if self.handle:
            self.handle.close()
            self.handle = None

    def save(self, report):
        self.close()
        if self.directory:
            report['report_dir'] = str(self.directory)
            rows = report['qualification'].get('steps', [])
            if rows:
                with (self.directory / 'settled-sine-results.csv').open('w', newline='', encoding='utf-8') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            (self.directory / 'test-report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            write_html(report)


def write_html(report):
    directory = Path(report['report_dir'])
    q = report['qualification']
    def esc(value):
        return escape(str(value))
    rows = q.get('steps', [])
    headings = ('Hz', 'Target g peak', 'Settled g peak', 'Error %', 'Raw transient peak g', 'Result')
    table = '<table><tr>' + ''.join(f'<th>{h}</th>' for h in headings) + '</tr>'
    for row in rows:
        values = [f'{row["frequency_hz"]:.5g}', f'{row["target_peak_g"]:.5g}', f'{row["settled_peak_g"]:.5g}',
                  f'{row["settled_error_percent"]:+.3f}', f'{row["transient_peak_g"]:.5g}', row['status'].upper()]
        table += '<tr>' + ''.join(f'<td>{esc(v)}</td>' for v in values) + '</tr>'
    table += '</table>'
    chart = '<img src="tracking.png" alt="Measured tracking with tolerance band">' if (directory / 'tracking.png').is_file() else ''
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Shaker test report</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#24344b}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border:1px solid #ccd5df;text-align:left}}img{{width:100%}}pre{{white-space:pre-wrap}}</style>
<h1>Shaker test · {esc(q['status'].upper())}</h1><p>{esc(report['started_at'])} · {esc(report['backend'])} · Setup {esc(report['config_id'])}</p>
<p>Run: {esc(report['outcome'])} · {report['blocks']} blocks · {report['duration_s']:.2f} wall seconds · {report.get('sample_seconds', report['blocks'] * report['settings']['block_seconds']):.2f} sample seconds<br>Tracking tolerance: ±{q['tolerance_percent']:g}%</p>
<p>{esc(report['error'])}</p>{chart}{table if rows else '<pre>' + esc(json.dumps({k:v for k,v in q.items() if k != 'chart'}, indent=2)) + '</pre>'}
<p>{esc(q['definition'])}</p><p>Raw peak means the largest absolute acquired sample; it differs from fitted sine amplitude and RMS.</p>
<p>Calibration: {esc(report['settings']['calibration_source'] or 'Not recorded')}<br>Input: {esc(report['input_mode'])}<br>Clocks: {esc(report['clock_status'])}</p>
<p><a href="block-metrics.csv">Per-block measurements</a> · <a href="test-report.json">Full settings and assessment</a></p></html>'''
    (directory / 'test-report.html').write_text(text, encoding='utf-8')


def render_report_chart(report):
    """Called by the desktop on the GUI thread, after acquisition has stopped."""
    if not report.get('report_dir'):
        return
    points = report.get('qualification', {}).get('chart', {}).get('points', [])
    if not points:
        return
    import pyqtgraph as pg
    import pyqtgraph.exporters
    from PySide6.QtCore import Qt, QEventLoop
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFont, QFontDatabase
    import os
    plot = pg.PlotWidget(background='w')
    try:
        family = QApplication.font().family()
        font_path = Path(os.environ.get('WINDIR', '')) / 'Fonts/segoeui.ttf'
        if font_path.is_file():
            font_id = QFontDatabase.addApplicationFont(str(font_path))
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                family = families[0]
        plot.setFont(QFont(family, 11))
        plot.setAttribute(Qt.WA_DontShowOnScreen)
        plot.resize(1100, 450)
        plot.addLegend(offset=(-12, 12), labelTextColor='#24344b', brush=pg.mkBrush('w'))
        plot.setTitle(f'Tracking · {report["qualification"]["status"].upper()}', color='#24344b', size='16pt')
        chart = report['qualification']['chart']
        plot.setLabel('bottom', chart['x_label'], color='#24344b')
        plot.setLabel('left', chart['y_label'], color='#24344b')
        for axis in ('left', 'bottom'):
            plot.getAxis(axis).setTextPen('#24344b')
            plot.getAxis(axis).setPen('#65748a')
            plot.getAxis(axis).setTickFont(QFont(family, 10))
        data = np.array(sorted(points) if report['settings']['test_mode'] == 'sine_sweep' else points)
        x, measured, target = data.T
        tolerance = report['qualification']['tolerance_percent'] / 100
        lower = plot.plot(x, target * (1 - tolerance), pen=pg.mkPen('#b0cabd'))
        upper = plot.plot(x, target * (1 + tolerance), pen=pg.mkPen('#b0cabd'))
        plot.addItem(pg.FillBetweenItem(lower, upper, brush=pg.mkBrush(36, 134, 89, 30)))
        plot.plot(x, target, pen=pg.mkPen('#65748a', width=2), name=f'Target · band ±{report["qualification"]["tolerance_percent"]:g}%')
        plot.plot(x, measured, pen=pg.mkPen('#2175ad', width=2), symbol='o', symbolSize=4, name='Measured')
        plot.show()
        QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
        exporter = pg.exporters.ImageExporter(plot.plotItem)
        exporter.parameters()['width'] = 1600
        exporter.export(str(Path(report['report_dir']) / 'tracking.png'))
        write_html(report)
    finally:
        plot.close()
