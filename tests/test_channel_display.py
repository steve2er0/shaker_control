from threading import Thread

import numpy as np
import pytest

import app
from channel_display import ChannelMailbox, peak_envelope
from run_settings import RunSettings, ConfigurationError
from workspace_ui import ChannelView, ResultsView


def test_peak_preview_is_bounded_and_retains_single_sample_spikes():
    values = np.zeros(102400)
    values[41111], values[81003] = 12., -15.
    times = np.arange(len(values)) / 51200
    x, y = peak_envelope(times, values)
    assert len(x) <= 4096
    assert np.all(np.diff(x) >= 0)
    assert y.min() == -15 and y.max() == 12
    assert times[41111] in x and times[81003] in x


def test_mailbox_discards_only_obsolete_display_data_and_keeps_sample_clock():
    mailbox = ChannelMailbox(200, 1)
    for index in range(10):
        mailbox.submit(np.full(100, index))
    blocks, samples, count, dropped, _, error = mailbox.drain()
    assert samples == 1000 and count == 10 and dropped and not error
    assert len(blocks) == 2 and blocks[-1][0, -1] == 9
    assert mailbox.drain()[:4] == ([], 0, 0, False)


@pytest.mark.parametrize('channels', [1, 4])
def test_hidden_channel_mailbox_cannot_queue_unbounded_plot_work(qt_app, settings_data, channels):
    settings_data.update(fs=51200, block_seconds=1, welch_nperseg=4096,
                         input_channel_labels=[f'Input {i}' for i in range(channels)])
    view = ChannelView(RunSettings.capture(settings_data))
    view.start_streaming()
    data = np.zeros((channels, 51200))
    data[:, 12501] = 10
    data[:, 37501] = -8
    def publish():
        for _ in range(500):
            view.queue_data_point(data)
    producer = Thread(target=publish)
    producer.start()
    producer.join()
    assert len(view.mailbox.blocks) <= 2
    view.refresh_display()
    assert view.samples == 500 * 51200 and view.blocks == 500
    assert view.buffer.shape == (channels, 102400)
    assert all(curve.xData is None for curve in view.time_curves), 'Hidden view must not draw'
    view.show()
    qt_app.processEvents()
    view.refresh_display()
    for curve in view.time_curves:
        assert len(curve.xData) <= 4096
        assert curve.yData.max() == 10 and curve.yData.min() == -8
    assert view.time_curves[0].xData[-1] > 498
    view.stop_streaming()
    view.close()


def test_channel_psd_keeps_latest_packet_without_gui_signal_backlog(qt_app, settings_data):
    view = ChannelView(RunSettings.capture(settings_data))
    view.start_streaming()
    frequencies = np.arange(20, dtype=float)
    for value in range(1, 100):
        view.queue_psd_data((frequencies, np.full(20, value), None, None))
    view.stop_streaming()
    np.testing.assert_array_equal(view.psd_curves[0].yData, np.full(19, 99))
    view.close()


def test_mode_switch_cannot_start_target_above_inherited_sine_limits(settings_data):
    settings_data.update(response_limits_enabled=True, response_max_rms_g=.55, response_max_peak_g=.85)
    with pytest.raises(ConfigurationError, match='above the measured'):
        RunSettings.capture(settings_data)


def test_historical_report_remains_viewable_after_stricter_setup_checks(qt_app, settings_data):
    values = dict(settings_data, response_limits_enabled=False, response_max_rms_g=.55, response_max_peak_g=.85)
    s = RunSettings.capture(values)
    historical = s.as_dict()
    historical['response_limits_enabled'] = True
    view = ResultsView()
    view.add_report({'outcome': 'fault', 'backend': 'Hardware', 'requested_backend': 'Hardware',
        'settings': historical, 'latest_metrics': None, 'started_at': '', 'blocks': 0,
        'duration_s': 0, 'config_id': s.config_id, 'log_dir': None, 'error': 'Original recorded fault'})
    assert 'Original recorded fault' in view.summary_text.toPlainText()
    view.close()
