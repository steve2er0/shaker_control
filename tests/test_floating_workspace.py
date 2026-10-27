"""User-facing navigation and run-control regressions for the floating shell."""
import pytest
from PySide6.QtCore import QPoint
from PySide6.QtTest import QTest

from conftest import wait_until


def test_inspector_navigation_keeps_one_draft_and_discard_restores_selectors(window):
    panel = window.config_tab
    window.inspect_settings('target')
    before = panel.fields['target_psd_points'].toPlainText()
    panel.fields['target_psd_points'].setPlainText('100, .002\n500, .002')
    window.inspect_settings('acquisition')
    assert panel.dirty and not window.start_button.isEnabled()
    assert not panel.sections['acquisition'][0].isHidden()
    assert panel.sections['target'][0].isHidden()
    window.inspect_settings('target')
    assert panel.fields['target_psd_points'].toPlainText() == '100, .002\n500, .002'
    window.mode_combo.setCurrentIndex(1)
    window.backend_combo.setCurrentIndex(0)
    panel.discard_changes()
    assert not panel.dirty
    assert panel.fields['target_psd_points'].toPlainText() == before
    assert window.mode_combo.currentData() == 'random'
    assert window.backend_combo.currentData() is True
    assert window.start_button.isEnabled()


def test_invalid_field_in_another_context_is_revealed_without_applying(window):
    window.inspect_settings('acquisition')
    panel = window.config_tab
    panel.fields['target_psd_points'].setPlainText('500, .002\n100, .002')
    assert not panel.apply_config()
    assert panel.context == 'all'
    assert not panel.sections['target'][0].isHidden()
    assert panel.sections['target'][1].isChecked()
    assert panel.error_labels['target_psd_points'].text()
    assert window.settings.target_psd_points[0][0] == 100


@pytest.mark.parametrize('width', [900, 1000, 1360])
def test_floating_panels_leave_stop_and_apply_visible(window, width):
    window.show()
    window.resize(width, 760)
    window.inspect_settings('target')
    QTest.qWait(60)
    for widget in (window.start_button, window.stop_button, window.config_tab.apply_button):
        origin = widget.mapTo(window, QPoint(0, 0))
        assert widget.isVisible()
        assert window.rect().contains(origin)
        assert window.rect().contains(origin + QPoint(widget.width()-1, widget.height()-1))
    assert window.test_browser.isVisible() == (width >= 1200)
    window.config_dock.hide()
    QTest.qWait(30)
    assert window.test_browser.isVisible()
    assert window.viewport_card.width() > 350


def test_browser_channel_and_captured_run_navigation_during_a_live_run(window):
    window.show()
    assert window.start_controller()
    wait_until(lambda: bool(window.controller_tab.history))
    window.test_browser.choose('channel', 2)
    assert window.realtime_tab.channel_combo.currentText() == 'Response 2'
    assert window.pages.currentWidget() is window.realtime_tab
    assert window.stop_button.isEnabled()
    assert not window.config_tab.fs_spin.isEnabled()
    window.show_equalizer()
    assert window.controller_tab.diagnostics_widget.currentIndex() == 3
    assert window.stop_button.isEnabled()
    assert window.stop_controller()
    wait_until(lambda: not window.busy)
    window.test_browser.choose('report', 0)
    assert window.pages.currentWidget() is window.results_tab
    assert 'Simulation' in window.inspector_text.toPlainText()
    assert window.results_tab.reports[0]['settings']['fs'] == 8000
    window.show_run_history()
    assert 'Stop requested' in window.inspector_text.toPlainText()
    assert window.controller_tab.value_labels['target'].text() == '0.999'
