"""Presentation components for the floating Shaker Control workspace.

The shell owns layout and navigation only. Settings validation, acquisition and
recording remain in their existing modules; no device work happens here.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QSize, Signal, QTimer, QByteArray, QSignalBlocker, QTime
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QApplication, QFrame, QWidget, QLabel, QPushButton, QToolButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QScrollArea, QSizePolicy,
    QStackedWidget, QComboBox, QMenu, QPlainTextEdit,
)


def workspace_colors():
    dark = QApplication.palette().window().color().lightness() < 128
    if dark:
        return dict(background='#1d232c', paper='#272f3a', soft='#222a35',
                    line='#424e60', ink='#e3e9f1', muted='#a8b6c9',
                    blue='#78b8ec', blue_soft='#253f55', blue_ink='#172635',
                    red='#f1a099', red_soft='#452e30', green='#8bc9b4',
                    amber='#efc578', header='#141a22')
    return dict(background='#e8ecf2', paper='#ffffff', soft='#f5f8fc',
                line='#d4dce7', ink='#303b4a', muted='#65748a',
                blue='#2479bb', blue_soft='#e2f0fc', blue_ink='#ffffff',
                red='#b3423c', red_soft='#fff0ed', green='#2d7762',
                amber='#936b25', header='#303640')


def workspace_stylesheet():
    c = workspace_colors()
    return f"""
        QMainWindow, QWidget#workspaceRoot {{ background: {c['background']}; color: {c['ink']}; }}
        QWidget#workspaceRoot {{ font-size: 12px; }}
        QFrame#appBar {{ background: {c['header']}; border: 0; }}
        QFrame#appBar QLabel {{ color: #eff3f8; background: transparent; }}
        QLabel#appBrand {{ font-size: 14px; font-weight: 600; }}
        QLabel#setupName {{ background: #454d59; border-radius: 4px; padding: 5px 12px; }}
        QLabel#backendBadge {{ color: #d3dce8; font-size: 11px; }}
        QFrame#workspaceCard, QFrame#ribbonCard, QFrame#testBrowser,
        QFrame#responseCard, QFrame#contextInspector {{
            background: {c['paper']}; border: 1px solid {c['line']}; border-radius: 8px;
        }}
        QFrame#workspaceCard {{ border-top: 3px solid {c['blue']}; }}
        QFrame#workspaceCard QLabel {{ color: {c['blue']}; background: transparent; }}
        QFrame#statusStrip {{ background: {c['paper']}; border-top: 1px solid {c['line']}; }}
        QLabel#mutedText, QLabel#metricTitle, QLabel#browserNote {{ color: {c['muted']}; background: transparent; font-size: 11px; }}
        QLabel#metricValue {{ color: {c['ink']}; background: transparent; font-size: 20px; font-weight: 500; }}
        QLabel#viewTitle, QLabel#inspectorTitle {{ font-size: 13px; font-weight: 600; background: transparent; }}
        QLabel#statusSummary {{ color: {c['muted']}; font-size: 11px; background: transparent; }}
        QLabel#setupNotice {{ color: {c['muted']}; padding: 5px 8px; }}
        QFrame#faultBanner {{ background: {c['red_soft']}; border: 1px solid {c['red']}; border-radius: 6px; }}
        QFrame#faultBanner QLabel {{ color: {c['red']}; background: transparent; }}
        QPushButton, QToolButton {{
            color: {c['ink']}; background: transparent; border: 1px solid transparent;
            border-radius: 4px; padding: 5px 8px;
        }}
        QPushButton:hover, QToolButton:hover {{ background: {c['blue_soft']}; }}
        QPushButton:checked, QToolButton:checked {{ color: {c['blue']}; background: {c['blue_soft']}; }}
        QPushButton:disabled, QToolButton:disabled {{ color: {c['muted']}; }}
        QPushButton#startButton, QPushButton#applyButton {{ background: {c['blue']}; color: {c['blue_ink']}; padding: 7px 12px; }}
        QPushButton#startButton:disabled, QPushButton#applyButton:disabled {{ background: {c['soft']}; color: {c['muted']}; border-color: {c['line']}; }}
        QPushButton#stopButton {{ color: {c['red']}; border-color: {c['line']}; padding: 7px 12px; }}
        QPushButton#stopButton:disabled {{ color: {c['muted']}; }}
        QPushButton#ribbonTab {{ padding: 3px 12px; font-size: 12px; }}
        QPushButton#viewButton {{ padding: 3px 7px; font-size: 11px; }}
        QToolButton#ribbonCommand {{ font-size: 11px; padding: 5px 6px; }}
        QToolButton#ribbonCommand::menu-button {{ width: 13px; border: 0; background: {c['paper']}; border-radius: 3px; }}
        QToolButton#ribbonCommand::menu-button:hover {{ background: {c['blue_soft']}; }}
        QPushButton#browserItem {{ text-align: left; padding: 6px 10px; border-radius: 0; }}
        QPushButton#browserItem QLabel {{ background: transparent; color: {c['ink']}; }}
        QPushButton#browserItem QLabel#browserNote {{ color: {c['muted']}; font-size: 10px; }}
        QLabel#browserGroup {{ color: {c['muted']}; padding: 10px 10px 3px; font-size: 11px; }}
        QScrollArea {{ border: 0; background: transparent; }}
        QScrollArea > QWidget > QWidget {{ background: transparent; }}
        QFrame#contextInspector QLineEdit, QFrame#contextInspector QPlainTextEdit,
        QFrame#contextInspector QSpinBox, QFrame#contextInspector QDoubleSpinBox,
        QComboBox {{ background: {c['paper']}; color: {c['ink']}; border: 1px solid {c['line']}; border-radius: 4px; padding: 4px 6px; min-height: 20px; }}
        QFrame#contextInspector QLineEdit:disabled, QFrame#contextInspector QPlainTextEdit:disabled,
        QFrame#contextInspector QSpinBox:disabled, QFrame#contextInspector QDoubleSpinBox:disabled {{ color: {c['muted']}; background: {c['soft']}; }}
        QComboBox::drop-down {{ border: 0; width: 20px; }}
        QTabWidget::pane {{ border: 0; }}
        QTabBar::tab {{ background: {c['soft']}; color: {c['muted']}; padding: 6px 10px; border: 0; }}
        QTabBar::tab:selected {{ background: {c['blue_soft']}; color: {c['blue']}; }}
        QPlainTextEdit, QListWidget {{ background: {c['paper']}; color: {c['ink']}; border: 1px solid {c['line']}; border-radius: 4px; }}
        QMenu {{ background: {c['paper']}; color: {c['ink']}; border: 1px solid {c['line']}; padding: 4px; }}
        QMenu::item {{ padding: 6px 18px; }}
        QMenu::item:selected {{ background: {c['blue_soft']}; }}
    """


# Original line drawings with a common view box and stroke. Qt renders them at
# device pixel ratio; there are no network assets or extra icon dependencies.
_ICON_PATHS = {
    'activity': '<path d="M2 12h5l3-8 4 16 3-8h5"/>',
    'target': '<path d="M3 3v18h18M5 17l4-4 4 2 4-8 4 4"/>',
    'preset': '<path d="M3 7h7l2 2h9v11H3zM3 7V4h7l2 3"/>',
    'recording': '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 14h18M16 17h2"/>',
    'browser': '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 3v18M5 8h2M5 12h2M5 16h2"/>',
    'inspector': '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M15 3v18M18 8h1M18 12h1M18 16h1"/>',
    'control': '<path d="M3 6h18M3 12h18M3 18h18"/><circle cx="8" cy="6" r="2"/><circle cx="16" cy="12" r="2"/><circle cx="10" cy="18" r="2"/>',
    'clock': '<circle cx="12" cy="12" r="9"/><path d="M12 6v6l4 2"/>',
    'calibration': '<path d="M9 5h12M9 12h12M9 19h12M2 5l2 2 3-4M2 12l2 2 3-4M2 19l2 2 3-4"/>',
    'results': '<path d="M5 2h10l4 4v16H5zM15 2v5h4M8 17l3-4 3 2 3-5"/>',
    'history': '<path d="M3 4v5h5M3 9a9 9 0 1 1 0 7M12 7v5l3 2"/>',
    'fit': '<path d="M3 9V3h6M15 3h6v6M21 15v6h-6M9 21H3v-6"/>',
    'play': '<path d="M7 3l14 9L7 21z"/>',
    'stop': '<rect x="5" y="5" width="14" height="14" rx="1"/>',
    'close': '<path d="M6 6l12 12M18 6L6 18"/>',
}


def workspace_icon(name, color=None, size=24):
    color = color or workspace_colors()['blue']
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" '
           f'fill="none" stroke="{color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">'
           + _ICON_PATHS.get(name, _ICON_PATHS['activity']) + '</svg>')
    renderer = QSvgRenderer(QByteArray(svg.encode()))
    result = QIcon()
    for ratio in (1, 2):
        pixmap = QPixmap(size * ratio, size * ratio)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        pixmap.setDevicePixelRatio(ratio)
        result.addPixmap(pixmap)
    return result


def framed(name):
    frame = QFrame()
    frame.setObjectName(name)
    return frame


def small_label(text='', name='mutedText'):
    label = QLabel(text)
    label.setObjectName(name)
    label.setWordWrap(True)
    return label


def command_button(text, icon, callback=None):
    button = QToolButton()
    button.setObjectName('ribbonCommand')
    button.setText(text)
    button.setAccessibleName(text)
    button.setIcon(workspace_icon(icon))
    button.setIconSize(QSize(25, 25))
    button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
    button.setMinimumWidth(64)
    if callback:
        button.clicked.connect(callback)
    return button


class AdaptiveRibbon(QFrame):
    """Keep run controls intact when a desktop window becomes narrow."""

    def __init__(self, command_stack, run_controls, tabs):
        super().__init__()
        self.setObjectName('ribbonCard')
        self.commands, self.run_controls = command_stack, run_controls
        self._stacked = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 4, 5, 5)
        layout.setSpacing(3)
        layout.addLayout(tabs)
        self.tools = QGridLayout()
        self.tools.setContentsMargins(0, 0, 0, 0)
        self.tools.setHorizontalSpacing(8)
        self.tools.setColumnStretch(0, 1)
        layout.addLayout(self.tools)
        self.tools.addWidget(command_stack, 0, 0)
        self.tools.addWidget(run_controls, 0, 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Defer until the parent layout has assigned both children's sizes.
        QTimer.singleShot(0, self.reflow)

    def reflow(self):
        required = self.commands.minimumSizeHint().width() + self.run_controls.sizeHint().width() + 28
        stacked = self.width() < required
        if stacked == self._stacked:
            return
        self._stacked = stacked
        self.tools.removeWidget(self.run_controls)
        if stacked:
            self.tools.addWidget(self.run_controls, 1, 0, 1, 2, Qt.AlignRight)
        else:
            self.tools.addWidget(self.run_controls, 0, 1)


class BrowserEntry(QPushButton):
    def __init__(self, title, detail, icon, parent=None):
        super().__init__(parent)
        self.setObjectName('browserItem')
        self.setCheckable(True)
        self.setAccessibleName(title)
        self.setAccessibleDescription(detail)
        self.setToolTip(f'{title}\n{detail}')
        # QPushButton's native hint ignores its child layout. Reserve room for
        # the two text lines instead of painting the detail over the next row.
        self.setMinimumHeight(50)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 8, 6)
        layout.setSpacing(8)
        image = QLabel()
        image.setPixmap(workspace_icon(icon).pixmap(QSize(18, 18)))
        image.setFixedSize(18, 20)
        layout.addWidget(image)
        text = QVBoxLayout()
        text.setSpacing(2)
        self.title, self.detail = QLabel(title), QLabel(detail)
        self.title.setWordWrap(True)
        self.detail.setObjectName('browserNote')
        self.detail.setWordWrap(True)
        text.addWidget(self.title)
        text.addWidget(self.detail)
        layout.addLayout(text, 1)
        for widget in (image, self.title, self.detail):
            widget.setAttribute(Qt.WA_TransparentForMouseEvents)


class TestBrowser(QFrame):
    selected = Signal(str, object)
    collapse_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName('testBrowser')
        self.setAccessibleName('Test browser')
        self.setMinimumWidth(190)
        self.setMaximumWidth(238)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 8)
        layout.setSpacing(0)
        header = QHBoxLayout()
        header.setContentsMargins(11, 6, 6, 4)
        header.addWidget(QLabel('Test browser'), 1)
        close = QToolButton()
        close.setIcon(workspace_icon('browser'))
        close.setAccessibleName('Collapse test browser')
        close.clicked.connect(self.collapse_requested)
        header.addWidget(close)
        layout.addLayout(header)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.content = QWidget()
        self.entries_layout = QVBoxLayout(self.content)
        self.entries_layout.setContentsMargins(0, 0, 0, 0)
        self.entries_layout.setSpacing(0)
        scroll.setWidget(self.content)
        layout.addWidget(scroll, 1)
        self.entries = {}
        self.selected_key = None

    def rebuild(self, settings, reports):
        while self.entries_layout.count():
            item = self.entries_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.entries = {}

        def group(title):
            label = QLabel(title)
            label.setObjectName('browserGroup')
            self.entries_layout.addWidget(label)

        def entry(key, title, detail, icon, data=None):
            button = BrowserEntry(title, detail, icon)
            button.clicked.connect(lambda checked=False, key=key, data=data: self.choose(key, data))
            self.entries[(key, data)] = button
            self.entries_layout.addWidget(button)

        group('Current test')
        entry('target', 'Sine sweep' if settings.test_mode == 'sine_sweep' else 'Random PSD target', settings.target_summary, 'target')
        entry('limits', 'Control & limits', f'±{settings.ao_volt_limit:g} V peak · {settings.max_rms_volts:g} V RMS', 'control')
        entry('calibration', 'Drive calibration', settings.calibration_source or 'Source not specified', 'calibration')
        group('Acquisition')
        for index, label in enumerate(settings.input_channel_labels):
            detail = ('Synthetic control input' if index == 0 else 'Synthetic response input') if settings.simulation_mode else f'{settings.device_ai} · {settings.accel_mv_per_g:g} mV/g'
            entry('channel', label, detail, 'activity', index)
        entry('acquisition', 'Sample clock', f'{settings.fs:,.0f} Hz · {settings.block_seconds:g} s blocks', 'clock')
        group('Session runs')
        if reports:
            for index, report in enumerate(reports):
                backend = report['backend'] or report['requested_backend'] + ' requested'
                entry('report', f'Run {index + 1} · {report["outcome"].title()}', f'{backend} · {report["blocks"]} blocks', 'results', index)
        else:
            self.entries_layout.addWidget(small_label('    No completed runs yet.'))
        self.entries_layout.addStretch()
        self.set_selection(*(self.selected_key or (None, None)))

    def choose(self, key, data=None):
        self.set_selection(key, data)
        self.selected.emit(key, data)

    def set_selection(self, key, data=None):
        self.selected_key = (key, data)
        for identity, button in self.entries.items():
            button.setChecked(identity == self.selected_key)


class ContextInspector(QFrame):
    visibilityChanged = Signal(bool)
    close_requested = Signal()

    def __init__(self, settings_panel):
        super().__init__()
        self.setObjectName('contextInspector')
        self.setAccessibleName('Contextual inspector')
        self.setMinimumWidth(306)
        self.setMaximumWidth(360)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = QHBoxLayout()
        header.setContentsMargins(12, 6, 6, 3)
        self.title = QLabel('Test setup')
        self.title.setObjectName('inspectorTitle')
        header.addWidget(self.title, 1)
        close = QToolButton()
        close.setIcon(workspace_icon('close'))
        close.setAccessibleName('Close inspector')
        close.clicked.connect(self.hide)
        header.addWidget(close)
        layout.addLayout(header)
        self.description = small_label()
        self.description.setContentsMargins(12, 3, 12, 3)
        self.description.hide()
        layout.addWidget(self.description)
        self.stack = QStackedWidget()
        self.stack.addWidget(settings_panel)
        layout.addWidget(self.stack, 1)

    def present(self, title, widget=None, description=''):
        self.title.setText(title)
        self.description.setText(description)
        self.description.setVisible(bool(description))
        self.stack.setCurrentIndex(0) if widget is None else self.stack.setCurrentWidget(widget)
        self.show()

    def showEvent(self, event):
        super().showEvent(event)
        self.visibilityChanged.emit(True)

    def hideEvent(self, event):
        super().hideEvent(event)
        self.visibilityChanged.emit(False)


class FloatingWorkspaceMixin:
    """Bind the floating presentation to the existing window's public actions."""

    CONTEXT_TITLES = {
        'all': 'Test setup', 'target': 'Target spectrum', 'limits': 'Control & limits',
        'calibration': 'Drive calibration', 'acquisition': 'Acquisition',
        'channel': 'Channel settings', 'equalizer': 'Equalizer tuning',
        'simulation': 'Simulation model', 'recording': 'Recording',
    }

    def _build_floating_workspace(self):
        self.setStyleSheet(workspace_stylesheet())
        self.setMinimumSize(900, 620)
        self.resize(1360, 860)
        self._browser_requested = True
        self._header_stacked = None
        self.session_events = []
        self._detail_kind = None
        central = QWidget()
        central.setObjectName('workspaceRoot')
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        appbar = framed('appBar')
        bar = QHBoxLayout(appbar)
        bar.setContentsMargins(12, 5, 12, 5)
        mark = QLabel()
        mark.setPixmap(workspace_icon('activity', '#b9ddfa').pixmap(QSize(23, 23)))
        bar.addWidget(mark)
        title = QLabel('Shaker Control')
        title.setObjectName('appBrand')
        bar.addWidget(title)
        self.setup_name = QLabel('Session setup')
        self.setup_name.setObjectName('setupName')
        self.setup_name.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        bar.addWidget(self.setup_name, 1)
        self.backend_label = QLabel()
        self.backend_label.setObjectName('backendBadge')
        self.backend_label.setAccessibleName('Actual acquisition backend')
        bar.addWidget(self.backend_label)
        outer.addWidget(appbar)

        ribbon_row = QHBoxLayout()
        ribbon_row.setContentsMargins(8, 8, 8, 8)
        ribbon_row.setSpacing(7)
        mode_card = framed('workspaceCard')
        mode_card.setFixedWidth(164)
        mode = QVBoxLayout(mode_card)
        mode.setContentsMargins(5, 6, 5, 8)
        mode.setSpacing(5)
        image = QLabel()
        image.setAlignment(Qt.AlignCenter)
        image.setPixmap(workspace_icon('activity').pixmap(QSize(29, 29)))
        mode.addWidget(image)
        caption = QLabel('Test mode')
        caption.setAlignment(Qt.AlignCenter)
        mode.addWidget(caption)
        self.mode_combo = QComboBox()
        self.mode_combo.setAccessibleName('Test mode')
        self.mode_combo.addItem('Random vibration', 'random')
        self.mode_combo.addItem('Sine sweep', 'sine_sweep')
        self.mode_combo.setCurrentIndex(1 if self.settings.test_mode == 'sine_sweep' else 0)
        mode.addWidget(self.mode_combo)
        ribbon_row.addWidget(mode_card)

        run_controls = QWidget()
        run = QHBoxLayout(run_controls)
        run.setContentsMargins(2, 2, 3, 2)
        run.setSpacing(7)
        backend = QVBoxLayout()
        backend.setSpacing(2)
        backend.addWidget(small_label('Backend'))
        self.backend_combo = QComboBox()
        self.backend_combo.setAccessibleName('Acquisition backend')
        self.backend_combo.addItem('Hardware', False)
        self.backend_combo.addItem('Simulation', True)
        self.backend_combo.setCurrentIndex(1 if self.settings.simulation_mode else 0)
        backend.addWidget(self.backend_combo)
        run.addLayout(backend)
        self.status_label = QLabel('Ready')
        self.status_label.setAccessibleName('Run status')
        self.status_label.setMinimumWidth(68)
        run.addWidget(self.status_label)
        self.start_button = QPushButton()
        self.start_button.setObjectName('startButton')
        self.start_button.setIcon(workspace_icon('play', workspace_colors()['blue_ink'], 16))
        self.stop_button = QPushButton('Stop')
        self.stop_button.setObjectName('stopButton')
        self.stop_button.setIcon(workspace_icon('stop', workspace_colors()['red'], 16))
        run.addWidget(self.start_button)
        run.addWidget(self.stop_button)

        tabs = QHBoxLayout()
        tabs.setSpacing(2)
        self.task_buttons = {}
        self.command_pages = QStackedWidget()
        self.command_pages.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.ribbon_commands = {}
        commands = {
            'Test': [('Target', 'target', lambda: self.inspect_settings('target')),
                     ('Preset', 'preset', None),
                     ('Recording', 'recording', lambda: self.inspect_settings('recording'))],
            'Acquisition': [('Channels', 'activity', lambda: self._browser_selected('channel', 0)),
                            ('Clock', 'clock', lambda: self.inspect_settings('acquisition')),
                            ('Backend setup', 'control', lambda: self.inspect_settings('channel'))],
            'Control': [('Control', 'control', lambda: self.inspect_settings('limits')),
                        ('Equalizer', 'target', self.show_equalizer),
                        ('Calibration', 'calibration', lambda: self.inspect_settings('calibration'))],
            'Results': [('Run report', 'results', self.show_run_report),
                        ('History', 'history', self.show_run_history),
                        ('Recording folder', 'recording', self.results_tab.open_folder)],
            'View': [('Fit plots', 'fit', self.fit_visible_plots)],
        }
        for index, (task, items) in enumerate(commands.items()):
            tab = QPushButton(task)
            tab.setObjectName('ribbonTab')
            tab.setCheckable(True)
            tab.clicked.connect(lambda checked=False, task=task: self.select_task(task))
            self.task_buttons[task] = tab
            tabs.addWidget(tab)
            page = QWidget()
            row = QHBoxLayout(page)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(2)
            for label, icon_name, callback in items:
                button = command_button(label, icon_name, callback)
                self.ribbon_commands[(task, label)] = button
                row.addWidget(button)
            row.addStretch()
            self.command_pages.addWidget(page)
        tabs.addStretch()
        # Common layout tools stay available without changing task or plot.
        self.browser_button = command_button('Browser', 'browser')
        self.browser_button.setCheckable(True)
        self.browser_button.toggled.connect(self._toggle_browser)
        self.setup_button = command_button('Inspector', 'inspector')
        self.setup_button.setCheckable(True)
        inspector_menu = QMenu(self.setup_button)
        inspector_menu.addAction('All test settings', lambda: self.inspect_settings('all'))
        self.setup_button.setMenu(inspector_menu)
        self.setup_button.setPopupMode(QToolButton.MenuButtonPopup)
        common = QWidget()
        common_row = QHBoxLayout(common)
        common_row.setContentsMargins(0, 0, 0, 0)
        common_row.setSpacing(1)
        common_row.addWidget(self.browser_button)
        common_row.addWidget(self.setup_button)
        tools = QWidget()
        tool_row = QHBoxLayout(tools)
        tool_row.setContentsMargins(0, 0, 0, 0)
        tool_row.setSpacing(6)
        tool_row.addWidget(self.command_pages)
        tool_row.addWidget(common)
        self.ribbon = AdaptiveRibbon(tools, run_controls, tabs)
        ribbon_row.addWidget(self.ribbon, 1)
        outer.addLayout(ribbon_row)

        notices = QWidget()
        notice_layout = QVBoxLayout(notices)
        notice_layout.setContentsMargins(8, 0, 8, 0)
        notice_layout.setSpacing(5)
        self.notice_label = small_label(name='setupNotice')
        self.notice_label.hide()
        notice_layout.addWidget(self.notice_label)
        self.fault_banner = framed('faultBanner')
        fault = QHBoxLayout(self.fault_banner)
        fault.setContentsMargins(10, 7, 8, 7)
        self.fault_label = QLabel()
        self.fault_label.setWordWrap(True)
        fault.addWidget(self.fault_label, 1)
        self.switch_simulation_button = QPushButton('Switch to simulation')
        self.switch_simulation_button.clicked.connect(self.switch_to_simulation)
        fault.addWidget(self.switch_simulation_button)
        self.dismiss_fault_button = QPushButton('Dismiss fault')
        self.dismiss_fault_button.clicked.connect(self.dismiss_fault)
        fault.addWidget(self.dismiss_fault_button)
        self.fault_banner.hide()
        notice_layout.addWidget(self.fault_banner)
        outer.addWidget(notices)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(8, 0, 8, 10)
        body_layout.setSpacing(10)
        self.test_browser = TestBrowser()
        self.test_browser.selected.connect(self._browser_selected)
        self.test_browser.collapse_requested.connect(lambda: self._toggle_browser(False))
        body_layout.addWidget(self.test_browser, 0, Qt.AlignTop)
        self.viewport_card = framed('responseCard')
        self.viewport_card.setMinimumWidth(350)
        viewport = QVBoxLayout(self.viewport_card)
        viewport.setContentsMargins(12, 10, 12, 8)
        viewport.setSpacing(8)
        self.header_grid = QGridLayout()
        self.header_grid.setContentsMargins(0, 0, 0, 0)
        titles = QWidget()
        titles_layout = QVBoxLayout(titles)
        titles_layout.setContentsMargins(0, 0, 0, 0)
        titles_layout.setSpacing(2)
        self.view_title = QLabel('Control response PSD')
        self.view_title.setObjectName('viewTitle')
        self.view_subtitle = small_label()
        titles_layout.addWidget(self.view_title)
        titles_layout.addWidget(self.view_subtitle)
        self.header_grid.addWidget(titles, 0, 0)
        self.view_controls = QWidget()
        views = QHBoxLayout(self.view_controls)
        views.setContentsMargins(0, 0, 0, 0)
        views.setSpacing(2)
        self.pages = QStackedWidget()
        self.nav_buttons = []
        for index, (label, page) in enumerate([
            ('Response', self.controller_tab), ('Channels', self.realtime_tab),
            ('Diagnostics', self.controller_tab.diagnostics_widget), ('Results', self.results_tab),
        ]):
            self.pages.addWidget(page)
            button = QPushButton(label)
            button.setObjectName('viewButton')
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, index=index: self.show_page(index))
            self.nav_buttons.append(button)
            views.addWidget(button)
        self.header_grid.addWidget(self.view_controls, 0, 1, Qt.AlignRight)
        viewport.addLayout(self.header_grid)
        viewport.addWidget(self.pages, 1)
        body_layout.addWidget(self.viewport_card, 1)
        self.config_dock = ContextInspector(self.config_tab)
        self.config_dock.visibilityChanged.connect(self._dock_visible)
        self.setup_button.toggled.connect(self._toggle_inspector)
        self.config_dock.hide()
        self.inspector_text = QPlainTextEdit()
        self.inspector_text.setReadOnly(True)
        self.inspector_text.setAccessibleName('Inspected run details')
        self.config_dock.stack.addWidget(self.inspector_text)
        body_layout.addWidget(self.config_dock)
        outer.addWidget(body, 1)

        footer = framed('statusStrip')
        footer_row = QHBoxLayout(footer)
        footer_row.setContentsMargins(8, 5, 10, 5)
        self.history_button = QPushButton('Run history')
        self.history_button.setIcon(workspace_icon('history', size=16))
        self.history_button.clicked.connect(self.show_run_history)
        footer_row.addWidget(self.history_button)
        self.setup_summary = small_label(name='statusSummary')
        self.setup_summary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        footer_row.addWidget(self.setup_summary, 1)
        self.recording_status = small_label(name='statusSummary')
        self.recording_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.recording_status.setMaximumWidth(340)
        footer_row.addWidget(self.recording_status, 1)
        outer.addWidget(footer)

        self.config_tab.test_mode_combo = self.mode_combo
        self.config_tab.applied.connect(self._settings_applied)
        self.config_tab.dirty_changed.connect(self._draft_changed)
        self.config_tab.discarded.connect(self._draft_discarded)
        self.config_tab.context_changed.connect(self._inspector_context_changed)
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        self.backend_combo.currentIndexChanged.connect(self._backend_changed)
        self.start_button.clicked.connect(self.start_controller)
        self.stop_button.clicked.connect(self.stop_controller)
        self.results_tab.run_list.currentRowChanged.connect(self._report_selection_changed)
        preset = self.ribbon_commands[('Test', 'Preset')]
        preset_menu = QMenu(preset)
        self.load_preset_action = preset_menu.addAction('Load preset…', self.config_tab.load_preset)
        self.save_preset_action = preset_menu.addAction('Save preset…', self.config_tab.save_preset)
        preset.setMenu(preset_menu)
        preset.setPopupMode(QToolButton.InstantPopup)
        for plot in (self.controller_tab.psd_plot, self.controller_tab.sine_plot):
            plot.setTitle(None)
        self.select_task('Test')
        self.show_page(0)
        self._rebuild_browser()
        self._record_shell_event('Workspace ready')
        QTimer.singleShot(0, self._reflow_shell)

    def select_task(self, task):
        for index, (name, button) in enumerate(self.task_buttons.items()):
            button.setChecked(name == task)
            if name == task:
                self.command_pages.setCurrentIndex(index)
        if task == 'Results':
            self.show_run_report()
        if hasattr(self, 'ribbon'):
            self.ribbon.reflow()

    def inspect_settings(self, context='all', description=''):
        self._detail_kind = None
        self.config_tab.set_context(context)
        title = self._context_title(context)
        self.config_dock.present(title, description=description)
        if (context, None) in self.test_browser.entries:
            self.test_browser.set_selection(context)
        self._reflow_shell()

    def _context_title(self, context):
        if context == 'target' and self.config_tab.test_mode == 'sine_sweep':
            return 'Sine sweep'
        return self.CONTEXT_TITLES.get(context, 'Test setup')

    def _inspector_context_changed(self, context):
        self.config_dock.title.setText(self._context_title(context))

    def _toggle_inspector(self, visible):
        if visible:
            if self.config_dock.stack.currentIndex() == 0 and self.config_tab.context == 'all':
                self.inspect_settings('target')
            else:
                self.config_dock.show()
        else:
            self.config_dock.hide()
        self._reflow_shell()

    def _toggle_browser(self, visible):
        self._browser_requested = visible
        if visible and self.width() < 1200 and not self.config_dock.isHidden():
            self.config_dock.hide()
        self._reflow_shell()

    def _reflow_shell(self):
        if not hasattr(self, 'config_dock'):
            return
        browser_visible = self._browser_requested and (self.config_dock.isHidden() or self.width() >= 1200)
        self.test_browser.setVisible(browser_visible)
        with QSignalBlocker(self.browser_button):
            self.browser_button.setChecked(browser_visible)
        if self.viewport_card.height() > 200:
            content_height = self.test_browser.content.sizeHint().height() + 48
            self.test_browser.setFixedHeight(min(self.viewport_card.height(), max(320, content_height)))
        stacked = self.viewport_card.width() < 680
        if stacked != self._header_stacked:
            self._header_stacked = stacked
            self.header_grid.removeWidget(self.view_controls)
            self.header_grid.addWidget(self.view_controls, 1 if stacked else 0,
                                       0 if stacked else 1, 1, 2 if stacked else 1,
                                       Qt.AlignLeft if stacked else Qt.AlignRight)

    def _sync_view_header(self, index):
        titles = ['Sine acceleration vs frequency' if self.settings.test_mode == 'sine_sweep' else 'Control response PSD',
                  'Channel response', 'Control diagnostics', 'Run results']
        self.view_title.setText(titles[index])
        self.view_subtitle.setText(self.settings.target_summary if index != 3 else 'Captured setup, actual backend and final outcome')
        QTimer.singleShot(0, self._reflow_shell)

    def _rebuild_browser(self):
        self.test_browser.rebuild(self.settings, self.results_tab.reports)
        QTimer.singleShot(0, self._reflow_shell)

    def _browser_selected(self, key, data):
        if key == 'channel':
            self.show_page(1)
            self.realtime_tab.channel_combo.setCurrentIndex(int(data) + 1)
            label = self.settings.input_channel_labels[int(data)]
            description = f'{label} · ' + ('synthetic acceleration' if self.settings.simulation_mode else f'{self.settings.device_ai} · hardware control input')
            self.inspect_settings('channel', description)
        elif key == 'report':
            self.results_tab.run_list.setCurrentRow(int(data))
            self.show_run_report()
        else:
            if key == 'target':
                self.show_page(0)
            self.inspect_settings(key)
        self.test_browser.set_selection(key, data)

    def _mode_changed(self):
        self.config_tab.set_mode(self.mode_combo.currentData())
        self.inspect_settings('target')

    def _backend_changed(self):
        self.config_tab.set_backend(self.backend_combo.currentData())
        self.inspect_settings('channel')

    def _draft_discarded(self):
        with QSignalBlocker(self.mode_combo), QSignalBlocker(self.backend_combo):
            self.mode_combo.setCurrentIndex(1 if self.settings.test_mode == 'sine_sweep' else 0)
            self.backend_combo.setCurrentIndex(1 if self.settings.simulation_mode else 0)
        self._inspector_context_changed(self.config_tab.context)
        self._refresh_controls()

    def _sync_shell_state(self):
        self.setup_name.setText('Draft setup · apply before starting' if self.config_tab.dirty else
                                (self.settings.preset_source.rsplit('/', 1)[-1] or 'Session setup'))
        self.setup_name.setToolTip(self.settings.preset_source or 'Launch defaults from config.py; session edits do not rewrite that file.')
        full_summary = self.setup_summary.text()
        self.setup_summary.setToolTip(full_summary)
        self.setup_summary.setText(f'{self.settings.fs / 1000:g} kHz · {self.settings.block_seconds:g} s blocks · '
                                   f'Δf {self.settings.fs / self.settings.welch_nperseg:g} Hz · {self.settings.num_input_channels} input(s)')
        self.backend_label.setToolTip(' · '.join(filter(None, (self.input_mode, self.clock_status))))
        self.status_label.setText('Draft setup' if self.config_tab.dirty and not self.busy and not self.fault_message else
                                  {'ready':'Ready', 'starting':'Starting…', 'running':'Running', 'stopping':'Stopping…',
                                   'stopped':'Stopped', 'complete':'Complete', 'fault':'Fault'}[self.run_state])
        self.status_label.setAccessibleName(f'Run status: {self.status_label.text()}')
        self.backend_label.setAccessibleName(f'Acquisition backend: {self.backend_label.text()}')
        color = workspace_colors()['red' if self.fault_message else 'green' if self.run_state == 'running' else 'ink']
        self.status_label.setStyleSheet(f'color: {color};')
        if self.active_log_dir:
            name = self.active_log_dir.rstrip('/').rsplit('/', 1)[-1]
            self.recording_status.setText(('Recording · ' if self.busy else 'Saved · ') + name)
            self.recording_status.setToolTip(self.active_log_dir)
        else:
            self.recording_status.setText('Recording requested · not started' if self.settings.data_log_enabled else 'Recording disabled')
            self.recording_status.setToolTip(self.settings.data_log_dir)
        for action in (self.load_preset_action, self.save_preset_action):
            action.setEnabled(not self.busy)
        self.ribbon_commands[('Control', 'Equalizer')].setEnabled(self.settings.test_mode == 'random')
        self.ribbon_commands[('Results', 'Recording folder')].setEnabled(self.results_tab.open_button.isEnabled())
        self._sync_view_header(self.pages.currentIndex())
        self.ribbon.reflow()

    def _record_shell_event(self, message):
        self.session_events.append(f'{QTime.currentTime().toString("HH:mm:ss")}  {message}')
        self.session_events[:] = self.session_events[-100:]
        self.history_button.setToolTip(self.session_events[-1])
        if self._detail_kind == 'history':
            self.inspector_text.setPlainText('\n\n'.join(self.session_events))

    def show_run_history(self):
        self._detail_kind = 'history'
        self.inspector_text.setPlainText('\n\n'.join(self.session_events) or 'No session events yet.')
        self.config_dock.present('Run history', self.inspector_text,
                                 'Session events · physical actions cannot be undone')
        self._reflow_shell()

    def show_run_report(self):
        self.show_page(3)
        self._detail_kind = 'report'
        self._report_selection_changed(self.results_tab.run_list.currentRow())
        self.config_dock.present('Captured run', self.inspector_text,
                                 'Run-specific settings and outcome')
        self._reflow_shell()

    def _report_selection_changed(self, index):
        if self._detail_kind == 'report':
            if 0 <= index < len(self.results_tab.reports):
                report = self.results_tab.reports[index]
                settings = report['settings']
                backend = report['backend'] or report['requested_backend'] + ' requested · not initialized'
                mode = 'Sine sweep' if settings['test_mode'] == 'sine_sweep' else 'Random vibration'
                details = [f'Run {index + 1} · {report["outcome"].title()}', f'Backend: {backend}',
                           f'Test: {mode}', f'Duration: {report["duration_s"]:.1f} s · {report["blocks"]} blocks',
                           f'Captured setup: {report["config_id"]}',
                           f'Calibration source: {settings["calibration_source"] or "Not recorded"}',
                           f'Recording: {report["log_dir"] or "Disabled or not initialized"}']
                if report['error']:
                    details.insert(1, f'Fault: {report["error"]}')
                self.inspector_text.setPlainText('\n\n'.join(details))
            else:
                self.inspector_text.setPlainText('No completed runs in this session.')
        if hasattr(self, 'ribbon_commands'):
            self.ribbon_commands[('Results', 'Recording folder')].setEnabled(self.results_tab.open_button.isEnabled())

    def show_equalizer(self):
        if self.settings.test_mode != 'random':
            return
        self.show_page(2)
        self.controller_tab.diagnostics_widget.setCurrentIndex(3)
        self.inspect_settings('equalizer')

    def fit_visible_plots(self):
        index = self.pages.currentIndex()
        if index == 0:
            plots = [self.controller_tab.plot_stack.currentWidget()]
        elif index == 1:
            plots = [self.realtime_tab.time_plot, self.realtime_tab.psd_plot]
        elif index == 2:
            plots = [self.controller_tab.diagnostics_widget.currentWidget()]
        else:
            return
        for plot in plots:
            plot.getPlotItem().autoRange()
