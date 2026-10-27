"""Filesystem paths shared by desktop launchers, setup capture and recording."""
from pathlib import Path, PureWindowsPath


APP_ROOT = Path(__file__).resolve().parent


def default_recording_directory():
    """Follow the OS Documents location, including redirected Windows folders.

    Query Qt lazily so numerical setup helpers do not initialize a GUI. Do not
    create directories until recording starts; a failed write must fault the run.
    """
    from PySide6.QtCore import QStandardPaths

    documents = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
    base = Path(documents) if documents else Path.home() / 'Documents'
    return base / 'ShakerControl' / 'Runs'


def resolve_recording_directory(value):
    """Resolve explicit relative folders against the app, never the launch cwd."""
    if value is None:
        return default_recording_directory().resolve()
    path = Path(value).expanduser()
    windows_path = PureWindowsPath(value)
    if windows_path.drive and not path.drive:
        raise ValueError('This is a Windows recording path. Choose a folder on this computer.')
    if path.drive and not path.is_absolute():
        raise ValueError('Use a full recording path, including the drive and root folder.')
    if path.root and not path.is_absolute():
        raise ValueError('Choose a recording folder with a drive or share name on this computer.')
    if not path.is_absolute():
        path = APP_ROOT / path
    return path.resolve()


def display_path_name(value):
    """Show just the filename for local paths and reports from either platform."""
    return PureWindowsPath(str(value)).name if value else ''
