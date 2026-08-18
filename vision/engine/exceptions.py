class OpenCVUnavailableError(RuntimeError):
    """Raised when OpenCV is required for live camera support but unavailable."""


class CVZoneUnavailableError(RuntimeError):
    """Raised when CVZone is required for stylized overlay rendering but unavailable."""


class PySimVerseUnavailableError(RuntimeError):
    """Raised when PySimVerse is required for drone control but unavailable."""