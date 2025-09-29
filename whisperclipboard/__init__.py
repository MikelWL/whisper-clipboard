"""
Whisper Clipboard - Voice-to-Text with Clipboard Integration

A voice recording and transcription tool that uses OpenAI Whisper
to convert speech to text and copies the result to the clipboard.
"""

from .recorder import VoiceRecorder
from .clipboard_manager import ClipboardManager
from .device_detector import get_optimal_device, get_device_info, check_environment

__version__ = "1.0.0"
__all__ = [
    "VoiceRecorder",
    "ClipboardManager",
    "get_optimal_device",
    "get_device_info",
    "check_environment"
]