#!/usr/bin/env python3
"""
Whisper Clipboard - Main Entry Point

Run this script to start the voice-to-text recorder with clipboard integration.
"""

import sys
import os
import argparse

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from whisperclipboard.recorder import VoiceRecorder


def main():
    """Main entry point for Whisper Clipboard."""
    parser = argparse.ArgumentParser(description="Whisper Clipboard - Voice-to-Text Recorder")
    parser.add_argument("--config", "-c", default="config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--debug", "-d", action="store_true",
                       help="Enable verbose debug logging")
    parser.add_argument("--once", action="store_true",
                       help="Record once and exit")

    args = parser.parse_args()

    # Create recorder
    recorder = VoiceRecorder(args.config, debug_mode=args.debug)

    if args.once:
        # Single recording mode (still uses Enter key)
        try:
            recorder.initialize_components()
            print("🎤 Single recording mode")
            recorder.record_once()
        except KeyboardInterrupt:
            print("\nCancelled")
        finally:
            recorder.cleanup()
    else:
        # Hotkey mode (uses configured hotkey)
        recorder.start_hotkey_mode()


if __name__ == "__main__":
    main()