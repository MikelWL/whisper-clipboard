#!/usr/bin/env python3
"""
Whisper Clipboard - Voice-to-Text with Clipboard Integration

A voice recording and transcription tool that uses OpenAI Whisper
to convert speech to text and copies the result to the clipboard.
"""

import os
import sys
import yaml
import logging
import argparse
import threading
import time
from pathlib import Path
from pynput import keyboard

from .device_detector import get_optimal_device, get_device_info, check_environment


class VoiceRecorder:
    """Voice recording and transcription with clipboard integration."""

    def __init__(self, config_path: str = "config.yaml", debug_mode: bool = False):
        """
        Initialize VoiceRecorder.

        Args:
            config_path: Path to configuration file
            debug_mode: Enable verbose debug logging
        """
        self.config = self._load_config(config_path)
        self.debug_mode = debug_mode
        self._setup_logging()
        self.device = self._setup_device()
        self.is_recording = False

        # Will be initialized later
        self.audio_recorder = None
        self.transcriber = None
        self.clipboard_manager = None
        self.keyboard_listener = None
        self.hotkey_pressed = False

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logging.error(f"Configuration file {config_path} not found")
            sys.exit(1)
        except yaml.YAMLError as e:
            logging.error(f"Error parsing configuration file: {e}")
            sys.exit(1)

    def _setup_logging(self):
        """Setup logging based on configuration."""
        if self.debug_mode:
            level = getattr(logging, self.config['system']['log_level'], logging.INFO)
            logging.basicConfig(
                level=level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.StreamHandler(sys.stdout),
                    logging.FileHandler('whisper_clipboard.log') if self.config['system']['debug_mode'] else logging.NullHandler()
                ]
            )
        else:
            # Minimal logging for normal use - only errors
            logging.basicConfig(
                level=logging.ERROR,
                format='❌ %(message)s',
                handlers=[logging.StreamHandler(sys.stdout)]
            )
        self.logger = logging.getLogger(__name__)

    def _setup_device(self) -> str:
        """Setup and validate compute device."""
        config_device = self.config['whisper']['device']

        if config_device == "auto":
            device = get_optimal_device()
        else:
            device = config_device

        if self.debug_mode:
            self.logger.info(f"Using device: {device}")
            if self.config['system']['debug_mode']:
                check_environment()

        return device

    def initialize_components(self):
        """Initialize audio, transcription, and clipboard components."""
        if self.debug_mode:
            self.logger.info("Initializing VoiceRecorder components...")

        # Import here to avoid circular dependencies and allow device setup first
        from .manual_audio_recorder import ManualAudioRecorder
        from .transcriber import WhisperTranscriber
        from .clipboard_manager import ClipboardManager

        # Initialize components
        self.audio_recorder = ManualAudioRecorder(self.config['audio'], debug_mode=self.debug_mode)
        self.transcriber = WhisperTranscriber(
            model_size=self.config['whisper']['model_size'],
            device=self.device,
            language=self.config['whisper']['language'],
            debug_mode=self.debug_mode,
            load_model=True  # Load model immediately on startup
        )
        self.clipboard_manager = ClipboardManager(debug_mode=self.debug_mode)

        # Set up audio recorder callbacks
        self.audio_recorder.on_recording_start = self._on_recording_start
        self.audio_recorder.on_recording_stop = self._on_recording_stop

        if self.debug_mode:
            self.logger.info("All components initialized successfully")

    def _setup_hotkey_listener(self):
        """Set up keyboard listener for hotkey detection."""
        record_key = self.config['hotkeys']['record_key']

        # Map common key names to pynput format
        key_mapping = {
            'ctrl_r': keyboard.Key.ctrl_r,
            'ctrl_l': keyboard.Key.ctrl_l,
            'right_ctrl': keyboard.Key.ctrl_r,
            'left_ctrl': keyboard.Key.ctrl_l,
            'alt_r': keyboard.Key.alt_r,
            'alt_l': keyboard.Key.alt_l,
            'right_alt': keyboard.Key.alt_r,
            'left_alt': keyboard.Key.alt_l,
            'f12': keyboard.Key.f12,
            'f11': keyboard.Key.f11,
            'f10': keyboard.Key.f10,
        }

        self.target_key = key_mapping.get(record_key, record_key)

        def on_press(key):
            if key == self.target_key and not self.hotkey_pressed:
                self.hotkey_pressed = True
                self._start_recording()

        def on_release(key):
            if key == self.target_key and self.hotkey_pressed:
                self.hotkey_pressed = False
                self._stop_recording()

        self.keyboard_listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release
        )
        self.keyboard_listener.start()
        if self.debug_mode:
            self.logger.info(f"Hotkey listener started for key: {record_key}")

    def start_hotkey_mode(self):
        """Start the hotkey-based recording mode."""
        try:
            self.initialize_components()
            self._setup_hotkey_listener()

            print(f"\n=== Whisper Clipboard - Voice Recorder ===")
            print(f"📋 Transcribed text will be copied to clipboard")
            print(f"🎯 Hold '{self.config['hotkeys']['record_key']}' to record")
            print(f"🛑 Press Ctrl+C to quit")
            print()

            # Keep running until interrupted
            while True:
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.cleanup()

    def _on_recording_start(self):
        """Callback when recording starts (from audio recorder)."""
        if self.debug_mode:
            self.logger.info("🎤 Recording started...")
        else:
            print("🎤 Recording...")
        self.is_recording = True

    def _on_recording_stop(self, audio_data):
        """
        Callback when recording stops (from audio recorder).

        Args:
            audio_data: Recorded audio data as numpy array
        """
        if self.debug_mode:
            self.logger.info("🛑 Recording stopped, transcribing...")
        else:
            print("🎯 Transcribing...")
        self.is_recording = False

        # Transcribe audio in background thread to avoid blocking
        threading.Thread(target=self._process_audio, args=(audio_data,), daemon=True).start()

    def _process_audio(self, audio_data):
        """
        Process recorded audio data.

        Args:
            audio_data: Audio data to transcribe
        """
        try:
            # Transcribe audio
            text = self.transcriber.transcribe(
                audio_data,
                text_config=self.config['text']
            )

            if text and text.strip():
                if self.debug_mode:
                    self.logger.info(f"💬 Transcribed: '{text}'")

                # Copy text to clipboard
                success = self.clipboard_manager.copy_to_clipboard(text)
                if success:
                    if self.debug_mode:
                        self.logger.info("✅ Text copied to clipboard successfully")
                        print(f"\n📋 COPIED TO CLIPBOARD: {text}\n")
                else:
                    self.logger.error("❌ Failed to copy text to clipboard")
            else:
                if self.debug_mode:
                    self.logger.warning("⚠️  No text transcribed")
                else:
                    print("⚠️  No speech detected")

        except Exception as e:
            self.logger.error(f"❌ Error processing audio: {e}")

    def _start_recording(self):
        """Start audio recording (called by hotkey handler)."""
        if not self.audio_recorder:
            self.logger.error("Audio recorder not initialized")
            return

        if self.is_recording:
            return  # Already recording

        success = self.audio_recorder.start_recording()
        if not success:
            self.logger.error("Failed to start recording")

    def _stop_recording(self):
        """Stop recording and process audio (called by hotkey handler)."""
        if not self.audio_recorder:
            return

        if not self.is_recording:
            return  # Not recording

        success = self.audio_recorder.stop_recording()
        if not success:
            self.logger.error("Failed to stop recording")

    def record_once(self):
        """Record a single audio clip and transcribe it."""
        if not self.audio_recorder:
            self.logger.error("Audio recorder not initialized")
            return False

        if self.is_recording:
            self.logger.warning("Already recording")
            return False

        try:
            print("🎤 Press Enter to start recording...")
            input()

            # Start recording
            success = self.audio_recorder.start_recording()
            if not success:
                self.logger.error("Failed to start recording")
                return False

            print("🎤 Recording... Press Enter to stop.")
            input()

            # Stop recording
            success = self.audio_recorder.stop_recording()
            if not success:
                self.logger.error("Failed to stop recording")
                return False

            return True

        except KeyboardInterrupt:
            self.logger.info("Recording cancelled by user")
            if self.audio_recorder:
                self.audio_recorder.cancel_recording()
            return False

    def interactive_mode(self):
        """Run in interactive mode for multiple recordings."""
        print("\n=== Whisper Clipboard - Voice Recorder ===")
        print("📋 Transcribed text will be copied to clipboard")
        print("🎤 Press Enter to record, Enter again to stop")
        print()

        try:
            self.initialize_components()

            while True:
                print("\nControls:")
                print("  [Enter] - Record and transcribe")
                print("  [q + Enter] - Quit")
                print()

                choice = input("Action: ").strip().lower()

                if choice == 'q':
                    break
                elif choice == '':
                    self.record_once()
                else:
                    print("Invalid choice. Press Enter to record or 'q' to quit.")

        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        if self.audio_recorder:
            self.audio_recorder.cleanup()
        if self.transcriber:
            self.transcriber.cleanup()

        if self.debug_mode:
            self.logger.info("VoiceRecorder cleaned up")

