#!/usr/bin/env python3
"""
WhisperDictate - Voice-to-Text Dictation for CLI environments

A background daemon that listens for hotkeys and transcribes speech
using OpenAI Whisper, then injects the text as keystrokes.
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from typing import Optional

from device_detector import get_optimal_device, get_device_info, check_environment


class WhisperDictate:
    """Main application class for WhisperDictate."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize WhisperDictate.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.device = self._setup_device()
        self.is_running = False
        
        # Will be initialized later
        self.audio_recorder = None
        self.transcriber = None
        self.keyboard_injector = None
        self.hotkey_handler = None
        
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
        level = getattr(logging, self.config['system']['log_level'], logging.INFO)
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('whisper_dictate.log') if self.config['system']['debug_mode'] else logging.NullHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_device(self) -> str:
        """Setup and validate compute device."""
        config_device = self.config['whisper']['device']
        
        if config_device == "auto":
            device = get_optimal_device()
        else:
            device = config_device
            
        self.logger.info(f"Using device: {device}")
        
        if self.config['system']['debug_mode']:
            check_environment()
            
        return device
    
    def initialize_components(self):
        """Initialize audio, transcription, keyboard, and hotkey components."""
        self.logger.info("Initializing WhisperDictate components...")
        
        # Import here to avoid circular dependencies and allow device setup first
        from manual_audio_recorder import ManualAudioRecorder
        from transcriber import WhisperTranscriber
        from keyboard_injector import KeyboardInjector
        from hotkey_handler import HotkeyHandler
        
        # Initialize components
        self.audio_recorder = ManualAudioRecorder(self.config['audio'])
        self.transcriber = WhisperTranscriber(
            model_size=self.config['whisper']['model_size'],
            device=self.device,
            language=self.config['whisper']['language']
        )
        self.keyboard_injector = KeyboardInjector(
            typing_delay=self.config['system']['typing_delay'],
            injection_mode=self.config['system'].get('injection_mode', 'type'),
            paste_key=self.config['system'].get('paste_key', 'ctrl+v'),
            restore_clipboard=self.config['system'].get('restore_clipboard', True),
            fail_fast=self.config['system'].get('fail_fast', False)
        )
        self.hotkey_handler = HotkeyHandler()
        
        # Configure hotkey handler
        self.hotkey_handler.configure_record_key(self.config['hotkeys']['record_key'])
        
        # Set up audio recorder callbacks
        self.audio_recorder.on_recording_start = self._on_recording_start
        self.audio_recorder.on_recording_stop = self._on_recording_stop
        
        # Set up hotkey callbacks
        self.hotkey_handler.on_start_recording = self._start_recording
        self.hotkey_handler.on_stop_recording = self._stop_recording
        self.hotkey_handler.on_cancel_recording = self._cancel_recording
        
        self.logger.info("All components initialized successfully")
    
    def start_daemon(self):
        """Start the background daemon process."""
        if self.is_running:
            self.logger.warning("Daemon is already running")
            return
            
        self.logger.info("Starting WhisperDictate daemon...")
        
        try:
            self.initialize_components()
            
            # Start hotkey listening
            self.hotkey_handler.start_listening()
            
            self.is_running = True
            
            self.logger.info("WhisperDictate daemon started successfully")
            self.logger.info(f"Hold '{self.config['hotkeys']['record_key']}' to record")
            self.logger.info(f"Press {self.config['hotkeys']['cancel_keys']} to cancel recording")
            
            # Keep the daemon running
            self._run_daemon_loop()
            
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, stopping daemon...")
            self.stop_daemon()
        except Exception as e:
            self.logger.error(f"Error starting daemon: {e}")
            sys.exit(1)
    
    
    def _run_daemon_loop(self):
        """Main daemon loop."""
        import time
        
        try:
            while self.is_running:
                time.sleep(0.1)  # Small sleep to prevent high CPU usage
        except KeyboardInterrupt:
            self.stop_daemon()
        except Exception as e:
            self.logger.error(f"Error in daemon loop: {e}")
            self.stop_daemon()
    
    def _on_recording_start(self):
        """Callback when recording starts (from audio recorder)."""
        self.logger.info("🎤 Recording started...")
    
    def _on_recording_stop(self, audio_data):
        """
        Callback when recording stops (from audio recorder).
        
        Args:
            audio_data: Recorded audio data as numpy array
        """
        self.logger.info("🛑 Recording stopped, transcribing...")
        
        # Transcribe audio in background thread to avoid blocking
        import threading
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
                self.logger.info(f"💬 Transcribed: '{text}'")
                
                # Inject text as keystrokes
                success = self.keyboard_injector.inject_text(text)
                if success:
                    self.logger.info("✅ Text injected successfully")
                else:
                    self.logger.error("❌ Failed to inject text")
            else:
                self.logger.warning("⚠️  No text transcribed")
                
        except Exception as e:
            self.logger.error(f"❌ Error processing audio: {e}")
    
    def _start_recording(self):
        """Start audio recording (called by hotkey handler)."""
        if not self.audio_recorder:
            self.logger.error("Audio recorder not initialized")
            return
        
        success = self.audio_recorder.start_recording()
        if not success:
            self.logger.error("Failed to start recording")
    
    def _stop_recording(self):
        """Stop recording and process audio (called by hotkey handler)."""
        if not self.audio_recorder:
            return
        
        success = self.audio_recorder.stop_recording()
        if not success:
            self.logger.error("Failed to stop recording")
    
    def _cancel_recording(self):
        """Cancel current recording without processing (called by hotkey handler)."""
        if not self.audio_recorder:
            return
        
        self.logger.info("🚫 Cancelling recording...")
        self.audio_recorder.cancel_recording()
    
    def stop_daemon(self):
        """Stop the daemon gracefully."""
        self.logger.info("Stopping WhisperDictate daemon...")
        self.is_running = False
        
        # Cleanup components
        if self.hotkey_handler:
            self.hotkey_handler.cleanup()
        if self.audio_recorder:
            self.audio_recorder.cleanup()
        if self.transcriber:
            self.transcriber.cleanup()
        
        self.logger.info("Daemon stopped successfully")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="WhisperDictate - Voice-to-Text for CLI")
    parser.add_argument("--config", "-c", default="config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--debug", "-d", action="store_true",
                       help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Override debug mode if specified
    if args.debug:
        # Load config temporarily to modify it
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            config['system']['debug_mode'] = True
            config['system']['log_level'] = 'DEBUG'
            # Write back temporarily (for this session)
        except:
            pass
    
    # Create and start the daemon
    dictate = WhisperDictate(args.config)
    dictate.start_daemon()


if __name__ == "__main__":
    main()