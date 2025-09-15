"""
Hotkey handler for WhisperDictate supporting both keyboard and mouse buttons.
"""

import logging
import threading
from typing import Callable, Optional, Dict, Any
import keyboard


class HotkeyHandler:
    """Unified hotkey handler for keyboard and mouse inputs."""
    
    def __init__(self):
        """Initialize HotkeyHandler."""
        self.logger = logging.getLogger(__name__)
        
        # State tracking
        self.is_active = False
        self.is_recording = False
        
        # Callbacks
        self.on_start_recording: Optional[Callable] = None
        self.on_stop_recording: Optional[Callable] = None
        self.on_cancel_recording: Optional[Callable] = None
        
        # Configuration
        self.record_key = "right_ctrl"  # Right Ctrl key
        self.cancel_key_combo = "ctrl+shift+c"
        
        self.logger.info("HotkeyHandler initialized")
    
    def configure_record_key(self, key_name: str):
        """
        Configure which key to use for recording.
        
        Args:
            key_name: Name of key ('right_ctrl', 'right_alt', 'f12', etc.)
        """
        # Map to keyboard library format
        key_map = {
            'right_ctrl': 'right_ctrl',
            'right_alt': 'right_alt', 
            'f12': 'f12',
            'f11': 'f11',
            'f10': 'f10',
        }
        
        if key_name.lower() in key_map:
            self.record_key = key_map[key_name.lower()]
            self.logger.info(f"Record key set to: {key_name}")
        else:
            self.logger.warning(f"Unknown key: {key_name}")
    
    def _on_record_key_press(self):
        """Handle record key press."""
        try:
            if not self.is_recording:
                if self.on_start_recording:
                    self.logger.info(f"Record key {self.record_key} pressed - starting recording")
                    self.is_recording = True
                    self.on_start_recording()
        except Exception as e:
            self.logger.error(f"Error in record key press handler: {e}")
    
    def _on_record_key_release(self):
        """Handle record key release."""
        try:
            if self.is_recording:
                if self.on_stop_recording:
                    self.logger.info(f"Record key {self.record_key} released - stopping recording")
                    self.is_recording = False
                    self.on_stop_recording()
        except Exception as e:
            self.logger.error(f"Error in record key release handler: {e}")
    
    def _on_cancel_key(self):
        """Handle cancel key combination."""
        try:
            if self.is_recording and self.on_cancel_recording:
                self.logger.info("Cancel key combination detected")
                self.on_cancel_recording()
        except Exception as e:
            self.logger.error(f"Error in cancel key handler: {e}")
    
    
    def start_listening(self):
        """Start listening for hotkey events."""
        if self.is_active:
            self.logger.warning("Already listening for hotkeys")
            return
        
        try:
            # Register global hotkeys using keyboard library
            keyboard.on_press_key(self.record_key, lambda _: self._on_record_key_press())
            keyboard.on_release_key(self.record_key, lambda _: self._on_record_key_release())
            keyboard.add_hotkey(self.cancel_key_combo, self._on_cancel_key)
            
            self.is_active = True
            self.logger.info(f"Global hotkey listening started - Record key: {self.record_key}")
            self.logger.info(f"Controls: Hold {self.record_key} to record, {self.cancel_key_combo} to cancel")
            
        except Exception as e:
            self.logger.error(f"Failed to start hotkey listeners: {e}")
            self.stop_listening()
            raise
    
    def stop_listening(self):
        """Stop listening for hotkey events."""
        if not self.is_active:
            return
        
        self.logger.info("Stopping hotkey listening...")
        
        try:
            # Remove all hotkeys registered by this handler
            keyboard.unhook_all()
        except Exception as e:
            self.logger.error(f"Error stopping keyboard listener: {e}")
        
        self.is_active = False
        self.is_recording = False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of hotkey handler.
        
        Returns:
            Dict[str, Any]: Status information
        """
        return {
            'is_active': self.is_active,
            'is_recording': self.is_recording,
            'record_key': str(self.record_key),
            'cancel_key_combo': self.cancel_key_combo,
        }
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_listening()
        self.logger.info("HotkeyHandler cleaned up")


# Test function for standalone testing
if __name__ == "__main__":
    import time
    import signal
    import sys
    
    logging.basicConfig(level=logging.DEBUG)
    
    handler = HotkeyHandler()
    
    # Set up test callbacks
    def on_start():
        print("🎤 RECORDING STARTED!")
    
    def on_stop():
        print("🛑 RECORDING STOPPED!")
    
    def on_cancel():
        print("🚫 RECORDING CANCELLED!")
    
    handler.on_start_recording = on_start
    handler.on_stop_recording = on_stop
    handler.on_cancel_recording = on_cancel
    
    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\nShutting down...")
        handler.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=== Hotkey Handler Test ===")
    print(f"Record key configured: {handler.record_key}")
    print("Controls:")
    print(f"  - Hold {handler.record_key} to start/stop recording")
    print("  - Press Ctrl+Shift+C to cancel recording")
    print("  - Press Ctrl+C to quit")
    print()
    
    try:
        handler.start_listening()
        print("Hotkey handler active. Try the controls above...")
        
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        pass
    finally:
        handler.cleanup()