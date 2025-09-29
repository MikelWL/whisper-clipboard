"""
Manual audio recorder for Whisper Clipboard.
Start/stop recording without continuous callbacks to avoid NumPy buffer issues.
"""

import pyaudio
import numpy as np
import threading
import time
import logging
import os
import sys
import contextlib
from typing import Optional, Callable, List


class ManualAudioRecorder:
    """Manual audio recorder with start/stop control."""
    
    def __init__(self, config: dict, debug_mode: bool = False):
        """
        Initialize ManualAudioRecorder.

        Args:
            config: Audio configuration dictionary
            debug_mode: Enable verbose debug logging
        """
        self.config = config
        self.debug_mode = debug_mode
        self.logger = logging.getLogger(__name__)
        
        # Audio parameters
        self.sample_rate = config['sample_rate']
        self.chunk_size = config['chunk_size']
        self.channels = config['channels']
        self.device_index = config['device_index']
        
        # Recording state
        self.is_recording = False
        self.audio_buffer = []
        self.recording_thread = None
        self.stop_recording_event = threading.Event()
        
        # PyAudio objects
        self.pyaudio = None
        self.stream = None
        
        # Callbacks
        self.on_recording_start: Optional[Callable] = None
        self.on_recording_stop: Optional[Callable[[np.ndarray], None]] = None
        
        self._initialize_audio()

    @contextlib.contextmanager
    def _suppress_alsa_warnings(self):
        """Context manager to suppress ALSA warnings."""
        if self.debug_mode:
            yield  # Don't suppress in debug mode
            return

        # Save original file descriptors
        null_fd = os.open(os.devnull, os.O_RDWR)
        stderr_fd = os.dup(2)

        try:
            # Redirect stderr to null
            os.dup2(null_fd, 2)
            yield
        finally:
            # Restore stderr
            os.dup2(stderr_fd, 2)
            os.close(null_fd)
            os.close(stderr_fd)

    def _initialize_audio(self):
        """Initialize PyAudio and discover audio devices."""
        try:
            with self._suppress_alsa_warnings():
                self.pyaudio = pyaudio.PyAudio()

            if self.debug_mode:
                self.logger.info("PyAudio initialized successfully")

                # Log available audio devices for debugging
                self._log_audio_devices()
            
            # Validate device index
            if self.device_index is not None:
                device_count = self.pyaudio.get_device_count()
                if self.device_index >= device_count:
                    self.logger.warning(f"Device index {self.device_index} out of range, using default")
                    self.device_index = None
                    
        except Exception as e:
            self.logger.error(f"Failed to initialize PyAudio: {e}")
            raise
    
    def _log_audio_devices(self):
        """Log available audio devices for debugging."""
        if not self.logger.isEnabledFor(logging.DEBUG):
            return
            
        device_count = self.pyaudio.get_device_count()
        self.logger.debug(f"Found {device_count} audio devices:")
        
        for i in range(device_count):
            try:
                device_info = self.pyaudio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:  # Input device
                    self.logger.debug(f"  {i}: {device_info['name']} "
                                    f"(channels: {device_info['maxInputChannels']}, "
                                    f"rate: {device_info['defaultSampleRate']})")
            except Exception:
                continue
    
    def _recording_worker(self):
        """Worker thread for recording audio data."""
        try:
            # Open audio stream
            with self._suppress_alsa_warnings():
                self.stream = self.pyaudio.open(
                    format=pyaudio.paFloat32,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=self.device_index,
                    frames_per_buffer=self.chunk_size
                )
            
            if self.debug_mode:
                self.logger.info("Audio stream opened, starting recording...")
            
            # Record until stop event is set
            while not self.stop_recording_event.is_set():
                try:
                    # Read audio data (blocking)
                    audio_data = self.stream.read(
                        self.chunk_size, 
                        exception_on_overflow=False
                    )
                    
                    # Convert to numpy array using safe method
                    try:
                        audio_chunk = np.frombuffer(audio_data, dtype=np.float32)
                    except Exception as e:
                        self.logger.warning(f"Buffer conversion issue: {e}, using fallback")
                        # Fallback conversion
                        import struct
                        num_samples = len(audio_data) // 4
                        audio_chunk = np.array(
                            struct.unpack(f'{num_samples}f', audio_data), 
                            dtype=np.float32
                        )
                    
                    # Add to buffer
                    self.audio_buffer.append(audio_chunk)
                    
                except Exception as e:
                    if not self.stop_recording_event.is_set():
                        self.logger.error(f"Error reading audio data: {e}")
                    break
            
        except Exception as e:
            self.logger.error(f"Error in recording worker: {e}")
        finally:
            # Clean up stream
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                    self.stream = None
                except Exception as e:
                    self.logger.error(f"Error closing stream: {e}")
    
    def start_recording(self):
        """Start recording audio."""
        if self.is_recording:
            self.logger.warning("Already recording")
            return False
        
        if self.debug_mode:
            self.logger.info("Starting manual recording...")

        # Reset state
        self.audio_buffer.clear()
        self.stop_recording_event.clear()
        self.is_recording = True
        
        # Start recording thread
        self.recording_thread = threading.Thread(
            target=self._recording_worker, 
            daemon=True
        )
        self.recording_thread.start()
        
        # Trigger callback
        if self.on_recording_start:
            try:
                self.on_recording_start()
            except Exception as e:
                self.logger.error(f"Error in recording start callback: {e}")
        
        return True
    
    def stop_recording(self):
        """Stop recording and return audio data."""
        if not self.is_recording:
            self.logger.warning("Not currently recording")
            return False
        
        if self.debug_mode:
            self.logger.info("Stopping manual recording...")

        # Signal stop and wait for thread
        self.stop_recording_event.set()
        self.is_recording = False
        
        if self.recording_thread:
            self.recording_thread.join(timeout=2.0)  # Wait max 2 seconds
        
        # Process collected audio
        if self.audio_buffer:
            try:
                # Combine all audio chunks
                combined_audio = np.concatenate(self.audio_buffer)
                duration = len(combined_audio) / self.sample_rate
                
                if self.debug_mode:
                    self.logger.info(f"Recording complete: {len(combined_audio)} samples "
                                   f"({duration:.2f}s)")
                
                # Trigger callback with audio data
                if self.on_recording_stop:
                    try:
                        self.on_recording_stop(combined_audio)
                    except Exception as e:
                        self.logger.error(f"Error in recording stop callback: {e}")
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error processing recorded audio: {e}")
                return False
        else:
            self.logger.warning("No audio data recorded")
            return False
    
    def cancel_recording(self):
        """Cancel current recording without processing."""
        if not self.is_recording:
            return
        
        self.logger.info("Cancelling recording...")
        
        # Signal stop and wait for thread
        self.stop_recording_event.set()
        self.is_recording = False
        
        if self.recording_thread:
            self.recording_thread.join(timeout=2.0)
        
        # Clear buffer without processing
        self.audio_buffer.clear()
    
    def get_audio_devices(self) -> List[dict]:
        """
        Get list of available input audio devices.
        
        Returns:
            List[dict]: List of device information dictionaries
        """
        devices = []
        if not self.pyaudio:
            return devices
        
        device_count = self.pyaudio.get_device_count()
        for i in range(device_count):
            try:
                device_info = self.pyaudio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    devices.append({
                        'index': i,
                        'name': device_info['name'],
                        'channels': device_info['maxInputChannels'],
                        'sample_rate': device_info['defaultSampleRate']
                    })
            except Exception:
                continue
        
        return devices
    
    def cleanup(self):
        """Clean up resources."""
        if self.is_recording:
            self.cancel_recording()
        
        if self.pyaudio:
            try:
                self.pyaudio.terminate()
            except Exception as e:
                self.logger.error(f"Error terminating PyAudio: {e}")
            finally:
                self.pyaudio = None
        
        if self.debug_mode:
            self.logger.info("ManualAudioRecorder cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


# Test function for standalone testing
if __name__ == "__main__":
    import signal
    import sys
    
    logging.basicConfig(level=logging.DEBUG)
    
    # Test configuration
    config = {
        'sample_rate': 16000,
        'chunk_size': 1024,
        'channels': 1,
        'device_index': 0  # Default audio device
    }
    
    recorder = ManualAudioRecorder(config)
    
    # Set up callbacks
    def on_start():
        print("🎤 Recording started!")
    
    def on_stop(audio_data):
        print(f"🛑 Recording stopped! Captured {len(audio_data)} samples "
              f"({len(audio_data) / config['sample_rate']:.2f}s)")
    
    recorder.on_recording_start = on_start
    recorder.on_recording_stop = on_stop
    
    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\nStopping...")
        recorder.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=== Manual Audio Recorder Test ===")
    print("Available devices:")
    for device in recorder.get_audio_devices():
        print(f"  {device['index']}: {device['name']}")
    
    print("\nPress 's' to START recording, 'q' to STOP and quit, 'c' to CANCEL")
    
    try:
        while True:
            command = input("> ").lower().strip()
            
            if command == 's':
                recorder.start_recording()
            elif command == 'q':
                if recorder.is_recording:
                    recorder.stop_recording()
                break
            elif command == 'c':
                recorder.cancel_recording()
            elif command == 'stop':
                if recorder.is_recording:
                    recorder.stop_recording()
            else:
                print("Commands: 's' = start, 'stop' = stop recording, 'c' = cancel, 'q' = quit")
                
    except KeyboardInterrupt:
        pass
    finally:
        recorder.cleanup()