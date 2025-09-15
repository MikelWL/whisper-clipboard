"""
Audio recording module for WhisperDictate.
Handles real-time audio capture, voice activity detection, and silence timeout.
"""

import pyaudio
import numpy as np
import threading
import time
import logging
from typing import Optional, Callable, List
from collections import deque


class AudioRecorder:
    """Real-time audio recorder with voice activity detection."""
    
    def __init__(self, config: dict):
        """
        Initialize AudioRecorder.
        
        Args:
            config: Audio configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Audio parameters
        self.sample_rate = config['sample_rate']
        self.chunk_size = config['chunk_size']
        self.channels = config['channels']
        self.device_index = config['device_index']
        
        # Recording state
        self.is_recording = False
        self.is_listening = False  # For VAD
        self.audio_data = deque()
        self.lock = threading.Lock()
        
        # PyAudio objects
        self.pyaudio = None
        self.stream = None
        
        # Voice activity detection
        self.vad_threshold = 0.01  # Will be set from config
        self.silence_start_time = None
        self.silence_timeout = 2.0  # Will be set from config
        
        # Callbacks
        self.on_recording_start: Optional[Callable] = None
        self.on_recording_stop: Optional[Callable[[np.ndarray], None]] = None
        self.on_silence_timeout: Optional[Callable] = None
        
        self._initialize_audio()
    
    def _initialize_audio(self):
        """Initialize PyAudio and discover audio devices."""
        try:
            self.pyaudio = pyaudio.PyAudio()
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
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio callback for real-time audio processing.
        
        Args:
            in_data: Audio data from microphone
            frame_count: Number of frames
            time_info: Timing information
            status: Stream status flags
            
        Returns:
            Tuple: (None, pyaudio.paContinue)
        """
        try:
            if status:
                self.logger.warning(f"Audio callback status: {status}")
            
            # Convert to numpy array (fix for PY_SSIZE_T_CLEAN error)
            try:
                audio_chunk = np.frombuffer(in_data, dtype=np.float32)
            except (SystemError, ValueError) as e:
                # Fallback for PY_SSIZE_T_CLEAN macro issue
                self.logger.warning(f"NumPy buffer conversion error: {e}, using fallback")
                try:
                    # Try alternative conversion using struct
                    import struct
                    num_samples = len(in_data) // 4  # 4 bytes per float32
                    audio_chunk = np.array(struct.unpack(f'{num_samples}f', in_data), dtype=np.float32)
                except Exception as e2:
                    self.logger.error(f"Fallback conversion also failed: {e2}")
                    # Return silence to prevent crash
                    audio_chunk = np.zeros(self.chunk_size, dtype=np.float32)
            
            # Voice activity detection
            if self._detect_voice_activity(audio_chunk):
                self._handle_voice_detected(audio_chunk)
            else:
                self._handle_silence_detected()
                
        except Exception as e:
            self.logger.error(f"Error in audio callback: {e}")
            # Continue processing to avoid stopping the stream
        
        return (None, pyaudio.paContinue)
    
    def _detect_voice_activity(self, audio_chunk: np.ndarray) -> bool:
        """
        Detect voice activity using RMS energy.
        
        Args:
            audio_chunk: Audio data chunk
            
        Returns:
            bool: True if voice activity detected
        """
        # Calculate RMS (Root Mean Square) energy
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        return rms > self.vad_threshold
    
    def _handle_voice_detected(self, audio_chunk: np.ndarray):
        """
        Handle voice activity detection.
        
        Args:
            audio_chunk: Audio data chunk with voice activity
        """
        with self.lock:
            # Start recording if not already started
            if not self.is_recording:
                self.is_recording = True
                self.audio_data.clear()
                self.logger.info("Voice detected - starting recording")
                if self.on_recording_start:
                    threading.Thread(target=self.on_recording_start, daemon=True).start()
            
            # Add audio data to buffer
            self.audio_data.append(audio_chunk.copy())
            
            # Reset silence timer
            self.silence_start_time = None
    
    def _handle_silence_detected(self):
        """Handle silence detection and timeout logic."""
        if not self.is_recording:
            return
        
        current_time = time.time()
        
        if self.silence_start_time is None:
            self.silence_start_time = current_time
        elif current_time - self.silence_start_time > self.silence_timeout:
            # Silence timeout reached
            self.logger.info(f"Silence timeout ({self.silence_timeout}s) - stopping recording")
            self._stop_recording_internal()
    
    def _stop_recording_internal(self):
        """Internal method to stop recording and process audio."""
        with self.lock:
            if not self.is_recording:
                return
            
            self.is_recording = False
            self.silence_start_time = None
            
            # Combine all audio chunks
            if self.audio_data:
                combined_audio = np.concatenate(list(self.audio_data))
                self.audio_data.clear()
                
                self.logger.info(f"Recording stopped - captured {len(combined_audio)} samples "
                               f"({len(combined_audio) / self.sample_rate:.2f}s)")
                
                # Callback with audio data
                if self.on_recording_stop:
                    threading.Thread(
                        target=self.on_recording_stop, 
                        args=(combined_audio,), 
                        daemon=True
                    ).start()
            else:
                self.logger.warning("No audio data captured")
    
    def start_listening(self, 
                       vad_threshold: float = 0.01,
                       silence_timeout: float = 2.0,
                       max_duration: float = 30.0):
        """
        Start listening for voice activity.
        
        Args:
            vad_threshold: Voice activity detection threshold
            silence_timeout: Seconds of silence before auto-stop
            max_duration: Maximum recording duration in seconds
        """
        if self.is_listening:
            self.logger.warning("Already listening")
            return
        
        self.vad_threshold = vad_threshold
        self.silence_timeout = silence_timeout
        
        try:
            # Open audio stream
            self.stream = self.pyaudio.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback,
                start=False
            )
            
            self.is_listening = True
            self.stream.start_stream()
            
            self.logger.info(f"Started listening (threshold: {vad_threshold}, "
                           f"timeout: {silence_timeout}s)")
            
        except Exception as e:
            self.logger.error(f"Failed to start listening: {e}")
            self.cleanup()
            raise
    
    def stop_listening(self):
        """Stop listening and clean up audio stream."""
        if not self.is_listening:
            return
        
        self.logger.info("Stopping audio listening")
        
        # Stop any ongoing recording
        if self.is_recording:
            self._stop_recording_internal()
        
        # Close audio stream
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                self.logger.error(f"Error closing audio stream: {e}")
            finally:
                self.stream = None
        
        self.is_listening = False
    
    def force_stop_recording(self):
        """Force stop current recording without processing."""
        with self.lock:
            if self.is_recording:
                self.logger.info("Force stopping recording")
                self.is_recording = False
                self.audio_data.clear()
                self.silence_start_time = None
    
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
        self.stop_listening()
        
        if self.pyaudio:
            try:
                self.pyaudio.terminate()
            except Exception as e:
                self.logger.error(f"Error terminating PyAudio: {e}")
            finally:
                self.pyaudio = None
        
        self.logger.info("AudioRecorder cleaned up")
    
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
        'device_index': None
    }
    
    recorder = AudioRecorder(config)
    
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
    
    # Start listening
    print("Starting audio recorder test...")
    print("Available devices:")
    for device in recorder.get_audio_devices():
        print(f"  {device['index']}: {device['name']}")
    
    try:
        recorder.start_listening(vad_threshold=0.01, silence_timeout=2.0)
        print("Listening for voice... (Ctrl+C to quit)")
        
        # Keep running
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        pass
    finally:
        recorder.cleanup()