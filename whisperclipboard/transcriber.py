"""
Whisper transcription module for Whisper Clipboard.
Handles GPU-accelerated speech-to-text transcription with device fallback.
"""

import whisper
import torch
import numpy as np
import logging
import threading
import time
import re
from typing import Optional, Dict, Any
from .device_detector import get_optimal_device, select_model_for_device


class WhisperTranscriber:
    """GPU-accelerated Whisper transcription with fallback support."""
    
    def __init__(self,
                 model_size: str = "base",
                 device: str = "auto",
                 language: Optional[str] = "en",
                 debug_mode: bool = False,
                 load_model: bool = True):
        """
        Initialize WhisperTranscriber.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Target device (auto, cuda, cpu)
            language: Target language (None for auto-detect)
            debug_mode: Enable verbose debug logging
            load_model: Load model immediately during initialization
        """
        self.debug_mode = debug_mode
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.model_size = model_size
        self.language = language
        self.device = device if device != "auto" else get_optimal_device()

        # Optimize model size for device
        self.model_size = select_model_for_device(self.device, model_size)

        # State
        self.model = None
        self.model_load_lock = threading.Lock()
        self.is_loading = False

        # Performance tracking
        self.transcription_times = []
        self.last_transcription_time = 0.0

        if self.debug_mode:
            self.logger.info(f"WhisperTranscriber initialized: model={self.model_size}, "
                            f"device={self.device}, language={self.language}")

        # Load model immediately if requested
        if load_model:
            self._load_model()
    
    def _load_model(self) -> bool:
        """
        Load Whisper model.

        Returns:
            bool: True if model loaded successfully
        """
        with self.model_load_lock:
            if self.model is not None:
                return True
            
            if self.is_loading:
                # Another thread is loading, wait for it
                while self.is_loading:
                    time.sleep(0.1)
                return self.model is not None
            
            try:
                self.is_loading = True

                # Always show loading message for startup loading (not just debug mode)
                print(f"🔄 Loading Whisper model '{self.model_size}' on {self.device}...")
                if self.debug_mode:
                    self.logger.info(f"Loading Whisper model '{self.model_size}' on {self.device}...")

                start_time = time.time()
                self.model = whisper.load_model(self.model_size, device=self.device)
                load_time = time.time() - start_time

                print(f"✅ Model loaded in {load_time:.1f}s")
                if self.debug_mode:
                    self.logger.info(f"Model loaded successfully in {load_time:.2f}s")

                    # Log model info
                    if hasattr(self.model, 'device'):
                        actual_device = next(self.model.parameters()).device
                        self.logger.info(f"Model loaded on device: {actual_device}")
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to load Whisper model: {e}")
                
                # Try fallback to CPU if GPU loading failed
                if self.device != "cpu":
                    self.logger.info("Attempting fallback to CPU...")
                    try:
                        self.device = "cpu"
                        self.model_size = select_model_for_device("cpu", self.model_size)
                        self.model = whisper.load_model(self.model_size, device="cpu")
                        self.logger.info(f"Fallback successful: model={self.model_size}, device=cpu")
                        return True
                    except Exception as fallback_error:
                        self.logger.error(f"CPU fallback also failed: {fallback_error}")
                
                return False
                
            finally:
                self.is_loading = False
    
    def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess audio data for Whisper.
        
        Args:
            audio_data: Raw audio data from recorder
            
        Returns:
            np.ndarray: Preprocessed audio data
        """
        # Ensure audio is float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Whisper expects audio in [-1, 1] range
        if np.abs(audio_data).max() > 1.0:
            audio_data = audio_data / np.abs(audio_data).max()
        
        # Remove DC bias
        audio_data = audio_data - np.mean(audio_data)
        
        # Basic noise gate (remove very quiet sections)
        threshold = np.std(audio_data) * 0.1
        audio_data = np.where(np.abs(audio_data) < threshold, 0, audio_data)
        
        return audio_data
    
    def _postprocess_text(self, text: str, config: Optional[Dict[str, Any]] = None) -> str:
        """
        Post-process transcribed text.
        
        Args:
            text: Raw transcribed text
            config: Text processing configuration
            
        Returns:
            str: Processed text
        """
        if not text or not text.strip():
            return ""
        
        # Default config if not provided
        if config is None:
            config = {
                'auto_capitalize': True,
                'auto_punctuate': True,
                'remove_filler_words': False
            }
        
        # Clean up whitespace
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        
        # Remove filler words if enabled
        if config.get('remove_filler_words', False):
            filler_words = ['um', 'uh', 'er', 'ah', 'like', 'you know']
            pattern = r'\b(?:' + '|'.join(filler_words) + r')\b'
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
            text = re.sub(r'\s+', ' ', text).strip()  # Clean up spaces again
        
        # Auto-capitalize if enabled
        if config.get('auto_capitalize', True):
            # Capitalize first letter
            if text:
                text = text[0].upper() + text[1:]
            
            # Capitalize after sentence endings
            text = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
        
        # Auto-punctuate if enabled (basic heuristics)
        if config.get('auto_punctuate', True):
            # Add period if no ending punctuation
            if text and not text.endswith(('.', '!', '?', ':', ';')):
                text += '.'
        
        return text
    
    def transcribe(self, 
                  audio_data: np.ndarray,
                  text_config: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Audio data from recorder
            text_config: Text processing configuration
            
        Returns:
            Optional[str]: Transcribed text or None if failed
        """
        if self.model is None:
            self.logger.error("Model not loaded - transcription unavailable")
            return None
        
        if len(audio_data) == 0:
            self.logger.warning("Empty audio data provided")
            return None
        
        try:
            start_time = time.time()
            
            # Preprocess audio
            processed_audio = self._preprocess_audio(audio_data)
            
            # Log audio info
            duration = len(processed_audio) / 16000  # Assuming 16kHz
            self.logger.debug(f"Transcribing {duration:.2f}s of audio...")
            
            # Transcribe
            transcribe_options = {
                "language": self.language,
                "task": "transcribe",
                "fp16": self.device != "cpu",  # Use fp16 on GPU for speed
            }
            
            result = self.model.transcribe(processed_audio, **transcribe_options)
            
            # Extract text
            raw_text = result.get("text", "").strip()
            
            # Post-process text
            final_text = self._postprocess_text(raw_text, text_config)
            
            # Performance tracking
            transcription_time = time.time() - start_time
            self.last_transcription_time = transcription_time
            self.transcription_times.append(transcription_time)
            
            # Keep only last 100 times for average calculation
            if len(self.transcription_times) > 100:
                self.transcription_times.pop(0)
            
            # Log results
            if final_text:
                if self.debug_mode:
                    avg_time = np.mean(self.transcription_times)
                    real_time_factor = transcription_time / duration if duration > 0 else 0

                    self.logger.info(f"Transcribed in {transcription_time:.2f}s "
                                   f"(RTF: {real_time_factor:.2f}, avg: {avg_time:.2f}s): "
                                   f'"{final_text}"')
            else:
                if self.debug_mode:
                    self.logger.warning(f"No transcription result in {transcription_time:.2f}s")
            
            return final_text if final_text else None
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get transcription performance statistics.
        
        Returns:
            Dict[str, Any]: Performance statistics
        """
        if not self.transcription_times:
            return {"transcriptions": 0}
        
        times = self.transcription_times
        return {
            "transcriptions": len(times),
            "last_time": self.last_transcription_time,
            "avg_time": np.mean(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "model_size": self.model_size,
            "device": self.device,
            "model_loaded": self.model is not None
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        info = {
            "model_size": self.model_size,
            "device": self.device,
            "language": self.language,
            "loaded": self.model is not None,
        }
        
        if self.model is not None:
            try:
                actual_device = next(self.model.parameters()).device
                info["actual_device"] = str(actual_device)
                
                # Count parameters (approximate)
                total_params = sum(p.numel() for p in self.model.parameters())
                info["parameters"] = total_params
                
            except Exception as e:
                self.logger.debug(f"Could not get detailed model info: {e}")
        
        return info
    
    def unload_model(self):
        """Unload the model to free memory."""
        with self.model_load_lock:
            if self.model is not None:
                if self.debug_mode:
                    self.logger.info("Unloading Whisper model...")
                del self.model
                self.model = None
                
                # Clear GPU cache if using CUDA
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.unload_model()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
        finally:
            if self.debug_mode:
                self.logger.info("WhisperTranscriber cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


# Test function for standalone testing
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.DEBUG)
    
    # Test with GPU if available
    transcriber = WhisperTranscriber(model_size="base", device="auto")
    
    print("Model info:", transcriber.get_model_info())
    
    # Generate test audio (1 second of sine wave at 440Hz)
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    test_audio = np.sin(440 * 2 * np.pi * t).astype(np.float32) * 0.1
    
    print(f"Testing with {len(test_audio)} samples ({duration}s)...")
    
    # Test transcription (will likely return nothing meaningful for sine wave)
    result = transcriber.transcribe(test_audio)
    print(f"Result: '{result}'")
    
    # Show performance stats
    print("Performance stats:", transcriber.get_performance_stats())
    
    # Cleanup
    transcriber.cleanup()