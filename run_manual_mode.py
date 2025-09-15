#!/usr/bin/env python3
"""
Manual mode WhisperDictate - bypasses automatic voice detection to test core functionality.
"""

import time
import yaml
import logging
import sys
import os

# Set environment variable for GPU
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_manual_dictation():
    """Test dictation with manual audio input."""
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    from transcriber import WhisperTranscriber
    from keyboard_injector import KeyboardInjector
    
    logger.info("Initializing components for manual test...")
    
    transcriber = WhisperTranscriber(
        model_size=config['whisper']['model_size'],
        device='cuda',  # Force GPU
        language=config['whisper']['language']
    )
    
    keyboard_injector = KeyboardInjector(
        typing_delay=config['system']['typing_delay']
    )
    
    logger.info("Components initialized successfully!")
    
    # Create some test audio (1 second of sine wave)
    import numpy as np
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create a more complex waveform that might produce some transcription
    test_audio = (
        0.3 * np.sin(440 * 2 * np.pi * t) +  # A4 note
        0.2 * np.sin(880 * 2 * np.pi * t) +  # A5 note  
        0.1 * np.sin(220 * 2 * np.pi * t) +  # A3 note
        0.1 * np.random.normal(0, 0.1, len(t))  # Some noise
    ).astype(np.float32)
    
    logger.info("Testing GPU transcription with synthetic audio...")
    
    # Test transcription
    result = transcriber.transcribe(test_audio, config['text'])
    
    if result:
        logger.info(f"Transcription result: '{result}'")
        logger.info("Testing keyboard injection...")
        
        print("KEYBOARD INJECTION TEST - Switch to a text editor in 3 seconds...")
        for i in range(3, 0, -1):
            print(f"  {i}...")
            time.sleep(1)
        
        success = keyboard_injector.inject_text(f"WhisperDictate GPU Test: {result}")
        if success:
            logger.info("✅ Keyboard injection successful!")
        else:
            logger.error("❌ Keyboard injection failed!")
            
    else:
        logger.warning("No transcription result (expected with synthetic audio)")
        
        # Test keyboard injection anyway
        print("KEYBOARD INJECTION TEST - Switch to a text editor in 3 seconds...")
        for i in range(3, 0, -1):
            print(f"  {i}...")
            time.sleep(1)
            
        test_text = "WhisperDictate Manual Test - GPU transcription system ready!"
        success = keyboard_injector.inject_text(test_text)
        if success:
            logger.info("✅ Keyboard injection test successful!")
        else:
            logger.error("❌ Keyboard injection test failed!")
    
    # Show performance stats
    logger.info(f"Transcriber stats: {transcriber.get_performance_stats()}")
    logger.info(f"Keyboard stats: {keyboard_injector.get_statistics()}")
    
    # Cleanup
    transcriber.cleanup()
    logger.info("Manual test completed!")

if __name__ == "__main__":
    try:
        test_manual_dictation()
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)