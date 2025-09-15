#!/usr/bin/env python3
"""
Test the manual audio recorder without continuous callbacks.
"""

import time
import logging
import sys
import os

# Set environment variable for GPU
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_manual_recorder():
    """Test manual recording functionality."""
    from manual_audio_recorder import ManualAudioRecorder
    
    config = {
        'sample_rate': 16000,
        'chunk_size': 1024,
        'channels': 1,
        'device_index': 8  # HD Pro Webcam C920
    }
    
    recorder = ManualAudioRecorder(config)
    
    def on_start():
        print("🎤 Recording started!")
    
    def on_stop(audio_data):
        print(f"🛑 Recording stopped! Captured {len(audio_data)} samples "
              f"({len(audio_data) / config['sample_rate']:.2f}s)")
    
    recorder.on_recording_start = on_start
    recorder.on_recording_stop = on_stop
    
    print("=== Manual Recorder Test ===")
    print("This test will record for 3 seconds without continuous callbacks")
    print("Starting recording in 2 seconds...")
    
    time.sleep(2)
    
    # Test manual start/stop
    print("Starting recording...")
    success = recorder.start_recording()
    
    if success:
        print("Recording for 3 seconds... Speak now!")
        time.sleep(3)
        
        print("Stopping recording...")
        recorder.stop_recording()
        
        print("✅ Manual recording test successful!")
    else:
        print("❌ Failed to start recording")
    
    # Cleanup
    recorder.cleanup()
    print("Test completed!")

if __name__ == "__main__":
    try:
        test_manual_recorder()
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)