#!/usr/bin/env python3
"""
Test audio device capabilities to find supported sample rates.
"""

import pyaudio
import logging

logging.basicConfig(level=logging.INFO)

def test_audio_devices():
    """Test available audio devices and their supported sample rates."""
    pa = pyaudio.PyAudio()
    
    print("=== Audio Device Information ===")
    
    # Get default input device
    try:
        default_info = pa.get_default_input_device_info()
        print(f"Default input device: {default_info['name']}")
        print(f"Default sample rate: {default_info['defaultSampleRate']}")
        print(f"Max input channels: {default_info['maxInputChannels']}")
        print()
    except Exception as e:
        print(f"Error getting default device: {e}")
    
    # List all input devices
    print("=== All Input Devices ===")
    device_count = pa.get_device_count()
    
    for i in range(device_count):
        try:
            info = pa.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"Device {i}: {info['name']}")
                print(f"  Channels: {info['maxInputChannels']}")
                print(f"  Default Rate: {info['defaultSampleRate']}")
                
                # Test common sample rates
                test_rates = [8000, 16000, 22050, 44100, 48000]
                supported_rates = []
                
                for rate in test_rates:
                    try:
                        if pa.is_format_supported(
                            rate,
                            input_device=i,
                            input_channels=1,
                            input_format=pyaudio.paFloat32
                        ):
                            supported_rates.append(rate)
                    except:
                        pass
                
                print(f"  Supported rates: {supported_rates}")
                print()
        except Exception as e:
            print(f"Error testing device {i}: {e}")
    
    pa.terminate()

if __name__ == "__main__":
    test_audio_devices()