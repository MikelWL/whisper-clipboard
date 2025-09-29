#!/usr/bin/env python3
"""
Whisper Clipboard Configuration Tool

One-time setup script to configure audio devices and hotkeys.
"""

import sys
import os
import yaml
import pyaudio
import time
from pynput import keyboard

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def list_audio_devices():
    """List all available audio input devices."""
    print("🎤 Scanning for audio devices...")

    try:
        p = pyaudio.PyAudio()
        device_list = []

        print("\nAvailable audio input devices:")
        print("-" * 50)

        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)

            # Only show input devices (microphones)
            if device_info['maxInputChannels'] > 0:
                device_list.append({
                    'index': i,
                    'name': device_info['name'],
                    'channels': device_info['maxInputChannels'],
                    'sample_rate': int(device_info['defaultSampleRate'])
                })

                print(f"[{i:2d}] {device_info['name']}")
                print(f"     Channels: {device_info['maxInputChannels']}, "
                      f"Sample Rate: {int(device_info['defaultSampleRate'])} Hz")
                print()

        p.terminate()
        return device_list

    except Exception as e:
        print(f"❌ Error scanning audio devices: {e}")
        return []


def get_user_device_choice(device_list):
    """Get user's choice of audio device."""
    if not device_list:
        print("❌ No audio input devices found!")
        return None

    while True:
        try:
            choice = input(f"Enter the device index you want to use (0-{len(device_list)-1}): ").strip()

            if not choice:
                continue

            device_index = int(choice)

            # Find device with this index
            selected_device = None
            for device in device_list:
                if device['index'] == device_index:
                    selected_device = device
                    break

            if selected_device:
                print(f"✅ Selected: [{device_index}] {selected_device['name']}")
                return selected_device
            else:
                print(f"❌ Invalid device index: {device_index}")

        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n❌ Setup cancelled")
            return None


def capture_hotkey():
    """Capture a hotkey from the user."""
    print("\n🎯 Hotkey Setup")
    print("-" * 20)
    print("Now listening for the next keystroke.")
    print("Press the key you would like to use as your dictation hotkey...")
    print("(Common choices: F12, Right Ctrl, Right Alt)")

    captured_key = None

    def on_press(key):
        nonlocal captured_key
        try:
            # Handle special keys
            if hasattr(key, 'name'):
                captured_key = key.name
            elif hasattr(key, 'char') and key.char:
                captured_key = key.char
            else:
                captured_key = str(key).replace('Key.', '')

            print(f"✅ Captured hotkey: {captured_key}")
            return False  # Stop listening
        except AttributeError:
            # Handle any unexpected key format
            captured_key = str(key).replace('Key.', '')
            print(f"✅ Captured hotkey: {captured_key}")
            return False

    try:
        # Create keyboard listener
        with keyboard.Listener(on_press=on_press) as listener:
            # Wait for key capture (with timeout)
            start_time = time.time()
            while captured_key is None and (time.time() - start_time) < 30:
                time.sleep(0.1)
                if not listener.running:
                    break

        if captured_key:
            return captured_key
        else:
            print("❌ No key captured (timeout)")
            return None

    except Exception as e:
        print(f"❌ Error capturing hotkey: {e}")
        print("💡 Note: No elevated permissions required")
        return None


def load_config(config_path="config.yaml"):
    """Load existing configuration."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Return default config if file doesn't exist
        return {
            'audio': {
                'sample_rate': 16000,
                'chunk_size': 1024,
                'channels': 1,
                'device_index': None
            },
            'whisper': {
                'model_size': "large-v3",
                'device': "auto",
                'language': "en"
            },
            'hotkeys': {
                'record_key': "right_ctrl",
                'cancel_keys': "ctrl+shift+c"
            },
            'recording': {
                'max_duration': 30
            },
            'text': {
                'auto_capitalize': True,
                'auto_punctuate': True,
                'remove_filler_words': False
            },
            'system': {
                'debug_mode': False,
                'log_level': "INFO"
            }
        }
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None


def save_config(config, config_path="config.yaml"):
    """Save configuration to file."""
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        print(f"✅ Configuration saved to {config_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving config: {e}")
        return False


def main():
    """Main configuration flow."""
    print("🔧 Whisper Clipboard Configuration Tool")
    print("=" * 40)
    print("This will set up your audio device and hotkey preferences.\n")

    # Load existing config
    config = load_config()
    if config is None:
        return 1

    # Step 1: Audio Device Setup
    print("Step 1: Audio Device Configuration")
    device_list = list_audio_devices()

    if device_list:
        selected_device = get_user_device_choice(device_list)
        if selected_device:
            config['audio']['device_index'] = selected_device['index']
            print(f"✅ Audio device configured: {selected_device['name']}")
        else:
            print("⚠️  Skipping audio device configuration")
    else:
        print("⚠️  No audio devices found, skipping audio configuration")

    # Step 2: Hotkey Setup
    print("\nStep 2: Hotkey Configuration")
    try:
        hotkey = capture_hotkey()
        if hotkey:
            config['hotkeys']['record_key'] = hotkey
            print(f"✅ Hotkey configured: {hotkey}")
        else:
            print("⚠️  Skipping hotkey configuration (will use default: right_ctrl)")
    except ImportError:
        print("⚠️  pynput library not available, skipping hotkey setup")
        print("💡 You can manually edit config.yaml to set your preferred hotkey")

    # Step 3: Save Configuration
    print("\nStep 3: Saving Configuration")
    if save_config(config):
        print("\n🎉 Configuration complete!")
        print("\nYour settings:")
        print(f"  Audio Device: {config['audio'].get('device_index', 'default')}")
        print(f"  Hotkey: {config['hotkeys']['record_key']}")
        print(f"  Whisper Model: {config['whisper']['model_size']}")
        print(f"  Language: {config['whisper']['language']}")
        print("\nYou can now run 'python main.py' to start Whisper Clipboard!")
    else:
        print("❌ Configuration failed")
        return 1

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n❌ Configuration cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)