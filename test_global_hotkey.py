#!/usr/bin/env python3
"""
Quick test to see if the 'keyboard' library can capture global hotkeys
"""
import keyboard
import time

print("Testing global hotkey capture with 'keyboard' library...")
print("Switch to ANY window (like a browser) and press F12")
print("Press Ctrl+C to quit")
print("Waiting for F12 key press...")

def on_f12():
    print("F12 pressed! This worked globally!")

# Try to register a global hotkey
try:
    keyboard.add_hotkey('f12', on_f12)
    
    # Keep the script running
    while True:
        time.sleep(0.1)
        
except KeyboardInterrupt:
    print("\nTest finished")
except Exception as e:
    print(f"Error: {e}")
    print("The 'keyboard' library might need different permissions")