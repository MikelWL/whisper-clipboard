#!/usr/bin/env python3
"""
Test text injection in isolation to debug the issue
"""
import time
import logging
from keyboard_injector import KeyboardInjector

logging.basicConfig(level=logging.DEBUG)

print("=== Text Injection Test ===")
print("Switch to a text editor (like gedit, VS Code, etc.) in 5 seconds...")

for i in range(5, 0, -1):
    print(f"Starting in {i}...")
    time.sleep(1)

print("Testing text injection now...")

injector = KeyboardInjector(typing_delay=0.02)
test_text = "Hello from WhisperDictate! This is a test."

success = injector.inject_text(test_text)
print(f"Injection result: {success}")

if success:
    print("✅ Text injection worked!")
else:
    print("❌ Text injection failed!")

# Show statistics
stats = injector.get_statistics()
print("Statistics:", stats)