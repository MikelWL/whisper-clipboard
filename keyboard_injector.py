"""
Keyboard injection module for WhisperDictate.
Handles system-level keystroke simulation to inject transcribed text.
"""

import time
import logging
from typing import Optional, Dict, Any
import keyboard
import re
import pyperclip


class KeyboardInjector:
    """System-level keyboard injection for transcribed text."""
    
    def __init__(self, 
                 typing_delay: float = 0.01,
                 injection_mode: str = "type",
                 paste_key: str = "ctrl+v",
                 restore_clipboard: bool = True,
                 fail_fast: bool = False):
        """
        Initialize KeyboardInjector.
        
        Args:
            typing_delay: Delay between keystrokes in seconds
            injection_mode: 'type' or 'paste'
            paste_key: Key combo to trigger paste ('ctrl+v', 'ctrl+shift+v', 'shift+insert')
            restore_clipboard: Whether to restore previous clipboard contents after paste
            fail_fast: If True, raise on paste failure rather than falling back
        """
        self.logger = logging.getLogger(__name__)
        self.typing_delay = max(0.001, typing_delay)  # Minimum 1ms delay
        self.injection_mode = injection_mode.lower()
        self.paste_key = paste_key
        self.restore_clipboard = restore_clipboard
        self.fail_fast = fail_fast
        
        # Initialize keyboard library
        try:
            self.logger.info(f"KeyboardInjector initialized with typing_delay={typing_delay}s, mode={self.injection_mode}, paste_key={self.paste_key}")
        except Exception as e:
            self.logger.error(f"Failed to initialize keyboard controller: {e}")
            raise
        
        # Special character mappings for keyboard library
        self.special_chars = {
            '\n': 'enter',
            '\t': 'tab',
        }
        
        # Statistics
        self.chars_typed = 0
        self.words_typed = 0
        self.injection_times = []

    def _inject_via_paste(self, text: str) -> bool:
        """
        Paste text via clipboard and paste key combo.
        Fail-fast if configured.
        """
        start_time = time.time()
        try:
            # Save current clipboard
            previous_clipboard: Optional[str] = None
            if self.restore_clipboard:
                try:
                    previous_clipboard = pyperclip.paste()
                except Exception:
                    # If reading fails, continue; restoration will be skipped
                    previous_clipboard = None
            
            # Set clipboard to text
            pyperclip.copy(text)
            # Small delay to ensure clipboard owner updates
            time.sleep(0.02)
            
            # Trigger paste
            keyboard.press_and_release(self.paste_key)
            
            # Optional: restore clipboard
            if self.restore_clipboard and previous_clipboard is not None:
                # Give target app a moment to consume the clipboard
                time.sleep(0.05)
                try:
                    pyperclip.copy(previous_clipboard)
                except Exception as e:
                    # Restoration failure is non-fatal
                    self.logger.debug(f"Clipboard restore failed: {e}")
            
            # Stats
            injection_time = time.time() - start_time
            self.injection_times.append(injection_time)
            if len(self.injection_times) > 100:
                self.injection_times.pop(0)
            self.chars_typed += len(text)
            self.words_typed += len(text.split())
            self.logger.info(f"Pasted {len(text)} characters in {injection_time:.2f}s via {self.paste_key}")
            return True
        except Exception as e:
            self.logger.error(f"Paste injection failed: {e}")
            if self.fail_fast:
                raise
            return False
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before injection.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            str: Preprocessed text ready for injection
        """
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Handle common transcription artifacts
        replacements = {
            # Fix common Whisper transcription issues
            ' .': '.',
            ' ,': ',',
            ' !': '!',
            ' ?': '?',
            ' ;': ';',
            ' :': ':',
            '( ': '(',
            ' )': ')',
            '[ ': '[',
            ' ]': ']',
            '{ ': '{',
            ' }': '}',
            '" ': '"',
            ' "': '"',
            "' ": "'",
            " '": "'",
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def inject_text(self, text: str, preprocess: bool = True) -> bool:
        """
        Inject text as keystrokes into the active window.
        
        Args:
            text: Text to inject
            preprocess: Whether to preprocess the text
            
        Returns:
            bool: True if injection successful
        """
        if not text:
            self.logger.warning("Empty text provided for injection")
            return False
        
        try:
            start_time = time.time()
            
            # Preprocess text if requested
            if preprocess:
                text = self._preprocess_text(text)
            
            if not text:
                self.logger.warning("Text became empty after preprocessing")
                return False
            
            self.logger.info(f"Injecting text: '{text}' ({len(text)} chars)")
            
            # Paste mode: clipboard + paste key
            if self.injection_mode == "paste":
                return self._inject_via_paste(text)
            
            # Type mode: use keyboard library for text injection with delay simulation
            if self.typing_delay > 0:
                # Character by character for typing delay
                char_count = 0
                for char in text:
                    if char in self.special_chars:
                        # Handle special keys
                        keyboard.press_and_release(self.special_chars[char])
                    else:
                        # Regular character
                        keyboard.write(char)
                    
                    char_count += 1
                    time.sleep(self.typing_delay)
            else:
                # Fast injection without delay
                keyboard.write(text)
                char_count = len(text)
            
            # Update statistics
            injection_time = time.time() - start_time
            self.chars_typed += char_count
            self.words_typed += len(text.split())
            self.injection_times.append(injection_time)
            
            # Keep only last 100 injection times
            if len(self.injection_times) > 100:
                self.injection_times.pop(0)
            
            chars_per_second = char_count / injection_time if injection_time > 0 else 0
            self.logger.info(f"Injected {char_count} characters in {injection_time:.2f}s "
                           f"({chars_per_second:.1f} chars/s)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to inject text: {e}")
            return False
    
    def inject_key(self, key: str) -> bool:
        """
        Inject a special key.
        
        Args:
            key: Special key to inject (e.g., 'enter', 'tab', 'backspace')
            
        Returns:
            bool: True if injection successful
        """
        try:
            self.logger.debug(f"Injecting key: {key}")
            keyboard.press_and_release(key)
            return True
        except Exception as e:
            self.logger.error(f"Failed to inject key {key}: {e}")
            return False
    
    def inject_key_combination(self, key_combo: str) -> bool:
        """
        Inject a key combination (e.g., 'ctrl+c').
        
        Args:
            key_combo: Key combination string (e.g., 'ctrl+c', 'ctrl+shift+v')
            
        Returns:
            bool: True if injection successful
        """
        try:
            self.logger.debug(f"Injecting key combination: {key_combo}")
            keyboard.press_and_release(key_combo)
            return True
        except Exception as e:
            self.logger.error(f"Failed to inject key combination {key_combo}: {e}")
            return False
    
    def inject_backspace(self, count: int = 1) -> bool:
        """
        Inject backspace keystrokes.
        
        Args:
            count: Number of backspaces to inject
            
        Returns:
            bool: True if injection successful
        """
        try:
            self.logger.debug(f"Injecting {count} backspace(s)")
            for _ in range(count):
                keyboard.press_and_release('backspace')
                if self.typing_delay > 0:
                    time.sleep(self.typing_delay)
            return True
        except Exception as e:
            self.logger.error(f"Failed to inject backspace: {e}")
            return False
    
    def inject_newline(self, count: int = 1) -> bool:
        """
        Inject newline characters.
        
        Args:
            count: Number of newlines to inject
            
        Returns:
            bool: True if injection successful
        """
        try:
            self.logger.debug(f"Injecting {count} newline(s)")
            for _ in range(count):
                self.keyboard.press(Key.enter)
                self.keyboard.release(Key.enter)
                if self.typing_delay > 0:
                    time.sleep(self.typing_delay)
            return True
        except Exception as e:
            self.logger.error(f"Failed to inject newline: {e}")
            return False
    
    def inject_space(self, count: int = 1) -> bool:
        """
        Inject space characters.
        
        Args:
            count: Number of spaces to inject
            
        Returns:
            bool: True if injection successful
        """
        try:
            self.logger.debug(f"Injecting {count} space(s)")
            for _ in range(count):
                self.keyboard.type(' ')
                if self.typing_delay > 0:
                    time.sleep(self.typing_delay)
            return True
        except Exception as e:
            self.logger.error(f"Failed to inject space: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get keyboard injection statistics.
        
        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        if not self.injection_times:
            return {
                "chars_typed": self.chars_typed,
                "words_typed": self.words_typed,
                "injections": 0,
                "typing_delay": self.typing_delay
            }
        
        return {
            "chars_typed": self.chars_typed,
            "words_typed": self.words_typed,
            "injections": len(self.injection_times),
            "total_time": sum(self.injection_times),
            "avg_injection_time": sum(self.injection_times) / len(self.injection_times),
            "min_injection_time": min(self.injection_times),
            "max_injection_time": max(self.injection_times),
            "typing_delay": self.typing_delay,
            "avg_chars_per_second": self.chars_typed / sum(self.injection_times) if self.injection_times else 0
        }
    
    def set_typing_delay(self, delay: float):
        """
        Set the delay between keystrokes.
        
        Args:
            delay: New delay in seconds (minimum 0.001)
        """
        old_delay = self.typing_delay
        self.typing_delay = max(0.001, delay)
        self.logger.info(f"Typing delay changed from {old_delay}s to {self.typing_delay}s")
    
    def test_injection(self) -> bool:
        """
        Test keyboard injection with a simple message.
        
        Returns:
            bool: True if test successful
        """
        test_message = "WhisperDictate keyboard injection test - this should appear as typed text"
        self.logger.info("Running keyboard injection test...")
        self.logger.info("Switch to a text editor and wait 3 seconds...")
        
        # Give user time to switch windows
        time.sleep(3)
        
        return self.inject_text(test_message)


# Test function for standalone testing
if __name__ == "__main__":
    import sys
    import signal
    
    logging.basicConfig(level=logging.DEBUG)
    
    injector = KeyboardInjector(typing_delay=0.02)  # Slower for demo
    
    def signal_handler(sig, frame):
        print("\nTest interrupted")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=== KeyboardInjector Test ===")
    print("This will test keyboard injection in 5 seconds...")
    print("Switch to a text editor (like gedit, nano, etc.) now!")
    
    for i in range(5, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)
    
    # Test basic text injection
    test_texts = [
        "Hello, this is a test of WhisperDictate keyboard injection!",
        "Testing special characters: ()[]{}\"'",
        "Numbers and symbols: 123 @#$%^&*",
        "Multiple words with punctuation.",
    ]
    
    for i, text in enumerate(test_texts):
        print(f"Injecting test text {i+1}: '{text}'")
        success = injector.inject_text(text)
        if success:
            injector.inject_newline()  # Add newline after each test
        time.sleep(2)  # Pause between tests
    
    # Show statistics
    print("\nStatistics:")
    stats = injector.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nTest complete!")