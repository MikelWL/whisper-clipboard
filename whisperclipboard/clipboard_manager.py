"""
Clipboard manager for Whisper Clipboard.
Handles copying transcribed text to clipboard for easy pasting.
"""

import time
import logging
import re
import pyperclip
from typing import Dict, Any


class ClipboardManager:
    """Clipboard manager for transcribed text."""

    def __init__(self, debug_mode: bool = False):
        """Initialize ClipboardManager."""
        self.debug_mode = debug_mode
        self.logger = logging.getLogger(__name__)

        # Statistics
        self.chars_copied = 0
        self.words_copied = 0
        self.copy_times = []

        if self.debug_mode:
            self.logger.info("ClipboardManager initialized")

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before copying to clipboard.

        Args:
            text: Raw text to preprocess

        Returns:
            str: Preprocessed text ready for clipboard
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

    def copy_to_clipboard(self, text: str, preprocess: bool = True) -> bool:
        """
        Copy text to clipboard.

        Args:
            text: Text to copy
            preprocess: Whether to preprocess the text

        Returns:
            bool: True if copy successful
        """
        if not text:
            self.logger.warning("Empty text provided for clipboard copy")
            return False

        try:
            start_time = time.time()

            # Preprocess text if requested
            if preprocess:
                text = self._preprocess_text(text)

            if not text:
                self.logger.warning("Text became empty after preprocessing")
                return False

            if self.debug_mode:
                self.logger.info(f"Copying text to clipboard: '{text}' ({len(text)} chars)")

            # Copy to clipboard
            pyperclip.copy(text)

            # Update statistics
            copy_time = time.time() - start_time
            self.copy_times.append(copy_time)
            if len(self.copy_times) > 100:
                self.copy_times.pop(0)
            self.chars_copied += len(text)
            self.words_copied += len(text.split())

            word_count = len(text.split())
            preview = text[:50] + "..." if len(text) > 50 else text

            # Terminal output for user feedback
            print(f"📋 Copied to clipboard: {word_count} words")
            print(f"   \"{preview}\"")

            if self.debug_mode:
                self.logger.info(f"✅ Copied {len(text)} characters to clipboard in {copy_time:.2f}s")
            return True

        except Exception as e:
            print(f"❌ Failed to copy to clipboard: {e}")
            self.logger.error(f"❌ Failed to copy text to clipboard: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get clipboard copy statistics.

        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        if not self.copy_times:
            return {
                "chars_copied": self.chars_copied,
                "words_copied": self.words_copied,
                "copies": 0
            }

        return {
            "chars_copied": self.chars_copied,
            "words_copied": self.words_copied,
            "copies": len(self.copy_times),
            "total_time": sum(self.copy_times),
            "avg_copy_time": sum(self.copy_times) / len(self.copy_times),
            "min_copy_time": min(self.copy_times),
            "max_copy_time": max(self.copy_times),
        }


# Test function for standalone testing
if __name__ == "__main__":
    import sys
    import signal

    logging.basicConfig(level=logging.DEBUG)

    clipboard = ClipboardManager()

    def signal_handler(sig, frame):
        print("\nTest interrupted")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print("=== ClipboardManager Test ===")
    print("Testing clipboard copying functionality...")

    # Test basic text copying
    test_texts = [
        "Hello, this is a test of Whisper Clipboard copying!",
        "Testing special characters: ()[]{}\"'",
        "Numbers and symbols: 123 @#$%^&*",
        "Multiple words with punctuation.",
        "Text with transcription artifacts: word ( parenthesis ) and word .",
    ]

    for i, text in enumerate(test_texts):
        print(f"\nTest {i+1}: '{text}'")
        success = clipboard.copy_to_clipboard(text)
        if success:
            print("✅ Copy successful - check your clipboard!")
        else:
            print("❌ Copy failed")

        if i < len(test_texts) - 1:
            input("Press Enter to continue to next test...")

    # Show statistics
    print("\nStatistics:")
    stats = clipboard.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nTest complete!")