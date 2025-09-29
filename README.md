# Whisper Clipboard

A voice-to-clipboard transcription tool I developed for my own Linux workflow. I created it to solve my specific needs for quick voice transcription and as a learning exercise with OpenAI's Whisper model. Hold a hotkey, speak, release, and your transcribed text is ready to paste anywhere. Sharing it here in case others find it useful or want to build something similar.

## Features

- **Hotkey-activated recording** - Hold a configurable key to record voice
- **GPU-accelerated transcription** - Uses CUDA when available, falls back to CPU
- **Direct clipboard integration** - Transcribed text automatically copied for pasting
- **Configurable audio settings** - Support for different microphones and audio devices
- **No sudo required** - Runs in userspace

## Getting Started

1. **Install dependencies:**
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install xclip

   # Install Python packages
   pip install -r requirements.txt
   ```

2. **Configure your setup:**
   ```bash
   python configure.py
   ```

3. **Start using:**
   ```bash
   python main.py
   ```

4. **Usage:**
   - Hold your hotkey (default: Right Ctrl)
   - Speak your message
   - Release the key
   - Transcribed text is copied to clipboard automatically

## Requirements

- **Python 3.8+** and **Linux/Ubuntu** (tested on Ubuntu 22.04)
- **Audio input device** (microphone)
- **CUDA GPU** (optional, for faster transcription)

Key dependencies: `openai-whisper`, `pyaudio`, `pyperclip`, `pynput`, `torch` (see `requirements.txt`)

## License

MIT License - feel free to use, modify, and distribute as you see fit.