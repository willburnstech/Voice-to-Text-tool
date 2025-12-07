# 🎙️ Voice-Paste — speak, release, and paste 🪄

Python utility for **macOS** that lets you hold **Right Option (⌥)**, talk (≤ 30 s), and have the transcript pasted wherever your text cursor is.
Powered by OpenAI's **`gpt-4o-transcribe`** speech-to-text model.

> **Why?** It’s much faster than typing quick notes, chat replies, TODOs, or email paragraphs.

---

## ✨ Features

* Global hot-key **Right Option** → record → auto-paste  
* Uses state-of-the-art OpenAI Audio API (`gpt-4o-transcribe`; fallback toggle for `gpt-4o-mini-transcribe` in code)  
* Multilingual, noise-robust, punctuation aware  
* Audio kept only in `~/Library/Caches/voice-paste/` and deleted after transcription  
* Console log with timestamps for debugging  
* Fails gracefully on network/API errors (daemon keeps running)

---

## Requirements

| Software | Notes |
|----------|-------|
| macOS 13 + | Accessibility & Microphone permissions |
| Python ≥ 3.9 | Homebrew: `brew install python` |
| FFmpeg (optional) | Only if you later swap in FFmpeg recording |
| OpenAI account | `OPENAI_API_KEY` in a `.env` file |
| Pip packages | listed below |

### Dependencies

```bash
pip install openai pyperclip pynput python-dotenv sounddevice soundfile numpy


# 1 · Clone & enter
git clone https://github.com/willburns05/Voice-to-Text-tool
cd Whisper

# 2 · Create a virtual env (optional but recommended)
python -m venv .venv && source .venv/bin/activate

# 3 · Install deps
pip install -r requirements.txt         # or the one-liner above

# 4 · Add your API key
echo "OPENAI_API_KEY=sk-..." > .env

