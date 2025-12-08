#!/usr/bin/env python3
"""
Voice-to-Text Tool: Records audio while hotkey is pressed, transcribes via OpenAI, and pastes the text.
Now with a Menu Bar interface!

Usage:
    Run this script to start the app in the menu bar.
    Press and hold Right Option to record audio (max 30 seconds).
    Release the hotkey to transcribe and paste the text.
"""

from __future__ import annotations

import os
import subprocess
import threading
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Callable, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    """Application configuration constants."""
    sample_rate: int = 16000  # Hz
    max_recording_seconds: int = 30
    min_recording_seconds: float = 0.5  # Skip accidental taps
    transcription_model: str = "gpt-4o-transcribe"
    api_max_retries: int = 2
    cache_dir: Path = Path("~/Library/Caches/voice-paste").expanduser()

    # System sounds
    sound_start: str = "/System/Library/Sounds/Tink.aiff"
    sound_stop: str = "/System/Library/Sounds/Pop.aiff"
    sound_error: str = "/System/Library/Sounds/Basso.aiff"


# Global config instance
config = Config()


# Try to import required dependencies
try:
    import pyperclip
    from pynput import keyboard
    from dotenv import load_dotenv
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    import numpy.typing as npt
    import rumps
    from openai import OpenAI
except ImportError as e:
    logger.error(f"Failed to import required dependency: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install openai pyperclip pynput python-dotenv sounddevice soundfile numpy rumps")
    sys.exit(1)

# Load environment variables from .env file
load_dotenv()


def validate_api_key() -> OpenAI | None:
    """Validate and initialize OpenAI client with API key."""
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        logger.error("OpenAI API key not found. Please set OPENAI_API_KEY in .env file.")
        return None

    if not api_key.startswith('sk-'):
        logger.error("Invalid API key format. OpenAI keys should start with 'sk-'.")
        return None

    if len(api_key) < 20:
        logger.error("API key appears too short. Please check your .env file.")
        return None

    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        logger.error(f"Error initializing OpenAI client: {e}")
        return None


# Initialize OpenAI client
client = validate_api_key()


# Import AppKit for Overlay
try:
    from AppKit import (
        NSWindow, NSFloatingWindowLevel, NSWindowStyleMaskBorderless,
        NSBackingStoreBuffered, NSColor, NSTextField, NSFont, NSScreen
    )
    from PyObjCTools import AppHelper
    APPKIT_AVAILABLE = True
except ImportError:
    logger.warning("AppKit not found, overlay will be disabled")
    APPKIT_AVAILABLE = False


class OverlayController:
    """Manages a floating overlay window with timer and audio level meter."""

    # Audio meter characters
    METER_FILLED = "█"
    METER_EMPTY = "░"
    METER_LENGTH = 20

    def __init__(self) -> None:
        self.window: Any | None = None
        self.title_label: Any | None = None
        self.meter_label: Any | None = None
        self.start_time: float = 0.0
        self.current_level: float = 0.0
        self.update_timer: threading.Timer | None = None
        self.is_visible: bool = False

        if not APPKIT_AVAILABLE:
            return
        try:
            self._create_window()
        except Exception as e:
            logger.error(f"Failed to create overlay: {e}")

    def _create_window(self) -> None:
        """Create the overlay window with timer and level meter."""
        screen_rect = NSScreen.mainScreen().frame()
        width = 280
        height = 90
        x = (screen_rect.size.width - width) / 2
        y = (screen_rect.size.height - height) / 2

        rect = ((x, y), (width, height))

        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            rect,
            NSWindowStyleMaskBorderless,
            NSBackingStoreBuffered,
            False
        )

        self.window.setLevel_(NSFloatingWindowLevel)
        self.window.setBackgroundColor_(NSColor.blackColor().colorWithAlphaComponent_(0.85))
        self.window.setOpaque_(False)
        self.window.setHasShadow_(True)
        self.window.setReleasedWhenClosed_(False)

        # Rounded corners
        self.window.contentView().setWantsLayer_(True)
        self.window.contentView().layer().setCornerRadius_(16.0)

        # Title label (Recording + timer)
        self.title_label = NSTextField.alloc().initWithFrame_(((0, 40), (width, 40)))
        self.title_label.setStringValue_("🎙️ Recording... 0:00")
        self.title_label.setTextColor_(NSColor.whiteColor())
        self.title_label.setFont_(NSFont.systemFontOfSize_(22))
        self.title_label.setBezeled_(False)
        self.title_label.setDrawsBackground_(False)
        self.title_label.setEditable_(False)
        self.title_label.setSelectable_(False)
        self.title_label.setAlignment_(2)  # Center

        # Audio level meter label
        self.meter_label = NSTextField.alloc().initWithFrame_(((0, 12), (width, 28)))
        self.meter_label.setStringValue_(self._build_meter(0.0))
        self.meter_label.setTextColor_(NSColor.colorWithRed_green_blue_alpha_(0.4, 0.8, 1.0, 1.0))
        self.meter_label.setFont_(NSFont.monospacedSystemFontOfSize_weight_(14, 0.0))
        self.meter_label.setBezeled_(False)
        self.meter_label.setDrawsBackground_(False)
        self.meter_label.setEditable_(False)
        self.meter_label.setSelectable_(False)
        self.meter_label.setAlignment_(2)  # Center

        self.window.contentView().addSubview_(self.title_label)
        self.window.contentView().addSubview_(self.meter_label)

    def _build_meter(self, level: float) -> str:
        """Build the audio level meter string."""
        # Clamp level between 0 and 1
        level = max(0.0, min(1.0, level))
        filled = int(level * self.METER_LENGTH)
        empty = self.METER_LENGTH - filled
        return self.METER_FILLED * filled + self.METER_EMPTY * empty

    def _format_time(self, seconds: float) -> str:
        """Format seconds as M:SS."""
        mins = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{mins}:{secs:02d}"

    def _update_display(self) -> None:
        """Update the timer and level meter display."""
        if not self.is_visible or not APPKIT_AVAILABLE:
            return

        elapsed = time.time() - self.start_time
        time_str = self._format_time(elapsed)
        # Capture level once to avoid race between meter and color
        level = self.current_level
        meter_str = self._build_meter(level)

        def do_update() -> None:
            if self.title_label:
                self.title_label.setStringValue_(f"🎙️ Recording... {time_str}")
            if self.meter_label:
                self.meter_label.setStringValue_(meter_str)
                # Color changes based on level (blue -> green -> yellow -> red)
                if level < 0.3:
                    color = NSColor.colorWithRed_green_blue_alpha_(0.4, 0.8, 1.0, 1.0)  # Blue
                elif level < 0.6:
                    color = NSColor.colorWithRed_green_blue_alpha_(0.4, 1.0, 0.6, 1.0)  # Green
                elif level < 0.85:
                    color = NSColor.colorWithRed_green_blue_alpha_(1.0, 0.9, 0.3, 1.0)  # Yellow
                else:
                    color = NSColor.colorWithRed_green_blue_alpha_(1.0, 0.4, 0.4, 1.0)  # Red
                self.meter_label.setTextColor_(color)

        try:
            AppHelper.callAfter(do_update)
        except Exception as e:
            logger.debug(f"Error updating display: {e}")

        # Schedule next update
        if self.is_visible:
            self.update_timer = threading.Timer(0.05, self._update_display)
            self.update_timer.daemon = True
            self.update_timer.start()

    def update_level(self, level: float) -> None:
        """Update the current audio level (0.0 to 1.0)."""
        self.current_level = level

    def _do_show(self) -> None:
        if self.window:
            self.window.makeKeyAndOrderFront_(None)

    def _do_hide(self) -> None:
        if self.window:
            self.window.orderOut_(None)

    def show(self) -> None:
        """Show the overlay window and start updates."""
        if self.window and APPKIT_AVAILABLE:
            try:
                self.start_time = time.time()
                self.current_level = 0.0
                self.is_visible = True
                AppHelper.callAfter(self._do_show)
                # Start update loop
                self._update_display()
            except Exception as e:
                logger.error(f"Error showing overlay: {e}")

    def hide(self) -> None:
        """Hide the overlay window and stop updates."""
        self.is_visible = False
        if self.update_timer:
            self.update_timer.cancel()
            self.update_timer = None

        if self.window and APPKIT_AVAILABLE:
            try:
                AppHelper.callAfter(self._do_hide)
            except Exception as e:
                logger.error(f"Error hiding overlay: {e}")


def play_sound(sound_path: str) -> None:
    """Play a system sound asynchronously."""
    try:
        if os.path.exists(sound_path):
            subprocess.Popen(
                ['afplay', sound_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
    except Exception as e:
        logger.error(f"Error playing sound: {e}")


def ensure_cache_dir() -> Path:
    """Ensure cache directory exists and return its path."""
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    return config.cache_dir


def paste_text() -> None:
    """Programmatically trigger Cmd+V to paste the clipboard contents."""
    try:
        cmd = [
            'osascript', '-e',
            'tell application "System Events" to keystroke "v" using command down'
        ]
        subprocess.run(cmd, check=True)
        logger.info("Text pasted successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error pasting text: {e}")


class VoiceRecorder:
    """Thread-safe voice recorder with persistent audio stream."""

    def __init__(
        self,
        app_callback: Callable[[str], None] | None = None,
        overlay: OverlayController | None = None
    ) -> None:
        self.recording: bool = False
        self.processing: bool = False
        self.audio_data: list[npt.NDArray[np.float32]] = []
        self.lock: threading.Lock = threading.Lock()
        self.stream: sd.InputStream | None = None
        self.app_callback: Callable[[str], None] | None = app_callback
        self.overlay: OverlayController | None = overlay
        self.running: bool = True

        # Start the persistent audio stream
        self._start_stream()

    def _start_stream(self) -> None:
        """Start the persistent audio stream."""
        try:
            self.stream = sd.InputStream(
                samplerate=config.sample_rate,
                channels=1,
                callback=self._audio_callback
            )
            self.stream.start()
            logger.info("Audio stream started and ready")
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            self.update_status("error")

    def _audio_callback(
        self,
        indata: npt.NDArray[np.float32],
        _frames: int,
        _time_info: Any,
        status: sd.CallbackFlags | None
    ) -> None:
        """Callback for audio stream to collect recorded data."""
        if status:
            logger.warning(f"Audio status: {status}")

        # Only collect data if we are actively recording
        if self.recording:
            with self.lock:
                # Enforce max recording limit
                current_samples = sum(len(chunk) for chunk in self.audio_data)
                if current_samples >= config.max_recording_seconds * config.sample_rate:
                    logger.info("Max recording time reached, stopping...")
                    threading.Thread(target=self.stop_recording).start()
                    return
                self.audio_data.append(indata.copy())

            # Calculate and update audio level for the overlay
            if self.overlay:
                # RMS (Root Mean Square) for audio level
                rms = np.sqrt(np.mean(indata ** 2))
                # Scale RMS to 0-1 range (typical speech is 0.01-0.1 RMS)
                # Using log scale for better visual response
                if rms > 0:
                    db = 20 * np.log10(rms + 1e-10)
                    # Map -60dB to 0dB range to 0-1
                    level = max(0.0, min(1.0, (db + 60) / 60))
                else:
                    level = 0.0
                self.overlay.update_level(level)

    def update_status(self, status: str) -> None:
        """Update app status via callback."""
        if self.app_callback:
            self.app_callback(status)

    def is_recording(self) -> bool:
        """Check if currently recording."""
        with self.lock:
            return self.recording

    def is_busy(self) -> bool:
        """Check if recording or processing."""
        with self.lock:
            return self.recording or self.processing

    def start_recording(self) -> bool:
        """Start recording immediately."""
        with self.lock:
            if self.recording or self.processing:
                return False

            self.recording = True
            self.processing = False
            self.audio_data = []
            logger.info("Starting recording...")

        self.update_status("recording")
        if self.overlay:
            self.overlay.show()
        play_sound(config.sound_start)
        return True

    def stop_recording(self) -> None:
        """Stop recording and trigger processing."""
        with self.lock:
            if not self.recording:
                return
            self.recording = False
            logger.info("Stop recording requested")

        if self.overlay:
            self.overlay.hide()
        play_sound(config.sound_stop)
        self.update_status("processing")

        # Process in a separate thread to not block the listener
        threading.Thread(target=self._process_audio).start()

    def close(self) -> None:
        """Clean up resources."""
        self.running = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error closing stream: {e}")

    def _process_audio(self) -> None:
        """Save and transcribe recorded audio."""
        with self.lock:
            if not self.audio_data:
                logger.warning("No audio recorded")
                self.processing = False
                self.update_status("idle")
                return
            audio_data_copy = self.audio_data.copy()
            self.audio_data = []
            self.processing = True

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ensure_cache_dir()
        temp_file = config.cache_dir / f"voice_recording_{timestamp}.wav"
        error_occurred = False

        try:
            audio_concat = np.concatenate(audio_data_copy, axis=0)
            duration = audio_concat.shape[0] / config.sample_rate
            logger.info(f"Processing {duration:.2f}s of audio")

            # Skip very short recordings (accidental taps)
            if duration < config.min_recording_seconds:
                logger.info(f"Recording too short ({duration:.2f}s), skipping")
                return

            sf.write(temp_file, audio_concat, config.sample_rate)

            self._transcribe_audio(temp_file)

        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            play_sound(config.sound_error)
            self.update_status("error")
            error_occurred = True
        finally:
            if temp_file.exists():
                try:
                    os.remove(temp_file)
                except OSError as e:
                    logger.debug(f"Failed to remove temp file: {e}")

            with self.lock:
                self.processing = False
            # Only reset to idle if no error occurred
            if not error_occurred:
                self.update_status("idle")

    def _transcribe_audio(self, audio_file: Path) -> None:
        """Transcribe audio file with OpenAI, with retry logic."""
        if not client:
            logger.error("OpenAI client not initialized")
            play_sound(config.sound_error)
            return

        last_error: Exception | None = None
        for attempt in range(config.api_max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt}/{config.api_max_retries}...")
                logger.info("Transcribing...")
                with open(audio_file, "rb") as file:
                    response = client.audio.transcriptions.create(
                        model=config.transcription_model,
                        file=file
                    )
                    text = response.text

                if text:
                    logger.info(f"Transcribed: {text[:50]}...")
                    pyperclip.copy(text)
                    time.sleep(0.05)  # Small delay to ensure clipboard is ready
                    paste_text()
                else:
                    logger.warning("Empty transcription")
                return  # Success, exit retry loop

            except Exception as e:
                last_error = e
                logger.warning(f"Transcription attempt failed: {str(e)}")
                if attempt < config.api_max_retries:
                    continue  # Try again

        # All retries exhausted
        logger.error(
            f"Transcription failed after {config.api_max_retries + 1} attempts: {str(last_error)}"
        )
        play_sound(config.sound_error)


class HotkeyListener:
    """Manages keyboard hotkey detection with debounce."""

    def __init__(self, recorder: VoiceRecorder) -> None:
        self.recorder: VoiceRecorder = recorder
        self.listener: keyboard.Listener | None = None
        self.key_held: bool = False  # Prevent key repeat triggering multiple starts

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        """Handle key press events."""
        if key == keyboard.Key.alt_r:
            if self.key_held:
                return  # Ignore key repeat
            self.key_held = True
            logger.info("Right Option pressed")
            self.recorder.start_recording()

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        """Handle key release events."""
        if key == keyboard.Key.alt_r:
            self.key_held = False
            logger.info("Right Option released")
            if self.recorder.is_recording():
                self.recorder.stop_recording()

    def start(self) -> None:
        """Start the keyboard listener."""
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.listener.start()

    def stop(self) -> None:
        """Stop the keyboard listener."""
        if self.listener:
            self.listener.stop()


class VoicePasteApp(rumps.App):
    """Main menu bar application."""

    def __init__(self) -> None:
        super(VoicePasteApp, self).__init__("🎙️", quit_button=None)
        self.menu = ["Reset Audio Engine", "Quit"]

        # Initialize Overlay
        self.overlay: OverlayController = OverlayController()

        # Pass overlay to recorder
        self.recorder: VoiceRecorder = VoiceRecorder(
            app_callback=self.update_icon,
            overlay=self.overlay
        )
        self.listener: HotkeyListener = HotkeyListener(self.recorder)
        self.listener.start()

        # Check API Key
        if not client:
            rumps.alert(
                "Missing OpenAI API Key",
                "Please set OPENAI_API_KEY in .env file and restart."
            )

    def update_icon(self, status: str) -> None:
        """Update menu bar icon based on status."""
        if status == "recording":
            self.title = "🔴"
        elif status == "processing":
            self.title = "⏳"
        elif status == "error":
            self.title = "⚠️"
            # Reset to idle after a delay
            rumps.Timer(lambda _: self.reset_icon(), 2).start()
        else:
            self.title = "🎙️"

    def reset_icon(self) -> None:
        """Reset icon to default state."""
        self.title = "🎙️"

    @rumps.clicked("Reset Audio Engine")
    def reset_audio(self, _: Any) -> None:
        """Reset the audio engine to clear any stuck state."""
        logger.info("Resetting audio engine...")
        # Hide overlay if visible
        self.overlay.hide()
        with self.recorder.lock:
            self.recorder.recording = False
            self.recorder.processing = False
            self.recorder.audio_data = []
        self.title = "🎙️"
        rumps.notification("Voice Paste", "Audio Engine Reset", "Ready to record.")

    @rumps.clicked("Quit")
    def quit_app(self, _: Any) -> None:
        """Clean up and quit the application."""
        self.listener.stop()
        self.recorder.close()
        self.overlay.hide()
        rumps.quit_application()


if __name__ == "__main__":
    print("\n🚀 Voice-to-Text App is starting...")
    print("👀 Look for the 🎙️ icon in your macOS Menu Bar (top right).")
    print("⌨️  Hotkey: Right Option (Press and Hold)")
    print("---------------------------------------------------\n")
    VoicePasteApp().run()
