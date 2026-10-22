"""
Audio notification manager: programmatic beeps via pygame + TTS via pyttsx3.
"""

from __future__ import annotations

import logging
import queue
import struct
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class SoundManager:
    """
    Manages audio feedback for aircraft detection events.
    Runs TTS in a background thread to avoid blocking the CV loop.
    """

    def __init__(self, sound_enabled: bool = True, tts_enabled: bool = True) -> None:
        self.sound_enabled = sound_enabled
        self.tts_enabled = tts_enabled
        self._mixer_ok = False
        self._tts_engine = None
        self._tts_queue: queue.Queue = queue.Queue()
        self._tts_thread: Optional[threading.Thread] = None

        self._init_pygame()
        if tts_enabled:
            self._init_tts()
            self._start_tts_thread()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_pygame(self) -> None:
        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            self._mixer_ok = True
            logger.info("pygame.mixer initialised.")
        except Exception as exc:
            logger.warning("pygame.mixer init failed: %s. Sound disabled.", exc)
            self._mixer_ok = False

    def _init_tts(self) -> None:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            engine.setProperty("volume", 0.9)
            self._tts_engine = engine
            logger.info("pyttsx3 TTS initialised.")
        except Exception as exc:
            logger.warning("pyttsx3 init failed: %s. TTS disabled.", exc)
            self._tts_engine = None

    def _start_tts_thread(self) -> None:
        if self._tts_engine is None:
            return
        self._tts_thread = threading.Thread(
            target=self._tts_worker, daemon=True, name="TTS-worker"
        )
        self._tts_thread.start()

    def _tts_worker(self) -> None:
        while True:
            text = self._tts_queue.get()
            if text is None:
                break
            try:
                if self._tts_engine and self.tts_enabled:
                    self._tts_engine.say(text)
                    self._tts_engine.runAndWait()
            except Exception as exc:
                logger.debug("TTS error: %s", exc)

    # ------------------------------------------------------------------
    # Beep generation
    # ------------------------------------------------------------------

    def _generate_beep(
        self,
        frequency: int = 880,
        duration_ms: int = 150,
        volume: float = 0.5,
    ) -> None:
        """Generate and play a sine-wave beep programmatically."""
        if not self._mixer_ok or not self.sound_enabled:
            return
        try:
            import math
            import pygame

            sample_rate = 44100
            n_samples = int(sample_rate * duration_ms / 1000)
            buf = bytearray(n_samples * 2)  # 16-bit mono

            for i in range(n_samples):
                t = i / sample_rate
                # Sine wave with simple ADSR envelope
                env = min(i / (sample_rate * 0.01), 1.0)  # 10ms attack
                env *= min((n_samples - i) / (sample_rate * 0.02), 1.0)  # 20ms release
                value = int(32767 * volume * env * math.sin(2 * math.pi * frequency * t))
                struct.pack_into("<h", buf, i * 2, value)

            sound = pygame.mixer.Sound(buffer=bytes(buf))
            sound.play()
        except Exception as exc:
            logger.debug("Beep generation error: %s", exc)

    # ------------------------------------------------------------------
    # Public sound methods
    # ------------------------------------------------------------------

    def play_detection_beep(self) -> None:
        """Short ascending beep on new aircraft detection."""
        if not self.sound_enabled:
            return
        threading.Thread(
            target=self._generate_beep, args=(880, 120, 0.4), daemon=True
        ).start()

    def play_matched_chime(self) -> None:
        """Pleasant two-note chime when FR24 match is found."""
        if not self.sound_enabled:
            return

        def _chime():
            self._generate_beep(1047, 100, 0.5)   # C6
            import time; time.sleep(0.12)
            self._generate_beep(1319, 180, 0.5)   # E6

        threading.Thread(target=_chime, daemon=True).start()

    def play_alert_tone(self, severity: str = "low") -> None:
        """
        Severity-based alert:
          low    - single mid tone
          medium - two rapid beeps
          high   - three urgent beeps
        """
        if not self.sound_enabled:
            return

        configs = {
            "low":    [(660, 200, 0.4)],
            "medium": [(880, 120, 0.6), (880, 120, 0.6)],
            "high":   [(1047, 100, 0.8), (1047, 100, 0.8), (1047, 200, 0.8)],
        }
        sequence = configs.get(severity, configs["low"])

        def _play():
            import time
            for freq, dur, vol in sequence:
                self._generate_beep(freq, dur, vol)
                time.sleep((dur + 80) / 1000.0)

        threading.Thread(target=_play, daemon=True).start()

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    def announce_flight(
        self,
        aircraft_type: str,
        airline: str,
        origin: str,
        dest: str,
    ) -> None:
        """
        Announce a matched flight via TTS.
        e.g. "Singapore Airlines Airbus A350, operating from London to Singapore"
        """
        if not self.tts_enabled or self._tts_engine is None:
            return

        # Clean up IATA codes into readable names where possible
        origin_str = origin if len(origin) > 3 else f"airport {origin}"
        dest_str = dest if len(dest) > 3 else f"airport {dest}"

        msg = f"{airline} {aircraft_type}, operating from {origin_str} to {dest_str}."
        self._tts_queue.put(msg)

    def speak(self, text: str) -> None:
        """Queue arbitrary text for TTS."""
        if self.tts_enabled and self._tts_engine:
            self._tts_queue.put(text)

    # ------------------------------------------------------------------
    # Toggles
    # ------------------------------------------------------------------

    def toggle_sound(self, enabled: bool) -> None:
        self.sound_enabled = enabled
        logger.info("Sound %s", "enabled" if enabled else "disabled")

    def toggle_tts(self, enabled: bool) -> None:
        self.tts_enabled = enabled
        if enabled and self._tts_engine is None:
            self._init_tts()
            self._start_tts_thread()
        logger.info("TTS %s", "enabled" if enabled else "disabled")

    def shutdown(self) -> None:
        """Clean shutdown of TTS worker thread."""
        if self._tts_thread:
            self._tts_queue.put(None)
            self._tts_thread.join(timeout=3.0)
        try:
            import pygame
            pygame.mixer.quit()
        except Exception:
            pass
