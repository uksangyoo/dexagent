"""Lightweight keyboard edge detector (pynput), no rio middleware required."""
from __future__ import annotations

import threading


class KeyPoller:
    """Track currently held alphanumeric keys; ``pressed(prev)`` returns key-downs."""

    def __init__(self):
        self._held: set[str] = set()
        self._lock = threading.Lock()
        self._listener = None
        try:
            from pynput import keyboard
        except ImportError:
            self._keyboard = None
            return
        self._keyboard = keyboard

        def on_press(key):
            ch = getattr(key, "char", None)
            if ch and len(ch) == 1:
                with self._lock:
                    self._held.add(ch.lower())

        def on_release(key):
            ch = getattr(key, "char", None)
            if ch and len(ch) == 1:
                with self._lock:
                    self._held.discard(ch.lower())

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.daemon = True
        self._listener.start()

    def held(self) -> set[str]:
        with self._lock:
            return set(self._held)

    def pressed(self, previous: set[str]) -> set[str]:
        return self.held() - previous

    def close(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
