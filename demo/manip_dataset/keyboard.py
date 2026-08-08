"""Non-blocking terminal keyboard reader (termios)."""

from __future__ import annotations

import select
import sys
import termios
import tty
from typing import Optional


HELP_TEXT = """
=== Manip teleop dataset collector keys ===
MANUAL_BASE:
  i/k     base forward / back
  j/l     base yaw left / right
  space   stop base
  A       start auto-approach (baseline depth now, crawl forward)
  v/b     grip CLOSE / OPEN (not recorded — use b to release a carried object)
ARM_TELEOP (after auto-approach stop):
  t/g     EE +/- X (fwd/back in arm_base)
  f/h     EE +/- Y
  r/y     EE +/- Z
  u/o     EE +/- yaw
  ,/.     EE +/- pitch
  v       grip CLOSE
  b       grip OPEN
  E       end attempt -> sensor success filter (grip stays as commanded)
ANY:
  C       cancel auto-approach (-> MANUAL_BASE)
  N       discard episode / abort
  ?       help
  Ctrl-C  quit
===========================================
"""


class Keyboard:
    def __init__(self, stream=None):
        self._fd = None
        self._old = None
        self._stream = stream or sys.stdin
        self.enabled = False

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, *args):
        self.disable()

    def enable(self):
        if not hasattr(self._stream, "fileno"):
            return
        try:
            self._fd = self._stream.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self.enabled = True
        except Exception as e:
            print(f"[keyboard] could not enable raw mode ({e}); keys may not work")
            self.enabled = False

    def disable(self):
        if self._fd is not None and self._old is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            except Exception:
                pass
        self.enabled = False
        self._fd = None
        self._old = None

    def poll(self) -> Optional[str]:
        """Return one character if available, else None."""
        if not self.enabled or self._fd is None:
            return None
        try:
            r, _, _ = select.select([self._stream], [], [], 0.0)
            if not r:
                return None
            ch = self._stream.read(1)
            return ch
        except Exception:
            return None
