#!/usr/bin/env python3
"""Delay local auto-start after registration while lazy CCTV streams warm."""

import os


TARGET = os.environ.get(
    "MARC_COMPETITION_MANAGER_TARGET",
    "/metacom2026/simulation_app/marc2026/runtime/competition_manager.py",
)
MARKER = "[marc-patch] CCTV auto-start warm-up"

WAIT_OLD = '''            plan = self._auto_plan
            self._auto_plan = None
'''
WAIT_NEW = '''            delay = float(os.environ.get("MARC_AUTO_START_DELAY_S", "0"))
            if self._auto_registered_at is None:
                self._auto_registered_at = time.monotonic()
                self.sim_ctx.play()
                log.info(f"[marc-patch] CCTV auto-start warm-up: {delay:.0f}s")
            if time.monotonic() - self._auto_registered_at < delay:
                return
            plan = self._auto_plan
            self._auto_plan = None
'''

INIT_OLD = "        self._auto_wait_logged = False\n"
INIT_NEW = INIT_OLD + "        self._auto_registered_at = None\n"

SET_OLD = '''        self._auto_wait_register = wait_register
        self._auto_wait_logged = False
        # local 패널 reset'''
SET_NEW = '''        self._auto_wait_register = wait_register
        self._auto_wait_logged = False
        self._auto_registered_at = None
        # local 패널 reset'''

RESET_OLD = '''            self._auto_wait_register = self._auto_wait_register_saved
            self._auto_wait_logged = False
            log.info('''
RESET_NEW = '''            self._auto_wait_register = self._auto_wait_register_saved
            self._auto_wait_logged = False
            self._auto_registered_at = None
            log.info('''


def patch_source(src: str) -> str:
    if MARKER in src:
        return src
    if WAIT_OLD not in src or SET_OLD not in src or RESET_OLD not in src:
        raise ValueError("competition-manager anchors not found")
    src = src.replace(INIT_OLD, INIT_NEW, 1)
    src = src.replace(WAIT_OLD, WAIT_NEW, 1)
    src = src.replace(SET_OLD, SET_NEW, 1)
    return src.replace(RESET_OLD, RESET_NEW, 1)


def main() -> int:
    if not os.path.exists(TARGET):
        print(f"[auto-start-delay-patch] source not found, skipping: {TARGET}")
        return 0
    with open(TARGET, encoding="utf-8") as f:
        original = f.read()
    try:
        patched = patch_source(original)
    except ValueError as exc:
        print(f"[auto-start-delay-patch] {exc}, skipping")
        return 0
    if patched == original:
        print("[auto-start-delay-patch] already applied, skipping")
        return 0
    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(patched)
    print("[auto-start-delay-patch] applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
