"""Stdout protocol shared with the Electron main process.

The renderer listens for two tagged JSON line formats, both of which predate the
script split and must stay byte-compatible:

    PROGRESS_UPDATE:{"percent": 42, "label": "Separating vocals... 42%"}
    JSON_STATUS:{"status": "SUCCESS", "path": "..."}

Because every action is now its own process, each script owns the full 0-100
range instead of the hand-tuned offsets the monolith used.
"""

import json
import sys


def report_progress(percent, label):
    """Emit a progress tick. *percent* is clamped to 0-100."""
    try:
        pct = int(max(0, min(100, percent)))
    except (TypeError, ValueError):
        pct = 0
    print("PROGRESS_UPDATE:" + json.dumps({"percent": pct, "label": str(label)}))
    sys.stdout.flush()


def emit_status(status, **fields):
    """Emit a terminal (or lifecycle) status line."""
    payload = {"status": status}
    payload.update({k: v for k, v in fields.items() if v is not None})
    print("JSON_STATUS:" + json.dumps(payload, ensure_ascii=False))
    sys.stdout.flush()


def log(scope, message):
    """Plain human-readable log line; ignored by the IPC parsers."""
    print(f"[{scope}] {message}")
    sys.stdout.flush()


class ScaledProgress:
    """Maps a 0-100 sub-task onto a slice of the overall progress bar.

    Lets a script say "the Demucs pass covers 10-55% of my run" without every
    call site doing the arithmetic.
    """

    def __init__(self, start, span, label):
        self.start = float(start)
        self.span = float(span)
        self.label = label

    def __call__(self, sub_percent):
        pct = self.start + (max(0.0, min(100.0, float(sub_percent))) / 100.0) * self.span
        report_progress(pct, f"{self.label}... {int(sub_percent)}%")
