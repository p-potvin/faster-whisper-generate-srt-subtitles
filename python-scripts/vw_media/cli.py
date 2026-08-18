"""Shared CLI scaffolding so each action script stays thin.

Every entrypoint accepts the same base contract::

    <script> <video_path|folder> [vault_root] [--output PATH] [--skip-existing]

``--output`` is the explicit destination flag; without it results land in the
hidden ``.enhanced/`` directory beside the source. ``vault_root`` stays a
positional for compatibility with the existing Electron spawn call.
"""

import argparse
import os
import sys
import traceback

from . import state
from .enhanced import resolve_output_path
from .media import iter_video_files
from .progress import emit_status, log


def build_parser(description, action):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("video_path", help="Path to a video file, or a folder to process recursively")
    parser.add_argument("vault_root", nargs="?", default=None,
                        help="Root vault path (informational; used for logging and relative reporting)")
    parser.add_argument("--output", "-o", default=None,
                        help="Explicit output path or directory. Defaults to the hidden .enhanced/ "
                             "directory beside the source file.")
    parser.add_argument("--skip-existing", action="store_true", default=False,
                        help=f"Skip files whose sidecar already records the '{action}' enhancement")
    parser.add_argument("--print-state", action="store_true", default=False,
                        help="Print the sidecar enhancement state for the target and exit")
    return parser


def _report_state(target):
    if os.path.isdir(target):
        files = list(iter_video_files(target))
    else:
        files = [target]
    for path in files:
        summary = state.summary(path)
        print(f"{path}: {summary}")
    return 0


def run(args, action, process_one, needs_output=True, skip_check=None):
    """Drive a single file or a whole folder through *process_one*.

    *process_one* is called as ``process_one(video_path, args, output_path)`` and
    should raise on failure. Batch runs keep going after a per-file failure and
    report the tally at the end, so one bad file cannot abort a long queue.

    *skip_check* overrides the ``--skip-existing`` test for actions where "already
    applied" is finer-grained than the action name — translation, for instance,
    is per target language.
    """
    target = os.path.abspath(args.video_path)

    if args.print_state:
        return _report_state(target)

    if not os.path.exists(target):
        emit_status("FAILED", error=f"Path not found: {target}")
        return 1

    if os.path.isfile(target):
        files = [target]
        batch = False
    else:
        files = sorted(iter_video_files(target))
        batch = True
        if not files:
            log(action, f"No video files found under {target}")
            emit_status("FAILED", error=f"No video files found under {target}")
            return 1
        log(action, f"Found {len(files)} video file(s) to process.")

    if batch and args.output and not os.path.isdir(args.output):
        # A single file path cannot receive many outputs.
        emit_status("FAILED", error="--output must be a directory when processing a folder")
        return 1

    processed = skipped = failed = 0

    for index, video_path in enumerate(files, start=1):
        if args.skip_existing:
            already = skip_check(video_path, args) if skip_check else state.is_applied(video_path, action)
            if already:
                log(action, f"[SKIP] {video_path} -> '{action}' already applied")
                skipped += 1
                continue

        if batch:
            log(action, f"[{index}/{len(files)}] {os.path.basename(video_path)}")

        output_path = resolve_output_path(video_path, args.output) if needs_output else None

        try:
            process_one(video_path, args, output_path)
            processed += 1
        except Exception as err:
            failed += 1
            if batch:
                log(action, f"Failed to process {video_path}: {err}")
                traceback.print_exc(file=sys.stderr)
            else:
                emit_status("FAILED", error=str(err))
                traceback.print_exc(file=sys.stderr)
                return 1

    if batch:
        log(action, f"Done. Processed: {processed}, Skipped: {skipped}, Failed: {failed}")
        if processed == 0 and failed > 0:
            emit_status("FAILED", error=f"All {failed} file(s) failed")
            return 1
        emit_status("SUCCESS", processed=processed, skipped=skipped, failed=failed)

    return 0
