"""Enhance Video — RTX VSR upscaling.

Backs the "Enhance Video 🪄" context-menu action and nothing else.

The upscaling itself lives in ``rtx_vsr_stream.py``, which owns the NVIDIA VFX
pipeline and is also used for live streaming. This script is the enhancement
*action*: it resolves the output path, honours ``--output`` and
``--skip-existing``, translates the VSR log into the progress protocol the app
listens for, and records the result in the sidecar so the menu can show that
video enhancement has been applied.

    python enhance_video.py <video|folder> [vault_root] [--quality HIGH]
                            [--scale 2] [--chroma yuv420p]
                            [--output PATH] [--skip-existing]
"""

import os
import re
import subprocess
import sys
import threading

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
for _path in (_SCRIPT_DIR, _PROJECT_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from vw_media import cli, enhanced, media, state                     # noqa: E402
from vw_media.progress import emit_status, log, report_progress      # noqa: E402

ACTION = "video"

VSR_SCRIPT = os.path.join(_SCRIPT_DIR, "rtx_vsr_stream.py")

_FRAME_RE = re.compile(r"Processed (\d+) frames")


def _python_exe():
    candidate = os.path.join(_PROJECT_ROOT, ".venv", "Scripts", "python.exe")
    if os.path.exists(candidate):
        return candidate
    candidate = os.path.join(_PROJECT_ROOT, ".venv", "bin", "python")
    if os.path.exists(candidate):
        return candidate
    return sys.executable


def estimate_total_frames(video_path, duration):
    """Frame count for the progress bar, from ffprobe's average frame rate."""
    try:
        res = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=avg_frame_rate',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True, text=True, check=True)
        num, _, den = res.stdout.strip().partition('/')
        fps = float(num) / float(den or 1)
        if fps > 0 and duration > 0:
            return int(fps * duration)
    except Exception:
        pass
    return 0


def run_vsr(source_path, temp_output_path, args, total_frames):
    """Run the VSR pipeline, forwarding its frame counter as progress."""
    if not os.path.exists(VSR_SCRIPT):
        # Not every project that uses vw_media ships the upscaler.
        raise RuntimeError(
            f"Video enhancement is unavailable here: {os.path.basename(VSR_SCRIPT)} "
            f"is not present in {_SCRIPT_DIR}")

    cmd = [
        _python_exe(), "-u", VSR_SCRIPT, "enhance", source_path, temp_output_path,
        "--quality", args.quality,
        "--scale", str(args.scale),
        "--chroma", args.chroma,
    ]
    log(ACTION, f"Spawning: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, encoding='utf-8', errors='replace')

    tail = []

    def reader():
        for line in process.stdout:
            tail.append(line)
            if len(tail) > 40:
                del tail[0]
            match = _FRAME_RE.search(line)
            if match and total_frames > 0:
                done = int(match.group(1))
                pct = 5 + min(90.0, done / total_frames * 90.0)
                report_progress(pct, f"Upscaling... {int(done / total_frames * 100)}%")

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    process.wait()
    thread.join()

    if process.returncode != 0:
        print("".join(tail))
        raise RuntimeError(f"RTX VSR upscaling failed with exit code {process.returncode}")


def process_one(video_path, args, output_path):
    media.require_streams(video_path, need_video=True, need_audio=False)

    duration = media.get_video_duration(video_path)
    total_frames = estimate_total_frames(video_path, duration)

    output_path, temp_output_path = enhanced.prepare_output(video_path, args.output)
    temp_dir = enhanced.make_temp_dir(output_path, prefix="video")

    emit_status("STARTING", path=output_path)
    report_progress(2, "Preparing video enhancement...")

    try:
        source_path = enhanced.resolve_source(video_path, output_path, temp_dir)

        # rtx_vsr_stream does its own atomic .tmp write, so it lands on
        # temp_output_path; we then promote that to the real output. The extra
        # hop is what lets us stack onto an existing enhanced copy without the
        # VSR pass ever reading and writing the same file.
        run_vsr(source_path, temp_output_path, args, total_frames)

        if not os.path.exists(temp_output_path):
            raise RuntimeError("RTX VSR reported success but produced no output file")
        enhanced.promote(temp_output_path, output_path)

        report_progress(97, "Recording enhancement state...")
        state.mark(video_path, ACTION,
                   enhanced_path=output_path,
                   params={"quality": args.quality, "scale": args.scale, "chroma": args.chroma},
                   outputs=[output_path])

        report_progress(100, "Video enhancement complete")
        emit_status("SUCCESS", path=output_path)
    except Exception:
        enhanced.discard_temp(temp_output_path)
        raise
    finally:
        enhanced.discard_temp(temp_dir)


def main():
    parser = cli.build_parser("Enhance video: RTX VSR upscaling", ACTION)
    parser.add_argument("--quality", default="HIGH", choices=["LOW", "MEDIUM", "HIGH", "ULTRA"],
                        help="VSR quality level (default: HIGH)")
    parser.add_argument("--scale", type=float, default=2.0,
                        help="Upscale factor, e.g. 2 or 4 (default: 2)")
    parser.add_argument("--chroma", default="yuv420p", choices=["yuv420p", "yuv422p", "yuv444p"],
                        help="Output chroma subsampling (default: yuv420p)")
    args = parser.parse_args()
    return cli.run(args, ACTION, process_one)


if __name__ == '__main__':
    sys.exit(main())
