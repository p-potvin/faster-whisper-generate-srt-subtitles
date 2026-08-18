"""Enhance Audio — vocal isolation and dynamic normalisation.

Backs the "Enhance Audio 🪄" context-menu action and nothing else. It separates
vocals with Demucs, normalises speech and background independently, remixes them
with the vocals boosted, and writes the result to the enhanced copy.

It does **not** transcribe, translate, or touch subtitles. Those are
generate_subtitles.py and translate_video.py, which run without ever importing
torch's ASR stack.

    python enhance_audio.py <video|folder> [vault_root] [--volume-boost 1.5]
                            [--output PATH] [--skip-existing]
"""

import os
import subprocess
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
for _path in (_SCRIPT_DIR, _PROJECT_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from vw_media import cli, enhanced, media, state                    # noqa: E402
from vw_media.progress import ScaledProgress, emit_status, log, report_progress  # noqa: E402

ACTION = "audio"

# Background is ducked this far below the isolated vocals before remixing.
BACKGROUND_VOLUME = "-20dB"


def _venv_python():
    candidate = os.path.join(_PROJECT_ROOT, ".venv", "Scripts", "python.exe")
    if os.path.exists(candidate):
        return candidate
    candidate = os.path.join(_PROJECT_ROOT, ".venv", "bin", "python")
    if os.path.exists(candidate):
        return candidate
    return sys.executable


def separate_vocals(source_path, temp_dir, duration):
    """Run Demucs, returning the isolated vocals WAV path.

    Falls back to CPU if the CUDA pass fails — a machine without a working GPU
    should be slow, not broken.
    """
    cmd = [
        _venv_python(), "-m", "demucs.separate",
        "-n", "htdemucs", "-d", "cuda", "--two-stems=vocals",
        "--shifts=1", "--overlap=0.25", "-o", temp_dir,
        "--filename", "{stem}.{ext}", source_path,
    ]

    try:
        media.run_command_with_progress(
            cmd, "Separating vocals",
            on_progress=ScaledProgress(5, 50, "Separating vocals (GPU)"),
            duration=duration)
    except Exception:
        report_progress(10, "Demucs GPU separation failed. Retrying on CPU...")
        cpu_cmd = [c for c in cmd]
        if "-d" in cpu_cmd:
            device_index = cpu_cmd.index("-d")
            del cpu_cmd[device_index:device_index + 2]
        media.run_command_with_progress(
            cpu_cmd, "Separating vocals",
            on_progress=ScaledProgress(10, 45, "Separating vocals (CPU)"),
            duration=duration)

    vocals_path = os.path.join(temp_dir, "htdemucs", "vocals.wav")
    if not os.path.exists(vocals_path):
        raise RuntimeError(f"Vocal separation failed: no output WAV at {vocals_path}")
    return vocals_path


def remix(source_path, vocals_path, temp_output_path, vocal_weight, duration):
    """Mix ducked background with normalised vocals and encode the result."""
    filter_complex = (
        f"[0:a]aformat=channel_layouts=stereo,dynaudnorm=f=250:g=31:p=0.95:m=100[bg_norm];"
        f"[bg_norm]volume={BACKGROUND_VOLUME}[bg];"
        f"[1:a]aformat=channel_layouts=stereo,highpass=f=100,"
        f"dynaudnorm=f=250:g=31:p=0.95:m=100[voc];"
        f"[bg][voc]amix=inputs=2:weights=1 {vocal_weight}:duration=first:normalize=0,"
        f"aformat=channel_layouts=stereo[out_a]"
    )

    base_args = [
        "ffmpeg", "-y", "-hide_banner", "-err_detect", "ignore_err", "-fflags", "+genpts",
        "-i", source_path, "-i", vocals_path,
        "-filter_complex", filter_complex,
        "-map", "0:v:0", "-map", "[out_a]",
    ]
    audio_args = ["-c:a", "aac", "-b:a", "320k", "-ac", "2"]

    gpu_args = base_args + ["-c:v", "h264_nvenc", "-preset", "p6", "-rc", "constqp", "-qp", "26"] \
        + audio_args + [temp_output_path]
    try:
        media.run_command_with_progress(
            gpu_args, "Encoding",
            on_progress=ScaledProgress(55, 40, "Encoding normalised audio (GPU)"),
            duration=duration)
        return
    except subprocess.CalledProcessError:
        report_progress(60, "NVENC encoding failed. Falling back to CPU...")

    cpu_args = base_args + ["-c:v", "libx264", "-crf", "23"] + audio_args + [temp_output_path]
    media.run_command_with_progress(
        cpu_args, "Encoding",
        on_progress=ScaledProgress(60, 35, "Encoding normalised audio (CPU)"),
        duration=duration)


def process_one(video_path, args, output_path):
    media.require_streams(video_path, need_video=True, need_audio=True)

    vocal_weight = min(2.5, max(1.0, float(args.volume_boost or 1.5)))
    duration = media.get_video_duration(video_path)

    output_path, temp_output_path = enhanced.prepare_output(video_path, args.output)
    temp_dir = enhanced.make_temp_dir(output_path, prefix="audio")

    emit_status("STARTING", path=output_path)
    report_progress(2, "Preparing audio enhancement...")

    try:
        source_path = enhanced.resolve_source(video_path, output_path, temp_dir)
        vocals_path = separate_vocals(source_path, temp_dir, duration)

        report_progress(55, "Applying dynamic normalisation and vocal mixing...")
        remix(source_path, vocals_path, temp_output_path, vocal_weight, duration)

        enhanced.promote(temp_output_path, output_path)

        report_progress(97, "Recording enhancement state...")
        state.mark(video_path, ACTION,
                   enhanced_path=output_path,
                   params={"volumeBoost": vocal_weight, "backgroundVolume": BACKGROUND_VOLUME},
                   outputs=[output_path])

        report_progress(100, "Audio enhancement complete")
        emit_status("SUCCESS", path=output_path)
    except Exception:
        enhanced.discard_temp(temp_output_path)
        raise
    finally:
        enhanced.discard_temp(temp_dir)


def main():
    parser = cli.build_parser(
        "Enhance audio: Demucs vocal isolation plus dynamic normalisation", ACTION)
    parser.add_argument("--volume-boost", type=float, default=1.5,
                        help="Vocal mix multiplier, clamped to 1.0-2.5 (1.5 is roughly +50%%)")
    args = parser.parse_args()
    return cli.run(args, ACTION, process_one)


if __name__ == '__main__':
    sys.exit(main())
