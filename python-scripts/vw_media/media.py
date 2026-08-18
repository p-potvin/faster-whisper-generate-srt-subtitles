"""ffmpeg/ffprobe helpers: probing, subprocess progress parsing, audio extraction."""

import json
import os
import subprocess
import threading

from .progress import log, report_progress

SUPPORTED_EXTENSIONS = ('.mp4', '.mkv', '.avi', '.mov', '.webm', '.wmv')

# 16 kHz mono is what every ASR model in the stack expects.
ASR_SAMPLE_RATE = 16000


def get_video_duration(path):
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', path,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(res.stdout.strip())
    except Exception:
        return 0.0


def check_streams(path):
    """Return ``(has_video, has_audio)`` for *path*."""
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type',
        '-of', 'default=noprint_wrappers=1:nokey=1', path,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        streams = [s.strip().lower() for s in res.stdout.strip().split('\n')]
        return 'video' in streams, 'audio' in streams
    except Exception:
        return False, False


def require_streams(path, need_video=True, need_audio=True):
    """Raise if *path* is missing a stream the caller depends on."""
    has_video, has_audio = check_streams(path)
    if need_video and not has_video:
        raise RuntimeError("Stream integrity violation: video stream missing")
    if need_audio and not has_audio:
        raise RuntimeError("Stream integrity violation: audio stream missing")
    return has_video, has_audio


def run_command_with_progress(cmd, desc, on_progress=None, duration=0.0):
    """Run *cmd*, streaming ffmpeg/tqdm progress through *on_progress*.

    *on_progress* receives a 0-100 sub-task percentage; wrap it in a
    :class:`~vw_media.progress.ScaledProgress` to place it on the overall bar.
    Raises :class:`subprocess.CalledProcessError` on a non-zero exit, after
    dumping the tail of the output so failures are diagnosable from the app log.
    """
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        encoding='utf-8',
        errors='replace',
    )

    output_lines = []

    def reader():
        for line in process.stdout:
            output_lines.append(line)
            if on_progress is None:
                continue
            # ffmpeg -progress output
            if "out_time_ms=" in line:
                try:
                    time_ms = int(line.split("=")[1].strip())
                    if duration > 0:
                        on_progress(min(100.0, (time_ms / 1000000.0) / duration * 100.0))
                except Exception:
                    pass
            # tqdm bars (Demucs)
            elif "%|" in line:
                try:
                    on_progress(int(line.split("%")[0].strip().split()[-1]))
                except Exception:
                    pass

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    process.wait()
    t.join()

    if process.returncode != 0:
        log("subprocess", f"Command failed: {' '.join(str(c) for c in cmd)}")
        print("".join(output_lines[-30:]))
        raise subprocess.CalledProcessError(process.returncode, cmd)

    return "".join(output_lines)


def extract_audio(video_path, wav_path, sample_rate=ASR_SAMPLE_RATE, denoise=True,
                  on_progress=None, duration=0.0):
    """Decode *video_path*'s audio to a mono WAV suitable for ASR.

    This is the cheap path the subtitle and translation actions use. It is a
    plain decode — no Demucs, no re-encode of the video — which is the whole
    point of splitting those actions out of the audio pipeline.
    """
    af = []
    if denoise:
        # Matches the live-subs daemon's front-end so offline and live
        # transcription see the same signal.
        af = ["highpass=f=90", "lowpass=f=7500", "afftdn=nf=-25", "dynaudnorm=f=200:g=15"]

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", video_path, "-vn"]
    if af:
        cmd += ["-af", ",".join(af)]
    cmd += ["-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le"]
    if duration > 0:
        cmd += ["-progress", "pipe:1", "-nostats"]
    cmd += [wav_path]

    run_command_with_progress(cmd, "Extracting audio", on_progress=on_progress, duration=duration)
    if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
        raise RuntimeError(f"Audio extraction produced no output at {wav_path}")
    return wav_path


def iter_video_files(root):
    """Yield every supported video under *root*, skipping dot-directories.

    Skipping dot-directories is what keeps a batch run from re-processing its own
    ``.enhanced/`` output.
    """
    for dirpath, _, filenames in os.walk(root):
        if any(part.startswith('.') for part in dirpath.split(os.sep)):
            continue
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in SUPPORTED_EXTENSIONS:
                yield os.path.join(dirpath, filename)
