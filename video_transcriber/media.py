import os
import subprocess

def get_media_duration_seconds(input_file):
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                input_file,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def extract_audio_to_wav(input_file, output_wav, normalize=False):
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_file,
        "-ar",
        "16000",
        "-ac",
        "1",
    ]
    if normalize:
        # Use Dynamic Audio Normalizer (dynaudnorm) or EBU R128 (loudnorm).
        # loudnorm is superb at making speech consistently audible without clipping.
        cmd.extend(["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"])
    
    cmd.extend([
        "-c:a",
        "pcm_s16le",
        output_wav,
    ])
    
    subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )


def isolate_vocals_with_demucs(input_audio, output_dir, device="cuda"):
    os.makedirs(output_dir, exist_ok=True)
    cmd = ["demucs", "--two-stems=vocals", "-o", output_dir, input_audio]
    if device == "cuda":
        cmd.insert(1, "-d")
        cmd.insert(2, "cuda")
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    vocals_path = None
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith("vocals.wav"):
                vocals_path = os.path.join(root, file)
                break
        if vocals_path:
            break
    if not vocals_path:
        raise RuntimeError("Voice isolation with Demucs did not produce vocals.wav")
    return vocals_path


def get_audio_volume_metrics(input_file):
    import re
    try:
        # Use FFmpeg volumedetect filter
        result = subprocess.run(
            [
                "ffmpeg",
                "-i",
                input_file,
                "-af",
                "volumedetect",
                "-vn",
                "-sn",
                "-dn",
                "-f",
                "null",
                "NUL", # Windows null device
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stderr
        mean_vol = re.search(r"mean_volume: ([\-\d\.]+) dB", output)
        max_vol = re.search(r"max_volume: ([\-\d\.]+) dB", output)
        
        return {
            "mean_volume": float(mean_vol.group(1)) if mean_vol else -25.0,
            "max_volume": float(max_vol.group(1)) if max_vol else 0.0
        }
    except Exception:
        return {"mean_volume": -25.0, "max_volume": 0.0}

def find_media_files(root_dir, extensions):
    ext_set = {ext.lower() for ext in extensions}
    for dirpath, _, files in os.walk(root_dir):
        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext in ext_set:
                yield os.path.join(dirpath, file_name)
