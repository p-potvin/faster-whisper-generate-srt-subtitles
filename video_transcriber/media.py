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
                os.path.abspath(input_file),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def extract_audio_to_wav(input_file, output_wav, normalize=True):
    # Improvement: Default to normalize=True for consistent volume
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        os.path.abspath(input_file),
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
        os.path.abspath(output_wav),
    ])
    
    subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

def isolate_vocals_with_demucs(input_audio, output_dir, device="cuda"):
    import sys
    os.makedirs(output_dir, exist_ok=True)
    input_audio_abs = os.path.abspath(input_audio)
    output_dir_abs = os.path.abspath(output_dir)
    
    cmd = [sys.executable, "-m", "demucs.separate", "--two-stems=vocals", "-o", output_dir_abs, input_audio_abs]
    if device == "cuda":
        cmd.insert(3, "-d")
        cmd.insert(4, "cuda")
    
    import subprocess
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        stderr_out = e.stderr if e.stderr else ""
        
        # If it failed due to CUDA (PyTorch CPU only, or out of VRAM), fallback to CPU
        if device == "cuda" and ("CUDA" in stderr_out or "device" in stderr_out.lower() or "AssertionError" in stderr_out or "RuntimeError" in stderr_out):
            print(f"\n[Demucs] CUDA error encountered, retrying on CPU. (Error: {stderr_out.strip()[-200:]})")
            if "-d" in cmd:
                idx = cmd.index("-d")
                cmd.pop(idx) # remove -d
                cmd.pop(idx) # remove cuda
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e2:
                raise RuntimeError(f"Demucs CPU fallback failed. Error:\n{e2.stderr}")
        else:
            raise RuntimeError(f"Demucs failed. Error:\n{stderr_out}")
    except FileNotFoundError:
        # Fallback if python module is not working
        cmd = ["demucs", "--two-stems=vocals", "-o", output_dir_abs, input_audio_abs]
        if device == "cuda":
            cmd.insert(1, "-d")
            cmd.insert(2, "cuda")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e3:
            raise RuntimeError(f"Demucs CLI failed. Error:\n{e3.stderr}")

    vocals_path = None
    for root, _, files in os.walk(output_dir_abs):
        for file in files:            
            if file == "vocals.wav":
                vocals_path = os.path.join(root, file) 
                break

        if vocals_path:
            break
    if not vocals_path:
        raise RuntimeError("Voice isolation with Demucs did not produce vocals.wav")
        
    # Create normalized path in the same directory as the input_audio to avoid 
    # nested Demucs structure issues and ensure cleanup works as expected.
    normalized_path = os.path.join(os.path.dirname(input_audio), os.path.splitext(os.path.basename(input_audio))[0] + "_vocals_norm.wav")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", os.path.abspath(vocals_path), 
                "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", 
                "-c:a", "pcm_s16le", os.path.abspath(normalized_path)
            ],
            check=True, capture_output=True, text=True
        )
        return normalized_path
    except subprocess.CalledProcessError as e:
        print(f"Warning: ffmpeg normalization failed, using raw vocals: {e.stderr}")
        return vocals_path


def get_audio_volume_metrics(input_file):
    import re
    try:
        input_file_abs = os.path.abspath(input_file)
        # Use FFmpeg volumedetect filter
        result = subprocess.run(
            [
                "ffmpeg",
                "-i",
                input_file_abs,
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
