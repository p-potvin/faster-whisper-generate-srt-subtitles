import os
import subprocess
import shutil

def get_audio_duration_seconds(input_file):
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                #"-show_streams",
                "-show_entries",
                "stream=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                os.path.abspath(input_file),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except Exception as exc:
        return None


def extract_audio_to_wav(input_file, output_wav, normalize=True):
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg.")
        
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
    
    # htdemucs is the latest and fastest model.
    # --shifts=0: Disables equivariant stabilization (which usually takes 10x longer) for speed.
    # --overlap=0.1: Minimum overlap to avoid artifacts but maximize speed.
    # --segment=10: Smaller segments can be processed faster on some GPUs.
    # --two-stems=vocals: Only extracts vocals, avoiding processing other stems.
    cmd = [
        sys.executable, "-m", "demucs.separate",
        "-n", "htdemucs", # Use Hybrid Transformer Demucs (fastest)
        "--two-stems=vocals",
        "--shifts=0", # SIGNIFICANT SPEEDUP: avoid random shifts
        "--overlap=0.1", # Lower overlap for faster processing
        "-o", output_dir_abs,
        input_audio_abs
    ]
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
    except Exception as exc:
        print(f"Unexpected error while running Demucs: {exc}")
        return None

    vocals_path = None
    # Demucs version 4+ uses htdemucs/model_name/filename/vocals.wav structure
    for root, _, files in os.walk(output_dir_abs):
        for file in files:            
            if file == "vocals.wav":
                vocals_path = os.path.join(root, file)
                break
        if vocals_path:
            break
            
    if not vocals_path:
        print("Voice isolation with Demucs did not produce vocals.wav")
        return None
        
    # Create normalized path inside the temporary directory (output_dir_abs) 
    # to avoid file collisions across multiple processing runs.
    normalized_path = os.path.join(output_dir_abs, os.path.splitext(os.path.basename(input_audio))[0] + "_vocals_norm.wav")
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

def find_media_files(root_dir, extensions):
    ext_set = {ext.lower() for ext in extensions}
    for dirpath, _, files in os.walk(root_dir):
        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext in ext_set:
                yield os.path.join(dirpath, file_name)

def srt_files_exist(input_file, output_file, languages, skip_original):
    base_path = os.path.splitext(input_file)[0]
    output_base = os.path.splitext(output_file)[0] if output_file else base_path
        
    check_paths = []
    if not skip_original:
        check_paths.append(output_base + ".srt")
    if languages:
        for lang in languages:
            lang_code = lang.strip().lower()
            if lang_code and lang_code not in ("orig", "original", "none"):
                check_paths.append(f"{output_base}.{lang_code}.srt")
    
    if check_paths and all(os.path.exists(p) for p in check_paths):
        return True
    
    return False
