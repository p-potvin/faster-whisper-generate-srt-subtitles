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

def extract_wav_for_asr(input_file):
    """
    Extracts a 16kHz mono WAV file from the video.
    This guarantees that Lhotse/torchaudio will use the 'soundfile' backend,
    completely bypassing the buggy FFmpeg C++ extension on Windows.
    """
    video_abs = os.path.abspath(input_file)
    video_dir = os.path.dirname(video_abs)
    video_name = os.path.splitext(os.path.basename(video_abs))[0]
    wav_path = os.path.join(video_dir, f"{video_name}_asr_temp.wav")
    
    if os.path.exists(wav_path):
        os.remove(wav_path)
        
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", video_abs,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        wav_path
    ], check=True)
    
    return wav_path

def fix_audio_and_reencode(video_path, output_path=None, delay_ms=0, bg_volume="-20dB", cq=26, fps=30, max_duration=None, progress_callback=None):
    import sys
    import math

    video_abs = os.path.abspath(video_path)
    video_dir = os.path.dirname(video_abs)
    video_name, video_ext = os.path.splitext(os.path.basename(video_abs))

    if not output_path:
        output_path = os.path.join(video_dir, f"{video_name}_fixed{video_ext}")
    output_abs = os.path.abspath(output_path)

    # 1. Fast Duration Check
    safe_duration = 0
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_abs],
            capture_output=True, text=True, check=True
        )
        detected_duration = float(res.stdout.strip())
        if max_duration and max_duration > 0:
            safe_duration = math.floor(min(detected_duration, max_duration))
        else:
            safe_duration = math.floor(detected_duration)
    except Exception as e:
        print(f"Warning: Could not determine duration for {video_name}. Using default processing.")
        if max_duration:
            safe_duration = max_duration

    temp_dir = os.path.join(video_dir, f"temp_demucs_{video_name}")
    os.makedirs(temp_dir, exist_ok=True)
    
    import threading
    def run_command_with_progress(cmd, desc, start_pct, scale):
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace'
        )
        
        def reader():
            for line in process.stdout:
                # 1. Update Progress
                # FFmpeg style
                if "out_time_ms=" in line:
                    try:
                        time_ms = int(line.split("=")[1].strip())
                        if safe_duration > 0:
                            pct = min(100, int((time_ms / 1000000) / safe_duration * scale) + start_pct)
                            if progress_callback:
                                progress_callback(f"{desc}... {pct-start_pct}%", pct)
                    except: pass
                # Demucs style (tqdm)
                elif "%|" in line:
                    try:
                        pct_str = line.split("%")[0].strip().split()[-1]
                        pct = int(pct_str)
                        scaled_pct = start_pct + int(pct * (scale / 100.0))
                        if progress_callback:
                            progress_callback(f"{desc}... {pct}%", scaled_pct)
                    except: pass
                
                # 2. Update Console/GUI Monitor (Selective to avoid signal flood)
                # We only print lines that look like status updates
                if any(x in line for x in ["frame=", "fps=", "size=", "time=", "bitrate=", "it/s", "s/it"]):
                    print(line, end='', flush=True)

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        process.wait()
        t.join()
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

    # 2. Execute Demucs
    print(f"--- Isolating Vocals for {video_name} ---")
    if progress_callback:
        progress_callback(f"Step 1.1: Separating Vocals (Demucs)...", 10)

    demucs_cmd = [
        sys.executable, "-m", "demucs.separate",
        "-n", "htdemucs", "-d", "cuda", "--two-stems=vocals",
        "--shifts=1", "--overlap=0.25", "-o", temp_dir,
        "--filename", "{stem}.{ext}", video_abs
    ]

    try:
        run_command_with_progress(demucs_cmd, "Step 1.1: Vocal Separation", 10, 35)
    except subprocess.CalledProcessError as e:
        # Fallback to CPU if CUDA fails
        print(f"Demucs CUDA failed, trying CPU.")
        if progress_callback:
            progress_callback("Demucs CUDA failed, falling back to CPU...", 15)
        demucs_cmd.remove("-d")
        demucs_cmd.remove("cuda")
        run_command_with_progress(demucs_cmd, "Step 1.1: Vocal Separation", 15, 30)
    
    if progress_callback:
        progress_callback("Step 1.1: Vocal Separation Complete", 45)

    vocals_path = os.path.join(temp_dir, "htdemucs", "vocals.wav")
    if not os.path.exists(vocals_path):
        raise RuntimeError(f"Demucs failed to produce vocals at {vocals_path}")

    # loudnorm requires buffered look-ahead, starving NVENC of audio frames.
    # dynaudnorm is a true streaming normalizer (zero lookahead), allowing NVENC
    # to encode at full hardware throughput without waiting on the audio graph.
    filter_complex = (
        f"[0:a]dynaudnorm=f=250:g=31:p=0.95:m=100[bg_norm];"
        f"[bg_norm]volume={bg_volume}[bg];"
        f"[1:a]highpass=f=100,dynaudnorm=f=250:g=31:p=0.95:m=100[voc];"
        f"[bg][voc]amix=inputs=2:weights=1 1.5:duration=first:normalize=0[out_a]"
    )

    # 4. Final Muxing Args
    off = round(delay_ms / 1000.0, 3)
    abs_off = abs(off)

    ffmpeg_args = ["ffmpeg", "-y", "-hide_banner", "-err_detect", "ignore_err", "-fflags", "+genpts", "-progress", "pipe:1"]
    if safe_duration > 0:
        ffmpeg_args.extend(["-t", str(safe_duration)])

    if delay_ms > 0:
        ffmpeg_args.extend(["-i", video_abs, "-itsoffset", str(abs_off), "-i", vocals_path])
    elif delay_ms < 0:
        ffmpeg_args.extend(["-itsoffset", str(abs_off), "-i", video_abs, "-i", vocals_path])
    else:
        ffmpeg_args.extend(["-i", video_abs, "-i", vocals_path])

    ffmpeg_args.extend([
        "-filter_complex", filter_complex,
        "-map", "0:v:0",
        "-map", "[out_a]",
        # p6 vs p7: p6 skips the slowest sub-pixel motion-estimation pass with near-zero
        # visible quality loss. Dropping -tune hq removes spatial/temporal AQ lookahead
        # which was stalling the GPU pipeline. constqp is equivalent to vbr+cq but
        # simpler — the GPU processes frames at a constant QP without rate-control overhead.
        "-c:v", "h264_nvenc", "-preset", "p6", "-rc", "constqp", "-qp", str(cq), "-vf", f"fps={fps}",
        "-c:a", "aac", "-b:a", "320k",
        output_abs
    ])

    print("--- Re-encoding Video with Fixed Audio ---")
    try:
        run_command_with_progress(ffmpeg_args, "Step 1.2: Encoding", 50, 50)
    except subprocess.CalledProcessError as e:
        print(f"NVENC failed, falling back to libx264.")
        idx = ffmpeg_args.index("h264_nvenc")
        ffmpeg_args[idx] = "libx264"
        # Remove nvenc-specific args
        for arg in ["-preset", "p6", "-rc", "constqp", "-qp", str(cq)]:
            if arg in ffmpeg_args:
                ffmpeg_args.remove(arg)
        ffmpeg_args.extend(["-crf", "23"]) # Standard CRF for libx264
        subprocess.run(ffmpeg_args, check=True)

    # Cleanup temp
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Replace original video
    shutil.move(output_abs, video_abs)
    print(f"Successfully fixed audio and replaced original: {video_abs}")
    return video_abs

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
