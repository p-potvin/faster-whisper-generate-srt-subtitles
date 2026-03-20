import os
import time
import tempfile
from faster_whisper import WhisperModel
from video_transcriber import utils
from video_transcriber import translation
from video_transcriber import media

_WHISPER_MODEL = None


def get_whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        _WHISPER_MODEL = WhisperModel("medium", device="cuda", compute_type="int8_float16")
    return _WHISPER_MODEL

def transcribe_video(
    input_file,
    output_file=None,
    languages=None,
    skip_original=False,
    translate_api="deep-translator",
    translate_mode="non-target",
    max_translate_chars=350000,
    max_translate_calls=500,
    overwrite=False,
    vad_filter=True,
    vad_threshold=None,
    normalize_audio=True,
):
    # Validate input file
    if not os.path.isfile(input_file):
        if os.path.isdir(input_file):
            raise ValueError(f"Input path is a directory, not a file: {input_file}")
        else:
            raise FileNotFoundError(f"Input file does not exist: {input_file}")
  
    # Check if all requested output files already exist when overwrite is disabled
    if not overwrite:
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
            print(f"All requested SRT files already exist, skipping: {input_file}")
            return []

    # --- Step 1: Transcribe ---
    model = get_whisper_model()

    # Dynamic VAD threshold detection and Audio Boosting
    transcription_file = input_file
    temp_audio_file = None
    
    try:
        vol_metrics = media.get_audio_volume_metrics(input_file)
        mean_vol = vol_metrics.get("mean_volume", -25.0)
        
        # Apply loudness normalization for all audio to ensure consistent speech-to-noise ratio
        if normalize_audio:
            print(f"Applying dynamic loudness normalization (Original Mean Vol: {mean_vol:.1f} dB)...")
            temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
            os.close(temp_fd)
            temp_audio_file = temp_path
            media.extract_audio_to_wav(input_file, temp_audio_file, normalize=True)
            transcription_file = temp_audio_file
            # Once normalized, audio levels are ideal (loudnorm targets ~ -16 LUFS)
            mean_vol = -16.0

        if vad_filter:
            if vad_threshold is None or vad_threshold == "auto" or vad_threshold == 0.0:
                # With normalized audio, the standard threshold of ~0.35 works perfectly.
                vad_threshold = max(0.15, min(0.35, 0.35 + (mean_vol + 16) * 0.01))
                print(f"Calculated auto-VAD threshold: {vad_threshold:.2f}")

        print(f"Transcribing: {input_file} (VAD filter: {vad_filter}, threshold: {vad_threshold})")
        start_ts = time.time()
        
        # Transcribe once to get segments and language info
        all_segments, info = model.transcribe(
            transcription_file, 
            beam_size=5, 
            task="transcribe", 
            vad_filter=vad_filter,
            vad_parameters=dict(threshold=vad_threshold) if vad_filter else None
        )
        all_segments = list(all_segments)
        elapsed_total = time.time() - start_ts
        print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
        print(f"Segments generated: {len(all_segments)}")
        print(f"Transcription elapsed: {elapsed_total:.1f}s")
        original_texts = [segment.text for segment in all_segments]
    finally:
        # Cleanup temporary boosted audio file
        if temp_audio_file and os.path.exists(temp_audio_file):
            try:
                os.remove(temp_audio_file)
            except Exception as e:
                print(f"Warning: Failed to clean up temporary audio file {temp_audio_file}: {e}")

    # --- Step 2: Prepare output file paths and contents ---
    base_path = os.path.splitext(input_file)[0]
    output_base = os.path.splitext(output_file)[0] if output_file else base_path

    outputs_to_generate = {}

    if not skip_original:
        original_srt_path = output_base + ".srt"
        if not overwrite and os.path.exists(original_srt_path):
            print(f"Skipping {original_srt_path} (overwrite is off).")
        else:
            path = output_base + ".srt"
            outputs_to_generate[path] = original_texts

    if languages:
        source_lang = str(getattr(info, "language", "") or "").strip().lower()
        source_lang_supported = translation.is_supported_language_code(source_lang, translate_api=translate_api)
        for lang_code in languages:
            lang_code = lang_code.strip().lower()
            if not lang_code or lang_code in ("orig", "original", "none"):
                continue

            path = f"{output_base}.{lang_code}.srt"
            
            is_target_lang_same_as_source = (lang_code == info.language) or \
                                            (lang_code.split('-')[0] == info.language.split('-')[0])

            if is_target_lang_same_as_source and translate_mode != "all":
                translated_texts = original_texts
            else:
                if translate_mode != "all" and source_lang and not source_lang_supported:
                    print(
                        f"Skipping translation to {lang_code}: detected source language '{info.language}' is not supported by {translate_api}."
                    )
                    continue
                print(f"Translating to {lang_code}...")
                import asyncio
                try:
                    translated_texts = asyncio.run(translation.translate_segments(
                        original_texts,
                        target_lang=lang_code,
                        translate_api=translate_api,
                        translate_mode=translate_mode,
                        max_chars=max_translate_chars,
                        max_calls=max_translate_calls,
                        detector=None
                    ))
                except translation.UnsupportedLanguageError as exc:
                    print(f"Skipping translation to {lang_code}: {exc}")
                    continue
                except Exception as exc:
                    print(f"Exception translating to {lang_code}: {exc}")
                    continue
            
            outputs_to_generate[path] = translated_texts
    
    # --- Step 3: Write all generated files ---
           
    if not outputs_to_generate:
        print("No new files to generate.")
        return []

    print(f"Writing {len(outputs_to_generate)} SRT file(s)...")
    output_paths = []
    for path, texts_to_write in outputs_to_generate.items():
        utils.write_srt(path, all_segments, texts_to_write)
        output_paths.append(path)
        print(f"Wrote: {path}")
        
    return output_paths
