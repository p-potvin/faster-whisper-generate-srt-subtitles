import os
import time
from faster_whisper import WhisperModel
import video_transcriber_utils as utils
import video_transcriber_translation as translation

def transcribe_video(
    input_file,
    output_file=None,
    languages=None,
    skip_original=False,
    translate_api="deep-translator",
    translate_mode="non-target",
    max_translate_chars=350000,
    max_translate_calls=250,
    overwrite=True,
    vad_filter=True,
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
    model_size = "medium"
    model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

    print(f"Transcribing: {input_file} (VAD filter: {vad_filter})")
    start_ts = time.time()
    
    # Transcribe once to get segments and language info
    all_segments, info = model.transcribe(input_file, beam_size=5, task="transcribe", vad_filter=vad_filter)
    all_segments = list(all_segments)
    elapsed_total = time.time() - start_ts
    print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
    print(f"Segments generated: {len(all_segments)}")
    print(f"Transcription elapsed: {elapsed_total:.1f}s")
    original_texts = [segment.text for segment in all_segments]

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
                print(f"Translating to {lang_code}...")
                import asyncio
                translated_texts = asyncio.run(translation.translate_segments(
                    original_texts,
                    target_lang=lang_code,
                    translate_api=translate_api,
                    translate_mode=translate_mode,
                    max_chars=max_translate_chars,
                    max_calls=max_translate_calls,
                    detector=None
                ))
            
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
