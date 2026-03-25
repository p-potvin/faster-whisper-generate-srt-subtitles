from ast import Tuple
import os
import tempfile
import time
import shutil
from tqdm import tqdm
from contextlib import contextmanager
from video_transcriber import utils
from video_transcriber import translation
from video_transcriber import media

_WHISPER_MODEL = None

@contextmanager
def temporary_directory(prefix = "transcriber_"):
    temp_dir = tempfile.mkdtemp(prefix = prefix)
    try:
        yield temp_dir
    finally:
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Failed to clean up temp directory {temp_dir}: {e}")

def get_whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        from faster_whisper import WhisperModel
        with utils.spinning_cursor("Initializing Whisper Machine Learning Model..."):
            _WHISPER_MODEL = WhisperModel("medium", device = "cuda", compute_type = "int8_float16")
    return _WHISPER_MODEL

def transcribe_video(
    input_file,
    output_file = None,
    languages = None,
    skip_original = False,
    translate_api = "deep-translator",
    translate_mode = "non-target",
    max_translate_chars = 350000,
    max_translate_calls = 500,
    overwrite = False,
    vad_filter = True,
    vad_threshold = None,
    isolate_vocals = True,
    source_language = ["en"]
):
    start_total = time.time()
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

    # --- Vocal Segmentation, Isolation, Normalization and Transcription ---
    model = get_whisper_model()
    all_segments = list()
    original_texts = []
    vad_params = dict(threshold = vad_threshold) if vad_threshold is not None else dict(threshold = 0.35)
    transcription_file = input_file
    
    if isolate_vocals:
        from pydub import AudioSegment
        from pydub.silence import split_on_silence
        
        with temporary_directory(prefix = "demucs_") as temp_dir:    
            try:               
                start_step = time.time()
                audio = AudioSegment.from_file(transcription_file)
                min_chunk_length_ms = 800
                min_silence_length_ms = 400  
                silence_threshold_db = -40

                # Defaults are 800ms and -40dBFS
                chunks = split_on_silence(
                    audio,
                    min_silence_len = min_silence_length_ms,
                    silence_thresh = silence_threshold_db,
                    keep_silence = 100
                )

                print(f"Audio split into {len(chunks)} chunks based on silence detection.")
                
                current_time_offset = 0.0
                segmentId = 0
                
                # Keep chunks >= min_chunk_length_ms                
                for i, chunk in enumerate(chunks):
                    print(f"Chunk {i}, duration: {chunk.duration_seconds} seconds, {chunk.dBFS:.2f} dBFS")
                    if len(chunk) >=  min_chunk_length_ms:
                        print(f"Chunk {i}")
                        chunkPath = os.path.join(temp_dir, f"chunk{i}")
                        os.makedirs(chunkPath, exist_ok = True)

                        chunkName = os.path.join(chunkPath, f"chunk.wav")
                        chunk.export(chunkName, format = "wav")                   
                                        
                        # Isolate vocals for this chunk
                        isolatedChunk = media.isolate_vocals_with_demucs(chunkName, chunkPath, device = "cuda")
                                               
                        # Transcribe the isolated chunk
                        segments_generator, info = model.transcribe(
                            isolatedChunk, 
                            beam_size = 5, 
                            task = "transcribe", 
                            vad_filter = vad_filter,
                            vad_parameters = vad_params if vad_filter else None,
                            multilingual = True,
                            condition_on_previous_text = True,
                            word_timestamps = True
                        )                     
                        
                        # Consume the generator into a list to allow multiple iterations and check content
                        chunk_segments = list(segments_generator)

                        # Offset the segment timestamps by the current total duration
                        # and collect them into a single list
                        for segment in chunk_segments:
                            
                            start_time = segment.start + current_time_offset
                            end_time = segment.end + current_time_offset
                            print(f"Current Time Offset: {current_time_offset}, Segment start: {start_time}, text: {segment.text}")
                            
                            # Use a dictionary or a SimpleNamespace to ensure attributes are accessible via .start/.end
                            from types import SimpleNamespace
                            updated_segment = SimpleNamespace(
                                id = segmentId,
                                start = start_time,
                                end = end_time,
                                text = segment.text,
                                words = segment.words,
                                tokens = segment.tokens,
                                avg_logprob = segment.avg_logprob,
                                no_speech_prob = segment.no_speech_prob
                            )

                            segmentId +=  1
                            all_segments.append(updated_segment)
                            original_texts.append(updated_segment.text)
                        
                    current_time_offset +=  chunk.duration_seconds
                        
            except Exception as e:
                print(f"Warning: Vocal isolation failed: {e}. Falling back to original audio.")

                # Fallback to standard transcription if chunking/isolation failed
                print(f"--- Transcription (Fallback) ---")
                vad_params = dict(threshold = vad_threshold) if vad_threshold is not None else dict(threshold = 0.35)
                segments, info = model.transcribe(
                    input_file, 
                    beam_size = 5, 
                    task = "transcribe", 
                    vad_filter = vad_filter,
                    vad_parameters = vad_params if vad_filter else None,
                    multilingual = True,
                    word_timestamps = True,
                )
                all_segments = list(segments)
                original_texts = [s.text for s in all_segments]
    else:
        print(f"--- Skip to Transcription ---")
        start_ts = time.time()
        
        vad_params = dict(threshold = vad_threshold) if vad_threshold is not None else dict(threshold = 0.35)

        # --- Transcribe exactly as is (no translation at this step) ---
        print("Running transcribing exactly as is...")
        all_segments, info = model.transcribe(
            transcription_file, 
            beam_size = 5, 
            task = "transcribe", 
            vad_filter = vad_filter,
            vad_parameters = vad_params if vad_filter else None,
            multilingual = True,
            word_timestamps = True,  # Crucial for precise SRT timing
            initial_prompt = "I am transcribing as-is clear dialogue, in its original language. There may be many languages spoken and they must remain in their original language.", # Guide the model
            condition_on_previous_text = True,
            compression_ratio_threshold = 2.4, # standard fallback trigger
            log_prob_threshold = -1.0,         # standard fallback trigger
            no_speech_threshold = 0.6          # filter out silence properly
        )
        
        all_segments = list(all_segments)
        elapsed_total = time.time() - start_ts
        print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
        print(f"Segments generated: {len(all_segments)}")
        print(f"Transcription completed in {elapsed_total:.2f}s")
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

    # Translate only if needed
    if languages:        
        # Default should include 'en' if no languages were provided
        target_languages = languages if isinstance(languages, list) and len(languages) > 0 else source_language
        
        for lang_code in target_languages:
            lang_code = lang_code.strip().lower()
            if not lang_code or lang_code in ("orig", "original", "none"):
                continue

            path = f"{output_base}.{lang_code}.srt"
            
            # --- FIXED LOGIC ---
            # We treat the original text as 'multilingual' (as transcribed).
            # We only generate a translation if we have non-target segments to translate.
            # If the whole video is ALREADY in the target language, we could just copy it,
            # but usually, users want the translation step to handle mixed languages.
            
            print(f"--- Step 3: Translation ({lang_code}) ---")
            import asyncio
            trans_start_ts = time.time()
            try:
                # 'non-target' mode in translation module correctly identifies 
                # segments that don't match lang_code and translates only those.
                print(f"Texts: {original_texts}")
                translated_texts = asyncio.run(translation.translate_segments(
                    original_texts,
                    target_lang = lang_code,
                    translate_api = translate_api,
                    translate_mode = translate_mode,
                    max_chars = max_translate_chars,
                    max_calls = max_translate_calls,
                    detector = None
                ))
                trans_elapsed = time.time() - trans_start_ts
                print(f"Translation to {lang_code} completed in {trans_elapsed:.2f}s")
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
    for path, texts_to_write in tqdm(outputs_to_generate.items(), desc = "Saving SRTs", unit = "file"):
        utils.write_srt(path, all_segments, texts_to_write)
        output_paths.append(path)
        print(f"\nWrote: {path}")
        
    print(f"\n--- Processing Finished ---")
    print(f"\nTotal time elapsed: {time.time() - start_total:.2f}s")
    return output_paths
