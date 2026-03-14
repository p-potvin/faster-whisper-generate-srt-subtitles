import argparse
import os
import subprocess
import time
from faster_whisper import WhisperModel


def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    milliseconds = (seconds - int(seconds)) * 1000
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(milliseconds):03d}"


def _translate_with_googletrans(texts, target_lang, translate_mode, max_calls):
    try:
        from googletrans import Translator
    except ImportError as exc:
        raise ImportError(
            "Missing googletrans. Install with: pip install googletrans==4.0.0-rc1"
        ) from exc

    translator = Translator()
    translated_texts = []
    calls = 0

    if translate_mode == "non-target":
        for text in texts:
            calls += 1
            if calls > max_calls:
                raise RuntimeError("Translation request limit reached.")
            result = translator.translate(text, dest=target_lang)
            detected = getattr(result, "src", "").lower().split("-")[0]
            if detected == target_lang.lower().split("-")[0]:
                translated_texts.append(text)
            else:
                translated_texts.append(result.text)
        return translated_texts

    chunk_size = 40
    for i in range(0, len(texts), chunk_size):
        calls += 1
        if calls > max_calls:
            raise RuntimeError("Translation request limit reached.")
        chunk = texts[i : i + chunk_size]
        result = translator.translate(chunk, dest=target_lang)
        if isinstance(result, list):
            translated_texts.extend([item.text for item in result])
        else:
            translated_texts.append(result.text)
    return translated_texts


def _translate_with_deep_translator(texts, target_lang, translate_mode, max_calls):
    try:
        from deep_translator import GoogleTranslator
    except ImportError as exc:
        raise ImportError(
            "Missing deep-translator. Install with: pip install deep-translator"
        ) from exc

    translated_texts = []
    calls = 0
    if translate_mode == "non-target":
        for text in texts:
            calls += 1
            if calls > max_calls:
                raise RuntimeError("Translation request limit reached.")
            detected = GoogleTranslator(source="auto", target=target_lang).detect(text)
            if detected.lower().split("-")[0] == target_lang.lower().split("-")[0]:
                translated_texts.append(text)
            else:
                translated_texts.append(
                    GoogleTranslator(source="auto", target=target_lang).translate(text)
                )
        return translated_texts

    chunk_size = 40
    for i in range(0, len(texts), chunk_size):
        calls += 1
        if calls > max_calls:
            raise RuntimeError("Translation request limit reached.")
        for text in texts[i : i + chunk_size]:
            translated_texts.append(
                GoogleTranslator(source="auto", target=target_lang).translate(text)
            )
    return translated_texts


def _translate_segments(
    texts,
    target_lang: str,
    translate_api="googletrans",
    max_chars=350000,
    max_calls=250,
    translate_mode="all",
):
    if not texts:
        return []

    total_chars = sum(len(t) for t in texts)
    if total_chars > max_chars:
        raise RuntimeError(
            f"Translation skipped: text length {total_chars} > max_chars {max_chars}. "
            "Use --max-translate-chars to raise this limit or skip translation."
        )

    # prefer line-by-line for non-target detection behavior
    if translate_mode == "non-target":
        if translate_api == "googletrans":
            try:
                from googletrans import Translator
            except ImportError as exc:
                raise ImportError(
                    "Missing googletrans. Install with: pip install googletrans==4.0.0-rc1"
                ) from exc
            translator = Translator()
            translated_texts = []
            calls = 0
            for text in texts:
                calls += 1
                if calls > max_calls:
                    raise RuntimeError("Translation request limit reached.")
                if not text.strip():
                    translated_texts.append(text)
                    continue
                result = translator.translate(text, dest=target_lang)
                detected = getattr(result, "src", "").lower().split("-")[0]
                target_base = target_lang.lower().split("-")[0]
                if detected == target_base:
                    translated_texts.append(text)
                else:
                    translated_texts.append(result.text)
            return translated_texts

        if translate_api == "deep-translator":
            try:
                from deep_translator import GoogleTranslator
            except ImportError as exc:
                raise ImportError(
                    "Missing deep-translator. Install with: pip install deep-translator"
                ) from exc
            translated_texts = []
            calls = 0
            for text in texts:
                calls += 1
                if calls > max_calls:
                    raise RuntimeError("Translation request limit reached.")
                if not text.strip():
                    translated_texts.append(text)
                    continue
                if translate_mode == "non-target":
                    detected = None
                    try:
                        from googletrans import Translator
                        detected = Translator().detect(text).lang
                    except Exception:
                        detected = None
                    if detected and detected.lower().split("-")[0] == target_lang.lower().split("-")[0]:
                        translated_texts.append(text)
                        continue
                translated_texts.append(
                    GoogleTranslator(source="auto", target=target_lang).translate(text)
                )
            return translated_texts

    # default all mode or fallback
    if translate_api == "googletrans":
        try:
            from googletrans import Translator
        except ImportError as exc:
            raise ImportError(
                "Missing googletrans. Install with: pip install googletrans==4.0.0-rc1"
            ) from exc
        translator = Translator()
        translated_texts = []
        calls = 0
        chunk_size = 40
        for i in range(0, len(texts), chunk_size):
            calls += 1
            if calls > max_calls:
                raise RuntimeError("Translation request limit reached.")
            chunk = texts[i : i + chunk_size]
            result = translator.translate(chunk, dest=target_lang)
            if isinstance(result, list):
                translated_texts.extend([item.text for item in result])
            else:
                translated_texts.append(result.text)
        return translated_texts

    if translate_api == "deep-translator":
        try:
            from deep_translator import GoogleTranslator
        except ImportError as exc:
            raise ImportError(
                "Missing deep-translator. Install with: pip install deep-translator"
            ) from exc
        translated_texts = []
        calls = 0
        for text in texts:
            calls += 1
            if calls > max_calls:
                raise RuntimeError("Translation request limit reached.")
            translated_texts.append(
                GoogleTranslator(source="auto", target=target_lang).translate(text)
            )
        return translated_texts

    raise ValueError(f"Unsupported translate API: {translate_api}")


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


def write_srt(output_path, segments, texts):
    with open(output_path, "w", encoding="utf-8") as output_file:
        for segment, text in zip(segments, texts):
            start_time = format_time(segment.start)
            end_time = format_time(segment.end)
            segment_id = segment.id + 1
            line_out = f"{segment_id}\n{start_time} --> {end_time}\n{text.lstrip()}\n\n"
            output_file.write(line_out)


import threading


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
    model_size = "medium"  # options: tiny, small, medium, large-v2
    model = WhisperModel(model_size, device="cpu", cpu_threads=12, compute_type="int8")

    print(f"Transcribing: {input_file}")
    print(f"VAD filter: {vad_filter}")
    start_ts = time.time()
    spinner = ["|", "/", "-", "\\"]
    stop_flag = threading.Event()

    def _show_progress():
        idx = 0
        while not stop_flag.is_set():
            elapsed = time.time() - start_ts
            print(
                f"\rTranscribing... {spinner[idx % len(spinner)]} Elapsed: {int(elapsed)}s",
                end="",
                flush=True,
            )
            idx += 1
            time.sleep(0.2)
        print("\rTranscribing... done. " + " " * 30, flush=True)

    prog_thread = threading.Thread(target=_show_progress)
    prog_thread.start()

    all_segments, info = model.transcribe(input_file, beam_size=5, task="transcribe", vad_filter=vad_filter)
    all_segments = list(all_segments)

    stop_flag.set()
    prog_thread.join()

    elapsed_total = time.time() - start_ts
    print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
    print(f"Segments generated: {len(all_segments)}")
    print(f"Transcription elapsed: {elapsed_total:.1f}s")
    original_texts = [segment.text for segment in all_segments]

    base_path = os.path.splitext(input_file)[0]
    output_dir = os.path.dirname(input_file)

    if output_file and os.path.isdir(output_file):
        raise ValueError(f"Output file cannot be a directory: {output_file}")

    if output_file:
        output_base = os.path.splitext(output_file)[0]
    else:
        output_base = base_path

    output_targets = []
    if not skip_original:
        output_targets.append(output_base + ".srt")
    if languages:
        for lang in languages:
            lang_code = lang.strip().lower()
            if not lang_code or lang_code in ("orig", "original", "none"):
                continue
            output_targets.append(f"{output_base}.{lang_code}.srt")

    # If all targets already exist and overwrite is disabled, skip entire transcription.
    if not overwrite and output_targets and all(os.path.exists(path) for path in output_targets):
        print(f"All output files already exist, skipping transcription: {input_file}")
        return []

    outputs = []
    if not skip_original:
        orig_path = output_base + ".srt"
        if os.path.exists(orig_path) and not overwrite:
            print(f"Skipping existing file: {orig_path}")
        else:
            write_srt(orig_path, all_segments, original_texts)
            outputs.append(orig_path)
            print(f"Wrote original srt: {orig_path}")

    if languages:
        for lang in languages:
            lang_code = lang.strip().lower()
            if not lang_code:
                continue
            if lang_code in ("orig", "original", "none"):
                continue
            if lang_code == "en" and info.language == "en":
                translated_texts = original_texts
            else:
                translated_texts = _translate_segments(
                    original_texts,
                    target_lang=lang_code,
                    translate_api=translate_api,
                    translate_mode=translate_mode,
                    max_chars=max_translate_chars,
                    max_calls=max_translate_calls,
                )

            out_path = f"{output_base}.{lang_code}.srt"
            if os.path.exists(out_path) and not overwrite:
                print(f"Skipping existing translation file: {out_path}")
                continue

            write_srt(out_path, all_segments, translated_texts)
            outputs.append(out_path)
            print(f"Wrote translated srt: {out_path}")

    return outputs


def find_media_files(root_dir, extensions):
    ext_set = {ext.lower() for ext in extensions}
    for dirpath, _, files in os.walk(root_dir):
        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext in ext_set:
                yield os.path.join(dirpath, file_name)


def print_progress(current, total, prefix="", width=30):
    ratio = current / total if total > 0 else 1
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r{prefix} [{bar}] {current}/{total}", end="", flush=True)
    if current >= total:
        print("", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Generate SRT for one video or scan a folder recursively.")
    parser.add_argument("input_file", nargs="?", help="Video/audio file path (if not scanning a folder)")
    parser.add_argument("--scan-dir", help="If set, recursively process all supported media files under this directory")
    parser.add_argument("--output-file", help="Output SRT file (for single file only)")
    parser.add_argument(
        "--languages",
        default="",
        help="Comma-separated target language codes for translation (e.g. en,es,fr). Generates <video>.<lang>.srt.",
    )
    parser.add_argument("--skip-original", action="store_true", help="Do not generate original-language SRT")
    parser.add_argument("--translate-api", default="deep-translator", help="Translator backend (googletrans or deep-translator)")
    parser.add_argument(
        "--translate-mode",
        choices=["all", "non-target"],
        default="non-target",
        help="all: translate all segments; non-target: only translate where detected src != target",
    )
    parser.add_argument("--max-translate-chars", type=int, default=350000, help="Max total chars per video before translation is skipped")
    parser.add_argument("--max-translate-calls", type=int, default=250, help="Max translator calls per video")
    parser.add_argument("--max-duration", type=float, default=3600, help="Max media duration in seconds to process (skip longer files)")
    parser.add_argument("--continue-on-error", action="store_true", help="For scan mode, continue to next file when one fails")
    parser.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing SRT files")
    parser.add_argument("--vad-filter", action="store_true", default=False, help="Enable VAD filtering in Whisper transcribe (default false for better coverage)")
    parser.add_argument(
        "--extensions",
        default=".mp4,.mkv,.avi,.mov,.flv,.webm,.mp3,.wav,.m4a",
        help="Comma-separated media extensions for scan mode",
    )

    args = parser.parse_args()
    overwrite = not args.no_overwrite
    languages = [lang.strip().lower() for lang in args.languages.split(",") if lang.strip()]

    if args.scan_dir:
        if not os.path.isdir(args.scan_dir):
            raise FileNotFoundError(f"Scan directory does not exist: {args.scan_dir}")

        media_exts = [ext.strip().lower() for ext in args.extensions.split(",") if ext.strip()]
        matched = list(find_media_files(args.scan_dir, media_exts))
        if not matched:
            print(f"No media files found under {args.scan_dir} with extensions {media_exts}")
            return

        print(f"Found {len(matched)} files. Processing recursively...")
        successes = []
        failures = []
        start_time = time.time()
        for idx, path in enumerate(matched, start=1):
            print_progress(idx, len(matched), prefix="Scan progress")
            if args.max_duration is not None:
                duration = get_media_duration_seconds(path)
                if duration is not None and duration > args.max_duration:
                    print(f"\nSkipping (too long > {args.max_duration}s): {path} ({duration:.1f}s)")
                    failures.append((path, f"duration {duration:.1f}s > max {args.max_duration}s"))
                    continue

            try:
                outputs = transcribe_video(
                    path,
                    output_file=None,
                    languages=languages,
                    skip_original=args.skip_original,
                    translate_api=args.translate_api,
                    translate_mode=args.translate_mode,
                    max_translate_chars=args.max_translate_chars,
                    max_translate_calls=args.max_translate_calls,
                    overwrite=overwrite,
                vad_filter=args.vad_filter,
                )
                print(f"\nDone: {path} -> {len(outputs)} output files")
                successes.append(path)
            except Exception as exc:
                print(f"\nFailed for {path}: {exc}")
                failures.append((path, str(exc)))
                if not args.continue_on_error:
                    break

        elapsed = time.time() - start_time
        print("\nScan completed.")
        print(f"Success: {len(successes)} files")
        print(f"Failed: {len(failures)} files")
        if failures:
            for fail_path, reason in failures:
                print(f" - {fail_path}: {reason}")
        print(f"Total time: {elapsed:.1f}s")

    else:
        if not args.input_file:
            parser.error("Provide input_file or --scan-dir")

        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"Input file does not exist: {args.input_file}")

        transcribe_video(
            args.input_file,
            output_file=args.output_file,
            languages=languages,
            skip_original=args.skip_original,
            translate_api=args.translate_api,
            translate_mode=args.translate_mode,
            max_translate_chars=args.max_translate_chars,
            max_translate_calls=args.max_translate_calls,
            overwrite=overwrite,
            vad_filter=args.vad_filter,
        )


if __name__ == "__main__":
    main()
