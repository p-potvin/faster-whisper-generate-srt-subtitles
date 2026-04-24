import argparse
from faulthandler import dump_traceback_later
from mimetypes import suffix_map
import os
import sys
import time

# VaultWares Agent Knowledge Sync
# Part of the official VaultWares media pipeline.
# Can call invoke_vaultwares_team for multi-step translation or quality assurance.

def main():
    # Only import heavy dependencies inside main() to speed up interpreter exit/error paths
    from video_transcriber import core
    from video_transcriber import utils
    from video_transcriber import media

    utils.set_correlation_id()
    utils.log_info("Starting video transcriber-translator")
    parser = argparse.ArgumentParser(description="Generate SRT for one video or scan a folder recursively.")
    parser.add_argument("input_file", nargs="?", help="Video/audio file path (if not scanning a folder)")
    parser.add_argument("--scan-dir", help="If set, recursively process all supported media files under this directory")
    parser.add_argument("--output-file", help="Output SRT file (for single file only)")
    parser.add_argument(
        "--languages",
        default="en",
        help="Comma-separated target language codes for translation (e.g. en,es,fr). Generates <video>.<lang>.srt.",
    )
    parser.add_argument("--skip-original", action="store_true", help="Do not generate original-language SRT")
    parser.add_argument("--skip-vocal-isolation", action="store_true", help="Skip vocal isolation with Demucs (faster but noisier)")
    parser.add_argument("--translate-api", default="deep-translator", help="Translator backend (googletrans or deep-translator)")
    parser.add_argument(
        "--translate-mode",
        choices=["all", "non-target"],
        default="all",
        help="all: translate all segments; non-target: only translate where detected src != target",
    )
    parser.add_argument("--max-translate-chars", type=int, default=1000000, help="Max total chars per video before translation is skipped")
    parser.add_argument("--max-translate-calls", type=int, default=500, help="Max translator calls per video")
    parser.add_argument("--max-duration", type=float, default=7200, help="Max media duration in seconds to process (skip longer files)")
    parser.add_argument("--continue-on-error", action="store_true", help="For scan mode, continue to next file when one fails")
    parser.add_argument("--overwrite", action="store_true", dest="overwrite", default=False, help="Overwrite existing SRT files")
    parser.add_argument("--source-language", default=None, help="Force Whisper to use a specific source language (e.g. 'en', 'es'). Prevents language-switching hallucinations.")
    parser.add_argument("--engine", choices=["whisper", "parakeet"], default="whisper", help="Transcription engine to use")
    parser.add_argument("--extensions", default=".mp4,.mkv,.avi,.mov,.flv,.webm,.mp3,.wav,.m4a", help="Comma-separated media extensions for scan mode",
    )

    args = parser.parse_args()
    
    # Default to scanning 'to_process' folder if no input is provided
    if not args.input_file and not args.scan_dir:
        default_dir = "to_process"
        if os.path.isdir(default_dir):
            args.scan_dir = default_dir
            print(f"No input specified. Defaulting to scan directory: {default_dir}")
        else:
            parser.error("No input file or scan directory specified, and default 'to_process' folder not found.")

    languages = [lang.strip().lower() for lang in args.languages.split(",") if lang.strip()]

    if args.scan_dir:
        if not os.path.isdir(args.scan_dir):
            raise FileNotFoundError(f"Scan directory does not exist: {args.scan_dir}")

        media_exts = [ext.strip().lower() for ext in args.extensions.split(",") if ext.strip()]
        matched = list(media.find_media_files(args.scan_dir, media_exts))
        if not matched:
            raise FileNotFoundError(f"No media files found under {args.scan_dir} with accepted extensions: {media_exts}")

        print(f"Found {len(matched)} files. Processing recursively...")
        successes = []
        failures = []
        start_time = time.time()
        total_media_duration = 0.0

        for path in matched:
            # Check audio duration before processing           
            duration = media.get_audio_duration_seconds(path)

            if duration is None or duration <= 0:
                print(f"Skipping (invalid duration: {duration}): {path}")
                failures.append((path, f"invalid duration: {duration}"))
                continue

            if args.max_duration is not None:
                if duration is not None and duration > args.max_duration:
                    print(f"Skipping (too long > {args.max_duration}s): {path} ({duration:.2f}s)")
                    failures.append((path, f"duration {duration:.2f}s > max {args.max_duration}s"))
                    continue

            try:
                outputs = core.transcribe_video(
                    path,
                    output_file=None,
                    languages=languages,
                    skip_original=args.skip_original,
                    skip_vocal_isolation=args.skip_vocal_isolation,
                    translate_api=args.translate_api,
                    translate_mode=args.translate_mode,
                    max_translate_chars=args.max_translate_chars,
                    max_translate_calls=args.max_translate_calls,
                    overwrite=args.overwrite,
                    source_language=args.source_language,
                    engine=args.engine
                )

                print(f"Done: {path} -> {len(outputs)} output files\n")
                successes.append(path)
                total_media_duration += duration

            except Exception as exc:
                print(f"Failed for {path}: {exc}")
                failures.append((path, str(exc)))
                if not args.continue_on_error:
                    break            

        elapsed = time.time() - start_time
        print("\n\nTranscription completed.")
        print(f"Success: {len(successes)} files")
        print(f"Failed: {len(failures)} files")

        if failures:
            for fail_path, reason in failures:
                print(f" - {os.path.abspath(fail_path)}: {reason}")

        print(f"Total time: {elapsed:.2f}s")
        print(f"Total media duration processed: {utils.format_time(total_media_duration)}")

    else:
        core.transcribe_video(
            args.input_file,
            output_file=args.output_file,
            languages=languages,
            skip_original=args.skip_original,
            skip_vocal_isolation=args.skip_vocal_isolation,
            translate_api=args.translate_api,
            translate_mode=args.translate_mode,
            max_translate_chars=args.max_translate_chars,
            max_translate_calls=args.max_translate_calls,
            overwrite=args.overwrite,
            source_language=args.source_language,
            engine=args.engine
        )


if __name__ == "__main__":
    sys.exit(main())
