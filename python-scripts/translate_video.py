"""Translate Video — translated subtitle tracks.

Backs the "Translate this video" context-menu action and nothing else.

Two deliberate constraints:

* **No audio track is produced.** The old pipeline synthesised a dubbed vocal
  track with Kokoro TTS and mixed it into a re-encoded copy. Dubbing was never a
  promised feature and that code has been removed; translation means translated
  subtitles.
* **No redundant ASR pass.** If a source-language SRT already exists (because
  Generate Subtitles ran earlier) it is reused, so translating is nearly free.
  Only when there is nothing to translate does the model get loaded.

    python translate_video.py <video|folder> [vault_root] --to fr
                              [--from en] [--output PATH] [--skip-existing]
"""

import os
import sys
import tempfile

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
for _path in (_SCRIPT_DIR, _PROJECT_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from vw_media import asr, cli, enhanced, media, state, subtitles     # noqa: E402
from vw_media.progress import ScaledProgress, emit_status, log, report_progress  # noqa: E402
from vw_media.translate import Translator                            # noqa: E402

ACTION = "translation"


def find_source_subtitles(video_path, source_language):
    """Locate an existing SRT to translate, newest-usable first."""
    base = os.path.splitext(video_path)[0]
    code = subtitles.external_code(source_language)
    candidates = [f"{base}.{code}.srt", f"{base}.srt"]

    enhanced_path = state.load(video_path).get('enhancedPath')
    if enhanced_path and os.path.exists(enhanced_path):
        enhanced_base = os.path.splitext(enhanced_path)[0]
        candidates += [f"{enhanced_base}.{code}.srt", f"{enhanced_base}.srt"]

    for candidate in candidates:
        segments = subtitles.read_srt(candidate)
        if segments:
            return candidate, segments
    return None, []


def transcribe_source(video_path, source_language):
    """Fall back to ASR when there is no subtitle track to translate."""
    duration = media.get_video_duration(video_path)
    work_dir = tempfile.mkdtemp(prefix="vw_translate_")
    wav_path = os.path.join(work_dir, "audio.wav")
    try:
        media.extract_audio(
            video_path, wav_path,
            on_progress=ScaledProgress(5, 10, "Extracting audio"),
            duration=duration)
        report_progress(16, "Loading speech recognition model...")
        return asr.transcribe(wav_path, language=source_language)
    finally:
        enhanced.discard_temp(work_dir)


def resolve_srt_targets(video_path, args, target_language):
    if args.output:
        out = os.path.abspath(os.path.expanduser(args.output))
        if os.path.isdir(out) or args.output.endswith((os.sep, '/')):
            base = os.path.splitext(os.path.basename(video_path))[0]
            os.makedirs(out, exist_ok=True)
            return [os.path.join(out, f"{base}.{subtitles.external_code(target_language)}.srt")]
        os.makedirs(os.path.dirname(out), exist_ok=True)
        return [out]

    existing = state.load(video_path).get('enhancedPath')
    if not existing or not os.path.exists(existing):
        existing = None
    # include_default stays False: a translation must not overwrite the
    # source-language track that players load by default.
    return subtitles.sidecar_targets(video_path, existing, target_language)


def already_translated(video_path, args):
    return state.is_applied(video_path, ACTION,
                            language=subtitles.external_code(args.translate_to))


def process_one(video_path, args, _output_path):
    media.require_streams(video_path, need_video=False, need_audio=True)

    source_language = subtitles.source_code(args.source_language)
    target_language = args.translate_to
    target_code = subtitles.external_code(target_language)

    emit_status("STARTING", path=video_path)
    report_progress(2, f"Preparing translation to {target_code.upper()}...")

    srt_path, segments = find_source_subtitles(video_path, source_language)
    if segments:
        log(ACTION, f"Reusing existing subtitles: {srt_path} ({len(segments)} cues)")
        report_progress(20, "Reusing existing subtitle track...")
    else:
        log(ACTION, "No source subtitles found; transcribing first.")
        segments = transcribe_source(video_path, source_language)
        if not segments:
            raise RuntimeError("No speech was recognised in this file")

    report_progress(45, f"Translating {len(segments)} cues to {target_code.upper()}...")
    translator = Translator(target_language)
    translated = translator.translate_segments(
        segments, on_progress=ScaledProgress(45, 40, f"Translating to {target_code.upper()}"))

    if translator.degraded:
        log(ACTION, "Warning: some cues could not be translated and kept their source text.")

    report_progress(88, "Writing translated subtitle track...")
    written = []
    for target in resolve_srt_targets(video_path, args, target_language):
        subtitles.write_srt(target, translated)
        written.append(target)
        log(ACTION, f"Wrote {target}")

    report_progress(96, "Recording enhancement state...")
    state.mark(video_path, ACTION,
               languages=[target_code],
               params={"from": source_language, "to": target_code,
                       "reusedSubtitles": bool(srt_path), "degraded": translator.degraded},
               outputs=written)

    report_progress(100, f"Translation to {target_code.upper()} complete")
    emit_status("SUCCESS", path=written[0] if written else video_path, language=target_code)


def main():
    parser = cli.build_parser(
        "Translate video: produce translated subtitle tracks", ACTION)
    parser.add_argument("--to", "--translate-to", dest="translate_to", required=True,
                        help="Target language code, e.g. fr, es, ja (qc maps to fr)")
    parser.add_argument("--from", "--source-language", dest="source_language", default="en",
                        help="Spoken language of the source audio (default: en)")
    args = parser.parse_args()
    try:
        return cli.run(args, ACTION, process_one, needs_output=False,
                       skip_check=already_translated)
    finally:
        asr.release()


if __name__ == '__main__':
    sys.exit(main())
