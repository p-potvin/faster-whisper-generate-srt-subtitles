"""Generate Subtitles — transcribe speech to SRT sidecars.

Backs the "Generate Subtitles" context-menu action and nothing else.

The important property is what it *doesn't* do: no Demucs, no video re-encode,
no enhanced copy. Previously this menu item ran the entire audio pipeline just to
reach the ASR step at the end, which meant asking for subtitles cost a full GPU
encode of the file. Here the audio is decoded straight to a 16 kHz mono WAV, fed
to the model, and written out as SRT.

    python generate_subtitles.py <video|folder> [vault_root] [--language en]
                                 [--output PATH] [--skip-existing]
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

ACTION = "subtitles"


def resolve_srt_targets(video_path, args, language):
    """Where the SRT files for *language* should be written.

    Without ``--output`` we write beside the original and beside the enhanced
    copy (when one exists), so subtitles are found whichever version plays.
    """
    if args.output:
        out = os.path.abspath(os.path.expanduser(args.output))
        if os.path.isdir(out) or args.output.endswith((os.sep, '/')):
            base = os.path.splitext(os.path.basename(video_path))[0]
            os.makedirs(out, exist_ok=True)
            return [os.path.join(out, f"{base}.{subtitles.external_code(language)}.srt")]
        os.makedirs(os.path.dirname(out), exist_ok=True)
        return [out]

    existing = state.load(video_path).get('enhancedPath')
    if not existing or not os.path.exists(existing):
        existing = None
    return subtitles.sidecar_targets(video_path, existing, language, include_default=True)


def process_one(video_path, args, _output_path):
    media.require_streams(video_path, need_video=False, need_audio=True)

    language = subtitles.source_code(args.language)
    duration = media.get_video_duration(video_path)

    emit_status("STARTING", path=video_path)
    report_progress(2, "Preparing transcription...")

    work_dir = tempfile.mkdtemp(prefix="vw_subs_")
    wav_path = os.path.join(work_dir, "audio.wav")

    try:
        media.extract_audio(
            video_path, wav_path,
            on_progress=ScaledProgress(2, 10, "Extracting audio"),
            duration=duration)

        report_progress(14, "Loading speech recognition model...")
        segments = asr.transcribe(
            wav_path, language=language,
            status_callback=lambda msg: report_progress(18, msg))

        if not segments:
            raise RuntimeError("No speech was recognised in this file")

        report_progress(88, "Writing subtitle tracks...")
        written = []
        for target in resolve_srt_targets(video_path, args, language):
            subtitles.write_srt(target, segments)
            written.append(target)
            log(ACTION, f"Wrote {target}")

        report_progress(96, "Recording enhancement state...")
        state.mark(video_path, ACTION,
                   languages=[subtitles.external_code(language)],
                   params={"language": language, "segments": len(segments)},
                   outputs=written)

        report_progress(100, f"Subtitles generated ({len(segments)} cues)")
        emit_status("SUCCESS", path=written[0] if written else video_path,
                    segments=len(segments))
    finally:
        enhanced.discard_temp(work_dir)


def main():
    parser = cli.build_parser(
        "Generate subtitles: transcribe speech to SRT sidecars", ACTION)
    parser.add_argument("--language", default="en",
                        help="Spoken language tag recorded on the subtitle track (default: en)")
    args = parser.parse_args()
    try:
        return cli.run(args, ACTION, process_one, needs_output=False)
    finally:
        asr.release()


if __name__ == '__main__':
    sys.exit(main())
