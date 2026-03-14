import argparse
from faster_whisper import WhisperModel
import os


def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    milliseconds = (seconds - int(seconds)) * 1000
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(milliseconds):03d}"


def _get_translator():
    """Return a translator instance.

    Uses `googletrans` to translate text into different languages.
    """
    try:
        from googletrans import Translator
    except ImportError as e:
        raise ImportError(
            "Missing translator dependency. Install it with: `pip install googletrans==4.0.0-rc1`"
        ) from e

    return Translator()


def _translate_segments(texts, target_lang: str):
    translator = _get_translator()
    translated = translator.translate(texts, dest=target_lang)

    # `googletrans` can return a single object or a list depending on input
    if isinstance(translated, list):
        return [t.text for t in translated]
    return [translated.text]


def transcribe_video(input_file, translate_to=None):
    model_size = "medium"  # other models: small, medium, large-v2, tiny

    # Run on CPU with INT8
    model = WhisperModel(model_size, device="cpu", cpu_threads=12, compute_type="int8")

    # If the user wants English, we can use faster-whisper's built-in translate task.
    task = "translate" if translate_to == "en" else "transcribe"
    all_segments, info = model.transcribe(input_file, beam_size=5, task=task, vad_filter=True)
    all_segments = list(all_segments)

    print("Detected language '{}' with probability {:.2f}".format(info.language, info.language_probability))

    texts = [segment.text for segment in all_segments]
    if translate_to and translate_to != "en":
        texts = _translate_segments(texts, translate_to)

    srt_filename = os.path.splitext(input_file)[0] + '.srt'

    with open(srt_filename, 'w', encoding='utf-8') as srt_file:
        for segment, text in zip(all_segments, texts):
            start_time = format_time(segment.start)
            end_time = format_time(segment.end)
            segment_id = segment.id + 1
            line_out = f"{segment_id}\n{start_time} --> {end_time}\n{text.lstrip()}\n\n"
            print(line_out)
            srt_file.write(line_out)
            srt_file.flush()  # flush so we don't lose data if it crashes midway


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio from a video file and generate an SRT file.")
    parser.add_argument("input_file", help="Path to the video file for transcription")
    parser.add_argument(
        "--translate-to",
        default=None,
        help=(
            "Translate the generated subtitles to a target language (ISO 639-1 code, e.g. 'en', 'es', 'fr'). "
            "If 'en' is selected, the faster-whisper built-in translate task is used."
        ),
    )

    args = parser.parse_args()
    transcribe_video(args.input_file, translate_to=args.translate_to)


if __name__ == "__main__":
    main()
