"""SRT sidecar writing and subtitle language-code normalisation.

Self-contained on purpose: the monolith imported ``vault_explorer.utils`` for
``write_srt``, which is exactly the kind of host coupling that stops this
directory from being copied into the other repos as-is.
"""

import os


def external_code(language):
    """Map an internal language tag to the code players expect in a filename.

    Quebec French has no distinct subtitle code — it ships as ``fr``.
    """
    code = str(language or '').strip().lower()
    return 'fr' if code in {'qc', 'fr-ca', 'ca-fr'} else code


def source_code(language):
    """Map an internal tag to the code the translation backend expects."""
    code = str(language or '').strip().lower()
    return 'fr' if code == 'qc' else code


def format_timestamp(seconds):
    if seconds is None or seconds < 0:
        seconds = 0.0
    ms = int(round((seconds - int(seconds)) * 1000))
    total = int(seconds)
    hours, total = divmod(total, 3600)
    minutes, secs = divmod(total, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_srt(output_path, segments):
    """Write *segments* (dicts with start/end/text) as an SRT file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as fh:
        index = 0
        for seg in segments:
            text = str(seg.get('text', '')).strip()
            if not text:
                continue
            index += 1
            fh.write(
                f"{index}\n"
                f"{format_timestamp(seg.get('start', 0.0))} --> {format_timestamp(seg.get('end', 0.0))}\n"
                f"{text}\n\n"
            )
    return output_path


def read_srt(path):
    """Parse an SRT back into segment dicts, or return [] if unreadable.

    Used so translation can reuse subtitles that were already generated instead
    of paying for a second ASR pass.
    """
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            raw = fh.read()
    except Exception:
        return []

    segments = []
    for block in raw.replace('\r\n', '\n').split('\n\n'):
        lines = [l for l in block.split('\n') if l.strip()]
        if len(lines) < 2:
            continue
        timing = next((l for l in lines if '-->' in l), None)
        if not timing:
            continue
        try:
            start_raw, end_raw = [p.strip() for p in timing.split('-->')[:2]]
            text = ' '.join(lines[lines.index(timing) + 1:]).strip()
            if text:
                segments.append({
                    'start': _parse_timestamp(start_raw),
                    'end': _parse_timestamp(end_raw),
                    'text': text,
                })
        except Exception:
            continue
    return segments


def _parse_timestamp(value):
    value = value.replace(',', '.')
    hours, minutes, secs = value.split(':')
    return int(hours) * 3600 + int(minutes) * 60 + float(secs)


def sidecar_targets(video_path, enhanced_path, language, include_default=False):
    """Every SRT path a given language should be written to.

    Both the original and the enhanced copy get a sidecar so subtitles show up
    whichever version the user plays.

    *include_default* additionally writes the extension-less ``<base>.srt`` that
    players load when no language is chosen. Only the transcription pass sets it
    — a translation must never clobber the source-language default track.
    """
    code = external_code(language)
    bases = [os.path.splitext(video_path)[0]]
    if enhanced_path and os.path.abspath(enhanced_path) != os.path.abspath(video_path):
        bases.append(os.path.splitext(enhanced_path)[0])

    targets = []
    for base in bases:
        targets.append(f"{base}.{code}.srt")
        if include_default:
            targets.append(f"{base}.srt")
    return targets
