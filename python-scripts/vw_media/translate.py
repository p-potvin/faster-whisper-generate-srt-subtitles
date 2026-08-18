"""Subtitle text translation.

Text only. There is deliberately no speech synthesis here: dubbed audio tracks
were never a product feature, and the Kokoro TTS path that used to live in the
audio pipeline has been removed from the codebase.
"""

from .progress import log
from .subtitles import source_code


class Translator:
    """Lazy GoogleTranslator with a per-run cache.

    Caching matters because subtitle tracks repeat short lines constantly, and
    each miss is a network round trip.
    """

    def __init__(self, target):
        self.target = source_code(target)
        self._impl = None
        self._cache = {}
        self._failed = False

    def _ensure(self):
        if self._impl is None:
            from deep_translator import GoogleTranslator
            self._impl = GoogleTranslator(source='auto', target=self.target)
        return self._impl

    def translate(self, text):
        """Translate *text*, falling back to the source string on failure.

        Returning the source text keeps a partially-translated track usable; the
        alternative (a placeholder like ``[fr]: ...``) writes junk into a file the
        user will actually read.
        """
        if not text or not text.strip():
            return text
        if text in self._cache:
            return self._cache[text]
        try:
            result = self._ensure().translate(text) or text
        except Exception as err:
            if not self._failed:
                log("translate", f"Translation failed ({err}); keeping source text")
                self._failed = True
            result = text
        self._cache[text] = result
        return result

    def translate_segments(self, segments, on_progress=None):
        """Translate a list of segment dicts, preserving timings."""
        out = []
        total = max(1, len(segments))
        for idx, seg in enumerate(segments):
            out.append({
                'start': seg.get('start', 0.0),
                'end': seg.get('end', 0.0),
                'text': self.translate(seg.get('text', '')),
            })
            if on_progress:
                on_progress((idx + 1) / total * 100.0)
        return out

    @property
    def degraded(self):
        """True if any translation call fell back to the source text."""
        return self._failed
