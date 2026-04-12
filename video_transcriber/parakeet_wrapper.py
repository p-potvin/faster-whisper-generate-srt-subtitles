import os
import logging
import subprocess
from typing import List

# Monkey-patch subprocess.Popen to NEVER open a console window on Windows
if os.name == 'nt':
    _original_popen = subprocess.Popen
    class _HushPopen(_original_popen):
        def __init__(self, *args, **kwargs):
            if 'creationflags' not in kwargs:
                kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW | getattr(subprocess, 'DETACHED_PROCESS', 0x00000008)
            super().__init__(*args, **kwargs)
    subprocess.Popen = _HushPopen


class TranscriptSegment:
    """
    Segment object compatible with the write_srt / translation pipeline.
    Mirrors the attributes used from faster_whisper segments.
    """
    def __init__(self, id: int, start: float, end: float, text: str, language: str = "en"):
        self.id = id
        self.start = start
        self.end = end
        self.text = text
        self.language = language

    def __repr__(self):
        return f"TranscriptSegment(id={self.id}, start={self.start:.2f}, end={self.end:.2f}, text={self.text!r})"


class ParakeetTranscriber:
    """
    ASR wrapper around NVIDIA Parakeet-TDT-1.1B using NeMo.

    Parakeet-TDT produces true word-level timestamps (start/end in seconds per
    recognised word).  Segment boundaries are derived from actual *voice* pauses
    — gaps between consecutive recognised words — so background noise that is
    not decoded as speech never prevents a segment from ending.  This is the
    fundamental fix for the issue where the previous Silero-VAD approach
    treated any audible sound as "not silence" and kept segments open.

    Flow
    ----
    1. Load ``nvidia/parakeet-tdt-1.1b`` (English, TDT decoder, ~1.1 B params).
    2. Transcribe the audio file with ``timestamps=True`` to obtain per-word
       start/end times.
    3. Walk through the words and start a new segment whenever the gap between
       the *end* of the last word and the *start* of the next word exceeds
       ``min_silence_s``.  That gap is genuine voice absence, not just low dB.
    4. Return a list of :class:`TranscriptSegment` objects ready for SRT output
       and the translation pipeline.
    """

    DEFAULT_MODEL = "nvidia/parakeet-tdt-1.1b"

    # Voice-pause thresholds (tuned for natural speech)
    DEFAULT_MIN_SILENCE_S = 0.8   # gap between words that ends a segment
    DEFAULT_MIN_SEGMENT_S = 0.3   # discard segments shorter than this

    def __init__(self, model_name: str = DEFAULT_MODEL):
        import torch
        import nemo.collections.asr as nemo_asr

        self.logger = logging.getLogger("video_transcriber.parakeet")
        self.logger.info(f"Loading Parakeet model: {model_name}")

        self.model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

        self.logger.info("Parakeet model loaded successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe_file(
        self,
        audio_path: str,
        min_silence_s: float = DEFAULT_MIN_SILENCE_S,
        min_segment_s: float = DEFAULT_MIN_SEGMENT_S,
        language: str = "en",
    ) -> List[TranscriptSegment]:
        """
        Transcribe *audio_path* and return voice-pause-segmented results.

        Parameters
        ----------
        audio_path:
            Path to a 16 kHz mono WAV file.
        min_silence_s:
            Minimum gap between consecutive recognised words (in seconds) that
            triggers a segment boundary.  This is true **voice** silence — not
            merely low audio level — because Parakeet only emits timestamps for
            decoded words.
        min_segment_s:
            Segments shorter than this (seconds) are discarded.
        language:
            Language tag stored on each returned segment (used by the
            translation layer to decide whether to translate).
        """
        self.logger.info(f"Transcribing: {audio_path}  (min_silence={min_silence_s}s)")

        import torch
        with torch.no_grad():
            hypotheses = self.model.transcribe([audio_path], timestamps=True)

        if not hypotheses:
            return []

        hyp = hypotheses[0]
        word_timestamps = self._extract_word_timestamps(hyp)

        if not word_timestamps:
            # No word timestamps available — fall back to a single segment.
            # End time is left as 0.0 and should be treated as a best-effort
            # placeholder; Parakeet-TDT always returns timestamps so this path
            # is only reached if an unexpected model variant is used.
            text = hyp.text if hasattr(hyp, "text") else str(hyp)
            if text.strip():
                return [TranscriptSegment(1, 0.0, 0.0, text.strip(), language)]
            return []
        if audio_data is None or len(audio_data) == 0:
            return ""

        try:
            # NeMo models often expect a manifest or a file path, 
            # but we can pass raw audio tensors to some methods.
            # Using the transcribe method with a list of raw audio
            with torch.no_grad():
                # Wrap audio in a list
                # Note: Canary-1B MultiTaskModel expects specific task/lang tags
                # For basic ASR, source_lang == target_lang
                # We use the internal _transcribe or similar if available for raw tensors
                
                # Simple implementation: write to a temporary buffer or use NeMo's streaming API
                # For this wrapper, we use the standard transcribe interface
                # which handles the internal preprocessing
                
                # Convert to torch tensor
                audio_tensor = torch.from_numpy(audio_data).to(DEVICE)
                
                # Canary-1B specific: requires taskname, source_lang, target_lang
                # We'll use the model's high-level transcribe method which can take audio paths
                # To avoid disk IO, we'd ideally use the frame-based forward pass, 
                # but for simplicity in this V1 wrapper, we'll use a memory-efficient path.
                
                # For now, let's use the standard transcribe which is robust
                # (Optimization: use NeMo's Streaming ASR buffers for V2)
                
                # Note: Parakeet-TDT (v3) is significantly faster for this.
                # Canary-1B MultiTask ASR requires task and language settings
                # We also provide 'pnc' (punctuation and capitalization) as a literal string
                results = self.model.transcribe(
                    [audio_data], 
                    batch_size=1,
                    task="asr",
                    pnc="yes",
                    source_lang=source_lang,
                    target_lang=target_lang,
                    verbose=False
                )
                
                # Check for list or raw result
                if isinstance(results, list) and len(results) > 0:
                    res = results[0]
                    return res if isinstance(res, str) else (res.text if hasattr(res, 'text') else str(res))
                return str(results) if results else ""
                
        except Exception as e:
            self.logger.error(f"Parakeet transcription error: {e}")
            return f"[Error: {e}]"

class ParakeetV3Wrapper:
    """
    Main interface for the application to interact with Parakeet natively (no Ray).
    """
    def __init__(self, model_name: str = "nvidia/canary-1b"):
        # Load worker directly into current thread
        self.worker = ParakeetWorker(model_name=model_name)
        self.logger = logging.getLogger("vaultwares.parakeet_wrapper")

        return self._group_into_segments(word_timestamps, min_silence_s, min_segment_s, language)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_word_timestamps(hypothesis) -> list:
        """
        Extract a normalised list of ``{'word', 'start', 'end'}`` dicts from a
        NeMo hypothesis object.  Handles both the direct-seconds format used by
        TDT models and the frame-offset format used by some older models.
        """
        ts = getattr(hypothesis, "timestamp", None)
        if not ts:
            return []

        word_list = ts.get("word", []) if isinstance(ts, dict) else []
        if not word_list:
            return []

        # Normalise: TDT models already give seconds; CTC/RNNT may give frame
        # offsets.  Detect by checking whether values look like small integers
        # (frames) vs. floats (seconds).
        first = word_list[0]
        if "start" in first and "end" in first:
            # Already in seconds
            return word_list

        if "start_offset" in first and "end_offset" in first:
            # Frame offsets — convert using model window stride (typically 0.01s
            # for standard NeMo ASR preprocessors; parakeet-tdt-1.1b returns
            # seconds directly so this branch is only reached for CTC/RNNT
            # variants). Stride should match the model's window_stride config.
            stride = 0.01  # 10 ms hop
            return [
                {
                    "word": w["word"],
                    "start": w["start_offset"] * stride,
                    "end": w["end_offset"] * stride,
                }
                for w in word_list
            ]

        return []

    @staticmethod
    def _group_into_segments(
        word_timestamps: list,
        min_silence_s: float,
        min_segment_s: float,
        language: str,
    ) -> List[TranscriptSegment]:
        """
        Walk through word-level timestamps and emit a new
        :class:`TranscriptSegment` whenever the gap to the next word exceeds
        *min_silence_s*.  Only segments at least *min_segment_s* long are kept.
        """
        segments: List[TranscriptSegment] = []
        seg_id = 1
        buf_words: List[str] = []
        seg_start: float = 0.0
        seg_end: float = 0.0

        for i, w in enumerate(word_timestamps):
            word = w["word"]
            w_start = float(w["start"])
            w_end = float(w["end"])

            if not buf_words:
                seg_start = w_start
                buf_words.append(word)
                seg_end = w_end
            else:
                gap = w_start - seg_end
                if gap >= min_silence_s:
                    # Voice pause detected — flush current segment
                    if (seg_end - seg_start) >= min_segment_s:
                        text = " ".join(buf_words)
                        segments.append(
                            TranscriptSegment(seg_id, seg_start, seg_end, text, language)
                        )
                        seg_id += 1
                    # Begin fresh segment
                    seg_start = w_start
                    buf_words = [word]
                    seg_end = w_end
                else:
                    buf_words.append(word)
                    seg_end = w_end

        # Flush the final pending segment
        if buf_words and (seg_end - seg_start) >= min_segment_s:
            text = " ".join(buf_words)
            segments.append(TranscriptSegment(seg_id, seg_start, seg_end, text, language))

        return segments
