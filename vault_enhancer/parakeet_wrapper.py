import os
import logging
import subprocess
from typing import List

# Silence noisy NeMo / PyTorch / Lightning startup logs
os.environ["NEMO_LOGGING_LEVEL"] = "ERROR"
os.environ["TORCHAUDIO_DEBUG"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1" 

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nemo_logging").setLevel(logging.ERROR)
logging.getLogger("torchaudio").setLevel(logging.ERROR)
logging.getLogger("torio").setLevel(logging.ERROR)

try:
    import torch
    torch.set_warn_always(False)
except ImportError:
    pass

try:
    import lhotse
    # Permanently force the soundfile backend (bypasses FFmpeg C++ extension bugs entirely)
    lhotse.set_current_audio_backend("LibsndfileBackend")
    
    # Patch Lhotse to avoid CUDA illegal memory access during random seeding on Windows
    import lhotse.utils
    lhotse.utils.fix_random_seed = lambda seed: None
except ImportError:
    pass

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

    DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"

    # Voice-pause thresholds (tuned for natural speech)
    DEFAULT_MIN_SILENCE_S = 0.8   # gap between words that ends a segment
    DEFAULT_MIN_SEGMENT_S = 0.3   # discard segments shorter than this

    def __init__(self, model_name: str = DEFAULT_MODEL):
        import torch
        import nemo.collections.asr as nemo_asr

        self.logger = logging.getLogger("vault_enhancer.parakeet")
        self.logger.info(f"Loading Parakeet model: {model_name}")

        # Use generic ASRModel to support Canary, Parakeet-TDT, etc.
        if model_name.endswith(".nemo") and os.path.exists(model_name):
            self.logger.info(f"Loading local model file: {model_name}")
            self.model = nemo_asr.models.ASRModel.restore_from(model_name)
        else:
            self.logger.info(f"Restoring model from cache or downloading: {model_name}")
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_name)
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
        import soundfile as sf
        import os
        import numpy as np
        
        # Read the entire audio to chunk it
        data, samplerate = sf.read(audio_path)
        chunk_duration = 60  # seconds
        chunk_size = chunk_duration * samplerate
        
        all_word_timestamps = []
        chunk_paths = []
        
        try:
            # 1. Slice audio into 60-second temporary WAV files
            for i in range(0, len(data), chunk_size):
                chunk_data = data[i:i+chunk_size]
                c_path = f"{audio_path}_chunk_{len(chunk_paths)}.wav"
                sf.write(c_path, chunk_data, samplerate)
                chunk_paths.append((c_path, i / samplerate))
                  # 2. Transcribe using batch_size=1. 
            # Sequential processing (batch_size=1) is the most stable on Windows
            # and prevents the "missed words" issue caused by batch interference.
            # We avoid a manual loop to prevent CUDA illegal memory access errors.
            with torch.no_grad():
                paths_only = [p[0] for p in chunk_paths]
                # num_workers=0 is mandatory on Windows to avoid WinError 32
                torch.cuda.synchronize()
                hypotheses = self.model.transcribe(paths_only, timestamps=True, batch_size=1, num_workers=0)
            
            # Defragment once after the large batch
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            import gc
            gc.collect()

            # 3. Merge timestamps with their respective time offsets
            for (c_path, offset), hyp in zip(chunk_paths, hypotheses):
                w_timestamps = self._extract_word_timestamps(hyp)
                if w_timestamps:
                    for w in w_timestamps:
                        all_word_timestamps.append({
                            "start": w["start"] + offset,
                            "end": w["end"] + offset,
                            "word": w["word"]
                        })
        finally:
            # Cleanup temporary chunks
            for p, _ in chunk_paths:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except: pass

        if not all_word_timestamps:
            # No word timestamps available — fall back to an empty list since the chunked
            # logic already processed all valid hypotheses.
            return []

        return self._group_into_segments(all_word_timestamps, min_silence_s, min_segment_s, language)
    
    def transcribe_audio_data(
        self,
        audio_data, # This is expected to be a numpy array or similar raw audio data
        min_silence_s: float = DEFAULT_MIN_SILENCE_S,
        min_segment_s: float = DEFAULT_MIN_SEGMENT_S,
        language: str = "en",
    ) -> List[TranscriptSegment]:
        """
        Transcribe raw audio data (e.g., numpy array) and return voice-pause-segmented results.
        """
        self.logger.info(f"Transcribing audio data (min_silence={min_silence_s}s)")

        import torch
        with torch.no_grad():
            # NeMo's transcribe method can take a list of audio data (numpy arrays)
            hypotheses = self.model.transcribe([audio_data], timestamps=True)

        if not hypotheses:
            return []

        hyp = hypotheses[0]
        word_timestamps = self._extract_word_timestamps(hyp)
        return self._group_into_segments(word_timestamps, min_silence_s, min_segment_s, language)



    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_word_timestamps(self, hypothesis) -> list:
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

    def _group_into_segments(self,
        word_timestamps: list,
        min_silence_s: float,
        min_segment_s: float,
        language: str,
        max_chars: int = 180,
    ) -> List[TranscriptSegment]:
        """
        Walk through word-level timestamps and emit a new
        :class:`TranscriptSegment` whenever the gap to the next word exceeds
        *min_silence_s*.  Only segments at least *min_segment_s* long are kept.

        Any segment whose text exceeds *max_chars* is then hard-split at word
        boundaries (preferring sentence-ending punctuation) with timestamps
        interpolated linearly between the first and last word of the sub-segment.
        """
        raw_segments: List[TranscriptSegment] = []
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
                        raw_segments.append(
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
            raw_segments.append(TranscriptSegment(seg_id, seg_start, seg_end, text, language))

        # Hard-split any segment that exceeds max_chars
        return self._split_long_segments(raw_segments, max_chars, language)

    def _split_long_segments(
        self,
        segments: List[TranscriptSegment],
        max_chars: int,
        language: str,
    ) -> List[TranscriptSegment]:
        """
        Split segments whose text exceeds *max_chars* at natural word boundaries.
        Timestamps are interpolated linearly assuming uniform word duration across
        the segment span.
        """
        result: List[TranscriptSegment] = []
        seg_id = 1

        for seg in segments:
            text = seg.text
            if len(text) <= max_chars:
                result.append(TranscriptSegment(seg_id, seg.start, seg.end, text, language))
                seg_id += 1
                continue

            # Split into chunks of ≤ max_chars at word boundaries
            words = text.split()
            duration = seg.end - seg.start
            chars_total = len(text)

            chunks: List[List[str]] = []
            buf: List[str] = []
            buf_len = 0

            for word in words:
                word_len = len(word) + (1 if buf else 0)
                if buf and buf_len + word_len > max_chars:
                    # Prefer splitting after sentence-ending punctuation
                    # within the last few words of the buffer
                    split_at = len(buf)
                    for k in range(len(buf) - 1, max(len(buf) - 5, -1), -1):
                        if buf[k].endswith((".", "!", "?", ",", ";")):
                            split_at = k + 1
                            break
                    chunks.append(buf[:split_at])
                    buf = buf[split_at:] + [word]
                    buf_len = len(" ".join(buf))
                else:
                    buf.append(word)
                    buf_len += word_len

            if buf:
                chunks.append(buf)

            # Distribute timestamps proportionally by character count
            chunk_texts = [" ".join(c) for c in chunks]
            chunk_lens = [len(t) for t in chunk_texts]
            total_len = sum(chunk_lens)
            cursor = seg.start

            for chunk_text, chunk_len in zip(chunk_texts, chunk_lens):
                chunk_duration = duration * (chunk_len / total_len) if total_len > 0 else 0
                chunk_end = round(cursor + chunk_duration, 3)
                result.append(TranscriptSegment(seg_id, round(cursor, 3), chunk_end, chunk_text, language))
                seg_id += 1
                cursor = chunk_end

        return result


class ParakeetV3Wrapper:
    """
    Main interface for the application to interact with Parakeet natively (no Ray).
    """
    def __init__(self, model_name: str = "nvidia/parakeet-tdt-0.6b-v3"):
        # Load transcriber directly into the current thread
        self.transcriber = ParakeetTranscriber(model_name=model_name)
        self.logger = logging.getLogger("vaultwares.parakeet_wrapper")

    def transcribe_file(self, audio_path: str, min_silence_s: float = ParakeetTranscriber.DEFAULT_MIN_SILENCE_S, min_segment_s: float = ParakeetTranscriber.DEFAULT_MIN_SEGMENT_S, language: str = "en") -> List[TranscriptSegment]:
        return self.transcriber.transcribe_file(audio_path, min_silence_s, min_segment_s, language)

    def transcribe_audio_data(self, audio_data, min_silence_s: float = ParakeetTranscriber.DEFAULT_MIN_SILENCE_S, min_segment_s: float = ParakeetTranscriber.DEFAULT_MIN_SEGMENT_S, language: str = "en") -> List[TranscriptSegment]:
        return self.transcriber.transcribe_audio_data(audio_data, min_silence_s, min_segment_s, language)
