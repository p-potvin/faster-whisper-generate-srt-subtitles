import os
import logging
import subprocess
import shutil
from typing import List

# Pre-import datasets to resolve the Windows MKL/OpenMP PyTorch/PyArrow import conflict causing silent exit code 1
try:
    import datasets
except Exception:
    pass

# Patch shutil.rmtree on Windows to ignore PermissionErrors during temp directory cleanup
# which avoids WinError 32 from NeMo's temporary manifest.json locks.
original_rmtree = shutil.rmtree
def patched_rmtree(path, *args, **kwargs):
    try:
        return original_rmtree(path, *args, **kwargs)
    except PermissionError:
        if "temp" in path.lower() or "tmp" in path.lower():
            pass
        else:
            raise
shutil.rmtree = patched_rmtree


# Silence noisy NeMo / PyTorch / Lightning startup logs
import warnings
warnings.filterwarnings("ignore")

os.environ["NEMO_LOGGING_LEVEL"] = "ERROR"
os.environ["TORCHAUDIO_DEBUG"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1" 

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nemo_logging").setLevel(logging.ERROR)
logging.getLogger("torchaudio").setLevel(logging.ERROR)
logging.getLogger("torio").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

try:
    from nemo.utils import logging as nemo_logging
    nemo_logging.setLevel(nemo_logging.ERROR)
except ImportError:
    pass

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
    ASR wrapper around ``nvidia/parakeet-tdt-0.6b-v3`` using NeMo.

    Parakeet-TDT produces true word-level timestamps (start/end in seconds per
    recognised word).  Segment boundaries are derived from actual *voice* pauses
    — gaps between consecutive recognised words — so background noise that is
    not decoded as speech never prevents a segment from ending.  This is the
    fundamental fix for the issue where the previous Silero-VAD approach
    treated any audible sound as "not silence" and kept segments open.

    Flow
    ----
    1. Load ``nvidia/parakeet-tdt-0.6b-v3`` (multilingual, ~600 M params).
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

    # Places a converted model directory may live, in preference order. The
    # shared store is last so a project-local copy always wins, but its presence
    # means every project finds the same converted model without any per-repo
    # configuration — the hard links are a disk-space optimisation, not a
    # requirement for the lookup to work.
    @staticmethod
    def _converted_search_roots():
        roots = [os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools", "models"))]
        model_dir = os.environ.get("VAULT_MODEL_DIR")
        if model_dir:
            roots.append(model_dir)
        store = os.environ.get("VW_MODEL_STORE")
        if store:
            roots.append(store)
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            roots.append(os.path.join(local_appdata, "VaultWares", "models"))
        return roots

    def __init__(self, model_name: str = DEFAULT_MODEL, status_callback=None):
        import time
        t_import_start = time.perf_counter()
        import torch
        import nemo.collections.asr as nemo_asr
        t_import_end = time.perf_counter()

        print(f"[ASR Telemetry] Library Import Latency: {t_import_end - t_import_start:.4f} seconds")

        self.logger = logging.getLogger("vaultwares_media_processing.parakeet")
        self.logger.info(f"Loading Parakeet model: {model_name}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        t_load_start = time.perf_counter()

        # Fast path: a directory produced by tools/convert_nemo_to_safetensors.py.
        # Building the modules with empty (meta) parameters and then streaming
        # safetensors straight to the GPU skips both the tar unpack and the
        # torch.load pickle, and never materialises the 2.4 GB state dict in
        # host RAM. Measured on parakeet-tdt-0.6b-v3: 45.8s / +2699 MB RSS ->
        # 6.5s / +85 MB, with every one of the 725 tensors bit-identical.
        self.model = self._load_converted(model_name, status_callback)

        if self.model is None:
            self.model = self._load_nemo_archive(model_name, nemo_asr, status_callback)

        self.model.eval()
        t_load_end = time.perf_counter()

        # Hand back anything the load left behind before transcription starts.
        self._release_load_scratch()

        print(f"[ASR Telemetry] Model Loading & Device Allocation: {t_load_end - t_load_start:.4f} seconds")
        print(f"[ASR Telemetry] Combined Engine Initialization: {time.perf_counter() - t_import_start:.4f} seconds")

        self.logger.info("Parakeet model loaded successfully.")

    # ------------------------------------------------------------------
    # Loading strategies
    # ------------------------------------------------------------------

    def _converted_dir_for(self, model_name: str):
        """Locate a converted model directory for *model_name*, if one exists."""
        base = os.path.basename(str(model_name)).replace(".nemo", "")
        for root in self._converted_search_roots():
            candidate = os.path.join(root, base)
            if (os.path.isfile(os.path.join(candidate, "model_config.yaml"))
                    and os.path.isfile(os.path.join(candidate, "model.safetensors"))):
                return candidate
        return None

    def _load_converted(self, model_name: str, status_callback=None):
        """Build on meta parameters, then stream safetensors onto the device.

        Returns None (having logged why) if anything about the fast path does not
        hold, so the caller can fall back to the stock NeMo restore.
        """
        converted = self._converted_dir_for(model_name)
        if not converted:
            return None

        try:
            import torch
            from omegaconf import OmegaConf, open_dict
            from nemo.utils import model_utils
            from accelerate import init_empty_weights
            from safetensors import safe_open

            if status_callback:
                status_callback("Step 0: Loading model weights (safetensors)...")
            self.logger.info(f"Loading from converted model dir: {converted}")

            cfg = OmegaConf.load(os.path.join(converted, "model_config.yaml"))
            target = cfg.get("target")
            if not target:
                raise RuntimeError("model_config.yaml has no `target` class path")

            # Asset references are stored as bare filenames so the directory can
            # be moved or hard-linked between projects. Resolve them against the
            # directory the config actually lives in.
            self._resolve_asset_paths(cfg, converted)

            # The dataset sections have to come off before construction — the
            # released config has no manifest_filepath, so NeMo raises while
            # setting up dataloaders — but transcribe() later clones
            # `validation_ds` to build its temporary dataloader. So: pop them to
            # construct, then put them back on the model's own config.
            dataset_cfg = {}
            with open_dict(cfg):
                for key in ("train_ds", "validation_ds", "test_ds"):
                    if key in cfg:
                        dataset_cfg[key] = cfg.pop(key)

            # torch.device('meta') is too broad here — NeMo's ConvSubsampling
            # calls .item() while computing its output length, which meta tensors
            # cannot do. init_empty_weights redirects only parameters.
            cls = model_utils.import_class_by_path(target)
            with init_empty_weights(include_buffers=False):
                model = cls(cfg=cfg, trainer=None)

            with open_dict(model.cfg):
                for key, value in dataset_cfg.items():
                    model.cfg[key] = value

            state = {}
            with safe_open(os.path.join(converted, "model.safetensors"),
                           framework="pt", device=self.device) as handle:
                for key in handle.keys():
                    state[key] = handle.get_tensor(key)

            missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
            state.clear()

            if unexpected:
                raise RuntimeError(f"{len(unexpected)} unexpected weights, e.g. {unexpected[:3]}")

            leftover = self._materialise_generated_buffers(model)
            if leftover:
                raise RuntimeError(f"parameters left uninitialised: {leftover[:5]}")

            # `missing` is expected to hold only non-persistent generated buffers,
            # which the step above rebuilt. A missing real parameter is fatal.
            real_missing = [n for n, p in model.named_parameters() if p.is_meta]
            if real_missing:
                raise RuntimeError(f"parameters left on meta: {real_missing[:5]}")

            model.to(self.device)
            self.logger.info(
                f"Loaded {len(missing) and 'with regenerated buffers' or 'cleanly'} "
                f"from safetensors on {self.device}")
            return model

        except Exception as err:
            # A converted directory that cannot be used is a performance
            # regression, not a failure — fall back and say so loudly enough to
            # be actionable.
            self.logger.warning(
                f"Fast safetensors load failed ({err}); falling back to the .nemo archive. "
                f"Re-run tools/convert_nemo_to_safetensors.py --force to rebuild it.")
            print(f"[ASR Telemetry] Fast load unavailable: {err}")
            self._release_load_scratch()
            return None

    @staticmethod
    def _resolve_asset_paths(cfg, converted_dir):
        """Turn bare asset filenames in *cfg* into absolute paths.

        Walks the config and rewrites any string that names a file sitting in
        *converted_dir* (tokenizer model, vocab, SPE vocab). Keeping these
        relative on disk is what lets one converted directory be hard-linked
        into several projects without each needing its own edited config.
        """
        from omegaconf import DictConfig, ListConfig, open_dict

        present = {name for name in os.listdir(converted_dir)}

        def walk(node):
            if isinstance(node, DictConfig):
                with open_dict(node):
                    for key in list(node.keys()):
                        value = node.get(key)
                        if isinstance(value, str) and value in present:
                            node[key] = os.path.join(converted_dir, value).replace("\\", "/")
                        elif isinstance(value, (DictConfig, ListConfig)):
                            walk(value)
            elif isinstance(node, ListConfig):
                for index, value in enumerate(node):
                    if isinstance(value, str) and value in present:
                        node[index] = os.path.join(converted_dir, value).replace("\\", "/")
                    elif isinstance(value, (DictConfig, ListConfig)):
                        walk(value)

        walk(cfg)
        return cfg

    @staticmethod
    def _materialise_generated_buffers(model, max_len: int = 5000):
        """Rebuild buffers that are computed rather than stored.

        ``encoder.pos_enc.pe`` is registered with ``persistent=False``, so it is
        absent from the checkpoint by design and stays on meta after loading.
        NeMo regenerates it via ``extend_pe``; doing it here means the first
        transcription does not have to.
        """
        import torch

        leftovers = [n for n, b in model.named_buffers() if b is not None and b.is_meta]
        for name in leftovers:
            owner = model
            parts = name.split('.')
            for part in parts[:-1]:
                owner = getattr(owner, part)
            leaf = parts[-1]
            if hasattr(owner, 'extend_pe'):
                # extend_pe short-circuits on hasattr(self, 'pe'), so the meta
                # buffer has to be removed outright rather than set to None.
                owner._buffers.pop(leaf, None)
                device = next(model.parameters()).device
                owner.extend_pe(max_len, device, torch.float32)

        return [n for n, b in model.named_buffers() if b is not None and b.is_meta]

    def _load_nemo_archive(self, model_name: str, nemo_asr, status_callback=None):
        """Stock NeMo restore: slower, but works without a converted directory."""
        if not (model_name.endswith(".nemo") and os.path.exists(model_name)):
            import glob
            candidates = []
            model_dir = os.environ.get("VAULT_MODEL_DIR")
            if model_dir:
                candidates.append(os.path.join(model_dir, "parakeet-tdt-0.6b-v3.nemo"))
            candidates += glob.glob(os.path.expanduser(
                "~/.cache/huggingface/hub/models--nvidia--parakeet-tdt-0.6b-v3/snapshots/*/*.nemo"))
            candidates += glob.glob(os.path.expanduser(
                "~/.cache/huggingface/hub/parakeet-tdt-0.6b-v3/snapshots/*/*.nemo"))

            local_model = next((c for c in candidates if c and os.path.exists(c)), None)
            if local_model:
                model_name = local_model
                self.logger.info(f"Found local model: {model_name}")

        if model_name.endswith(".nemo") and os.path.exists(model_name):
            if status_callback:
                status_callback("Step 0: Restoring model from local path...")
            self.logger.info(f"Loading model from local .nemo archive: {model_name}")
            model = nemo_asr.models.ASRModel.restore_from(model_name, map_location=self.device)
        else:
            if status_callback:
                status_callback("Step 0: Loading model from cache/remote...")
            self.logger.info(f"Restoring model from cache or downloading: {model_name}")
            model = nemo_asr.models.ASRModel.from_pretrained(model_name, map_location=self.device)

        if self.device == "cuda":
            model = model.cuda()
        return model

    @staticmethod
    def _release_load_scratch():
        """Drop loader temporaries so they do not sit in RAM for the whole run."""
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

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
        # Downmix multi-channel audio to mono if necessary
        if len(data.shape) > 1 and data.shape[1] > 1:
            data = data.mean(axis=1)
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
        
        import numpy as np
        if isinstance(audio_data, np.ndarray) and len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = audio_data.mean(axis=1)

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
    def __init__(self, model_name: str = "nvidia/parakeet-tdt-0.6b-v3", status_callback=None):
        # Load transcriber directly into the current thread
        self.transcriber = ParakeetTranscriber(model_name=model_name, status_callback=status_callback)
        self.logger = logging.getLogger("vaultwares.parakeet_wrapper")

    def transcribe_file(self, audio_path: str, min_silence_s: float = ParakeetTranscriber.DEFAULT_MIN_SILENCE_S, min_segment_s: float = ParakeetTranscriber.DEFAULT_MIN_SEGMENT_S, language: str = "en") -> List[TranscriptSegment]:
        return self.transcriber.transcribe_file(audio_path, min_silence_s, min_segment_s, language)

    def transcribe_audio_data(self, audio_data, min_silence_s: float = ParakeetTranscriber.DEFAULT_MIN_SILENCE_S, min_segment_s: float = ParakeetTranscriber.DEFAULT_MIN_SEGMENT_S, language: str = "en") -> List[TranscriptSegment]:
        return self.transcriber.transcribe_audio_data(audio_data, min_silence_s, min_segment_s, language)
