import os
import torch
import numpy as np
import logging
import subprocess
from typing import Optional

# Monkey-patch subprocess.Popen to NEVER open a console window on Windows
if os.name == 'nt':
    _original_popen = subprocess.Popen
    class _HushPopen(_original_popen):
        def __init__(self, *args, **kwargs):
            if 'creationflags' not in kwargs:
                kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW | getattr(subprocess, 'DETACHED_PROCESS', 0x00000008)
            super().__init__(*args, **kwargs)
    subprocess.Popen = _HushPopen

# Check for CUDA availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ParakeetWorker:
    """
    Dedicated worker for running NVIDIA Parakeet V3 / Canary V2 models.
    """
    def __init__(self, model_name: str = "nvidia/canary-1b"):
        import nemo.collections.asr as nemo_asr
        self.logger = logging.getLogger("vaultwares.parakeet_worker")
        self.logger.info(f"Initializing ParakeetWorker with model: {model_name} on {DEVICE}")
        
        # Load the model (Canary-1B supports EN, DE, ES, FR ASR and Translation)
        try:
            self.model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(model_name=model_name)
            self.model = self.model.to(DEVICE)
            self.model.eval()
            self.logger.info("Parakeet/Canary model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load Parakeet model: {e}")
            raise

    def transcribe(self, audio_data: np.ndarray, source_lang: str = "en", target_lang: str = "en") -> str:
        """
        Transcribes a NumPy audio buffer (16kHz float32).
        """
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

    def transcribe(self, audio_data: np.ndarray, source_lang: str = "en", target_lang: str = "en") -> str:
        """
        Synchronous dispatch transcription to the worker.
        """
        return self.worker.transcribe(audio_data, source_lang, target_lang)

    def shutdown(self):
        pass
