"""
vw_media — shared primitives for the Vault media enhancement scripts.

This package is deliberately self-contained and free of repo-specific imports so
the whole directory can be copied verbatim into vault-streaming,
vaultwares-media-processing and vw-cli. The only outside coupling is the ASR
wrapper lookup in :mod:`vw_media.asr`, which probes for whichever host package
happens to be present.

Each user-facing action is its own entrypoint script (enhance_audio,
generate_subtitles, translate_video, enhance_video). They share the helpers here
but never call into one another: clicking "Generate Subtitles" must not start
Demucs, and clicking "Enhance Audio" must not start the ASR model.
"""

__all__ = [
    "asr",
    "cli",
    "enhanced",
    "media",
    "progress",
    "state",
    "subtitles",
    "translate",
]
