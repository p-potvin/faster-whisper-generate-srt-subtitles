<img src="https://raw.githubusercontent.com/p-potvin/vaultwares-docs/main/logo/vaultwares-logo.svg">

# Vault Video Enhancer

**High-Performance Video Transcription & Multi-Language Translation Tool**  
Part of the VaultWares Ecosystem • [docs.vaultwares.com](https://docs.vaultwares.com) • [vaultwares.com](https://vaultwares.com)

**Generate high-quality .srt subtitles from audio/video files and translate them into any number of languages using local models. Powers subtitles in vault-player and feeds into vaultwares-pipelines.**

---

## 🚀 Features

- **Dual Engines:** High-speed transcription via `faster-whisper` or high-precision via NVIDIA `Parakeet-TDT`.
- **Vocal Isolation:** Integrated `Demucs` support for cleaner transcription in noisy environments.
- **Multi-Language:** Translate into multiple target languages simultaneously.
- **Batch Processing:** Recursive folder scanning with extension filtering.
- **VaultWares Native:** Built-in compliance with VaultWares Brand & Privacy guidelines.

---

## 🛠️ Usage

### Command Line Interface

```bash
python enhancer.py --input video.mp4 --languages en,fr,es --engine parakeet
```

#### Advanced Options

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--input_file` | None | Path to a single video/audio file. |
| `--scan-dir` | `to_process` | Recursively scan this directory for media. |
| `--languages` | `en` | Comma-separated target languages (e.g., `en,fr,es`). |
| `--engine` | `parakeet` | Transcription engine: `whisper` or `parakeet`. |
| `--source-language` | `None` | Force a specific source language to prevent hallucinations. |
| `--skip-original` | `False` | Do not generate the original language SRT file. |
| `--skip-vocal-isolation` | `False` | Skip Demucs vocal isolation (faster but noisier). |
| `--translate-api` | `deep-translator` | Backend: `googletrans` or `deep-translator`. |
| `--translate-mode` | `all` | `all` (translate everything) or `non-target` (detected != target). |
| `--max-translate-chars` | `1000000` | Skip translation if total chars exceed this limit. |
| `--max-translate-calls` | `500` | Max translator calls per video. |
| `--max-duration` | `7200` | Skip media files longer than this (in seconds). |
| `--continue-on-error` | `False` | Continue to the next file if one fails in scan mode. |
| `--overwrite` | `False` | Overwrite existing SRT files. |
| `--extensions` | `.mp4,.mkv...` | Comma-separated extensions for scan mode. |

### Graphical User Interface

A native **PySide6** GUI is provided for an interactive experience:

```bash
python vault_gui.py
```

---

## 🔒 Privacy & Security

- **100% Local Processing:** Model weights stay on-device.
- **No Telemetry:** No tracking, no data exfiltration.
- **Bilingual by Design:** Fully supports English and French UI and processing.

---

## 🎨 Branding

The **VaultWares Theme** design system has been added as a git submodule to `vault-themes/`.

When incorporating logos (such as favicons or when the full logo doesn't fit), use the **minimal logos** which feature only the "V" part of the logo.

- **Minimal Logos:** Located at `vault-themes/Brand/minimal-logos/`
- **Favicons:** Located at `vault-themes/Brand/favicons/`

**Note:** The `-ink-filled`, `-mono-filled`, and `gold-filled` versions should be the default variations used.

---

## 🤝 Contributing

See `CONTRIBUTING.md` and the central **VaultWares Brand Guidelines**.

---

*Built with ❤️ for privacy and accuracy by the VaultWares team.*
