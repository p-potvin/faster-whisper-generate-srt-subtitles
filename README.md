<img src="https://raw.githubusercontent.com/p-potvin/vaultwares-docs/main/logo/vaultwares-logo.svg">

# video-transcriber-translator

**Video Transcription &amp; Multi-Language Translation Tool**  
**Part of the VaultWares Ecosystem** • <a href="https://docs.vaultwares.com">docs.vaultwares.com</a> • <a href="https://vaultwares.com">vaultwares.com</a>

**Generate high-quality .srt subtitles from audio/video files and translate them into any number of languages using local models. Powers subtitles in vault-player and feeds into vaultwares-pipelines.**

## Overview
This repository (forked from faster-whisper SRT generator) provides fast, accurate, local-first transcription + translation capabilities for media files across the VaultWares ecosystem.

## Features
- High-accuracy transcription via faster-whisper
- Multi-language translation (any target languages)
- .srt subtitle output
- Batch processing support
- Integration with realtime-stt for live streams
- Seamless handoff to vault-player and vault-flows
- Agent coordination ready (via vaultwares-agentciation)

## Quick Start

```bash
git clone https://github.com/p-potvin/video-transcriber-translator.git
cd video-transcriber-translator
git submodule update --init --recursive
pip install -r requirements.txt

python transcribe_and_translate.py --input video.mp4 --languages en fr es
```

Architecture &amp; Agent Integration
Fully synchronized with the VaultWares Agent Knowledge Dissemination System:
→ https://raw.githubusercontent.com/p-potvin/vaultwares-docs/main/agents/knowledge-dissemination.mdx
Can invoke the full Google ADK-powered VaultWares agent team via invoke_vaultwares_team skill for complex translation or quality improvement tasks.
Privacy &amp; Security

100% local processing by default
No cloud APIs unless explicitly enabled
Full model weights stay on-device
Threat model available in central docs

Contributing
See CONTRIBUTING.md and the central Brand Guidelines.
License
GPL-3.0 (see LICENSE)
Built with ❤️ for privacy

