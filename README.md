# video-transcriber-translator
Generate subtitles (SRT) from an audio/video file. It uses the `faster-whisper` library, which is much faster than OpenAI's original Whisper implementation.

It can also translate the generated subtitles into other languages.

The model `large-v2` has the best quality, while `medium` is a good compromise between quality and speed.

Read more about faster-whisper: https://github.com/guillaumekln/faster-whisper

## Quick Start
```bash
pip install -r requirements.txt
python generate-srt.py input.mp4
```

## Usage

### 1) Single file transcription (default original language)
```bash
python generate-srt.py input.mp4
```

### 2) Single file with explicit output file
```bash
python generate-srt.py "ToProcess\video.mp4" --output-file "ToProcess\video.srt"
```

### 3) Translate a full video to one language
```bash
python generate-srt.py "ToProcess\video.mp4" --languages es
```

### 4) Translate to multiple languages (one command)
```bash
python generate-srt.py "ToProcess\video.mp4" --languages en,es,fr
```

### 5) Translate only non-target segments (use autodetect per segment)
```bash
python generate-srt.py "ToProcess\video.mp4" --languages en --translate-mode non-target
```

> **Note:** The script always writes the original transcription to `video.srt`. When translation is enabled, translated subtitles are written to `video.<lang>.srt` (e.g. `video.en.srt`).
> - `--translate-mode all` translates all segments.
> - `--translate-mode non-target` keeps segments already in the target language (best-effort detection) and translates only other segments.
> - During transcription, the script shows an in-place progress spinner and elapsed seconds.
> - Default is `--vad-filter` off for broader coverage. Use `--vad-filter` to enable Whisper VAD filtering if you want fewer, more compact segments.

### 6) Scan entire library recursively and generate SRTs next to each file
```bash
python generate-srt.py --scan-dir "C:\Media\Library" --languages en,es --overwrite
```

### 7) Safe scan with per-file duration and translate limits (continue on errors)
```bash
python generate-srt.py --scan-dir "C:\Media\Library" --languages en --max-duration 3600 --max-translate-chars 200000 --max-translate-calls 150 --continue-on-error
```

### 8) Limit translation usage to avoid free-tier rate limits
```bash
python generate-srt.py "ToProcess\video.mp4" --languages en --translate-mode non-target --max-translate-chars 200000 --max-translate-calls 150
```

### 9) Install quick command wrapper on Windows
Create `generate-srt.cmd` next to `generate-srt.py`:
```cmd
@echo off
python "%~dp0generate-srt.py" %*
```
Then add that folder to PATH and run:
```powershell
generate-srt "C:\path\in.mp4" --output-file "C:\path\out.srt"
```

## Installation

Install the required dependencies using the provided requirements file:

```bash
pip install -r requirements.txt
```

# Create generate-srt.cmd next to generate-srt.py:

```bash
@echo off
python "%~dp0generate-srt.py" %*
```
# Then add this folder to your PATH (System Environment Variables).

# Now you can run from anywhere:
```bash
generate-srt "C:\path\in.mp4" --output-file "C:\path\out.srt"
```

> ⚠️ `googletrans` needs an internet connection to work (it uses Google Translate's web API).
> 
> If `googletrans` breaks on your environment, use deep-translator by adding `--translate-api deep-translator` (installed in requirements now).

### Output files
| Command | Output | Notes |
|---|---|---|
| `python generate-srt.py input.mp4` | `input.srt` | original transcription |
| `python generate-srt.py input.mp4 --languages en` | `input.srt`, `input.en.srt` | `input.en.srt` is translated; `input.srt` remains original |
| `--translate-mode non-target` | `input.srt`, `input.en.srt` | only non-target-language segments are translated when possible |


