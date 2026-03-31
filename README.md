# EPUB to Audiobook

Convert any EPUB file into an M4B audiobook using [Microsoft VibeVoice](https://github.com/microsoft/VibeVoice) TTS. Runs locally on your Mac (Apple Silicon via MPS), NVIDIA GPU (CUDA), or CPU — no cloud APIs, no subscriptions.

## Features

- Parses EPUB chapters automatically, skipping TOC/copyright/cover pages
- Generates natural-sounding speech with customizable voice presets
- Resumes interrupted conversions (skips already-generated chapters)
- Assembles chapters into a single `.m4b` audiobook with ffmpeg
- Cleans text for TTS (removes URLs, normalizes quotes, handles whitespace)

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager
- [ffmpeg](https://ffmpeg.org/) (`brew install ffmpeg`)
- A [Hugging Face](https://huggingface.co/settings/tokens) token (recommended, for faster model downloads)

## Setup

```bash
# Clone and enter the project
git clone <repo-url> && cd epub-to-audiobook

# Install dependencies
uv sync

# Set your Hugging Face token
echo 'HF_TOKEN=hf_your_token_here' > .env
```

### Download Voice Files

VibeVoice uses `.pt` speaker embedding files to define voices. Download the experimental presets:

```bash
mkdir -p voices && cd voices
curl -LO https://raw.githubusercontent.com/microsoft/VibeVoice/main/demo/download_experimental_voices.sh
bash download_experimental_voices.sh
cd ..
```

Available voices: `en1`, `en2` (English), `de` (German), `fr` (French), `sp` (Spanish), `pt` (Portuguese), `pl` (Polish), `jp` (Japanese), `kr` (Korean).

## Usage

```bash
# Default voice (en1)
uv run --env-file .env python epub_to_audiobook.py book.epub

# Choose a voice by name
uv run --env-file .env python epub_to_audiobook.py book.epub en2

# Use a custom voice file
uv run --env-file .env python epub_to_audiobook.py book.epub ./my_voice.pt

# Start from a specific chapter
uv run --env-file .env python epub_to_audiobook.py book.epub en1 --start 5

# Adjust generation guidance (default: 1.5)
uv run --env-file .env python epub_to_audiobook.py book.epub en1 --cfg-scale 1.3
```

Output is saved to `<book-name>_audio/<book-name>.m4b` alongside the input EPUB.

## Options

| Argument | Description |
| --- | --- |
| `epub` | Path to the EPUB file (required) |
| `voice` | Voice name or path to `.pt` file (default: `en1`) |
| `--start N` | Chapter number to start from (default: 1) |
| `--cfg-scale F` | Generation guidance scale (default: 1.5). Lower = more natural variation, higher = tighter voice adherence |

## How It Works

1. **Parse** — Extracts chapter text from the EPUB using ebooklib + BeautifulSoup
2. **Clean** — Removes URLs, normalizes quotes, collapses whitespace for natural speech
3. **Chunk** — Splits text into ~2000-character pieces at sentence boundaries (fits VibeVoice's 8K token context)
4. **Generate** — Feeds each chunk through VibeVoice with the chosen voice, saving WAV files per chapter
5. **Assemble** — Concatenates chapter WAVs into a single M4B audiobook via ffmpeg

## Model

Uses [VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) — a 0.5B parameter model that handles ~10 minutes of audio per call. The model is downloaded automatically on first run (~1 GB).

For longer content or higher quality, Microsoft also offers [VibeVoice-TTS-1.5B](https://huggingface.co/microsoft/VibeVoice-TTS-1.5B) (~90 minutes per call).
