# EPUB to Audiobook

Convert any EPUB into a chaptered `.m4b` audiobook narrated by OpenAI's `gpt-audio`, reached through [OpenRouter](https://openrouter.ai). Output is tagged so Apple Books and Audiobookshelf recognize it as a real audiobook with working chapter markers.

This replaces an earlier local VibeVoice implementation. The tradeoff is deliberate: this version needs an internet connection and costs money per character, and it sounds considerably better.

## Cost

Billing was sampled against the live API at several chunk sizes. Both models held a steady rate per 1,000 characters of input text.

| Model | Per 1,000 chars | Per hour of audio | 500K char novel |
| --- | --- | --- | --- |
| `gpt-audio-mini` (default) | ~$0.0039 | ~$0.18 | ~$2 |
| `gpt-audio` | ~$0.10 | ~$4.80 | ~$50 |

Mini is 26 times cheaper and read test passages with perfect fidelity. Stay on it unless you listen to a sample and hear something you dislike.

Always run `--estimate` first. It prints the exact character count and projected cost without spending anything.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- `ffmpeg` and `mp4v2` (`brew install ffmpeg mp4v2`)
- An [OpenRouter API key](https://openrouter.ai/keys) with credit on the account

`mp4v2` supplies `mp4chaps`, `mp4tags`, and `mp4art`. Without it the file gets built but Apple Books will not treat it as an audiobook.

## Setup

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

Put that in `~/.zshrc` so it survives new shells. No `uv sync` is needed: the script declares its own dependencies inline (PEP 723), so `uv run` builds a throwaway environment on demand.

## Usage

```bash
# Always start here: cost and chapter count, spends nothing
uv run epub2audiobook.py book.epub --estimate

# Convert one chapter to audition the voice for pennies
uv run epub2audiobook.py book.epub --max-chapters 1

# Full conversion
uv run epub2audiobook.py book.epub -o ./output
```

Output lands in `output/book.m4b`, with per-chapter WAVs kept in `output/chapters/`.

## Options

| Argument | Description |
| --- | --- |
| `epub` | Path to the EPUB file (required) |
| `-m`, `--model` | `mini` or `full`, or any OpenRouter model id (default: `mini`) |
| `-v`, `--voice` | Voice name (default: `ash`) |
| `-o`, `--output-folder` | Where WAVs and the `.m4b` are written (default: `./output`) |
| `-w`, `--workers` | Concurrent API calls (default: 4). Lower it if you hit rate limits |
| `--cover` | Path to a cover image to embed |
| `--max-chapters N` | Stop after N chapters, for cheap test runs |
| `--min-chars N` | Skip sections shorter than N chars (default: 500). Drops title pages, copyright, TOC |
| `--fallback` | `say` (default) reads refused passages with a local voice, `silence` leaves a gap |
| `--fallback-voice` | macOS voice for refused passages (default: `Samantha`, see `say -v '?'`) |
| `--estimate` | Print character count and cost, then exit |

## Moderation

Both `gpt-audio` models are moderated. On sex, profanity, and some violence they answer instead of reading, either with a flat "I'm sorry" or a quiet paraphrase. Converting a 147,000 word novel produced 28 such passages, about 0.3% of the text.

You cannot prompt around it. Four framings were tested against 28 known-refused passages, including explicitly stating the text is a published literary novel:

| Framing | Passages read cleanly |
| --- | --- |
| `Read aloud verbatim:` (current) | 3/28 |
| Audiobook narrator framing | 0/28 |
| Bare text, no instruction | 0/28 |
| System prompt as TTS engine | 0/28 |

Refusals are probabilistic rather than fixed, so the same sentence occasionally reads fine on a later attempt. That is why the script retries before giving up.

When it does give up, `--fallback say` renders the sentence with the macOS `say` command, which is local and unmoderated. The output is resampled to 24 kHz mono and loudness-matched to the API narration (-24.8 LUFS, about 190 words per minute), so the handoff is a voice change rather than a jolt. Nothing is rewritten or dropped, and every substitution is recorded in `refusals.json`.

The alternative is going direct to OpenAI's `/v1/audio/speech`, a dedicated TTS endpoint that is far less likely to refuse. It costs roughly 3.5x more: about $11.31 for a 147,000 word novel versus $3.24 on OpenRouter, because `gpt-4o-mini-tts` bills $12.00 per 1M audio tokens against $2.40 for `gpt-audio-mini`.

## Voices

All thirteen OpenAI voices work and were verified against the live API: `ash`, `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`, `coral`, `sage`, `ballad`, `verse`, `marin`, `cedar`.

`ash` is the default and the strongest general-purpose narrator. Since a one-chapter test costs pennies on mini, auditioning three voices on the same chapter is the cheap way to choose.

## How It Works

1. **Parse** extracts chapter text from the EPUB with ebooklib and BeautifulSoup, skipping sections below `--min-chars`.
2. **Chunk** splits each chapter into pieces of 800 characters or less, always at sentence boundaries.
3. **Generate** streams each chunk through OpenRouter and collects the returned PCM audio.
4. **Verify** compares the model's own transcript against the input text and retries anything below 92% similarity.
5. **Assemble** concatenates chapter WAVs, encodes to AAC, and writes Apple-native chapters and audiobook metadata with the `mp4v2` tools.

## Resuming

Each chapter is written to its own WAV, atomically via a `.part` file that is renamed only on success. Re-running skips chapters that already exist, so an interrupted run only bills you for what is left.

Do not switch models mid-book. Finished chapters are skipped on re-run, so a book started on mini and finished on full ends up with two voices spliced together. Delete `output/chapters/` if you change models.

## Working Around the Model

`gpt-audio` is a conversational model that emits audio, not a dedicated text-to-speech endpoint. Four behaviors had to be handled, each one a real failure observed in testing.

- **Audio output requires streaming.** A non-streaming request returns `Audio output requires stream: true` with a 400.
- **Long inputs truncate.** Around 1,200 characters the stream dies with `Could not finish the message because max_tokens or model output limit was reached`. Raising `max_tokens` does not help, and neither does `max_completion_tokens`. Chunks of 800 characters completed cleanly every time.
- **A system prompt makes truncation worse.** Adding one caused failures on text that succeeded without it, repeatedly. The script sends a single user message and no system prompt.
- **It can paraphrase.** Being a chat model, it sometimes reads something close to your text rather than your text. Every response carries a transcript of what was actually spoken, so the script diffs that against the input and retries on drift.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `OPENROUTER_API_KEY is not set` | Export it, and add the line to `~/.zshrc` |
| `Could not finish the message because max_tokens...` | Lower `MAX_CHUNK_CHARS` from 800 toward 500, delete that chapter's WAV, re-run |
| `transcript drift (similarity 0.74)` | Usually passes on retry. If one chunk always fails, the source text is likely a table or a block of symbols |
| 402 / insufficient credits | Add OpenRouter credit and re-run. Finished chapters are skipped |
| Rate limit errors | Drop to `-w 2` |
| `.m4b` empty in Apple Books | Confirm `mp4v2` is installed. Check with `mp4info out.m4b`, which should show `Media Type: Audio Book` |

## Guide

A full walkthrough lives in the Obsidian vault at `Guides/Convert EPUBs to Audiobooks with Audiblez.md`.
