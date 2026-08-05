# /// script
# requires-python = ">=3.10"
# dependencies = ["ebooklib", "beautifulsoup4", "requests"]
# ///
"""Convert an EPUB to an .m4b audiobook using gpt-audio (voice "ash") via OpenRouter."""

import argparse
import base64
import difflib
import json
import os
import re
import subprocess
import sys
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ebooklib
import requests
from bs4 import BeautifulSoup
from ebooklib import epub

API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Short aliases so you can type --model mini instead of the full id.
MODELS = {
    "mini": "openai/gpt-audio-mini",
    "full": "openai/gpt-audio",
}
# Measured $/1,000 characters of input text, averaged over test runs.
PRICE_PER_1K_CHARS = {
    "openai/gpt-audio-mini": 0.0039,
    "openai/gpt-audio": 0.10,
}

SAMPLE_RATE = 24000          # gpt-audio pcm16 is 24 kHz mono
MAX_CHUNK_CHARS = 800        # tested safe; larger chunks start truncating
MAX_RETRIES = 4


# ---------- EPUB -> text ----------

def extract_chapters(path, min_chars=500):
    book = epub.read_epub(path, options={"ignore_ncx": True})
    chapters = []
    for item in book.get_items():
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        soup = BeautifulSoup(item.get_body_content(), "html.parser")
        text = "\n".join(
            p.get_text(" ", strip=True)
            for p in soup.find_all(["p", "h1", "h2", "h3", "li"])
        )
        text = re.sub(r"\s+\n", "\n", text).strip()
        if len(text) >= min_chars:
            title = soup.find(["h1", "h2", "h3"])
            chapters.append({
                "title": title.get_text(" ", strip=True) if title else item.get_name(),
                "text": text,
            })
    meta = {
        "title": (book.get_metadata("DC", "title") or [("Unknown", None)])[0][0],
        "author": (book.get_metadata("DC", "creator") or [("Unknown", None)])[0][0],
    }
    return meta, chapters


def chunk_text(text, limit=MAX_CHUNK_CHARS):
    """Split into <=limit chunks on sentence boundaries. Never cut mid-sentence."""
    sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    chunks, current = [], ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        while len(sentence) > limit:            # pathological long sentence
            cut = sentence.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            chunks.append(sentence[:cut])
            sentence = sentence[cut:].strip()
        if len(current) + len(sentence) + 1 <= limit:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


# ---------- OpenRouter TTS ----------

def synthesize(text, voice, api_key, model):
    """Return raw pcm16 bytes for one chunk. Raises on truncation or mismatch."""
    body = {
        "model": model,
        "modalities": ["text", "audio"],
        "audio": {"voice": voice, "format": "pcm16"},
        "stream": True,
        "messages": [{"role": "user", "content": "Read aloud verbatim:\n\n" + text}],
    }
    resp = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        stream=True,
        timeout=300,
    )
    resp.raise_for_status()

    audio, transcript, error = [], [], None
    for raw in resp.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8").strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        event = json.loads(payload)
        if event.get("error"):
            error = event["error"]
        for choice in event.get("choices", []):
            if choice.get("error"):
                error = choice["error"]
            delta = (choice.get("delta") or {}).get("audio") or {}
            if delta.get("data"):
                audio.append(base64.b64decode(delta["data"]))
            if delta.get("transcript"):
                transcript.append(delta["transcript"])

    if error:
        raise RuntimeError(error.get("message", "stream error"))

    spoken = "".join(transcript)
    fidelity = difflib.SequenceMatcher(None, text, spoken).ratio()
    if fidelity < 0.92:
        raise RuntimeError(f"transcript drift (similarity {fidelity:.2f})")
    return b"".join(audio)


def synthesize_with_retry(index, text, voice, api_key, model):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return index, synthesize(text, voice, api_key, model)
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"chunk {index} failed after {MAX_RETRIES} tries: {exc}")
            print(f"    chunk {index} retry {attempt}: {exc}", file=sys.stderr)
    return index, b""


def render_chapter(chapters_dir, number, text, voice, api_key, workers, model):
    out = chapters_dir / f"chapter_{number:03d}.wav"
    if out.exists():
        print(f"  chapter {number}: already done, skipping")
        return out

    chunks = chunk_text(text)
    print(f"  chapter {number}: {len(text):,} chars in {len(chunks)} chunks")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(synthesize_with_retry, i, c, voice, api_key, model)
            for i, c in enumerate(chunks)
        ]
        pieces = [f.result() for f in futures]

    pieces.sort(key=lambda p: p[0])
    pcm = b"".join(p[1] for p in pieces)

    tmp = out.with_suffix(".part")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    tmp.rename(out)                       # atomic: a partial file never looks finished
    return out


# ---------- packaging ----------

def build_m4b(wavs, meta, output, cover=None):
    listing = output.parent / "wav_list.txt"
    listing.write_text("".join(f"file '{w.resolve()}'\n" for w in wavs))

    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-c:a", "aac", "-b:a", "64k", str(output)],
        check=True, capture_output=True,
    )

    # Nero-format chapter file, which is what mp4chaps expects
    lines, offset = [], 0.0
    for i, w in enumerate(wavs, start=1):
        with wave.open(str(w), "rb") as f:
            duration = f.getnframes() / f.getframerate()
        h, rem = divmod(offset, 3600)
        m, s = divmod(rem, 60)
        lines.append(f"{int(h):02d}:{int(m):02d}:{s:06.3f} Chapter {i}")
        offset += duration
    output.with_suffix(".chapters.txt").write_text("\n".join(lines) + "\n")

    subprocess.run(["mp4chaps", "-i", str(output)], check=True, capture_output=True)
    subprocess.run(
        ["mp4tags", "-type", "Audiobook", "-genre", "Audiobook",
         "-song", meta["title"], "-artist", meta["author"], str(output)],
        check=True, capture_output=True,
    )
    if cover and Path(cover).exists():
        subprocess.run(["mp4art", "--add", str(cover), str(output)],
                       check=True, capture_output=True)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="EPUB to .m4b via gpt-audio on OpenRouter")
    ap.add_argument("epub")
    ap.add_argument("-v", "--voice", default="ash")
    ap.add_argument("-m", "--model", default="mini",
                    help="mini (cheap, ~$0.004/1k chars) or full (~$0.10/1k chars). "
                         "A full OpenRouter model id also works. Default: mini")
    ap.add_argument("-o", "--output-folder", default="./output")
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("--cover")
    ap.add_argument("--max-chapters", type=int, help="stop after N chapters (for test runs)")
    ap.add_argument("--min-chars", type=int, default=500,
                    help="skip sections shorter than this (drops title pages, copyright, TOC)")
    ap.add_argument("--estimate", action="store_true", help="print cost estimate and exit")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key and not args.estimate:
        sys.exit("OPENROUTER_API_KEY is not set")

    model = MODELS.get(args.model, args.model)
    rate = PRICE_PER_1K_CHARS.get(model)

    meta, chapters = extract_chapters(args.epub, min_chars=args.min_chars)
    if args.max_chapters:
        chapters = chapters[: args.max_chapters]

    total_chars = sum(len(c["text"]) for c in chapters)
    print(f"{meta['title']} by {meta['author']}")
    print(f"{len(chapters)} chapters, {total_chars:,} characters")
    print(f"model: {model}  voice: {args.voice}")
    if rate:
        cost = total_chars / 1000 * rate
        print(f"estimated cost: ${cost:,.2f}  "
              f"(~${cost * 0.9:,.2f}-${cost * 1.1:,.2f})")
        if model != MODELS["mini"]:
            mini_cost = total_chars / 1000 * PRICE_PER_1K_CHARS[MODELS["mini"]]
            print(f"  with --model mini: ${mini_cost:,.2f}")
    else:
        print("estimated cost: unknown (no measured rate for this model)")
    if args.estimate:
        return

    out_dir = Path(args.output_folder)
    chapters_dir = out_dir / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)

    wavs = []
    for i, chapter in enumerate(chapters, start=1):
        wavs.append(render_chapter(chapters_dir, i, chapter["text"],
                                   args.voice, api_key, args.workers, model))

    output = out_dir / (Path(args.epub).stem + ".m4b")
    print("packaging m4b...")
    build_m4b(wavs, meta, output, args.cover)
    print(f"done: {output}")


if __name__ == "__main__":
    main()
