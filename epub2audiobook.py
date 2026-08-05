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
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

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

# gpt-audio is a moderated model. On text it objects to it answers you instead of
# reading, so these openers mean "refused", not "misheard".
REFUSAL_MARKERS = (
    "i'm sorry", "i am sorry", "i can't assist", "i cannot assist",
    "i can't help", "i cannot help", "i won't be able", "i'm not able",
    "i can't read", "i cannot read", "i'm unable", "i am unable",
)

# Half a second of silence stands in for a refused sentence when there is no fallback.
SILENCE = b"\x00\x00" * (SAMPLE_RATE // 2)

# Matched to the gpt-audio narration measured off a finished book, so the local
# fallback does not jump out at you: -24.8 LUFS integrated, roughly 190 words/min.
FALLBACK_RATE = 190
FALLBACK_LUFS = -24.8


class Refused(Exception):
    """The model answered instead of reading it.

    Moderation is probabilistic, not a fixed blocklist: the same sentence sometimes
    reads fine on a later attempt, so this is retried like any other failure. It is a
    distinct type only so the log can say why a passage needed the local fallback.
    """


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
        if is_refusal(spoken):
            raise Refused(spoken.strip())
        raise RuntimeError(f"transcript drift (similarity {fidelity:.2f})")
    return b"".join(audio)


def local_speech(text, voice):
    """Speak text with the macOS `say` command, level-matched to the API narration.

    The API model is moderated and will not read some passages. `say` runs on your
    machine and reads anything, so a refused sentence becomes a different voice rather
    than a hole in the book.
    """
    with tempfile.TemporaryDirectory() as tmp:
        aiff = Path(tmp) / "s.aiff"
        subprocess.run(["say", "-v", voice, "-r", str(FALLBACK_RATE), "-o", str(aiff), text],
                       check=True, capture_output=True)
        done = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(aiff),
             "-af", f"loudnorm=I={FALLBACK_LUFS}:TP=-6.7:LRA=7",
             "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-"],
            check=True, capture_output=True,
        )
    return done.stdout


def make_fallback(mode, voice):
    """Return a function that renders text the API refused, or None for plain silence."""
    if mode == "silence":
        return None
    if not shutil.which("say"):
        print("warning: `say` not found, refusals will be silent", file=sys.stderr)
        return None

    def fallback(text):
        try:
            return local_speech(text, voice)
        except Exception as exc:
            print(f"    local fallback failed ({exc}), using silence", file=sys.stderr)
            return SILENCE

    return fallback


def is_refusal(spoken):
    """gpt-audio is moderated. It answers instead of reading when it objects to the text."""
    low = spoken.lower()
    return any(p in low for p in REFUSAL_MARKERS)


def synthesize_with_retry(index, text, voice, api_key, model):
    """Retry every failure, including refusals, since moderation is probabilistic.

    The final failure is re-raised keeping its type, so the caller can tell a refusal
    apart from a paraphrase when it decides what to log.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return index, synthesize(text, voice, api_key, model)
        except Exception as exc:
            if attempt == MAX_RETRIES:
                if isinstance(exc, Refused):
                    raise
                raise RuntimeError(f"chunk {index} failed after {MAX_RETRIES} tries: {exc}")
            print(f"    chunk {index} retry {attempt}: {exc}", file=sys.stderr)
    return index, b""


def synthesize_salvaging_refusals(index, text, voice, api_key, model, refusals, fallback):
    """A chunk the model will not read cleanly is retried sentence by sentence.

    Moderation shows up two ways: a flat refusal, or a quiet paraphrase that trips the
    fidelity check. Both are deterministic, so both fall back to per-sentence synthesis
    rather than killing a run that may be hours deep. Sentences that still fail become a
    beat of silence and are recorded verbatim. Nothing is rewritten or silently dropped.
    """
    try:
        return synthesize_with_retry(index, text, voice, api_key, model)[1]
    except (Refused, RuntimeError) as exc:
        reason = "refused" if isinstance(exc, Refused) else "drifted"

    def cover(sentence):
        """Whatever the API will not say, say locally rather than leave a hole."""
        return fallback(sentence) if fallback else SILENCE

    how = "read locally" if fallback else "replaced with silence"
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) == 1:
        refusals.append({"chunk": index, "reason": reason, "covered": bool(fallback),
                         "text": text})
        print(f"    chunk {index}: {reason.upper()}, {how}", file=sys.stderr, flush=True)
        return cover(text)

    pieces, substituted = [], 0
    for sentence in sentences:
        try:
            pieces.append(synthesize_with_retry(index, sentence, voice, api_key, model)[1])
        except Exception as exc:
            refusals.append({"chunk": index, "reason": reason, "covered": bool(fallback),
                             "text": sentence, "error": str(exc)})
            pieces.append(cover(sentence))
            substituted += 1
    kept = len(sentences) - substituted
    print(f"    chunk {index}: {reason}, salvaged {kept}/{len(sentences)} sentences"
          + (f", {substituted} {how}" if substituted else ""),
          file=sys.stderr, flush=True)
    return b"".join(pieces)


def cached_chunk(cache_dir, number, index, text, voice, api_key, model, progress,
                 refusals, fallback):
    """Synthesize one chunk, or reuse it from disk if a previous run already paid for it."""
    cached = cache_dir / f"chapter_{number:03d}_{index:04d}.pcm"
    if cached.exists():
        progress(cached=True)
        return index, cached.read_bytes()

    pcm = synthesize_salvaging_refusals(index, text, voice, api_key, model, refusals,
                                        fallback)
    tmp = cached.with_suffix(".tmp")
    tmp.write_bytes(pcm)
    tmp.rename(cached)                    # atomic, so a killed run never caches a partial chunk
    progress(cached=False)
    return index, pcm


def render_chapter(chapters_dir, cache_dir, number, text, voice, api_key, workers,
                   model, progress, refusals, fallback):
    out = chapters_dir / f"chapter_{number:03d}.wav"
    if out.exists():
        print(f"  chapter {number}: already done, skipping", flush=True)
        return out

    chunks = chunk_text(text)
    print(f"  chapter {number}: {len(text):,} chars in {len(chunks)} chunks", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(cached_chunk, cache_dir, number, i, c, voice, api_key, model,
                        progress, refusals, fallback)
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
    ap.add_argument("--fallback", choices=["say", "silence"], default="say",
                    help="what to do with passages the moderated model refuses: read them "
                         "with the local macOS voice (default) or leave silence")
    ap.add_argument("--fallback-voice", default="Samantha",
                    help="macOS voice for refused passages (see `say -v '?'`)")
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
    cache_dir = out_dir / "chunks"
    chapters_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    total_chunks = sum(len(chunk_text(c["text"])) for c in chapters)
    counter = {"done": 0, "reused": 0}
    lock = Lock()
    started = time.time()

    def progress(cached):
        with lock:
            counter["done"] += 1
            counter["reused"] += 1 if cached else 0
            done, billed = counter["done"], counter["done"] - counter["reused"]
            if done % 10 and done != total_chunks:
                return
            elapsed = time.time() - started
            per_chunk = elapsed / billed if billed else 0
            eta = (total_chunks - done) * per_chunk / 60
            spent = f", ~${billed * MAX_CHUNK_CHARS / 1000 * rate:,.2f} spent" if rate else ""
            print(f"    {done}/{total_chunks} chunks "
                  f"({counter['reused']} reused{spent}), eta ~{eta:.0f} min", flush=True)

    refusals = []
    fallback = make_fallback(args.fallback, args.fallback_voice)
    if fallback:
        print(f"refused passages will be read locally by '{args.fallback_voice}'", flush=True)

    wavs = []
    for i, chapter in enumerate(chapters, start=1):
        wavs.append(render_chapter(chapters_dir, cache_dir, i, chapter["text"], args.voice,
                                   api_key, args.workers, model, progress, refusals,
                                   fallback))

    output = out_dir / (Path(args.epub).stem + ".m4b")
    print("packaging m4b...", flush=True)
    build_m4b(wavs, meta, output, args.cover)
    print(f"done: {output}")
    print(f"took {(time.time() - started) / 60:.0f} min, "
          f"{counter['done'] - counter['reused']} chunks billed this run")

    if refusals:
        report = out_dir / "refusals.json"
        report.write_text(json.dumps(refusals, indent=2))
        words = sum(len(r["text"].split()) for r in refusals)
        covered = sum(1 for r in refusals if r.get("covered"))
        print(f"\nNOTE: the model refused {len(refusals)} passage(s), about {words} words.")
        if covered:
            print(f"{covered} were read by the local '{args.fallback_voice}' voice instead, "
                  f"so no text is missing. You will hear the voice change briefly.")
        if covered < len(refusals):
            print(f"{len(refusals) - covered} are silent gaps.")
        print(f"Full list: {report}")


if __name__ == "__main__":
    main()
