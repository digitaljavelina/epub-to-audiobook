import sys
import os
import re
import gc
import copy
import subprocess
import time
import glob

import numpy as np
import soundfile as sf
import torch
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

MODEL_ID = "microsoft/VibeVoice-Realtime-0.5B"
SAMPLE_RATE = 24000
VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")


def extract_chapters(epub_path):
    """Read an EPUB file and return a list of (title, text) tuples."""
    book = epub.read_epub(epub_path)
    chapters = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "lxml")

        title_tag = soup.find(["h1", "h2", "h3"])
        title = title_tag.get_text(strip=True) if title_tag else item.get_name()

        text = soup.get_text(separator="\n", strip=True)

        if len(text) < 100:
            continue
        skip_titles = {"contents", "table of contents", "toc", "cover",
                       "title page", "copyright", "also by", "other titles"}
        if title.lower().strip() in skip_titles:
            continue

        chapters.append((title, text))

    return chapters


def clean_text_for_tts(text):
    """Clean up text so it sounds natural when read aloud."""
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"&\w+;", " ", text)
    # Normalize smart quotes (VibeVoice expects plain quotes)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return text.strip()


def split_into_chunks(text, max_chars=2000):
    """Split text into chunks sized for VibeVoice's 8K token context."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += (" " + sentence) if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if len(sentence) > max_chars:
                words = sentence.split()
                current_chunk = ""
                for word in words:
                    if len(current_chunk) + len(word) + 1 <= max_chars:
                        current_chunk += (" " + word) if current_chunk else word
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = word
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def detect_device():
    """Pick the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(device):
    """Load the VibeVoice model and processor."""
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_streaming_processor import (
        VibeVoiceStreamingProcessor,
    )

    processor = VibeVoiceStreamingProcessor.from_pretrained(MODEL_ID)

    if device == "mps":
        dtype, attn = torch.float32, "sdpa"
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            MODEL_ID, torch_dtype=dtype, attn_implementation=attn, device_map=None,
        )
        model.to("mps")
    elif device == "cuda":
        dtype, attn = torch.bfloat16, "flash_attention_2"
        try:
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                MODEL_ID, torch_dtype=dtype, device_map="cuda", attn_implementation=attn,
            )
        except Exception:
            print("flash_attention_2 unavailable, falling back to sdpa")
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                MODEL_ID, torch_dtype=dtype, device_map="cuda", attn_implementation="sdpa",
            )
    else:
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            MODEL_ID, torch_dtype=torch.float32, device_map="cpu", attn_implementation="sdpa",
        )

    model.set_ddpm_inference_steps(num_steps=5)
    return model, processor


def resolve_voice(voice_arg):
    """Resolve a voice argument to a .pt file path.

    Accepts either:
      - A direct path to a .pt file
      - A voice name to look up in the voices/ directory
    """
    # Direct path
    if voice_arg.endswith(".pt") and os.path.isfile(voice_arg):
        return voice_arg

    # Search voices/ directory
    os.makedirs(VOICES_DIR, exist_ok=True)
    pt_files = glob.glob(os.path.join(VOICES_DIR, "**", "*.pt"), recursive=True)
    voice_map = {}
    for pt_file in pt_files:
        name = os.path.splitext(os.path.basename(pt_file))[0].lower()
        voice_map[name] = pt_file

    lookup = voice_arg.lower()
    if lookup in voice_map:
        return voice_map[lookup]

    # Partial match
    for name, path in voice_map.items():
        if lookup in name or name in lookup:
            return path

    if voice_map:
        available = ", ".join(sorted(voice_map.keys()))
        print(f"Error: Voice '{voice_arg}' not found. Available: {available}")
    else:
        print(f"Error: No voice files found in {VOICES_DIR}/")
        print("Download voices from the VibeVoice repo:")
        print("  https://github.com/microsoft/VibeVoice/tree/main/demo/voices")
        print("Place .pt files in the voices/ directory.")
        print("Or provide a direct path to a .pt voice file.")
    sys.exit(1)


def generate_chapter_audio(model, processor, text, voice_prompt, device,
                           chapter_num, output_dir, cfg_scale=1.5):
    """Generate audio for a single chapter using VibeVoice."""
    cleaned = clean_text_for_tts(text)
    chunks = split_into_chunks(cleaned)

    print(f"    {len(chunks)} chunks, longest: {max(len(c) for c in chunks)} chars")

    wav_path = os.path.join(output_dir, f"chapter_{chapter_num:03d}.wav")
    chunk_wavs = []

    for i, chunk in enumerate(chunks):
        print(f"    Chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)", end="\r")

        inputs = processor.process_input_with_cached_prompt(
            text=chunk,
            cached_prompt=voice_prompt,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(device)

        start = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=cfg_scale,
            tokenizer=processor.tokenizer,
            generation_config={"do_sample": False},
            all_prefilled_outputs=(
                copy.deepcopy(voice_prompt) if voice_prompt is not None else None
            ),
        )
        elapsed = time.time() - start

        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
            audio = outputs.speech_outputs[0]
            chunk_path = os.path.join(
                output_dir, f"chapter_{chapter_num:03d}_chunk_{i:04d}.wav"
            )
            processor.save_audio(audio, output_path=chunk_path)
            chunk_wavs.append(chunk_path)

            samples = audio.shape[-1] if len(audio.shape) > 0 else len(audio)
            duration = samples / SAMPLE_RATE
            print(f"    Chunk {i + 1}/{len(chunks)}: {duration:.1f}s audio in {elapsed:.1f}s")

        del outputs, inputs
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # Concatenate chunks into a single chapter WAV
    if chunk_wavs:
        silence = np.zeros(int(SAMPLE_RATE * 0.5))
        with sf.SoundFile(wav_path, mode="w", samplerate=SAMPLE_RATE, channels=1) as out:
            for j, chunk_path in enumerate(chunk_wavs):
                data, _ = sf.read(chunk_path)
                out.write(data)
                if j < len(chunk_wavs) - 1:
                    out.write(silence)
                os.remove(chunk_path)

        total_duration = os.path.getsize(wav_path) / (SAMPLE_RATE * 2)  # 16-bit PCM
        print(f"    Saved: {wav_path} (~{total_duration:.0f}s)")

    return wav_path


def assemble_audiobook(wav_files, output_path):
    """Combine chapter WAVs into a single M4B audiobook."""
    m4a_files = []
    for wav_path in wav_files:
        m4a_path = wav_path.replace(".wav", ".m4a")
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-c:a", "aac", "-b:a", "64k", m4a_path],
            capture_output=True,
        )
        m4a_files.append(m4a_path)

    concat_path = os.path.join(os.path.dirname(output_path), "concat_list.txt")
    with open(concat_path, "w") as f:
        for m4a_path in m4a_files:
            f.write(f"file '{m4a_path}'\n")

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_path,
            "-c", "copy",
            output_path,
        ],
        capture_output=True,
    )

    os.remove(concat_path)
    for m4a_path in m4a_files:
        os.remove(m4a_path)

    print(f"\nAudiobook saved to: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert EPUB to audiobook using Microsoft VibeVoice TTS",
        epilog="""Voice setup:
  Place .pt voice files in the voices/ directory, or pass a direct path.
  Download experimental voices from: https://github.com/microsoft/VibeVoice

Examples:
  %(prog)s book.epub                     # uses default 'en1' voice
  %(prog)s book.epub en2                 # use a different voice by name
  %(prog)s book.epub ./my_voice.pt       # use a custom voice file""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("epub", help="Path to EPUB file")
    parser.add_argument("voice", nargs="?", default="en1",
                        help="Voice name or path to .pt file (default: en1)")
    parser.add_argument("--start", type=int, default=1,
                        help="Chapter number to start from (default: 1)")
    parser.add_argument("--cfg-scale", type=float, default=1.5,
                        help="Generation guidance scale (default: 1.5)")
    args = parser.parse_args()

    epub_path = args.epub
    if not os.path.exists(epub_path):
        print(f"Error: File not found: {epub_path}")
        sys.exit(1)

    # Resolve voice
    voice_path = resolve_voice(args.voice)
    print(f"Voice: {voice_path}")

    # Detect device
    device = detect_device()
    print(f"Device: {device}")

    # Create output directory
    book_name = os.path.splitext(os.path.basename(epub_path))[0]
    output_dir = os.path.join(os.path.dirname(epub_path) or ".", f"{book_name}_audio")
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Parse EPUB
    print(f"\nParsing: {epub_path}")
    chapters = extract_chapters(epub_path)
    print(f"Found {len(chapters)} chapters\n")

    # Step 2: Load model
    print(f"Loading VibeVoice model ({MODEL_ID})...")
    model, processor = load_model(device)
    print("Model loaded!")

    # Load voice prompt
    voice_prompt = torch.load(voice_path, map_location=device, weights_only=False)

    # Step 3: Generate audio for each chapter
    wav_files = []
    for i, (title, text) in enumerate(chapters, 1):
        if i < args.start:
            print(f"Chapter {i}/{len(chapters)}: {title} (skipped)")
            continue
        existing_wav = os.path.join(output_dir, f"chapter_{i:03d}.wav")
        if os.path.exists(existing_wav) and os.path.getsize(existing_wav) > 0:
            print(f"Chapter {i}/{len(chapters)}: {title} (already exists)")
            wav_files.append(existing_wav)
            continue
        print(f"Chapter {i}/{len(chapters)}: {title}")
        wav_path = generate_chapter_audio(
            model, processor, text, voice_prompt, device,
            i, output_dir, cfg_scale=args.cfg_scale,
        )
        wav_files.append(wav_path)

    # Step 4: Assemble into audiobook
    print("\nAssembling audiobook...")
    output_path = os.path.join(output_dir, f"{book_name}.m4b")
    assemble_audiobook(wav_files, output_path)

    # Clean up individual WAV files
    for wav_path in wav_files:
        os.remove(wav_path)

    print("Done!")


if __name__ == "__main__":
    main()
