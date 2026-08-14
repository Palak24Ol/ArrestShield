"""Whisper speech-to-text adapter for local audio files."""

from __future__ import annotations

from pathlib import Path
import time


def transcribe_audio(
    audio_path: Path,
    model_source: str,
    language_hint: str | None = None,
) -> dict:
    resolved = audio_path.resolve(strict=True)
    if resolved.suffix.lower() not in {".wav", ".flac", ".mp3", ".m4a", ".ogg"}:
        raise ValueError(f"Unsupported audio type: {resolved.suffix}")
    import torch
    from transformers import pipeline

    source_path = Path(model_source)
    source = str(source_path.resolve()) if source_path.exists() else model_source
    asr = pipeline(
        "automatic-speech-recognition",
        model=source,
        device=-1,
        torch_dtype=torch.float32,
    )
    generate_kwargs = {"task": "transcribe"}
    if language_hint:
        generate_kwargs["language"] = language_hint
    started = time.perf_counter()
    output = asr(str(resolved), return_timestamps=True, generate_kwargs=generate_kwargs)
    text = " ".join(str(output.get("text") or "").split())
    return {
        "text": text,
        "model": source,
        "runtime_seconds": time.perf_counter() - started,
        "llm_used": False,
    }
