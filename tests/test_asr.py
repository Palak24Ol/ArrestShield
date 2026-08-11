from pathlib import Path

import pytest

from arrestshield.asr import (
    AudioMetadata,
    WhisperASR,
    character_error_rate,
    normalize_asr_text,
    validate_audio_file,
    word_error_rate,
)


AUDIO_CONFIG = {
    "allowed_extensions": [".wav", ".flac"],
    "maximum_file_bytes": 1_000_000,
    "maximum_duration_seconds": 180.0,
    "chunk_length_seconds": 30.0,
}


def metadata_for(path: Path, duration: float = 2.0) -> AudioMetadata:
    return AudioMetadata(str(path.resolve()), duration, 16_000, 1, path.stat().st_size)


def test_asr_normalization_and_error_rates() -> None:
    assert normalize_asr_text("  HELLO,   Duniya! ") == "hello duniya"
    assert normalize_asr_text("नमस्ते, दुनिया!") == "नमस्ते दुनिया"
    assert word_error_rate("one two three", "one too three") == pytest.approx(1 / 3)
    assert character_error_rate("cat", "cut") == pytest.approx(1 / 3)
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "hallucination") == 1.0


def test_validate_audio_enforces_extension_and_duration(tmp_path: Path) -> None:
    invalid = tmp_path / "call.exe"
    invalid.write_bytes(b"audio")
    with pytest.raises(ValueError, match="Unsupported audio extension"):
        validate_audio_file(invalid, AUDIO_CONFIG, lambda path: metadata_for(path))

    valid = tmp_path / "call.wav"
    valid.write_bytes(b"audio")
    with pytest.raises(ValueError, match="exceeds maximum_duration"):
        validate_audio_file(
            valid, AUDIO_CONFIG, lambda path: metadata_for(path, duration=181.0)
        )


def test_whisper_adapter_is_lazy_and_never_marks_llm_usage(tmp_path: Path) -> None:
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"audio")
    captured = {}

    class FakePipeline:
        def __call__(self, path: str, **kwargs):
            captured["call"] = {"path": path, **kwargs}
            return {"text": "  aapka account block hoga ", "chunks": [{"timestamp": [0, 1]}]}

    def factory(**kwargs):
        captured["load"] = kwargs
        return FakePipeline()

    backend = {
        "enabled": True,
        "model_id": "fake/whisper",
        "local_pretrained_path": "artifacts/missing",
        "languages": ["en", "hi"],
        "task": "transcribe",
    }
    transcriber = WhisperASR(
        "fake",
        backend,
        AUDIO_CONFIG,
        tmp_path,
        pipeline_factory=factory,
        probe=lambda path: metadata_for(path),
    )
    assert transcriber._pipeline is None
    result = transcriber.transcribe(audio, language_hint="hi")
    assert result.text == "aapka account block hoga"
    assert result.llm_used is False
    assert captured["load"]["task"] == "automatic-speech-recognition"
    assert captured["call"]["generate_kwargs"] == {"task": "transcribe", "language": "hi"}
    with pytest.raises(ValueError, match="does not declare language"):
        transcriber.transcribe(audio, language_hint="fr")
