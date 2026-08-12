"""Local Whisper-family transcription with strict audio validation and metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence
import unicodedata


@dataclass(frozen=True)
class AudioMetadata:
    path: str
    duration_seconds: float
    sample_rate: int | None
    channels: int | None
    bytes: int


@dataclass(frozen=True)
class ASRResult:
    text: str
    backend: str
    model: str
    language_hint: str | None
    runtime_seconds: float
    audio: AudioMetadata
    chunks: tuple[Mapping[str, Any], ...]
    llm_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["chunks"] = [dict(value) for value in self.chunks]
        return payload


def normalize_asr_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    # Python's ``\w`` omits several combining-mark categories that are
    # semantically required by Indic scripts (for example matras and virama).
    text = "".join(
        character
        if character.isspace() or unicodedata.category(character)[0] in {"L", "M", "N"}
        else " "
        for character in text
    )
    return " ".join(text.split())


def edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row_index, ref_value in enumerate(reference, start=1):
        current = [row_index]
        for column_index, hyp_value in enumerate(hypothesis, start=1):
            insertion = current[column_index - 1] + 1
            deletion = previous[column_index] + 1
            substitution = previous[column_index - 1] + (ref_value != hyp_value)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_tokens = normalize_asr_text(reference).split()
    hypothesis_tokens = normalize_asr_text(hypothesis).split()
    if not reference_tokens:
        return 0.0 if not hypothesis_tokens else 1.0
    return edit_distance(reference_tokens, hypothesis_tokens) / len(reference_tokens)


def character_error_rate(reference: str, hypothesis: str) -> float:
    reference_characters = list(normalize_asr_text(reference).replace(" ", ""))
    hypothesis_characters = list(normalize_asr_text(hypothesis).replace(" ", ""))
    if not reference_characters:
        return 0.0 if not hypothesis_characters else 1.0
    return edit_distance(reference_characters, hypothesis_characters) / len(reference_characters)


def probe_audio(path: Path, ffprobe_binary: str = "ffprobe") -> AudioMetadata:
    completed = subprocess.run(
        [
            ffprobe_binary,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "format=duration:stream=sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    stream = streams[0] if streams else {}
    duration = float((payload.get("format") or {}).get("duration") or 0.0)
    return AudioMetadata(
        path=str(path.resolve()),
        duration_seconds=duration,
        sample_rate=int(stream["sample_rate"]) if stream.get("sample_rate") else None,
        channels=int(stream["channels"]) if stream.get("channels") else None,
        bytes=path.stat().st_size,
    )


def validate_audio_file(
    path: Path,
    audio_config: Mapping[str, Any],
    probe: Callable[[Path], AudioMetadata] = probe_audio,
) -> AudioMetadata:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"Audio path is not a file: {resolved}")
    allowed = {str(value).lower() for value in audio_config["allowed_extensions"]}
    if resolved.suffix.lower() not in allowed:
        raise ValueError(f"Unsupported audio extension: {resolved.suffix}")
    if resolved.stat().st_size > int(audio_config["maximum_file_bytes"]):
        raise ValueError("Audio file exceeds maximum_file_bytes")
    metadata = probe(resolved)
    if metadata.duration_seconds <= 0:
        raise ValueError("Audio duration must be positive")
    if metadata.duration_seconds > float(audio_config["maximum_duration_seconds"]):
        raise ValueError("Audio duration exceeds maximum_duration_seconds")
    return metadata


class WhisperASR:
    """Lazy local wrapper for OpenAI Whisper and compatible IndicWhisper models."""

    def __init__(
        self,
        backend_name: str,
        backend_config: Mapping[str, Any],
        audio_config: Mapping[str, Any],
        project_root: Path,
        pipeline_factory: Callable[..., Any] | None = None,
        probe: Callable[[Path], AudioMetadata] = probe_audio,
    ) -> None:
        self.backend_name = backend_name
        self.backend_config = dict(backend_config)
        self.audio_config = dict(audio_config)
        self.project_root = project_root
        self.pipeline_factory = pipeline_factory
        self.probe = probe
        self._pipeline: Any | None = None

    @property
    def model_source(self) -> str:
        local_path = self.project_root / str(self.backend_config["local_pretrained_path"])
        return str(local_path) if local_path.exists() else str(self.backend_config["model_id"])

    def load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        if not self.backend_config.get("enabled", False):
            reason = self.backend_config.get("disabled_reason") or "backend is disabled"
            raise RuntimeError(f"ASR backend {self.backend_name} is unavailable: {reason}")
        factory = self.pipeline_factory
        if factory is None:
            import torch
            from transformers import pipeline

            factory = pipeline
            dtype = torch.float32
        else:
            dtype = None
        kwargs: dict[str, Any] = {
            "task": "automatic-speech-recognition",
            "model": self.model_source,
            "device": -1,
        }
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        self._pipeline = factory(**kwargs)
        # Transformers' external chunk iterator omits Whisper attention masks
        # in 4.48.3. Let Whisper perform its native long-form segmentation so
        # the pipeline supplies an explicit mask. We also pass task/language on
        # every request, so checkpoint-level forced IDs are redundant and emit
        # a conflict warning unless cleared.
        model = getattr(self._pipeline, "model", None)
        for owner in (
            getattr(model, "generation_config", None),
            getattr(model, "config", None),
            getattr(self._pipeline, "generation_config", None),
        ):
            if owner is not None and hasattr(owner, "forced_decoder_ids"):
                owner.forced_decoder_ids = None
        return self._pipeline

    def transcribe(self, audio_path: Path, language_hint: str | None = None) -> ASRResult:
        metadata = validate_audio_file(audio_path, self.audio_config, self.probe)
        if language_hint and language_hint not in set(self.backend_config["languages"]):
            raise ValueError(
                f"Backend {self.backend_name} does not declare language {language_hint}"
            )
        generate_kwargs: dict[str, Any] = {"task": self.backend_config.get("task", "transcribe")}
        if language_hint:
            generate_kwargs["language"] = language_hint
        started = time.perf_counter()
        output = self.load()(
            str(Path(metadata.path)),
            return_timestamps=True,
            generate_kwargs=generate_kwargs,
        )
        runtime = time.perf_counter() - started
        text = " ".join(str(output.get("text") or "").split())
        chunks = tuple(output.get("chunks") or ())
        return ASRResult(
            text=text,
            backend=self.backend_name,
            model=self.model_source,
            language_hint=language_hint,
            runtime_seconds=runtime,
            audio=metadata,
            chunks=chunks,
            llm_used=False,
        )
