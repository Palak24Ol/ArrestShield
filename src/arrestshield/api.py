"""FastAPI service for text/audio scam detection without LLM classification."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from .asr import WhisperASR
from .inference import DetectorEngine, InferencePolicy
from .transformer_inference import MultiTaskPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TurnInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    speaker_role: str = Field(default="unknown", min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=10_000)


class DetectTextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conversation_id: str | None = Field(default=None, max_length=128)
    turns: list[TurnInput] = Field(min_length=1, max_length=100)
    include_sensitive_entities: bool = False


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_service_components(
    config_path: Path,
) -> tuple[DetectorEngine, WhisperASR | None, dict[str, Any]]:
    config = read_json(config_path)
    policy_config = config["policy"]
    policy = InferencePolicy(
        detector_status=str(policy_config["detector_status"]),
        allow_research_fusion=bool(policy_config["allow_research_fusion"]),
        enable_honeypot_handoff=bool(policy_config["enable_honeypot_handoff"]),
        maximum_turns=int(policy_config["maximum_turns"]),
        maximum_characters=int(policy_config["maximum_characters"]),
    )
    fusion_path = PROJECT_ROOT / config["models"]["risk_fusion_path"]
    auxiliary_predictor = None
    transformer_path = PROJECT_ROOT / config["models"]["multitask_transformer_path"]
    if (
        bool(config["models"].get("use_multitask_auxiliary_if_available", True))
        and (transformer_path / "manifest.json").exists()
    ):
        auxiliary_predictor = MultiTaskPredictor(
            transformer_path,
            torch_threads=int(config["models"].get("multitask_torch_threads", 4)),
        ).predict
    engine = DetectorEngine.from_paths(
        PROJECT_ROOT / config["models"]["base_detector_path"],
        policy,
        fusion_path=fusion_path,
        auxiliary_predictor=auxiliary_predictor,
    )
    transcriber = None
    if config["asr"]["enabled"]:
        asr_config = read_json(PROJECT_ROOT / config["asr"]["config_path"])
        backend_name = str(config["asr"]["backend"])
        transcriber = WhisperASR(
            backend_name,
            asr_config["backends"][backend_name],
            asr_config["audio"],
            PROJECT_ROOT,
        )
    return engine, transcriber, config


def create_app(
    engine: DetectorEngine | None = None,
    transcriber: WhisperASR | None = None,
    config: dict[str, Any] | None = None,
) -> FastAPI:
    if engine is None:
        engine, transcriber, config = load_service_components(
            PROJECT_ROOT / "configs/deployment/api.json"
        )
    config = config or {
        "service": {"name": "ArrestShield ML API", "version": "1.0.0"},
        "asr": {"enabled": transcriber is not None},
    }
    app = FastAPI(
        title=str(config["service"]["name"]),
        version=str(config["service"]["version"]),
        description=(
            "Research ML inference service. The detector and risk fusion are trained models; "
            "the service never uses an LLM to decide whether a conversation is a scam."
        ),
    )
    app.state.engine = engine
    app.state.transcriber = transcriber
    app.state.config = config

    @app.get("/healthz")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "asr_enabled": app.state.transcriber is not None,
            "llm_used_for_detection": False,
        }

    @app.get("/v1/model")
    def model_info() -> dict[str, Any]:
        result = app.state.engine.model_info()
        result["multitask_auxiliary_loaded"] = app.state.engine.auxiliary_predictor is not None
        return result

    @app.post("/v1/detect/text")
    def detect_text(request: DetectTextRequest) -> dict[str, Any]:
        try:
            return app.state.engine.detect(
                [turn.model_dump() for turn in request.turns],
                conversation_id=request.conversation_id,
                include_sensitive_entities=request.include_sensitive_entities,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/v1/detect/audio")
    async def detect_audio(
        file: Annotated[UploadFile, File(description="WAV, FLAC, MP3, M4A, or OGG")],
        language_hint: Annotated[str | None, Form()] = None,
        conversation_id: Annotated[str | None, Form()] = None,
        include_sensitive_entities: Annotated[bool, Form()] = False,
    ) -> dict[str, Any]:
        if app.state.transcriber is None:
            raise HTTPException(status_code=503, detail="ASR is disabled")
        suffix = Path(file.filename or "").suffix.lower()
        allowed = set(app.state.transcriber.audio_config["allowed_extensions"])
        if suffix not in allowed:
            raise HTTPException(status_code=415, detail="Unsupported audio extension")
        maximum_bytes = int(app.state.transcriber.audio_config["maximum_file_bytes"])
        try:
            with tempfile.TemporaryDirectory(prefix="arrestshield-audio-") as temp_dir:
                path = Path(temp_dir) / f"upload{suffix}"
                size = 0
                with path.open("wb") as handle:
                    while chunk := await file.read(1024 * 1024):
                        size += len(chunk)
                        if size > maximum_bytes:
                            raise HTTPException(status_code=413, detail="Audio file is too large")
                        handle.write(chunk)
                result = app.state.transcriber.transcribe(path, language_hint=language_hint)
                detection = app.state.engine.detect(
                    [{"speaker_role": "caller", "text": result.text}],
                    conversation_id=conversation_id,
                    include_sensitive_entities=include_sensitive_entities,
                )
                detection["asr"] = result.to_dict()
                detection["asr"]["audio"]["path"] = "temporary_upload_deleted"
                return detection
        except HTTPException:
            raise
        except (ValueError, OSError, RuntimeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        finally:
            await file.close()

    return app


def create_default_app() -> FastAPI:
    return create_app()
