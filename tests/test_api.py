from fastapi.testclient import TestClient

from arrestshield.api import create_app
from arrestshield.inference import DetectorEngine, InferencePolicy

from test_inference import base_bundle


def client() -> TestClient:
    engine = DetectorEngine(base_bundle(), InferencePolicy())
    app = create_app(
        engine=engine,
        transcriber=None,
        config={
            "service": {"name": "Test API", "version": "1.0.0"},
            "asr": {"enabled": False},
        },
    )
    return TestClient(app)


def test_health_model_and_text_contract() -> None:
    api = client()
    health = api.get("/healthz")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "asr_enabled": False,
        "llm_used_for_detection": False,
    }
    model = api.get("/v1/model")
    assert model.status_code == 200
    assert model.json()["llm_used_for_detection"] is False

    response = api.post(
        "/v1/detect/text",
        json={
            "conversation_id": "api-demo",
            "turns": [{"speaker_role": "caller", "text": "CBI says transfer now"}],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["conversation_id"] == "api-demo"
    assert payload["is_scam"] is True
    assert payload["honeypot"]["invoked"] is False


def test_api_rejects_unknown_fields_and_disabled_audio() -> None:
    api = client()
    invalid = api.post(
        "/v1/detect/text",
        json={
            "turns": [{"speaker_role": "caller", "text": "hello", "unexpected": True}]
        },
    )
    assert invalid.status_code == 422

    audio = api.post(
        "/v1/detect/audio",
        files={"file": ("call.wav", b"not-real-audio", "audio/wav")},
    )
    assert audio.status_code == 503
    assert audio.json()["detail"] == "ASR is disabled"
