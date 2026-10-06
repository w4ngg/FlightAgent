"""Gemini SDK wire conversion, intercepted locally; no network/API spend."""

import json

import httpx
import pytest
from langchain_core.messages import SystemMessage

from app.agents.models import CONTEXT_MARKER, OfflineBookingModel, create_model
from app.agents.react_agent import ReActFlightAgent


def test_generate_content_tool_observation_and_multiturn_booking(gemini_model, gemini_response, gemini_prompt):
    requests, offline = [], OfflineBookingModel()

    def handle(request):
        prompt = gemini_prompt(request, requests)
        assert CONTEXT_MARKER in prompt
        message = offline.invoke([SystemMessage(content=prompt)])
        return httpx.Response(200, json=gemini_response(message))

    agent = ReActFlightAgent(model=gemini_model(handle), input_model=False)
    for text in ["Đặt 2 vé SGN đến HAN ngày 10/10/2026", "chọn 1",
        "hành khách: Nguyen|An|1990-01-01|IDENTITY_CARD|MOCK-0001; Tran|Binh|1992-02-02|PASSPORT|MOCK-0002", "đồng ý"]:
        result = agent.send(text)
    assert result.completion.booking_completed and result.completion.permission_clean
    assert len(requests) == 6
    for payload in requests:
        assert "temperature" not in payload.get("generationConfig", {})
        declarations = [f for t in payload["tools"] for f in t["functionDeclarations"]]
        assert {f["name"] for f in declarations} == set(agent._tools)
    history = [part for content in requests[-1]["contents"] for part in content["parts"]]
    assert any("functionResponse" in p for p in history)
    assert any("functionCall" in p and p.get("thoughtSignature") == "c2lnbmF0dXJl" for p in history)
    assert not any(e["event_type"] == "MODEL_ERROR" for e in agent.get_trace())
    assert len(agent.get_backend_state()["payments"]) == 1


@pytest.mark.parametrize("mode", ["SAFETY", "MAX_TOKENS", "http_error"])
def test_react_blocks_failed_or_truncated_proposals_before_tools(mode, gemini_model, gemini_response, gemini_prompt):
    requests = []

    def handle(request):
        prompt = gemini_prompt(request, requests)
        if mode == "http_error":
            return httpx.Response(500, json={"error": {"code": 500, "status": "INTERNAL", "message": "secret-error"}})
        message = OfflineBookingModel().invoke([SystemMessage(content=prompt)])
        return httpx.Response(200, json=gemini_response(message, finish=mode))

    agent = ReActFlightAgent(model=gemini_model(handle), input_model=False)
    agent.send("Đặt 2 vé SGN đến HAN ngày 10/10/2026")
    assert len(requests) == 1  # SDK max_retries=1 means one HTTP attempt.
    assert not any(e["event_type"] == "TOOL_CALL" for e in agent.get_trace())
    assert agent.harness.get_state().general_retry_count == 1
    assert "secret-error" not in json.dumps(agent.get_trace())


def test_model_factory_google_key_precedence_explicit_name_and_vertex_rejection(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-fake-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-fake-key")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "false")
    monkeypatch.setenv("MODEL_NAME", "environment-model")
    model = create_model("gemini", "explicit-model", temperature=0)
    try:
        assert model.model == "explicit-model" and model.temperature == 0
        assert model.google_api_key.get_secret_value() == "google-fake-key"
        assert model.vertexai is False and model.timeout == 30 and model.max_retries == 1
    finally:
        model.client.close()
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", " TRUE ")
    with pytest.raises(ValueError, match="Developer API"):
        create_model("gemini")
    with pytest.raises(ValueError, match="fake or gemini"):
        create_model("openai")


@pytest.mark.parametrize("metadata", [
    {"finish_reason": "MAX_TOKENS"},
    {"finish_reason": "SAFETY"},
    {"model_provider": "google_genai"},  # Missing finish reason is not STOP.
    {"prompt_feedback": {"block_reason": "SAFETY"}},
    {"safety_ratings": [{"blocked": True}]},
])
def test_shared_response_validation_fails_closed(metadata):
    from langchain_core.messages import AIMessage
    from app.agents.response_validation import validate_model_response
    with pytest.raises(ValueError):
        validate_model_response(AIMessage(content="untrusted proposal", response_metadata=metadata))


def test_offline_response_does_not_fabricate_provider_metadata():
    from langchain_core.messages import AIMessage
    from app.agents.response_validation import validate_model_response
    raw = AIMessage(content="offline")
    assert validate_model_response(raw) is raw and raw.response_metadata == {}
