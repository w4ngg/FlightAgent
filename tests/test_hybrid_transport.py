"""Native Gemini routing traffic intercepted locally; no paid tokens."""

import json

import httpx
import pytest
from langchain_core.messages import SystemMessage

from app.agents.hybrid_agent import HybridFlightAgent
from app.agents.router_model import OfflineRouterModel
from app.agents.routing import ConversationDecision, ROUTER_CONTEXT_MARKER
from app.evaluation.episode import run_scenario
from app.evaluation.oracle import events
from app.evaluation.scenarios import get_scenario


@pytest.mark.parametrize("scenario_id", ["S01", "S04", "S08"])
def test_native_router_only_receives_conversation_schema_and_replays_signatures(
    scenario_id, gemini_model, gemini_response, gemini_prompt
):
    requests, offline = [], OfflineRouterModel()

    def handle(request):
        prompt = gemini_prompt(request, requests)
        assert ROUTER_CONTEXT_MARKER in prompt
        context = json.loads(prompt.split(ROUTER_CONTEXT_MARKER, 1)[1])
        assert context["public_state"]["harness"]["booking_id"] is None
        output = offline.invoke([SystemMessage(content=prompt)])
        return httpx.Response(200, json=gemini_response(output))

    model = gemini_model(handle)
    run = run_scenario(get_scenario(scenario_id), lambda h: HybridFlightAgent(h, model=model, input_model=False))
    assert run.result.scenario_passed, run.result.violations
    assert len(requests) == 3
    assert not events(run.evidence.trace, "MODEL_ERROR")
    for payload in requests:
        config = payload["generationConfig"]
        schema = config["responseJsonSchema"]
        assert config["responseMimeType"] == "application/json"
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"]) == {
            "action", "request_fields", "selection_index", "selection_evidence", "clarification"}
        assert set(schema["$defs"]["GroundedRequestField"]["required"]) == {"field", "value", "evidence"}
        assert not payload.get("tools")
    assert any(p.get("thoughtSignature") == "c2lnbmF0dXJl"
               for c in requests[-1]["contents"] if c["role"] == "model" for p in c["parts"])
    assert len(run.evidence.final_backend_state["payments"]) == 1
    assert all(e["payload"]["usage"]["total_tokens"] == 130 for e in events(run.evidence.trace, "MODEL_OUTPUT"))


@pytest.mark.parametrize("mode", ["prompt_block", "bad_schema", "http_error", "MAX_TOKENS", "SAFETY"])
def test_router_errors_fail_closed_without_hidden_http_retry_or_secret_echo(
    mode, gemini_model, gemini_response, gemini_prompt
):
    requests = []

    def handle(request):
        prompt = gemini_prompt(request, requests)
        if mode == "http_error":
            return httpx.Response(500, json={"error": {"code": 500, "status": "INTERNAL", "message": "secret-provider-error"}})
        if mode == "prompt_block":
            return httpx.Response(200, json={"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}})
        output = OfflineRouterModel().invoke([SystemMessage(content=prompt)])
        if mode == "bad_schema":
            output = '{"action":"PAY","confirmation":true}'
        return httpx.Response(200, json=gemini_response(output, finish=mode if mode in {"MAX_TOKENS", "SAFETY"} else "STOP"))

    agent = HybridFlightAgent(model=gemini_model(handle), input_model=False)
    result = agent.send("Đặt 2 vé SGN đến HAN ngày 10/10/2026")
    assert len(requests) == 1 and result.waiting_for_user
    assert agent.harness.get_state().general_retry_count == 1
    assert events(agent.get_trace(), "HYBRID_ROUTER_REJECTED")
    assert not events(agent.get_trace(), "TOOL_CALL")
    assert not agent.get_backend_state()["bookings"]
    assert "secret-provider-error" not in json.dumps(agent.get_trace())


def test_router_factory_preserves_explicit_model_and_sdk_defaults(monkeypatch):
    from app.agents.models import create_router_model
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-fake-key")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "false")
    monkeypatch.setenv("MODEL_NAME", "user-selected-router-model")
    model = create_router_model("gemini")
    try:
        assert model.model == "user-selected-router-model"
        assert model.vertexai is False and model.max_retries == 1 and model.timeout == 30
    finally:
        model.client.close()
    explicit = create_router_model("gemini", "explicit-model")
    try:
        assert explicit.model == "explicit-model"
    finally:
        explicit.client.close()
    assert isinstance(create_router_model("fake"), OfflineRouterModel)
    with pytest.raises(ValueError):
        create_router_model("unknown")


def test_offline_router_wrapper_is_typed_read_only_and_explicitly_not_nlp():
    model = OfflineRouterModel()
    prompt = SystemMessage(content=ROUTER_CONTEXT_MARKER + json.dumps({"local_input_recognized": False}))
    value = model.with_structured_output(ConversationDecision).invoke([prompt])
    assert isinstance(value, ConversationDecision) and value.action == "CLARIFY"
    packet = model.with_structured_output(ConversationDecision, include_raw=True).invoke([prompt])
    assert packet["parsed"] == value and packet["parsing_error"] is None
    with pytest.raises(ValueError):
        model.with_structured_output(ConversationDecision, method="function_calling")
