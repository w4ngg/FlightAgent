"""Gemini native JSON-schema planning tests; all HTTP intercepted, not run."""

import json

import httpx
import pytest
from langchain_core.messages import SystemMessage

from app.agents.plan_execute_agent import PlanExecuteFlightAgent
from app.agents.planner_model import OfflinePlannerModel
from app.agents.planning import BookingPlan, PLANNER_CONTEXT_MARKER
from app.evaluation.episode import run_scenario
from app.evaluation.oracle import events
from app.evaluation.scenarios import get_scenario


@pytest.mark.parametrize("scenario_id,requests_count", [("S01", 1), ("S04", 2), ("S08", 2)])
def test_native_json_schema_planning_replanning_and_signature_history(
    scenario_id, requests_count, gemini_model, gemini_response, gemini_prompt
):
    requests, offline = [], OfflinePlannerModel()

    def handle(request):
        prompt = gemini_prompt(request, requests)
        assert PLANNER_CONTEXT_MARKER in prompt
        output = offline.invoke([SystemMessage(content=prompt)])
        return httpx.Response(200, json=gemini_response(output))

    model = gemini_model(handle)
    run = run_scenario(get_scenario(scenario_id), lambda h: PlanExecuteFlightAgent(h, model=model, input_model=False))
    assert run.result.scenario_passed, run.result.violations
    assert len(requests) == requests_count
    assert not events(run.evidence.trace, "MODEL_ERROR")
    assert not events(run.evidence.trace, "PLAN_REJECTED")
    for payload in requests:
        config = payload["generationConfig"]
        schema = config["responseJsonSchema"]
        assert config["responseMimeType"] == "application/json"
        assert schema["additionalProperties"] is False
        step = schema["$defs"]["PlanStep"]
        assert set(step["required"]) == {"action", "reason", "endpoint"}
        assert step["additionalProperties"] is False
        assert not payload.get("tools")  # Planner is not a business tool caller.
    if len(requests) > 1:
        assert any(p.get("thoughtSignature") == "c2lnbmF0dXJl"
                   for c in requests[-1]["contents"] if c["role"] == "model" for p in c["parts"])
    assert len(run.evidence.final_backend_state["payments"]) == 1
    assert all(e["payload"]["usage"]["total_tokens"] == 130 for e in events(run.evidence.trace, "MODEL_OUTPUT"))


@pytest.mark.parametrize("mode", ["prompt_block", "bad_schema", "http_error", "MAX_TOKENS", "SAFETY"])
def test_planner_errors_even_valid_truncated_json_never_execute_tools(
    mode, gemini_model, gemini_response, gemini_prompt
):
    requests = []

    def handle(request):
        prompt = gemini_prompt(request, requests)
        if mode == "http_error":
            return httpx.Response(500, json={"error": {"code": 500, "status": "INTERNAL", "message": "secret-provider-error"}})
        if mode == "prompt_block":
            return httpx.Response(200, json={"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}})
        output = OfflinePlannerModel().invoke([SystemMessage(content=prompt)])
        if mode == "bad_schema":
            output = '{"steps":[{"action":"PAY","reason":"unsafe","endpoint":null,"confirmation":true}]}'
        return httpx.Response(200, json=gemini_response(output, finish=mode if mode in {"MAX_TOKENS", "SAFETY"} else "STOP"))

    agent = PlanExecuteFlightAgent(model=gemini_model(handle), input_model=False)
    result = agent.send("Đặt 2 vé SGN đến HAN ngày 10/10/2026")
    assert len(requests) == 1 and result.terminal_status is None
    assert agent.get_state()["plan"] is None and agent.harness.get_state().general_retry_count == 1
    assert not events(agent.get_trace(), "TOOL_CALL")
    assert not agent.get_backend_state()["bookings"]
    assert "secret-provider-error" not in json.dumps(agent.get_trace())


def test_planner_factory_preserves_provider_model_and_retry_defaults(monkeypatch):
    from app.agents.models import create_planner_model
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-fake-key")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "false")
    monkeypatch.setenv("MODEL_NAME", "user-selected-planning-model")
    model = create_planner_model("gemini")
    try:
        assert model.model == "user-selected-planning-model"
        assert model.vertexai is False and model.max_retries == 1 and model.timeout == 30
    finally:
        model.client.close()
    explicit = create_planner_model("gemini", "explicit-model")
    try:
        assert explicit.model == "explicit-model"
    finally:
        explicit.client.close()
    assert isinstance(create_planner_model("fake"), OfflinePlannerModel)
    with pytest.raises(ValueError):
        create_planner_model("other")


def test_offline_structured_wrapper_supports_validated_plan_and_explicit_raw_packet():
    model = OfflinePlannerModel()
    agent = PlanExecuteFlightAgent()
    prompt = SystemMessage(content=PLANNER_CONTEXT_MARKER + json.dumps({"public_state": agent._public_context()}))
    value = model.with_structured_output(BookingPlan).invoke([prompt])
    assert isinstance(value, BookingPlan)
    packet = model.with_structured_output(BookingPlan, include_raw=True).invoke([prompt])
    assert packet["parsed"] == value and packet["parsing_error"] is None
    with pytest.raises(ValueError):
        model.with_structured_output(BookingPlan, method="function_calling")
