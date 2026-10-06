"""Input/API regression checks; every SDK HTTP call uses MockTransport."""

import json

import httpx
import pytest
from langchain_core.messages import AIMessage, SystemMessage
from pydantic import ValidationError

from app.agents.diagnostics import InterpretationError, failure_diagnostic
from app.agents.factory import create_agent, create_pattern_model
from app.agents.interpretation import INPUT_CONTEXT_MARKER, InputDecision, InputInterpreter, input_wire_schema
from app.agents.models import OfflineBookingModel
from app.agents.react_agent import ReActFlightAgent
from test_input_interpretation import JSON_REQUEST, PROFILE, REQUEST, ScriptedInput, search_decision


def walk_schema(node):
    yield node
    for value in node.get("properties", {}).values():
        yield from walk_schema(value)
    if "items" in node:
        yield from walk_schema(node["items"])
    for value in node.get("anyOf", []):
        yield from walk_schema(value)


def test_wire_contract_is_small_but_local_contract_remains_strict():
    schema = input_wire_schema()
    nodes = list(walk_schema(schema))
    assert all(set(n) <= {"type", "enum", "required", "properties", "items", "anyOf", "additionalProperties"} for n in nodes)
    assert all(n.get("additionalProperties") is False for n in nodes if "properties" in n)
    assert schema["properties"]["search_patch"]["type"] == ["object", "null"]
    assert schema["properties"]["selection_index"]["type"] == ["integer", "null"]
    assert input_wire_schema() == schema  # No destructive changes to Pydantic's schema.
    strict = InputDecision.model_json_schema()
    assert strict["properties"]["evidence"]["maxItems"] == 120
    assert strict["$defs"]["FieldEvidence"]["properties"]["text"]["maxLength"] == 200


@pytest.mark.parametrize("mutate", [
    lambda d: d["search_patch"].update(passenger_count=True),
    lambda d: d["search_patch"].update(passenger_count=0),
    lambda d: d["search_patch"].update(max_budget_vnd=-1),
    lambda d: d["search_patch"].update(destination="DND"),
    lambda d: d.update(extra_field="private-value"),
    lambda d: d.update(intent="ASK_INFO", topic="STATUS"),
    lambda d: d.update(evidence=d["evidence"] * 21),
    lambda d: d["evidence"][0].update(text="private-value" * 100),
])
def test_simplified_wire_never_weakens_local_schema(mutate):
    data = search_decision().model_dump(mode="json")
    mutate(data)
    interpreter = InputInterpreter(ScriptedInput(data))
    with pytest.raises(InterpretationError) as failed:
        interpreter.interpret({}, [])
    assert failed.value.stage == "schema_validation" and failed.value.code == "INPUT_SCHEMA_INVALID"
    assert isinstance(failed.value.__cause__, ValidationError)
    diagnostic = failure_diagnostic(failed.value, node="conversation.input", stage="api_call")
    assert diagnostic["category"] == "validation" and diagnostic["response_received"]
    assert "private-value" not in json.dumps(diagnostic)


@pytest.mark.parametrize("pattern", ["react", "plan_execute", "hybrid"])
def test_actual_google_400_has_safe_diagnostics_and_never_runs_tools(
    pattern, gemini_model, gemini_prompt, tmp_path, monkeypatch
):
    requests = []
    monkeypatch.setenv("GEMINI_API_KEY", "environment-test-secret")

    def handle(request):
        gemini_prompt(request, requests)
        return httpx.Response(400, json={"error": {
            "code": 400, "status": "INVALID_ARGUMENT",
            "message": "Invalid response_json_schema: schema has too many states. "
                       "Unknown name 'exclusiveMinimum'. key=test-fake-key "
                       "token=arbitrary-token https://example.invalid?key=test-fake-key "
                       "environment-test-secret Authorization: Bearer arbitrary-bearer "
                       "'arbitrary-unregistered-secret'",
            "details": [{"secret": "never-serialize-error-details"}],
        }})

    agent = create_agent(pattern, model=gemini_model(handle, "gemini-3.1-flash-lite"))
    answer = agent.send("Đặt 1 vé từ SGN đến DAD ngày 10/10/2026, chỉ Vietnam Airlines")
    assert "HTTP 400" in answer.assistant_message and "Model chưa" not in answer.assistant_message
    state = agent.get_state()
    error = state["last_model_error"]
    assert error["exception_type"] == "GoogleInvalidRequestError"
    assert error["category"] == "provider" and error["stage"] == "api_call"
    assert error["http_status"] == 400 and error["provider_status"] == "INVALID_ARGUMENT"
    assert error["error_code"] == "PROVIDER_BAD_REQUEST" and error["retryable"] is False
    assert "too many states" in error["provider_message"] and "exclusiveMinimum" in error["provider_message"]
    assert len(error["schema_hash"]) == 64 and error["structured_output_method"] == "json_schema"
    assert state["harness"]["general_retry_count"] == state["harness"]["step_count"] == 1
    assert state["harness"]["input_blocked"] and state["harness"]["constraints"] is None
    assert state["request_draft"] == {} and state["pending_interpretation"] is None
    assert len(requests) == 1 and not any(e["event_type"] == "TOOL_CALL" for e in agent.get_trace())
    for event in ("MODEL_ERROR", "INPUT_INTERPRETATION_REJECTED"):
        assert next(e["payload"]["diagnostic"] for e in agent.get_trace() if e["event_type"] == event) == error
    assert "last_model_error" not in agent._public_context()
    target = tmp_path / pattern
    agent.export_artifacts(target)
    serialized = json.dumps(state) + json.dumps(agent.get_trace()) + "".join(p.read_text() for p in target.glob("*.json"))
    for secret in ("test-fake-key", "environment-test-secret", "arbitrary-token", "arbitrary-bearer",
                   "arbitrary-unregistered-secret", "never-serialize-error-details", "example.invalid"):
        assert secret not in serialized


@pytest.mark.parametrize("status,code,retryable", [
    (401, "PROVIDER_AUTHENTICATION", False), (403, "PROVIDER_PERMISSION", False),
    (404, "PROVIDER_MODEL_NOT_FOUND", False), (429, "PROVIDER_RATE_LIMIT", True),
    (503, "PROVIDER_ERROR", True),
])
def test_sdk_http_errors_are_classified_without_hidden_retries(status, code, retryable, gemini_model):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(status, json={"error": {"code": status, "status": "UNAVAILABLE",
                                                      "message": "Provider error"}})

    agent = ReActFlightAgent(model=gemini_model(handle))
    agent.send(REQUEST)
    error = agent.get_state()["last_model_error"]
    assert error["http_status"] == status and error["error_code"] == code and error["retryable"] is retryable
    assert len(requests) == 1


def test_bad_json_safety_schema_and_grounding_are_not_reported_as_api_failure(gemini_model, gemini_response):
    responses = [
        gemini_response("not-json-private-value"),
        gemini_response("private-value", finish="SAFETY"),
        gemini_response(json.dumps({"intent": "UPDATE_SEARCH", "search_patch": {"passenger_count": True}})),
        gemini_response(search_decision().model_dump_json()),
    ]
    agent = ReActFlightAgent(model=gemini_model(lambda request: httpx.Response(200, json=responses.pop(0))))
    # Separate fresh episodes avoid exhausting the intended shared retry budget.
    for text, category, stage in [
        (REQUEST, "parse", "parse"), (REQUEST, "response", "response_validation"),
        (REQUEST, "validation", "schema_validation"), ("a different actual user turn", "validation", "grounding"),
    ]:
        agent.reset()
        reply = agent.send(text)
        error = agent.get_state()["last_model_error"]
        assert error["category"] == category and error["stage"] == stage
        assert "http_status" not in error and "Google từ chối" not in reply.assistant_message
        assert "private-value" not in json.dumps(error)
        assert not agent.get_backend_state()["bookings"]


def test_400_preserves_accepted_data_withdraws_consent_and_can_recover(gemini_model, gemini_response, gemini_prompt):
    requests, fail = [], True

    def handle(request):
        nonlocal fail
        prompt = gemini_prompt(request, requests)
        if INPUT_CONTEXT_MARKER in prompt:
            if fail:
                fail = False
                return httpx.Response(400, json={"error": {"code": 400, "status": "INVALID_ARGUMENT",
                    "message": "Invalid response_json_schema"}})
            return httpx.Response(200, json=gemini_response(search_decision().model_dump_json()))
        return httpx.Response(200, json=gemini_response(OfflineBookingModel().invoke([SystemMessage(content=prompt)])))

    agent = ReActFlightAgent(model=gemini_model(handle, "gemini-3.1-flash-lite"))
    agent.send(JSON_REQUEST); agent.send("/select 1"); agent.send(PROFILE)
    before = agent.harness.get_state()
    agent.send(REQUEST)
    after = agent.harness.get_state()
    assert after.constraints == before.constraints and after.quote == before.quote
    assert after.selected_offer_id == before.selected_offer_id and after.input_blocked and after.confirmation is None
    count = len(requests)
    agent.send("đồng ý")
    assert len(requests) == count and not agent.get_backend_state()["bookings"]
    agent.send(REQUEST)
    assert agent.get_state()["last_model_error"] is None and agent.get_state()["pending_interpretation"]
    agent.send("áp dụng")
    assert not agent.harness.get_state().input_blocked
    agent.reset()
    assert agent.get_state()["last_model_error"] is None


def test_generic_exception_is_not_an_error_text_exfiltration_channel():
    agent = ReActFlightAgent(input_model=ScriptedInput(RuntimeError("unregistered-provider-secret")))
    agent.send(REQUEST)
    assert "unregistered-provider-secret" not in json.dumps(agent.get_state()) + json.dumps(agent.get_trace())
    assert "provider_message" not in agent.get_state()["last_model_error"]


@pytest.mark.parametrize("pattern", ["react", "plan_execute", "hybrid"])
def test_original_query_happy_route_with_simplified_schema_over_mock_sdk(
    pattern, gemini_model, gemini_response, gemini_prompt
):
    requests, offline = [], create_pattern_model(pattern, "fake")
    query = "Đặt 1 vé từ SGN đến DAD ngày 10/10/2026, chỉ Vietnam Airlines"
    decision = {"intent": "UPDATE_SEARCH", "search_patch": {
        "origin": "SGN", "destination": "DAD", "departure_date": "2026-10-10",
        "passenger_count": 1, "allowed_airlines": ["Vietnam Airlines"],
    }, "evidence": [{"field": k, "text": v} for k, v in {
        "origin": "SGN", "destination": "DAD", "departure_date": "10/10/2026",
        "passenger_count": "1 vé", "allowed_airlines": "chỉ Vietnam Airlines",
    }.items()]}

    def handle(request):
        assert "gemini-3.1-flash-lite" in request.url.path
        prompt = gemini_prompt(request, requests)
        if INPUT_CONTEXT_MARKER in prompt:
            schema = requests[-1]["generationConfig"]["responseJsonSchema"]
            assert schema == input_wire_schema() and not requests[-1].get("tools")
            message = json.dumps(decision)
        else:
            message = offline.invoke([SystemMessage(content=prompt)])
        return httpx.Response(200, json=gemini_response(message, model="gemini-3.1-flash-lite"))

    agent = create_agent(pattern, model=gemini_model(handle, "gemini-3.1-flash-lite"))
    assert "áp dụng" in agent.send(query).assistant_message
    assert agent.harness.get_state().constraints is None and len(requests) == 1
    agent.send("áp dụng")
    constraints = agent.harness.get_state().constraints
    assert constraints.allowed_airlines == ["Vietnam Airlines"] and constraints.preferred_airlines == []
    assert all(o["flight"]["airline"] == "Vietnam Airlines" for o in agent.get_state()["offers"])
    agent.send("chọn 1"); agent.send(PROFILE)
    assert not agent.get_backend_state()["bookings"]
    assert agent.send("đồng ý").completion.booking_completed
    assert len(agent.get_backend_state()["payments"]) == 1
    assert len(agent.harness.get_state().ticket_numbers) == 1
    assert agent.get_state()["last_model_error"] is None


def test_provider_diagnostic_bounds_and_masks_encoded_keys_and_user_data():
    from google.genai.errors import ClientError
    message = "Invalid response_json_schema " + " 'private\nvalue' " + "key=unknown-secret " + \
              "test%2Fkey%3Fvalue SecretFamilyName " + "x" * 2000
    exc = ClientError(400, {"error": {"status": "INVALID_ARGUMENT", "message": message}})
    result = failure_diagnostic(exc, node="conversation.input", stage="api_call", secrets=("test/key?value",),
                                private_values=("SecretFamilyName",))
    assert len(result["provider_message"]) <= 600 and "response_json_schema" in result["provider_message"]
    assert all(s not in result["provider_message"] for s in ("private", "value", "unknown-secret", "SecretFamilyName", "test%2F"))


def test_provider_error_request_dump_is_never_written_to_diagnostic():
    from google.genai.errors import ClientError
    exc = ClientError(400, {"error": {"status": "INVALID_ARGUMENT", "message": "contents: raw-private-input"}})
    result = failure_diagnostic(exc, node="conversation.input", stage="api_call")
    assert "raw-private-input" not in json.dumps(result)
