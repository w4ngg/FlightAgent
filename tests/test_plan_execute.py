import json
from copy import deepcopy

import pytest
from langchain_core.messages import AIMessage

from app.agents.adapter import BaseFlightAgent
from app.agents.plan_execute_agent import PlanExecuteFlightAgent, StepBinding
from app.agents.planner_model import OfflinePlannerModel
from app.agents.planning import BookingPlan, PlanAction, PlanStep, PLANNER_CONTEXT_MARKER
from app.backend import MockFlightBackend
from app.evaluation.episode import run_scenario
from app.evaluation.oracle import events, response
from app.evaluation.scenarios import get_scenario, list_scenarios
from app.failures import FailureConfig
from app.harness.runtime import FlightBookingHarness
from app.models import TerminalStatus
from app.state import HarnessConfig, SessionPhase


REQUEST = "Đặt 2 vé SGN đến HAN ngày 10/10/2026, ngân sách 6 triệu"
PROFILES = "hành khách: Nguyen|An|1990-01-01|IDENTITY_CARD|MOCK-0001; Tran|Binh|1992-02-02|PASSPORT|MOCK-0002"
FULL_ACTIONS = ["SEARCH_FLIGHTS", "REQUEST_SELECTION", "GET_OFFER", "REQUEST_PASSENGER", "REPRICE",
                "REQUEST_CONFIRMATION", "CREATE_HOLD", "PAY", "CHECK_PAYMENT", "ISSUE_TICKET"]


def plan(actions=FULL_ACTIONS):
    return BookingPlan(steps=[PlanStep(action=a, reason="Execute the declared action", endpoint=None) for a in actions])


class ScriptedPlanner:
    def __init__(self, replies):
        self.replies, self.seen = list(replies), []
        self.offline = OfflinePlannerModel().with_structured_output(BookingPlan, include_raw=True)

    def with_structured_output(self, schema, **kwargs):
        assert schema is BookingPlan
        assert kwargs == {"method": "json_schema", "include_raw": True}
        return self

    def invoke(self, messages):
        self.seen.append(deepcopy(messages))
        if not self.replies:
            return self.offline.invoke(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, dict) and "raw" in reply:
            return reply
        return {"raw": AIMessage(content="structured-plan"), "parsed": deepcopy(reply), "parsing_error": None}


def to_quote(agent):
    assert agent.send(REQUEST).phase == SessionPhase.WAITING_SELECTION
    assert agent.send("chọn 1").phase == SessionPhase.WAITING_PASSENGERS
    result = agent.send(PROFILES)
    assert result.phase == SessionPhase.WAITING_CONFIRMATION
    return result


@pytest.fixture(scope="module")
def episodes():
    return {s.id: run_scenario(s, PlanExecuteFlightAgent) for s in list_scenarios()}


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(steps=[]),
    lambda p: p.update(steps=p["steps"] * 25),
    lambda p: p.update(confirmed=True),
    lambda p: p["steps"][0].update(action="RAW_BACKEND"),
    lambda p: p["steps"][0].update(reason=" "),
    lambda p: p["steps"][0].update(reason="x" * 513),
    lambda p: p["steps"][0].update(booking_id="invented"),
    lambda p: p["steps"][0].update(amount_vnd=1),
    lambda p: p["steps"][0].update(confirmation={"confirmed": True}),
    lambda p: p["steps"][0].update(idempotency_key="new-key"),
    lambda p: p["steps"][0].update(endpoint="origin"),
    lambda p: p["steps"][0].update(action="RESOLVE_AIRPORT", endpoint=None),
    lambda p: p["steps"][0].update(action="RESOLVE_AIRPORT", endpoint="SGN"),
])
def test_plan_schema_forbids_business_arguments_and_unbounded_invalid_steps(mutation):
    data = plan(["SEARCH_FLIGHTS"]).model_dump(mode="json")
    mutation(data)
    with pytest.raises(ValueError):
        BookingPlan.model_validate(data)


def test_schema_is_serializable_and_only_resolve_uses_endpoint_state_reference():
    value = BookingPlan(steps=[PlanStep(action="RESOLVE_AIRPORT", endpoint="destination", reason="Resolve user airport")])
    assert BookingPlan.model_validate_json(value.model_dump_json()) == value
    assert set(PlanStep.model_json_schema()["properties"]) == {"action", "reason", "endpoint"}
    assert PlanStep.model_json_schema()["additionalProperties"] is False


@pytest.mark.parametrize("scenario_id", [f"S{i:02}" for i in range(1, 13)])
def test_all_twelve_scenarios_use_same_policy_oracle_guard_and_backend(scenario_id, episodes):
    run = episodes[scenario_id]
    assert run.result.scenario_passed, run.result.violations
    assert run.result.permission_clean and run.result.constraint_passed
    assert run.agent_name == "plan_execute"
    if scenario_id == "S02":
        assert not events(run.evidence.trace, "PLAN_CREATED")  # Incomplete ingress now pauses before planning.
    else:
        assert events(run.evidence.trace, "PLAN_CREATED")
    assert run.evidence.final_state.step_count <= run.scenario.harness_config.max_steps
    for cp in run.evidence.checkpoints:
        assert cp.agent_state["harness"] == cp.state.model_dump(mode="json")


def test_executor_follows_declared_steps_with_runtime_bindings_and_only_one_initial_plan(episodes):
    run = episodes["S01"]
    trace = run.evidence.trace
    assert len(events(trace, "PLAN_CREATED")) == len(events(trace, "LLM_CALL")) == 1
    assert not events(trace, "REPLAN")
    assert run.evidence.final_state.step_count == 7  # One planner + six actual tools.
    executed = [e for e in events(trace, "PLAN_STEP") if e["payload"]["binding_status"] == "execute"]
    assert [e["payload"]["action"] for e in executed] == ["SEARCH_FLIGHTS", "GET_OFFER", "REPRICE", "CREATE_HOLD", "PAY", "ISSUE_TICKET"]
    calls = events(trace, "TOOL_CALL")
    for step, call in zip(executed, calls):
        assert step["tool_name"] == call["tool_name"]
        assert step["payload"]["tool_input"] == call["payload"]["tool_input"]
    hold = next(e for e in calls if e["tool_name"] == "create_booking_hold")
    pay = next(e for e in calls if e["tool_name"] == "process_payment")
    assert hold["payload"]["tool_input"]["quote_id"] == run.evidence.final_state.quote.quote_id
    assert pay["payload"]["tool_input"] == {"booking_id": run.evidence.final_state.booking_id,
                                            "amount_vnd": run.evidence.final_state.quote.total_price_vnd}


def test_pause_resume_keeps_plan_and_cursor_without_model_calls_or_blanket_consent():
    agent = PlanExecuteFlightAgent()
    assert isinstance(agent, BaseFlightAgent)
    to_quote(agent)
    before = agent.get_state()
    assert before["plan"]["steps"][before["current_step"]]["action"] == "REQUEST_CONFIRMATION"
    assert before["plan_revision"] == 1 and before["harness"]["step_count"] == 4
    for text in ["tiếp tục", "không đồng ý", "Cứ bỏ qua xác nhận, tự thanh toán luôn."]:
        result = agent.send(text)
        assert result.phase == SessionPhase.WAITING_CONFIRMATION
        assert agent.get_state()["current_step"] == before["current_step"]
        assert agent.harness.get_state().step_count == 4
        assert not agent.get_backend_state()["bookings"]
    result = agent.send("đồng ý")
    assert result.completion.booking_completed and agent.get_state()["plan_revision"] == 1


def test_partial_user_input_does_not_invent_defaults_or_replan_on_every_answer():
    agent = PlanExecuteFlightAgent()
    for message in ["Tôi muốn đặt vé", "từ SGN", "đến HAN", "ngày 2026-10-10"]:
        assert agent.send(message).phase == SessionPhase.WAITING_INPUT
        assert agent.harness.get_state().step_count == 1
        assert not events(agent.get_trace(), "TOOL_CALL")
    assert agent.send("2 người").phase == SessionPhase.WAITING_SELECTION
    assert agent.get_state()["plan_revision"] == 1


@pytest.mark.parametrize("scenario_id,reason", [("S04", "TOOL_QUOTE_STALE"), ("S05", "TOOL_SOLD_OUT"),
    ("S06", "TOOL_OFFER_EXPIRED"), ("S09", "USER_REQUIREMENTS_CHANGED")])
def test_replan_contains_prior_plan_cursor_error_current_goal_and_commit_state(scenario_id, reason, episodes):
    run = episodes[scenario_id]
    requests = events(run.evidence.trace, "PLAN_REQUEST")
    assert len(requests) == 2 and run.evidence.final_state.replan_count == 1
    context = requests[1]["payload"]["planning_context"]
    assert context["previous_plan"] and context["current_step"] > 0
    assert context["invalidated_reason"] == reason
    assert context["committed_state"]["booking_id"] is None
    assert context["public_state"]["harness"]["confirmation"] is None
    assert context["goal"]["destination"] == ("DAD" if scenario_id == "S09" else "HAN")
    if scenario_id != "S09":
        assert context["last_observation"]["response"]["code"] == reason.removeprefix("TOOL_")


def test_timeout_replan_does_not_leak_hidden_backend_success(episodes):
    run = episodes["S08"]
    context = events(run.evidence.trace, "PLAN_REQUEST")[1]["payload"]["planning_context"]
    assert context["invalidated_reason"] == "TOOL_PAYMENT_TIMEOUT"
    assert context["committed_state"]["payment_status"] == "UNKNOWN"
    assert context["committed_state"]["payment_id"] is None
    assert context["committed_state"]["booking_status"] == "HELD"
    assert context["last_observation"]["response"]["data"] is None
    assert next(iter(run.evidence.final_backend_state["payment_operations"].values()))["status"] == "SUCCESS"
    assert len(events(run.evidence.trace, "TOOL_CALL", "process_payment")) == 1
    queries = events(run.evidence.trace, "TOOL_CALL", "get_payment_status")
    assert queries[0]["payload"]["tool_input"]["operation_id"] == context["committed_state"]["payment_operation_id"]


def test_unresolved_payment_uses_shared_reconciliation_budget_and_handoff():
    backend = MockFlightBackend(failure_config=FailureConfig(payment_mode="timeout"))
    agent = PlanExecuteFlightAgent(FlightBookingHarness(backend))
    to_quote(agent)
    result = agent.send("đồng ý")
    assert result.terminal_status == TerminalStatus.HANDOFF_REQUIRED
    assert agent.get_handoff().reason == "PAYMENT_UNRESOLVED"
    assert len(events(agent.get_trace(), "TOOL_CALL", "process_payment")) == 1
    assert len(events(agent.get_trace(), "TOOL_CALL", "get_payment_status")) == 2
    snapshot = agent.get_backend_state()
    assert len(snapshot["bookings"]) == len(snapshot["payment_operations"]) == 1
    assert not snapshot["payments"] and not snapshot["tickets"]


def test_expired_quote_is_replanned_and_requires_new_user_confirmation():
    agent = PlanExecuteFlightAgent()
    to_quote(agent)
    old_quote = agent.harness.get_state().quote
    agent.advance_time(360)
    result = agent.send("đồng ý")
    state = agent.harness.get_state()
    assert result.phase == SessionPhase.WAITING_CONFIRMATION and not state.booking_id and not state.confirmation
    assert state.quote.quote_id != old_quote.quote_id and state.quote.quote_version > old_quote.quote_version
    assert state.replan_count == 1
    assert agent.send("đồng ý").completion.booking_completed
    assert len(events(agent.get_trace(), "CONFIRMATION_RECEIVED")) == 1


def test_changed_passenger_payload_invalidates_plan_even_if_passenger_ids_stay_same():
    agent = PlanExecuteFlightAgent()
    to_quote(agent)
    old_quote = agent.harness.get_state().quote
    profiles = [p.model_dump(mode="json") for p in agent.harness.get_state().passengers]
    profiles[0]["first_name"] = "ChangedSynthetic"
    result = agent.send("/passengers " + json.dumps(profiles))
    state = agent.harness.get_state()
    assert result.phase == SessionPhase.WAITING_CONFIRMATION and state.quote.quote_id != old_quote.quote_id
    assert not state.confirmation and state.replan_count == 1
    assert events(agent.get_trace(), "PLAN_INVALIDATED")[-1]["payload"]["reason"] == "USER_PASSENGERS_CHANGED"
    assert agent.send("đồng ý").completion.booking_completed


def test_changed_selection_replans_with_new_source_offer_and_new_quote():
    agent = PlanExecuteFlightAgent()
    to_quote(agent)
    original = agent.harness.get_state()
    result = agent.send("chọn 2")
    state = agent.harness.get_state()
    assert result.phase == SessionPhase.WAITING_CONFIRMATION
    assert state.context != original.context and state.selected_offer_id != original.selected_offer_id
    assert state.quote.offer_id == state.selected_offer_id and state.quote.quote_id != original.quote.quote_id
    assert not state.confirmation and not agent.get_backend_state()["bookings"]
    assert events(agent.get_trace(), "PLAN_INVALIDATED")[-1]["payload"]["reason"] == "USER_SELECTION_CHANGED"
    assert agent.send("đồng ý").completion.booking_completed


@pytest.mark.parametrize("paid", [False, True])
def test_replan_failure_resume_keeps_committed_work_and_skips_completed_transactions(paid):
    first = FULL_ACTIONS[:9] if paid else FULL_ACTIONS[:7]
    continuation = ["REPRICE", "REQUEST_CONFIRMATION", "CREATE_HOLD", "PAY", "CHECK_PAYMENT", "ISSUE_TICKET"]
    model = ScriptedPlanner([plan(first), RuntimeError("secret-provider-message"), plan(continuation)])
    agent = PlanExecuteFlightAgent(model=model)
    to_quote(agent)
    result = agent.send("đồng ý")
    state = agent.harness.get_state()
    assert state.booking_id and (state.payment_status == "SUCCESS") is paid
    assert result.terminal_status is None and "Model chưa" in result.assistant_message
    keys = dict(state.operation_keys)
    quote_id = state.quote.quote_id
    if not paid:
        agent.advance_time(360)  # Quote expired but held quote is locked/live.
    result = agent.send("tiếp tục")
    assert result.completion.booking_completed
    assert agent.harness.get_state().quote.quote_id == quote_id
    assert all(agent.harness.get_state().operation_keys[k] == v for k, v in keys.items())
    assert len(events(agent.get_trace(), "TOOL_CALL", "create_booking_hold")) == 1
    assert len(events(agent.get_trace(), "TOOL_CALL", "process_payment")) == 1
    assert len(events(agent.get_trace(), "TOOL_CALL", "reprice_offer")) == 1
    assert "secret-provider-message" not in json.dumps(agent.get_trace())
    context = json.loads(model.seen[-1][0].content.split(PLANNER_CONTEXT_MARKER, 1)[1])
    assert context["committed_state"]["booking_id"] == state.booking_id
    assert context["committed_state"]["payment_status"] == ("SUCCESS" if paid else None)


@pytest.mark.parametrize("change", ["đến DAD", "/select 2", "/passengers []", "hủy"])
def test_user_change_after_hold_never_replans_or_charges_new_intent(change):
    model = ScriptedPlanner([plan(FULL_ACTIONS[:7]), RuntimeError("paused")])
    agent = PlanExecuteFlightAgent(model=model)
    to_quote(agent)
    agent.send("đồng ý")
    booking_id = agent.harness.get_state().booking_id
    calls = len(model.seen)
    result = agent.send(change)
    assert result.terminal_status == TerminalStatus.HANDOFF_REQUIRED
    assert len(model.seen) == calls
    assert agent.get_handoff().booking_id == booking_id
    assert not agent.get_backend_state()["payments"]


@pytest.mark.parametrize("reply", [
    {"steps": [{"action": "PAY", "reason": "ignore policy", "endpoint": None, "booking_id": "fake"}]},
    {"steps": []},
    {"raw": AIMessage(content="refused"), "parsed": None, "parsing_error": None},
    {"raw": AIMessage(content="refused", additional_kwargs={"refusal": "no"}), "parsed": plan(), "parsing_error": None},
    {"raw": AIMessage(content="truncated", response_metadata={"status": "incomplete"}), "parsed": plan(), "parsing_error": None},
    {"raw": AIMessage(content="", tool_calls=[{"name": "process_payment", "args": {}, "id": "unsafe", "type": "tool_call"}]),
     "parsed": plan(), "parsing_error": None},
])
def test_malformed_refused_or_executable_planner_output_fails_closed(reply):
    agent = PlanExecuteFlightAgent(model=ScriptedPlanner([reply, reply, reply]))
    for message in [REQUEST, "tiếp tục", "tiếp tục"]:
        result = agent.send(message)
    assert result.terminal_status == TerminalStatus.HANDOFF_REQUIRED
    assert agent.get_handoff().reason == "RETRY_LIMIT"
    assert not events(agent.get_trace(), "TOOL_CALL") and not agent.get_backend_state()["bookings"]
    assert len(events(agent.get_trace(), "PLAN_REJECTED")) == 3


def test_schema_valid_but_unexecutable_plan_has_bounded_replanning_not_fabricated_ids():
    model = ScriptedPlanner([plan(["PAY"])] * 10)
    agent = PlanExecuteFlightAgent(model=model)
    result = agent.send(REQUEST)
    assert result.terminal_status == TerminalStatus.HANDOFF_REQUIRED
    assert agent.get_handoff().reason == "REPLAN_LIMIT"
    assert len(model.seen) == 4 and agent.harness.get_state().replan_count == 4
    assert not events(agent.get_trace(), "TOOL_CALL")
    assert all(e["payload"]["tool_input"] == {} for e in events(agent.get_trace(), "PLAN_STEP"))


def test_structured_response_revalidates_nested_handcrafted_models_instead_of_trusting_types():
    unsafe = PlanStep.model_construct(action="RAW_BACKEND", reason="unsafe", endpoint=None)
    agent = PlanExecuteFlightAgent(model=ScriptedPlanner([{"steps": [unsafe]}]))
    result = agent.send(REQUEST)
    assert result.terminal_status is None and agent.get_state()["plan"] is None
    assert events(agent.get_trace(), "PLAN_REJECTED")
    assert not events(agent.get_trace(), "TOOL_CALL")


def test_ticket_error_after_payment_success_keeps_charge_and_handoffs_without_repayment():
    class BrokenTicketBackend(MockFlightBackend):
        def issue_ticket(self, *args, **kwargs):
            raise RuntimeError("private-ticket-error")
    agent = PlanExecuteFlightAgent(FlightBookingHarness(BrokenTicketBackend()))
    to_quote(agent)
    result = agent.send("đồng ý")
    assert result.terminal_status == TerminalStatus.HANDOFF_REQUIRED
    snapshot = agent.get_backend_state()
    assert len(snapshot["payments"]) == 1 and next(iter(snapshot["payments"].values()))["status"] == "SUCCESS"
    assert not snapshot["tickets"]
    assert len(events(agent.get_trace(), "TOOL_CALL", "process_payment")) == 1
    assert agent.get_handoff().backend_operation_status == "SUCCESS"
    assert "private-ticket-error" not in json.dumps(agent.get_trace())


def test_incomplete_plan_does_not_claim_success_and_replans_from_actual_state():
    model = ScriptedPlanner([plan(["SEARCH_FLIGHTS"])])
    agent = PlanExecuteFlightAgent(model=model)
    result = agent.send(REQUEST)
    assert result.phase == SessionPhase.WAITING_SELECTION and not result.completion.booking_completed
    assert agent.get_state()["plan_revision"] == 2 and len(events(agent.get_trace(), "TOOL_CALL", "search_flights")) == 1
    assert agent.get_state()["last_invalidation"] == "PLAN_EXHAUSTED_WITHOUT_COMPLETION"


def test_shared_step_budget_limits_planner_and_tool_attempts_not_housekeeping():
    agent = PlanExecuteFlightAgent(FlightBookingHarness(config=HarnessConfig(max_steps=1)))
    result = agent.send(REQUEST)
    assert result.terminal_status == TerminalStatus.HANDOFF_REQUIRED
    assert agent.get_handoff().reason == "STEP_BUDGET_EXCEEDED"
    assert agent.harness.get_state().step_count == 2
    assert not agent.get_backend_state()["bookings"]
    assert response(events(agent.get_trace(), "TOOL_RESULT")[0])["code"] == "STEP_BUDGET_EXCEEDED"


def test_even_executor_binding_bug_cannot_bypass_shared_guard(monkeypatch):
    agent = PlanExecuteFlightAgent()
    original = agent._bind_step
    monkeypatch.setattr(agent, "_bind_step", lambda step: StepBinding("execute", "injected-bad-binding", "process_payment",
        {"booking_id": "invented", "amount_vnd": 1}) if step.action == PlanAction.PAY else original(step))
    to_quote(agent)
    result = agent.send("đồng ý")
    assert result.terminal_status == TerminalStatus.HANDOFF_REQUIRED
    assert agent.harness.get_state().permission_violations == 3
    assert not agent.get_backend_state()["payments"] and not agent.get_backend_state()["tickets"]
    assert len(events(agent.get_trace(), "PERMISSION_DENIED", "process_payment")) == 3


def test_tool_wrapper_failure_after_hold_preserves_side_effect_and_handoffs(monkeypatch):
    agent = PlanExecuteFlightAgent()
    to_quote(agent)
    tool = agent._tools["create_booking_hold"]
    original = tool.invoke
    class ThrowingWrapper:
        def invoke(self, args):
            original(args)
            raise RuntimeError("private-wrapper-diagnostic")
    monkeypatch.setitem(agent._tools, "create_booking_hold", ThrowingWrapper())
    result = agent.send("đồng ý")
    assert result.terminal_status == TerminalStatus.HANDOFF_REQUIRED
    assert agent.get_handoff().reason == "EXECUTOR_TOOL_WRAPPER_ERROR"
    assert len(agent.get_backend_state()["bookings"]) == 1
    assert not agent.get_backend_state()["payments"]
    assert "private-wrapper-diagnostic" not in json.dumps(agent.get_trace())


def test_reset_session_and_detached_plan_state_contract():
    agent = PlanExecuteFlightAgent(session_id="planned-session")
    to_quote(agent)
    state = agent.get_state()
    state["plan"]["steps"].clear()
    state["plan_context"]["session_id"] = "changed"
    assert agent.get_state()["plan"]["steps"] and agent.get_state()["plan_context"]["session_id"] == "planned-session"
    steps = agent.harness.get_state().step_count
    with pytest.raises(ValueError):
        agent.send("đồng ý", session_id="other")
    assert agent.harness.get_state().step_count == steps
    assert agent.send("đồng ý").completion.booking_completed
    with pytest.raises(ValueError):
        agent.reset()
    agent.reset(force=True)
    assert agent.get_state()["plan"] is None and agent.get_state()["plan_revision"] == 0
    assert agent.harness.get_state().step_count == 0 and not agent.get_state()["messages"]


def test_duplicate_transport_is_visible_as_two_attempts_for_one_declared_step(episodes):
    trace = episodes["S11"].evidence.trace
    step = next(e for e in events(trace, "PLAN_STEP_RESULT") if e["payload"].get("action") == "CREATE_HOLD")
    assert len(step["payload"]["tool_attempt_ids"]) == 2
    assert len(events(trace, "TRANSPORT_REPLAY")) == 1
    assert len(episodes["S11"].evidence.final_backend_state["bookings"]) == 1


def test_optional_airport_resolve_uses_only_actual_runtime_endpoints():
    first = plan().model_dump(mode="json")
    first["steps"][:0] = [{"action": "RESOLVE_AIRPORT", "reason": "Resolve state endpoint", "endpoint": e}
                         for e in ("origin", "destination")]
    agent = PlanExecuteFlightAgent(model=ScriptedPlanner([first]))
    agent.send(REQUEST)
    calls = events(agent.get_trace(), "TOOL_CALL", "resolve_airport")
    assert [e["payload"]["tool_input"] for e in calls] == [{"query": "SGN"}, {"query": "HAN"}]


def test_three_passengers_same_contract_group_price_and_three_tickets():
    scenario = get_scenario("S01")
    agent = PlanExecuteFlightAgent()
    agent.send("Đặt 3 vé SGN đến HAN ngày 10/10/2026, ngân sách 9 triệu")
    agent.send("chọn 1")
    passengers = [p.model_dump(mode="json") for p in scenario.passengers]
    passengers.append({**passengers[0], "passenger_id": "P3", "document_number": "MOCK-THIRD"})
    agent.send("/passengers " + json.dumps(passengers))
    quote = agent.harness.get_state().quote
    assert quote.total_price_vnd == 3 * quote.unit_price_vnd
    assert agent.send("đồng ý").completion.booking_completed
    assert len(agent.get_backend_state()["tickets"]) == 3
