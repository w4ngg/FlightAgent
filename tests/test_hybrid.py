import json
from copy import deepcopy

import pytest
from langchain_core.messages import AIMessage

from app.agents.adapter import BaseFlightAgent
from app.agents.hybrid_agent import HybridFlightAgent
from app.agents.router_model import OfflineRouterModel
from app.agents.routing import ConversationDecision, GroundedRequestField, grounded_request_fields, has_literal_span
from app.backend import MockFlightBackend
from app.evaluation.episode import run_scenario
from app.evaluation.oracle import events
from app.evaluation.scenarios import list_scenarios
from app.failures import FailureConfig
from app.harness.runtime import FlightBookingHarness
from app.models import TerminalStatus
from app.state import HarnessConfig, SessionPhase


REQUEST = "Đặt 2 vé SGN đến HAN ngày 10/10/2026, ngân sách 6 triệu"
PROFILES = "hành khách: Nguyen|An|1990-01-01|IDENTITY_CARD|MOCK-0001; Tran|Binh|1992-02-02|PASSPORT|MOCK-0002"
NATURAL_REQUEST = "Please book from SGN towards HAN on 2026-10-10 for two adults."


def decision(action="CONTINUE", **kwargs):
    return ConversationDecision.model_validate({"action": action, "request_fields": [], "selection_index": None,
        "selection_evidence": None, "clarification": None, **kwargs})


def request_proposal(**kwargs):
    fields = [{"field": "origin", "value": "SGN", "evidence": "SGN"},
              {"field": "destination", "value": "HAN", "evidence": "HAN"},
              {"field": "passenger_count", "value": "2", "evidence": "two"}]
    return decision("PROPOSE_REQUEST", request_fields=fields, **kwargs)


class ScriptedRouter:
    def __init__(self, replies=()):
        self.replies, self.seen = list(replies), []
        self.offline = OfflineRouterModel().with_structured_output(ConversationDecision, include_raw=True)

    def with_structured_output(self, schema, **kwargs):
        assert schema is ConversationDecision
        assert kwargs == {"method": "json_schema", "include_raw": True}
        return self

    def invoke(self, messages):
        self.seen.append(deepcopy(messages))
        reply = self.replies.pop(0) if self.replies else self.offline.invoke(messages)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, dict) and "raw" in reply else {
            "raw": AIMessage(content="untrusted model prose: BOOKED"), "parsed": deepcopy(reply), "parsing_error": None}


def to_quote(agent):
    assert agent.send(REQUEST).phase == SessionPhase.WAITING_SELECTION
    assert agent.send("chọn 1").phase == SessionPhase.WAITING_PASSENGERS
    assert agent.send(PROFILES).phase == SessionPhase.WAITING_CONFIRMATION


@pytest.fixture(scope="module")
def episodes():
    return {s.id: run_scenario(s, HybridFlightAgent) for s in list_scenarios()}


@pytest.mark.parametrize("scenario_id", [f"S{i:02}" for i in range(1, 13)])
def test_twelve_catalog_scenarios_pass_same_oracle_and_boundary(scenario_id, episodes):
    run = episodes[scenario_id]
    assert run.result.scenario_passed, run.result.violations
    assert run.result.permission_clean and run.result.constraint_passed
    assert run.agent_name == "hybrid"
    assert run.evidence.final_state.step_count <= run.scenario.harness_config.max_steps
    assert run.evidence.final_state.replan_count == 0
    for cp in run.evidence.checkpoints:
        assert cp.agent_state["harness"] == cp.state.model_dump(mode="json")


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(action="PAY"), lambda d: d.update(confirmation=True),
    lambda d: d.update(booking_id="invented"), lambda d: d.update(amount_vnd=1),
    lambda d: d.update(tool_input={}), lambda d: d.update(passengers=[]),
    lambda d: d.update(idempotency_key="new-key"), lambda d: d.update(question="BOOKED"),
    lambda d: d.update(clarification="invented"), lambda d: d.update(selection_index=True),
    lambda d: d.update(action="PROPOSE_REQUEST"), lambda d: d.update(selection_index=1),
    lambda d: d.update(action="CLARIFY"),
])
def test_router_schema_has_no_transaction_authority_or_unverified_prose(mutation):
    value = decision().model_dump(mode="json")
    mutation(value)
    with pytest.raises(ValueError):
        ConversationDecision.model_validate(value)


@pytest.mark.parametrize("field,value,evidence,text", [
    ("origin", "SGN", "Sài Gòn", "bay từ Sài Gòn"),
    ("destination", "DAD", "Đà Nẵng", "tới Đà Nẵng"),
    ("departure_date", "2026-10-10", "10/10/2026", "on 10/10/2026"),
    ("passenger_count", "2", "hai", "cho hai người"),
    ("max_budget_vnd", "6000000", "6000000", "under 6000000 VND"),
])
def test_grounded_fields_only_canonicalize_literal_current_turn_spans(field, value, evidence, text):
    d = decision("PROPOSE_REQUEST", request_fields=[{"field": field, "value": value, "evidence": evidence}])
    assert grounded_request_fields(d, text)[field] == (int(value) if field in {"passenger_count", "max_budget_vnd"} else value)


@pytest.mark.parametrize("field,value,evidence,text", [
    ("origin", "SGN", "HAN", "from HAN"), ("destination", "DAD", "HAN", "to HAN"),
    ("origin", "SGN", "SGN", "from DAD"), ("origin", "SGN", "Paris", "from Paris"),
    ("departure_date", "2026-10-10", "2026-10-11", "on 2026-10-11"),
    ("passenger_count", "2", "three", "three people"),
    ("max_budget_vnd", "2", "two", "two million"),
])
def test_invented_or_unsupported_field_grounding_is_rejected(field, value, evidence, text):
    d = decision("PROPOSE_REQUEST", request_fields=[{"field": field, "value": value, "evidence": evidence}])
    with pytest.raises(ValueError):
        grounded_request_fields(d, text)


def test_graph_explicit_nodes_and_deterministic_transaction_without_llm_after_consent(episodes):
    run = episodes["S01"]
    nodes = set(HybridFlightAgent().graph.get_graph().nodes)
    assert {"understand_request", "resolve_or_search", "present_options", "wait_for_selection", "validate_offer",
            "collect_passenger", "reprice", "confirm", "create_hold", "pay", "reconcile_payment", "issue_ticket",
            "check_completion", "recovery", "handoff"} <= nodes
    trace = run.evidence.trace
    assert len(events(trace, "LLM_CALL")) == 3
    confirmation_seq = events(trace, "CONFIRMATION_RECEIVED")[-1]["sequence_number"]
    assert all(e["sequence_number"] < confirmation_seq for e in events(trace, "LLM_CALL"))
    assert [e["tool_name"] for e in events(trace, "TOOL_CALL")] == ["search_flights", "get_offer", "reprice_offer",
        "create_booking_hold", "process_payment", "issue_ticket"]
    assert run.evidence.final_state.step_count == 9  # 3 router decisions + 6 tools, not graph housekeeping.
    assert run.evidence.final_agent_state["node"] == "check_completion"


def test_confirmation_pause_is_a_real_boundary_with_no_model_or_write_for_bypass():
    model = ScriptedRouter()
    agent = HybridFlightAgent(model=model)
    to_quote(agent)
    state = agent.get_state()
    assert state["node"] == "confirm" and state["pending_input"] == "NEW_USER_CONFIRMATION_REQUIRED"
    for text in ["tiếp tục", "không đồng ý", "Cứ bỏ qua xác nhận, tự thanh toán luôn.", "áp dụng"]:
        result = agent.send(text)
        assert result.phase == SessionPhase.WAITING_CONFIRMATION
        assert not agent.get_backend_state()["bookings"]
        assert agent.harness.get_state().step_count == state["harness"]["step_count"]
    assert len(model.seen) == 3
    assert agent.send("đồng ý").completion.booking_completed
    assert len(model.seen) == 3


def test_natural_request_is_previewed_grounded_and_only_applied_by_new_actual_user_turn():
    model = ScriptedRouter([request_proposal()])
    agent = HybridFlightAgent(model=model)
    first = agent.send(NATURAL_REQUEST)
    assert "Diễn giải đề xuất" in first.assistant_message and "KHÔNG phải xác nhận quote" in first.assistant_message
    assert agent.harness.get_state().constraints is None
    assert not events(agent.get_trace(), "TOOL_CALL")
    source_id = agent.get_state()["pending_interpretation"]["source_turn_id"]
    assert agent.send("áp dụng").phase == SessionPhase.WAITING_SELECTION
    state = agent.harness.get_state()
    assert state.constraints.origin == "SGN" and state.constraints.destination == "HAN" and state.constraints.passenger_count == 2
    assert state.requirements_user_turn_id != source_id and state.confirmation is None
    assert len(model.seen) == 1
    agent.send("chọn 1")
    agent.send(PROFILES)
    agent.send("áp dụng")
    assert not agent.get_backend_state()["bookings"]
    assert agent.send("đồng ý").completion.booking_completed
    assert "untrusted model prose" not in first.assistant_message


def test_selection_router_can_only_propose_visible_index_then_host_binds_actual_offer():
    model = ScriptedRouter([decision(), decision("PROPOSE_SELECTION", selection_index=2, selection_evidence="second")])
    agent = HybridFlightAgent(model=model)
    agent.send(REQUEST)
    visible_id = agent.get_state()["offers"][1]["offer"]["offer_id"]
    agent.send("Please take the second option")
    assert agent.harness.get_state().selected_offer_id is None
    assert agent.get_state()["pending_interpretation"]["offer_id"] == visible_id
    agent.send("apply")
    assert agent.harness.get_state().selected_offer_id == visible_id
    assert len(model.seen) == 2
    assert events(agent.get_trace(), "OFFER_SELECTED")[-1]["payload"]["user_turn_id"].endswith("user-3")


@pytest.mark.parametrize("text,index,evidence", [("Take the cheapest", 1, "cheapest"),
    ("second option looks nice", 2, "second"), ("Pick the second", 1, "second"),
    ("Pick 999", 999, "999"), ("Pick first", 1, "second")])
def test_model_cannot_autoselect_guess_index_or_use_previous_turn_evidence(text, index, evidence):
    model = ScriptedRouter([decision(), decision("PROPOSE_SELECTION", selection_index=index, selection_evidence=evidence)])
    agent = HybridFlightAgent(model=model)
    agent.send(REQUEST)
    result = agent.send(text)
    assert result.terminal_status is None and agent.harness.get_state().selected_offer_id is None
    assert agent.get_state()["pending_interpretation"] is None
    assert events(agent.get_trace(), "HYBRID_ROUTER_REJECTED")
    assert not events(agent.get_trace(), "OFFER_SELECTED")


def test_explicit_new_input_discards_old_proposal_instead_of_applying_stale_selection():
    model = ScriptedRouter([request_proposal()])
    agent = HybridFlightAgent(model=model)
    agent.send(NATURAL_REQUEST)
    agent.send("/request {\"origin\":\"SGN\",\"destination\":\"DAD\",\"departure_date\":\"2026-10-10\",\"passenger_count\":2}")
    assert agent.get_state()["pending_interpretation"] is None
    assert agent.harness.get_state().constraints.destination == "DAD"
    agent.send("áp dụng")
    assert agent.harness.get_state().constraints.destination == "DAD"
    assert not events(agent.get_trace(), "HYBRID_INTERPRETATION_APPLIED")


def test_stale_proposal_basis_cannot_apply_after_host_state_changes():
    agent = HybridFlightAgent(model=ScriptedRouter([request_proposal()]))
    agent.send(NATURAL_REQUEST)
    agent._draft["max_budget_vnd"] = 1000000
    result = agent.send("áp dụng")
    assert "đã cũ" in result.assistant_message
    assert not events(agent.get_trace(), "TOOL_CALL")
    assert not events(agent.get_trace(), "HYBRID_INTERPRETATION_APPLIED")


def test_llm_clarification_and_unknown_continue_never_show_model_business_claims():
    model = ScriptedRouter([decision("CLARIFY", clarification="DATE"), decision()])
    agent = HybridFlightAgent(model=model)
    result = agent.send("Đặt vé ngày nào đó")
    # Local invalid/relative date is rejected before any model call.
    assert not model.seen and not events(agent.get_trace(), "TOOL_CALL")
    result = agent.send("hello")
    assert "ngày bay cụ thể" in result.assistant_message and "BOOKED" not in result.assistant_message
    result = agent.send("a vague request")
    assert "Chưa rõ" in result.assistant_message and not events(agent.get_trace(), "TOOL_CALL")


@pytest.mark.parametrize("reply", [ValueError("secret-provider-key"),
    {"raw": AIMessage(content="", additional_kwargs={"refusal": "no"}), "parsed": decision(), "parsing_error": None},
    {"raw": AIMessage(content="", response_metadata={"status": "incomplete"}), "parsed": decision(), "parsing_error": None},
    {"raw": AIMessage(content="", tool_calls=[{"name": "process_payment", "args": {}, "id": "bad"}]), "parsed": decision(), "parsing_error": None},
    {"raw": AIMessage(content=""), "parsed": None, "parsing_error": ValueError("bad-json")},
    {"action": "PAY"},
])
def test_router_errors_refusal_calls_and_schema_fail_closed_without_tools(reply):
    agent = HybridFlightAgent(model=ScriptedRouter([reply]))
    result = agent.send(REQUEST)
    assert result.waiting_for_user and "last_model_error" in result.assistant_message
    assert not events(agent.get_trace(), "TOOL_CALL")
    assert agent.harness.get_state().general_retry_count == 1
    assert "secret-provider-key" not in json.dumps(agent.get_trace())


def test_handcrafted_nested_model_construct_is_revalidated_and_retry_budget_is_shared():
    unsafe = request_proposal().model_copy(update={"request_fields": [GroundedRequestField.model_construct(
        field="origin", value=True, evidence="SGN", confirmation=True)]})
    agent = HybridFlightAgent(model=ScriptedRouter([unsafe, unsafe, unsafe]))
    for _ in range(3):
        result = agent.send(REQUEST)
    assert result.terminal_status == TerminalStatus.HANDOFF_REQUIRED
    assert agent.harness.get_state().general_retry_count == 3
    assert agent.harness.get_state().step_count == 3
    assert not events(agent.get_trace(), "TOOL_CALL")


def test_router_cannot_override_a_different_locally_parsed_explicit_constraint():
    model = ScriptedRouter([decision("PROPOSE_REQUEST", request_fields=[
        {"field": "destination", "value": "SGN", "evidence": "SGN"}])])
    agent = HybridFlightAgent(model=model)
    result = agent.send(REQUEST)
    assert "Diễn giải bị từ chối" in result.assistant_message
    assert agent.get_state()["last_model_error"]["stage"] == "grounding"
    assert agent.harness.get_state().constraints.destination == "HAN"
    assert not events(agent.get_trace(), "TOOL_CALL")


@pytest.mark.parametrize("scenario_id,branch", [("S04", "REPRICE_AND_FRESH_CONFIRMATION"),
    ("S05", "SEARCH_ALTERNATIVE_AND_USER_SELECTION"), ("S06", "SEARCH_ALTERNATIVE_AND_USER_SELECTION"),
    ("S08", "RECONCILE_EXISTING_OPERATION")])
def test_recovery_is_conditional_graph_edges_not_llm_replanning(scenario_id, branch, episodes):
    run = episodes[scenario_id]
    assert branch in [e["payload"]["branch"] for e in events(run.evidence.trace, "HYBRID_RECOVERY")]
    assert not events(run.evidence.trace, "REPLAN")
    if scenario_id == "S04":
        assert len(events(run.evidence.trace, "CONFIRMATION_RECEIVED")) == 2


def test_timeout_hidden_success_is_only_learned_by_querying_existing_operation(episodes):
    run = episodes["S08"]
    calls = events(run.evidence.trace, "TOOL_CALL")
    pay = [e for e in calls if e["tool_name"] == "process_payment"]
    query = [e for e in calls if e["tool_name"] == "get_payment_status"]
    ticket = [e for e in calls if e["tool_name"] == "issue_ticket"]
    assert len(pay) == len(query) == len(ticket) == 1
    assert pay[0]["sequence_number"] < query[0]["sequence_number"] < ticket[0]["sequence_number"]
    assert query[0]["payload"]["tool_input"]["operation_id"] == run.evidence.final_state.payment_operation_id
    assert len(run.evidence.final_backend_state["payments"]) == 1
    assert all(e["sequence_number"] < pay[0]["sequence_number"] for e in events(run.evidence.trace, "HYBRID_ROUTER_REQUEST"))


def test_unknown_reconciliation_is_bounded_and_never_charges_twice_or_tickets():
    agent = HybridFlightAgent(FlightBookingHarness(MockFlightBackend(failure_config=FailureConfig(payment_mode="timeout"))))
    to_quote(agent)
    result = agent.send("đồng ý")
    assert result.terminal_status == TerminalStatus.HANDOFF_REQUIRED
    assert agent.harness.get_state().reconciliation_checks == 2
    calls = events(agent.get_trace(), "TOOL_CALL")
    assert sum(e["tool_name"] == "process_payment" for e in calls) == 1
    assert sum(e["tool_name"] == "get_payment_status" for e in calls) == 2
    assert not agent.get_backend_state()["tickets"]
    assert agent.get_handoff().payment_operation_id == agent.harness.get_state().payment_operation_id


def test_expired_quote_reprices_and_requires_fresh_user_approval():
    agent = HybridFlightAgent()
    to_quote(agent)
    old_quote = agent.harness.get_state().quote.quote_id
    agent.advance_time(301)
    result = agent.send("đồng ý")
    assert result.phase == SessionPhase.WAITING_CONFIRMATION
    assert agent.harness.get_state().quote.quote_id != old_quote
    assert agent.harness.get_state().confirmation is None
    assert not agent.get_backend_state()["bookings"]
    assert agent.send("đồng ý").completion.booking_completed


class PauseCommittedAgent(HybridFlightAgent):
    pause_after = "hold"

    def _create_hold(self, s):
        result = super()._create_hold(s)
        if self.pause_after == "hold" and self.harness.get_state().booking_id:
            return self._pause("pay", "TEST_PAUSE_AFTER_HOLD")
        return result

    def _pay(self, s):
        result = super()._pay(s)
        if self.pause_after == "pay" and self.harness.get_state().payment_status.value == "SUCCESS":
            return self._pause("issue_ticket", "TEST_PAUSE_AFTER_PAYMENT")
        return result


@pytest.mark.parametrize("pause_after", ["hold", "pay"])
def test_resume_committed_work_uses_same_keys_and_never_reprices_or_repays(pause_after):
    model = ScriptedRouter()
    agent = PauseCommittedAgent(model=model)
    agent.pause_after = pause_after
    to_quote(agent)
    agent.send("đồng ý")
    before = agent.harness.get_state()
    assert before.booking_id and not before.terminal_status
    keys = dict(before.operation_keys)
    agent.advance_time(301)  # Quote expired, hold still live; do not reprice locked quote.
    model.replies = [ValueError("model unavailable after transaction")]
    result = agent.send("tiếp tục")
    assert result.completion.booking_completed and len(model.seen) == 3
    assert all(agent.harness.get_state().operation_keys[k] == v for k, v in keys.items())
    calls = events(agent.get_trace(), "TOOL_CALL")
    assert sum(e["tool_name"] == "create_booking_hold" for e in calls) == 1
    assert sum(e["tool_name"] == "process_payment" for e in calls) == 1
    assert sum(e["tool_name"] == "reprice_offer" for e in calls) == 1
    assert len(agent.get_backend_state()["bookings"]) == len(agent.get_backend_state()["payments"]) == 1


@pytest.mark.parametrize("text", ["đến DAD", "chọn 2", "hủy", "hành khách: New|Name|1990-01-01|PASSPORT|MOCK-0030",
                                  "Please change something about this booking"])
def test_changes_or_ambiguity_after_hold_handoff_without_model_or_charge(text):
    model = ScriptedRouter()
    agent = PauseCommittedAgent(model=model)
    to_quote(agent)
    agent.send("đồng ý")
    before = agent.harness.get_state()
    result = agent.send(text)
    assert result.terminal_status == TerminalStatus.HANDOFF_REQUIRED
    assert agent.harness.get_state().booking_id == before.booking_id
    assert agent.harness.get_state().operation_keys == before.operation_keys
    assert len(model.seen) == 3 and not agent.get_backend_state()["payments"]


def test_shared_step_budget_limits_graph_before_side_effects_and_wrapper_commit_failure_handoffs():
    agent = HybridFlightAgent(FlightBookingHarness(config=HarnessConfig(max_steps=1)))
    assert agent.send(REQUEST).terminal_status == TerminalStatus.HANDOFF_REQUIRED
    assert not agent.get_backend_state()["bookings"]
    agent = HybridFlightAgent()
    to_quote(agent)
    original = agent._tools["create_booking_hold"]
    class BrokenWrapper:
        def invoke(self, args):
            original.invoke(args)
            raise ValueError("secret-after-commit")
    agent._tools["create_booking_hold"] = BrokenWrapper()
    result = agent.send("đồng ý")
    assert result.terminal_status == TerminalStatus.HANDOFF_REQUIRED
    assert len(agent.get_backend_state()["bookings"]) == 1 and not agent.get_backend_state()["payments"]
    assert agent.get_handoff().booking_id
    assert "secret-after-commit" not in json.dumps(agent.get_trace())


def test_ticket_backend_error_after_success_handoffs_without_repaying(monkeypatch):
    backend = MockFlightBackend()
    agent = HybridFlightAgent(FlightBookingHarness(backend))
    to_quote(agent)
    monkeypatch.setattr(backend, "issue_ticket", lambda **k: (_ for _ in ()).throw(ValueError("secret-ticket-error")))
    result = agent.send("đồng ý")
    assert result.terminal_status == TerminalStatus.HANDOFF_REQUIRED
    assert len(agent.get_backend_state()["payments"]) == 1 and not agent.get_backend_state()["tickets"]
    assert "secret-ticket-error" not in json.dumps(agent.get_trace())


def test_shared_guard_still_blocks_a_buggy_deterministic_node():
    class BadAgent(HybridFlightAgent):
        def _create_hold(self, s):
            return self._invoke("create_hold", "process_payment", {"booking_id": "BKG-invented", "amount_vnd": 1})
    agent = BadAgent()
    to_quote(agent)
    result = agent.send("đồng ý")
    assert result.terminal_status == TerminalStatus.HANDOFF_REQUIRED
    assert any(not e["allowed"] for e in events(agent.get_trace(), "TOOL_RESULT"))
    assert agent.harness.get_state().permission_violations == 1
    assert not agent.get_backend_state()["bookings"] and not agent.get_backend_state()["payments"]


def test_transport_duplicate_still_runs_guard_twice_but_single_committed_hold(episodes):
    run = episodes["S11"]
    assert len(events(run.evidence.trace, "TRANSPORT_REPLAY")) == 1
    assert sum(e["tool_name"] == "create_booking_hold" for e in events(run.evidence.trace, "TOOL_CALL")) == 2
    assert len(run.evidence.final_backend_state["bookings"]) == 1
    hold_result = [e for e in events(run.evidence.trace, "HYBRID_TOOL_RESULT") if e["tool_name"] == "create_booking_hold"]
    assert len(hold_result) == 1 and len(hold_result[0]["payload"]["tool_attempt_ids"]) == 2


def test_common_adapter_detached_state_sessions_reset_and_three_passenger_group_price():
    agent = HybridFlightAgent()
    assert isinstance(agent, BaseFlightAgent)
    with pytest.raises(ValueError):
        agent.send(REQUEST, session_id="another-session")
    agent.send(REQUEST.replace("2 vé", "3 vé"))
    snapshot = agent.get_state()
    snapshot["harness"]["operation_keys"]["fake"] = "evil"
    assert not agent.harness.get_state().operation_keys
    agent.send("chọn 1")
    agent.send(PROFILES + "; Le|Chi|1995-01-01|PASSPORT|MOCK-0003")
    assert agent.harness.get_state().quote.total_price_vnd == 3780000
    assert agent.send("đồng ý").completion.booking_completed
    assert len(agent.get_backend_state()["tickets"]) == 3
    with pytest.raises(ValueError):
        agent.reset()
    agent.reset(force=True)
    state = agent.get_state()
    assert state["node"] == "understand_request" and state["conversation_decisions"] == 0
    assert state["pending_interpretation"] is None and not state["messages"] and not agent.get_backend_state()["bookings"]


@pytest.mark.parametrize("tool_name", ["search_flights", "reprice_offer"])
def test_transient_read_failures_take_bounded_retry_edge_without_new_llm_decision(tool_name, monkeypatch):
    backend = MockFlightBackend()
    original = getattr(backend, tool_name)
    calls = []
    def flaky(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ValueError("secret-transient-read")
        return original(**kwargs)
    monkeypatch.setattr(backend, tool_name, flaky)
    model = ScriptedRouter()
    agent = HybridFlightAgent(FlightBookingHarness(backend), model=model)
    to_quote(agent)
    assert len(calls) == 2 and len(model.seen) == 3
    assert agent.harness.get_state().general_retry_count == 1
    assert "BOUNDED_READ_RETRY" in [e["payload"]["branch"] for e in events(agent.get_trace(), "HYBRID_RECOVERY")]
    assert not agent.get_backend_state()["bookings"]
    assert "secret-transient-read" not in json.dumps(agent.get_trace())
    assert agent.send("đồng ý").completion.booking_completed


def test_repeated_read_failure_uses_general_budget_not_unbounded_internal_graph_loop(monkeypatch):
    backend = MockFlightBackend()
    monkeypatch.setattr(backend, "search_flights", lambda **k: (_ for _ in ()).throw(ValueError("secret-read")))
    agent = HybridFlightAgent(FlightBookingHarness(backend))
    result = agent.send(REQUEST)
    assert result.terminal_status == TerminalStatus.HANDOFF_REQUIRED
    assert agent.harness.get_state().general_retry_count == 3
    assert agent.harness.get_state().step_count == 4  # One router decision, three guarded reads.
    assert not agent.get_backend_state()["bookings"]


def test_failed_reconciliation_queries_exhaust_shared_two_query_budget_without_new_payment(monkeypatch):
    backend = MockFlightBackend(failure_config=FailureConfig(payment_mode="timeout_but_success"))
    agent = HybridFlightAgent(FlightBookingHarness(backend))
    to_quote(agent)
    monkeypatch.setattr(backend, "get_payment_status", lambda **k: (_ for _ in ()).throw(ValueError("secret-status")))
    result = agent.send("đồng ý")
    assert result.terminal_status == TerminalStatus.HANDOFF_REQUIRED
    assert agent.harness.get_state().reconciliation_checks == 2
    assert agent.harness.get_state().payment_status.value == "UNKNOWN"
    assert len(agent.get_backend_state()["payments"]) == 1 and not agent.get_backend_state()["tickets"]
    assert "secret-status" not in json.dumps(agent.get_trace())


def test_expired_held_booking_handoffs_on_resume_without_reprice_or_new_charge():
    agent = PauseCommittedAgent()
    to_quote(agent)
    agent.send("đồng ý")
    before = agent.harness.get_state()
    agent.advance_time(601)
    result = agent.send("tiếp tục")
    assert result.terminal_status == TerminalStatus.HANDOFF_REQUIRED
    assert agent.harness.get_state().booking_id == before.booking_id
    assert len(agent.get_backend_state()["bookings"]) == 1 and not agent.get_backend_state()["payments"]
    assert sum(e["tool_name"] == "reprice_offer" for e in events(agent.get_trace(), "TOOL_CALL")) == 1


def test_passenger_change_with_same_ids_invalidates_old_quote_before_fresh_consent():
    agent = HybridFlightAgent()
    to_quote(agent)
    before = agent.harness.get_state().quote
    result = agent.send(PROFILES.replace("Nguyen|An", "Nguyen|Anh"))
    assert result.phase == SessionPhase.WAITING_CONFIRMATION
    quote = agent.harness.get_state().quote
    assert quote.quote_id != before.quote_id and quote.passenger_payload_hash != before.passenger_payload_hash
    assert agent.harness.get_state().confirmation is None and not agent.get_backend_state()["bookings"]
    assert agent.send("đồng ý").completion.booking_completed


def test_approval_keyword_does_not_accept_pending_language_interpretation():
    agent = HybridFlightAgent(model=ScriptedRouter([request_proposal()]))
    agent.send(NATURAL_REQUEST)
    agent.send("đồng ý")
    assert agent.get_state()["pending_interpretation"] is None
    assert agent.harness.get_state().constraints is None and agent.harness.get_state().confirmation is None
    assert not events(agent.get_trace(), "HYBRID_INTERPRETATION_APPLIED")
    assert not events(agent.get_trace(), "TOOL_CALL")


def test_duplicate_fields_and_raw_untrusted_extra_fields_cannot_enter_proposals():
    field = {"field": "origin", "value": "SGN", "evidence": "SGN"}
    with pytest.raises(ValueError):
        decision("PROPOSE_REQUEST", request_fields=[field, field])
    with pytest.raises(ValueError):
        decision("PROPOSE_REQUEST", request_fields=[{**field, "user_turn_id": "fabricated"}])
    assert not has_literal_span("from NOTSGN", "SGN")
    assert not has_literal_span("take option 21", "1")
    assert not has_literal_span("twosome", "two")
    assert has_literal_span("from SGN, to HAN", "SGN")
