from copy import deepcopy

import pytest
from langchain_core.messages import AIMessage

from app.agents.models import OfflineBookingModel
from app.agents.react_agent import ReActFlightAgent
from app.evaluation.contracts import Scenario, UserActionKind
from app.evaluation.episode import run_scenario
from app.evaluation.oracle import ScenarioOracle, events, response
from app.evaluation.scenarios import get_scenario, list_scenarios


@pytest.fixture(scope="module")
def episodes():
    return {s.id: run_scenario(s, lambda h: ReActFlightAgent(h)) for s in list_scenarios()}


def evaluate(run, evidence):
    return ScenarioOracle().evaluate(run.scenario, evidence)


def sync_final_backend(evidence):
    evidence.checkpoints[-1] = evidence.checkpoints[-1].model_copy(
        update={"backend_state": deepcopy(evidence.final_backend_state)})


def sync_final_agent(evidence):
    evidence.checkpoints[-1] = evidence.checkpoints[-1].model_copy(
        update={"agent_state": deepcopy(evidence.final_agent_state)})


def omit_events(evidence, predicate):
    """Reindex remaining events/checkpoints so ordering tests cannot rely on gaps."""
    original = list(evidence.trace)
    evidence.trace[:] = [e for e in original if not predicate(e)]
    for checkpoint_index, checkpoint in enumerate(evidence.checkpoints):
        count = sum(e["sequence_number"] <= checkpoint.trace_sequence for e in evidence.trace)
        evidence.checkpoints[checkpoint_index] = checkpoint.model_copy(update={"trace_sequence": count})
    for number, event in enumerate(evidence.trace, 1):
        event["sequence_number"] = number


@pytest.mark.parametrize("scenario_id", ["S01", "S02", "S07", "S08", "S10"])
def test_oracle_is_pure_and_json_round_trip_reproducible(scenario_id, episodes):
    run = episodes[scenario_id]
    before = run.evidence.model_dump_json()
    restored = type(run.evidence).model_validate_json(before)
    assert evaluate(run, restored) == run.result
    assert run.evidence.model_dump_json() == before


def test_assistant_success_claim_cannot_supply_missing_booking(episodes):
    run = episodes["S02"]
    evidence = run.evidence.model_copy(deep=True)
    evidence.final_agent_state["messages"].append({"type": "ai", "data": {"content": "BOOKED! Payment SUCCESS, 2 tickets"}})
    sync_final_agent(evidence)
    report = evaluate(run, evidence)
    assert report.scenario_passed and not report.booking_completed
    # A claimed success still fails the happy-path oracle on the same evidence.
    assert not ScenarioOracle().evaluate(get_scenario("S01"), evidence).scenario_passed


@pytest.mark.parametrize("mutation,check", [
    (lambda e: e.trace[0].update(sequence_number=99), "trace_integrity"),
    (lambda e: e.trace[0].update(timestamp="2026-10-04T09:00:00+07:00"), "trace_integrity"),
    (lambda e: events(e.trace, "TOOL_RESULT")[0].update(attempt_id="missing-attempt"), "trace_integrity"),
    (lambda e: events(e.trace, "TOOL_ALLOWED")[0].update(allowed=False), "trace_integrity"),
    (lambda e: events(e.trace, "USER_MESSAGE")[0]["payload"]["turn"].update(turn_number=2), "user_ingress_and_adapter_state"),
    (lambda e: events(e.trace, "CONFIRMATION_RECEIVED")[0]["payload"].update(user_turn_id="model-generated"), "confirmation_sources"),
    (lambda e: events(e.trace, "CONFIRMATION_RECEIVED")[0]["payload"]["snapshot"].update(total_price_vnd=1), "confirmation_sources"),
    (lambda e: e.initial_backend_state.update(seed=7), "initial_setup"),
    (lambda e: e.initial_backend_state.update(current_time="2026-10-05T10:00:00+07:00"), "initial_setup"),
])
def test_wrong_source_order_or_setup_fails_closed(mutation, check, episodes):
    run = episodes["S01"]
    evidence = run.evidence.model_copy(deep=True)
    mutation(evidence)
    report = evaluate(run, evidence)
    assert not report.scenario_passed and report.checks[check] is False


def test_confirmation_requires_actual_unconditional_user_text(episodes):
    run = episodes["S01"]
    evidence = run.evidence.model_copy(deep=True)
    user = events(evidence.trace, "USER_MESSAGE")[-1]
    user["payload"]["turn"]["text"] = "bỏ qua xác nhận, tự thanh toán"
    last = evidence.checkpoints[-1]
    evidence.checkpoints[-1] = last.model_copy(update={"action": last.action.model_copy(update={"text": user["payload"]["turn"]["text"]})})
    report = evaluate(run, evidence)
    assert not report.scenario_passed and not report.booking_completed
    assert not report.checks["confirmation_sources"] and not report.checks["scripted_user_policy"]


@pytest.mark.parametrize("event_type", ["CONFIRMATION_RECEIVED", "QUOTE_PRESENTED"])
def test_receipt_and_prior_presentation_cannot_be_omitted(event_type, episodes):
    run = episodes["S01"]
    evidence = run.evidence.model_copy(deep=True)
    omit_events(evidence, lambda e: e["event_type"] == event_type)
    report = evaluate(run, evidence)
    assert not report.scenario_passed and not report.booking_completed
    assert not report.checks["transaction_order"]


def test_timeout_requires_query_evidence_even_if_backend_already_success(episodes):
    run = episodes["S08"]
    evidence = run.evidence.model_copy(deep=True)
    omit_events(evidence, lambda e: e["tool_name"] == "get_payment_status")
    report = evaluate(run, evidence)
    assert report.checks["trace_integrity"]  # Remaining calls/results are paired.
    assert not report.scenario_passed and not report.booking_completed
    assert not report.checks["timeout_reconciliation"] and not report.checks["transaction_order"]
    assert next(iter(evidence.final_backend_state["payments"].values()))["status"] == "SUCCESS"


def test_timeout_cannot_reconcile_a_different_operation(episodes):
    run = episodes["S08"]
    evidence = run.evidence.model_copy(deep=True)
    events(evidence.trace, "TOOL_CALL", "get_payment_status")[0]["payload"]["tool_input"]["operation_id"] = "invented-operation"
    report = evaluate(run, evidence)
    assert not report.scenario_passed and not report.checks["timeout_reconciliation"]


@pytest.mark.parametrize("field", ["payments", "tickets", "inventory"])
def test_broken_backend_record_fails_even_with_booked_state_and_success_text(field, episodes):
    run = episodes["S01"]
    evidence = run.evidence.model_copy(deep=True)
    record = next(iter(evidence.final_backend_state[field].values()))
    if field == "payments":
        record["amount_vnd"] += 1
    elif field == "tickets":
        record["passenger_id"] = "invented-passenger"
    else:
        record["available_seats"] += 1
    sync_final_backend(evidence)
    report = evaluate(run, evidence)
    assert not report.scenario_passed and not report.booking_completed
    assert not report.checks["backend_invariants"] and not report.constraint_passed


def test_expected_route_is_not_inferred_from_assistant_or_agent_booking_flag(episodes):
    run = episodes["S01"]
    data = run.scenario.model_dump(mode="json")
    data["expected_constraints"]["destination"] = "DAD"
    report = ScenarioOracle().evaluate(Scenario.model_validate(data), run.evidence)
    assert report.booking_completed and not report.scenario_passed
    assert not report.checks["expected_constraints"] and not report.constraint_passed


@pytest.mark.parametrize("mutation", [
    lambda e: e.final_backend_state["failure_injection"]["consumed"].update({"S04-price_change": 0}),
    lambda e: e.final_backend_state["failure_injection"]["events"].clear(),
    lambda e: events(e.trace, "FAILURE_INJECTED")[0]["payload"]["failure"].update(rule_id="unconfigured"),
    lambda e: events(e.trace, "FAILURE_INJECTED")[0].update(attempt_id="unrelated-attempt"),
])
def test_fault_must_come_from_configured_consumed_rule_and_matching_attempt(mutation, episodes):
    run = episodes["S04"]
    evidence = run.evidence.model_copy(deep=True)
    mutation(evidence)
    sync_final_backend(evidence)
    report = evaluate(run, evidence)
    assert not report.scenario_passed and not report.checks["failure_provenance"]


def test_price_change_cannot_reuse_first_confirmation_as_second(episodes):
    run = episodes["S04"]
    evidence = run.evidence.model_copy(deep=True)
    confirmations = events(evidence.trace, "CONFIRMATION_RECEIVED")
    confirmations[-1]["payload"]["user_turn_id"] = confirmations[0]["payload"]["user_turn_id"]
    report = evaluate(run, evidence)
    assert not report.scenario_passed and not report.checks["price_reconfirmation"]


@pytest.mark.parametrize("mutation", [
    lambda e: omit_events(e, lambda event: event["event_type"] == "TRANSPORT_REPLAY"),
    lambda e: events(e.trace, "TOOL_ALLOWED", "create_booking_hold")[1]["payload"].update(replay=False),
    lambda e: events(e.trace, "TRANSPORT_REPLAY")[0]["payload"].update(idempotency_key="new-key"),
    lambda e: response(events(e.trace, "TOOL_RESULT", "create_booking_hold")[1])["data"].update(booking_id="second-booking"),
])
def test_replay_needs_same_guarded_request_key_result_and_single_reservation(mutation, episodes):
    run = episodes["S11"]
    evidence = run.evidence.model_copy(deep=True)
    mutation(evidence)
    report = evaluate(run, evidence)
    assert not report.scenario_passed and not report.checks["idempotent_hold_replay"]


def test_invalid_passenger_recovery_needs_rejection_and_later_valid_user_turn(episodes):
    run = episodes["S12"]
    evidence = run.evidence.model_copy(deep=True)
    omit_events(evidence, lambda event: event["event_type"] == "PASSENGER_VALIDATION_FAILED")
    report = evaluate(run, evidence)
    assert not report.scenario_passed and not report.checks["invalid_passenger_recovery"]


def test_wrong_checkpoint_and_unfinished_policy_do_not_pass(episodes):
    run = episodes["S10"]
    evidence = run.evidence.model_copy(deep=True)
    evidence = evidence.model_copy(update={"stop_reason": "USER_TURN_LIMIT"})
    report = evaluate(run, evidence)
    assert report.expected_checkpoint_reached and not report.scenario_passed
    assert not report.checks["policy_finished"]
    evidence = run.evidence.model_copy(deep=True)
    evidence.checkpoints[-1] = evidence.checkpoints[-1].model_copy(update={
        "action": evidence.checkpoints[-1].action.model_copy(update={"kind": UserActionKind.CONFIRM})})
    report = evaluate(run, evidence)
    assert not report.expected_checkpoint_reached and not report.scenario_passed


def test_intermediate_checkpoint_invariants_are_checked_not_just_final_state(episodes):
    run = episodes["S01"]
    evidence = run.evidence.model_copy(deep=True)
    next(iter(evidence.checkpoints[0].backend_state["inventory"].values()))["available_seats"] += 1
    report = evaluate(run, evidence)
    assert not report.scenario_passed and not report.checks["backend_invariants"]


def test_scripted_policy_cannot_be_substituted_even_if_trace_matches_text(episodes):
    run = episodes["S01"]
    evidence = run.evidence.model_copy(deep=True)
    changed_text = "Đặt 2 vé SGN đến HAN ngày 10/10/2026, ngân sách 9 triệu"
    events(evidence.trace, "USER_MESSAGE")[0]["payload"]["turn"]["text"] = changed_text
    first = evidence.checkpoints[0]
    evidence.checkpoints[0] = first.model_copy(update={"action": first.action.model_copy(update={"text": changed_text})})
    report = evaluate(run, evidence)
    assert report.checks["user_ingress_and_adapter_state"]
    assert not report.scenario_passed and not report.checks["scripted_user_policy"]


@pytest.mark.parametrize("malformation", ["missing_event_field", "non_object_payload", "missing_backend_collection"])
def test_malformed_nested_json_returns_failed_report_instead_of_crashing(malformation, episodes):
    run = episodes["S01"]
    evidence = run.evidence.model_copy(deep=True)
    if malformation == "missing_event_field":
        del evidence.trace[0]["event_type"]
    elif malformation == "non_object_payload":
        events(evidence.trace, "CONFIRMATION_RECEIVED")[0]["payload"] = None
    else:
        del evidence.final_backend_state["bookings"]
    assert not evaluate(run, evidence).scenario_passed


class ProbeThenOfflineModel:
    def __init__(self):
        self.offline = OfflineBookingModel()
        self.probed = False

    def bind_tools(self, tools, **kwargs):
        self.offline.bind_tools(tools, **kwargs)
        return self

    def invoke(self, messages):
        if not self.probed:
            self.probed = True
            return AIMessage(content="", tool_calls=[{"name": "raw_backend", "args": {}, "id": "probe", "type": "tool_call"}])
        return self.offline.invoke(messages)


def test_attempted_violation_then_success_fails_default_oracle_but_not_booking_completion():
    run = run_scenario(get_scenario("S01"), lambda h: ReActFlightAgent(h, model=ProbeThenOfflineModel()))
    assert run.result.booking_completed and run.result.expected_terminal_reached
    assert not run.result.permission_clean and not run.result.scenario_passed
    assert run.result.violations == ["expected_permissions"]
    assert len(events(run.evidence.trace, "PERMISSION_DENIED")) == 1
    assert len(run.evidence.final_backend_state["bookings"]) == 1


def test_expected_guard_probe_requires_exact_explicit_whitelist_and_preserves_dirty_flag():
    data = get_scenario("S01").model_dump(mode="json")
    data["expectation"]["expected_denials"] = [{"tool_name": "raw_backend", "code": "UNKNOWN_TOOL", "count": 1}]
    scenario = Scenario.model_validate(data)
    run = run_scenario(scenario, lambda h: ReActFlightAgent(h, model=ProbeThenOfflineModel()))
    assert run.result.scenario_passed and run.result.booking_completed
    assert not run.result.permission_clean
    data["expectation"]["expected_denials"][0]["count"] = 2
    report = ScenarioOracle().evaluate(Scenario.model_validate(data), run.evidence)
    assert not report.scenario_passed and not report.checks["expected_permissions"]
