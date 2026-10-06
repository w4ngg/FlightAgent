import json
from copy import deepcopy

import pytest

from app.agents.input import parse_user_input
from app.agents.react_agent import ReActFlightAgent
from app.evaluation.contracts import (
    EventExpectation, Scenario, ScenarioExpectation, ScenarioPredicate, UserActionKind,
)
from app.evaluation.environment import ScenarioHarness
from app.evaluation.episode import run_scenario
from app.evaluation.oracle import ScenarioOracle, events, response
from app.evaluation.scenarios import get_scenario, list_scenarios
from app.evaluation.user_policy import ScriptedUserPolicy
from app.models import TerminalStatus
from app.state import SessionPhase


@pytest.fixture(scope="module")
def episodes():
    return {s.id: run_scenario(s, lambda h: ReActFlightAgent(h)) for s in list_scenarios()}


def test_catalog_has_twelve_machine_readable_detached_contracts():
    catalog = list_scenarios()
    assert [s.id for s in catalog] == [f"S{i:02}" for i in range(1, 13)]
    for s in catalog:
        assert Scenario.model_validate_json(s.model_dump_json()) == s
        assert len(s.passengers) == 2 and s.max_user_turns == 12
        assert s.backend_seed == 42 and s.start_time.isoformat() == "2026-10-05T09:00:00+07:00"
    catalog[0].passengers.clear()
    catalog[0].expectation.required_events.clear()
    assert len(get_scenario("S01").passengers) == 2
    assert get_scenario("S01").expectation.required_events
    assert "initial_user_message" in Scenario.model_json_schema()["properties"]
    with pytest.raises(ValueError):
        get_scenario("S99")


@pytest.mark.parametrize("mutation", [
    lambda s: s.update(backend_seed=True), lambda s: s.update(start_time="2026-10-05T09:00:00"),
    lambda s: s.update(max_user_turns=0), lambda s: s.update(passengers=[]),
    lambda s: s["passengers"][1].update(document_number=s["passengers"][0]["document_number"]),
    lambda s: s["passengers"][0].update(date_of_birth="2030-01-01"),
    lambda s: s["expected_constraints"].update(passenger_count=3),
    lambda s: s["expectation"].update(phase="WAITING_INPUT"),
    lambda s: s["expectation"].update(terminal_status=None),
    lambda s: s["expectation"].update(ticket_count=1),
    lambda s: s.update(extra_unknown=True),
])
def test_scenario_schema_rejects_inconsistent_or_unsafe_setup(mutation):
    data = get_scenario("S01").model_dump(mode="json")
    mutation(data)
    with pytest.raises(ValueError):
        Scenario.model_validate(data)


@pytest.mark.parametrize("kwargs", [
    {"event_type": "TOOL_RESULT", "tool_name": "raw_backend"},
    {"event_type": "TOOL_CALL", "effect": "SOLD_OUT"},
    {"event_type": "USER_MESSAGE", "min_count": 2, "max_count": 1},
    {"event_type": "USER_MESSAGE", "min_count": True},
])
def test_event_filter_schema(kwargs):
    with pytest.raises(ValueError):
        EventExpectation(**kwargs)


@pytest.mark.parametrize("number", range(1, 13))
def test_every_catalog_scenario_passes_offline_with_real_guard_and_backend(number, episodes):
    run = episodes[f"S{number:02}"]
    assert run.result.scenario_passed, run.result.violations
    assert run.result.permission_clean and run.result.constraint_passed
    assert run.evidence.checkpoints
    assert all(cp.agent_state["harness"] == cp.state.model_dump(mode="json") for cp in run.evidence.checkpoints)
    assert run.evidence.final_state.step_count <= run.scenario.harness_config.max_steps


@pytest.mark.parametrize("scenario_id,phase", [("S02", SessionPhase.WAITING_INPUT), ("S10", SessionPhase.WAITING_CONFIRMATION)])
def test_checkpoint_pass_is_not_booking_or_terminal_success(scenario_id, phase, episodes):
    report = episodes[scenario_id].result
    assert report.scenario_passed and report.expected_checkpoint_reached
    assert not report.expected_terminal_reached and not report.booking_completed
    assert report.observed_phase == phase and report.observed_terminal is None
    assert not episodes[scenario_id].evidence.final_backend_state["bookings"]


def test_declined_failed_is_a_correct_scenario_outcome(episodes):
    run = episodes["S07"]
    assert run.result.scenario_passed and run.result.expected_terminal_reached
    assert run.result.observed_terminal == TerminalStatus.FAILED and not run.result.booking_completed
    assert next(iter(run.evidence.final_backend_state["payments"].values()))["status"] == "DECLINED"
    assert not run.evidence.final_backend_state["tickets"]


def test_price_change_has_two_real_confirmations_and_no_old_quote_commit(episodes):
    run = episodes["S04"]
    approvals = events(run.evidence.trace, "CONFIRMATION_RECEIVED")
    assert len(approvals) == 2 and approvals[0]["payload"]["user_turn_id"] != approvals[1]["payload"]["user_turn_id"]
    first_attempt = next(cp for cp in run.evidence.checkpoints if cp.action.kind == UserActionKind.CONFIRM)
    assert not first_attempt.backend_state["bookings"]
    assert first_attempt.state.confirmation is None and first_attempt.state.quote.quote_version == 2
    assert approvals[0]["payload"]["snapshot"]["total_price_vnd"] < approvals[1]["payload"]["snapshot"]["total_price_vnd"]


@pytest.mark.parametrize("scenario_id", ["S05", "S06"])
def test_alternative_offer_is_resolved_from_actual_search_not_guessed(scenario_id, episodes):
    run = episodes[scenario_id]
    selections = [cp.action.text.removeprefix("chọn ") for cp in run.evidence.checkpoints if cp.action.kind == UserActionKind.SELECT]
    assert len(selections) == 2 and selections[0] != selections[1]
    assert len(run.evidence.final_backend_state["bookings"]) == 1
    assert len(run.evidence.final_backend_state["tickets"]) == 2


def test_timeout_observation_is_unknown_until_query_same_operation(episodes):
    run = episodes["S08"]
    pay = events(run.evidence.trace, "TOOL_RESULT", "process_payment")[0]
    status = events(run.evidence.trace, "TOOL_RESULT", "get_payment_status")[0]
    ticket = events(run.evidence.trace, "TOOL_CALL", "issue_ticket")[0]
    assert response(pay)["data"] is None and response(pay)["commit_status"] == "UNKNOWN"
    assert response(pay)["operation_ref"] == response(status)["data"]["operation"]["operation_id"]
    assert pay["sequence_number"] < status["sequence_number"] < ticket["sequence_number"]
    assert len(run.evidence.final_backend_state["payments"]) == 1


def test_destination_change_has_new_intent_and_no_old_transaction(episodes):
    run = episodes["S09"]
    changed = next(cp for cp in run.evidence.checkpoints if cp.action.kind == UserActionKind.CHANGE)
    assert changed.state.constraints.destination == "DAD"
    assert changed.state.quote is None and changed.state.confirmation is None
    assert not changed.backend_state["bookings"]
    booking = next(iter(run.evidence.final_backend_state["bookings"].values()))
    assert booking["quote"]["destination"] == "DAD" and booking["context"]["transaction_intent_id"] == "intent-2"


def test_bypass_text_is_not_misparsed_as_tu_origin_change(episodes):
    text = get_scenario("S10").user_policy.bypass_message
    assert parse_user_input(text, []).requirements == {}
    run = episodes["S10"]
    assert run.evidence.final_state.constraints == run.scenario.initial_constraints
    assert run.evidence.final_state.quote and run.evidence.final_state.confirmation is None


def test_duplicate_delivery_goes_through_guard_once_more_with_same_key_and_result(episodes):
    run = episodes["S11"]
    trace = run.evidence.trace
    results = events(trace, "TOOL_RESULT", "create_booking_hold")
    assert len(results) == 2 and response(results[0]) == response(results[1])
    assert len(events(trace, "TRANSPORT_REPLAY")) == 1
    assert events(trace, "TOOL_ALLOWED", "create_booking_hold")[1]["payload"]["replay"] is True
    backend = run.evidence.final_backend_state
    assert len(backend["bookings"]) == 1
    reserves = [e for e in backend["inventory_events"] if e["event_type"] == "HOLD_RESERVED"]
    assert len(reserves) == 1 and reserves[0]["seats_delta"] == -2


def test_invalid_profile_checkpoint_has_no_side_effect_before_correction(episodes):
    run = episodes["S12"]
    rejected = next(cp for cp in run.evidence.checkpoints if cp.action.kind == UserActionKind.INVALID_PASSENGERS)
    assert rejected.state.invalid_passenger_attempts == 1 and rejected.state.quote is None
    assert all(not rejected.backend_state[name] for name in ("bookings", "payments", "payment_operations", "tickets"))
    assert run.result.booking_completed


def test_policy_selects_deterministically_from_observed_prices_and_does_not_read_backend(episodes):
    scenario = get_scenario("S01")
    policy = ScriptedUserPolicy(scenario)
    initial = episodes["S01"].evidence.checkpoints[0].agent_state
    policy.next_action(initial)
    public = deepcopy(initial)
    # Offer details are mock public observations; reorder them and choose a
    # new runtime ID to prove the policy did not hard-code seed fixture IDs.
    public["offers"] = list(reversed(public["offers"]))
    public["offers"][0]["offer"]["offer_id"] = "OFF-RUNTIME-SOURCE"
    public["offers"][0]["offer"]["unit_price_vnd"] = 1
    public["offers"][0]["total_price_vnd"] = 2
    action = policy.next_action(public)
    assert action.text == "chọn OFF-RUNTIME-SOURCE"
    assert not hasattr(policy, "backend") and not hasattr(policy, "harness")


def test_policy_does_not_confirm_undisplayed_or_expired_quote(episodes):
    public = deepcopy(episodes["S01"].evidence.checkpoints[2].agent_state)
    policy = ScriptedUserPolicy(get_scenario("S01"))
    policy.next_action(public)
    public["harness"]["displayed_quote_id"] = None
    assert policy.next_action(public).kind == UserActionKind.RESUME
    public["harness"]["displayed_quote_id"] = public["harness"]["quote"]["quote_id"]
    public["current_time"] = public["harness"]["quote"]["expires_at"]
    assert policy.next_action(public).kind == UserActionKind.RESUME


def test_policy_empty_offers_stops_without_relaxing_user_constraints(episodes):
    public = deepcopy(episodes["S01"].evidence.checkpoints[0].agent_state)
    policy = ScriptedUserPolicy(get_scenario("S01"))
    policy.next_action(public)
    public["offers"] = []
    before = deepcopy(public)
    assert policy.next_action(public) is None and policy.stop_reason == "NO_ELIGIBLE_OFFER"
    assert public == before


def test_user_turn_limit_is_not_automatic_handoff_or_reset():
    scenario = Scenario.model_validate({**get_scenario("S01").model_dump(), "max_user_turns": 1})
    run = run_scenario(scenario, lambda h: ReActFlightAgent(h))
    assert run.evidence.stop_reason == "USER_TURN_LIMIT" and not run.result.scenario_passed
    assert run.evidence.final_state.terminal_status is None
    assert len(run.evidence.checkpoints) == 1


def test_checkpoint_on_last_allowed_turn_is_not_misclassified_as_exhaustion():
    scenario = Scenario.model_validate({**get_scenario("S02").model_dump(), "max_user_turns": 1})
    run = run_scenario(scenario, lambda h: ReActFlightAgent(h))
    assert run.result.scenario_passed and run.evidence.stop_reason == "EXPECTED_CHECKPOINT"


def test_pattern_neutral_factory_uses_only_common_adapter_contract():
    class ProxyAdapter:
        name = "protocol-proxy"
        def __init__(self, harness):
            self.delegate = ReActFlightAgent(harness)
        def send(self, message, session_id=None):
            return self.delegate.send(message, session_id=session_id)
        def reset(self, seed=42, *, force=False):
            return self.delegate.reset(seed, force=force)
        def configure_failure(self, config=None, *, rules=None):
            return self.delegate.configure_failure(config, rules=rules)
        def get_state(self):
            return self.delegate.get_state()
        def get_backend_state(self):
            return self.delegate.get_backend_state()
        def get_trace(self):
            return self.delegate.get_trace()
        def get_handoff(self):
            return self.delegate.get_handoff()
        def check_completion(self):
            return self.delegate.check_completion()
        def export_artifacts(self, output_dir):
            return self.delegate.export_artifacts(output_dir)
    run = run_scenario(get_scenario("S01"), ProxyAdapter)
    assert run.agent_name == "protocol-proxy" and run.result.scenario_passed


def test_scenarios_use_fresh_seed_clock_fault_counters_and_no_cross_episode_mutation():
    first = run_scenario(get_scenario("S04"), lambda h: ReActFlightAgent(h))
    second = run_scenario(get_scenario("S04"), lambda h: ReActFlightAgent(h))
    assert first.evidence.initial_backend_state == second.evidence.initial_backend_state
    assert first.evidence.final_backend_state == second.evidence.final_backend_state
    assert first.evidence.trace == second.evidence.trace
    first.evidence.final_backend_state["bookings"].clear()
    assert second.result.scenario_passed and second.evidence.final_backend_state["bookings"]


def test_s11_duplicate_is_rearmed_only_by_explicit_operator_reset():
    scenario = get_scenario("S11")
    harness = ScenarioHarness(scenario, session_id="replay-reset")
    agent = ReActFlightAgent(harness)
    for _ in range(2):
        policy = ScriptedUserPolicy(scenario)
        while (action := policy.next_action(agent.get_state())) is not None:
            agent.send(action.text)
        assert len(events(harness.get_trace(), "TRANSPORT_REPLAY")) == 1
        agent.reset(force=True)


def test_export_preserves_all_authoritative_and_policy_evidence_without_overwrite(tmp_path, episodes):
    run = episodes["S08"]
    folder = tmp_path / "episode"
    run.export_artifacts(folder)
    assert len(list(folder.glob("*.json"))) == 11
    for filename in ("scenario.json", "scripted_user.json", "checkpoints.json", "scenario_result.json"):
        assert json.loads((folder / filename).read_text())
    assert json.loads((folder / "scenario_result.json").read_text())["scenario_passed"]
    assert json.loads((folder / "completion.json").read_text())["booking_completed"]
    original = (folder / "backend_state.json").read_bytes()
    with pytest.raises(FileExistsError):
        run.export_artifacts(folder)
    assert (folder / "backend_state.json").read_bytes() == original
