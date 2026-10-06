"""Pure scenario oracle: authoritative snapshots + ordered evidence, not prose.

This is the catalog's behavioral contract, not the aggregate evaluator/metrics
runner of milestone H. JSON evidence is audit data, not cryptographic proof.
"""

from collections import Counter
from datetime import datetime

from app.evaluation.contracts import (
    EventExpectation, Scenario, ScenarioEvidence, ScenarioOracleResult, ScenarioPredicate, UserActionKind,
)
from app.evaluation.user_policy import ScriptedUserPolicy
from app.failures import FailureEvent
from app.harness.completion import CompletionChecker
from app.harness.runtime import accepts_confirmation
from app.harness.trace import TraceEvent
from app.invariants import check_backend_invariants
from app.models import ConfirmationSnapshot, Quote, TerminalStatus, ToolResponse
from app.state import CompletionResult
from app.tools import TOOL_RESPONSE_TYPES


def events(trace, event_type, tool_name=None):
    return [e for e in trace if e["event_type"] == event_type and
            (tool_name is None or e["tool_name"] == tool_name)]


def response(event):
    return event["payload"]["response"]


def matches(event, specification: EventExpectation):
    if event["event_type"] != specification.event_type or (
        specification.tool_name and event["tool_name"] != specification.tool_name
    ):
        return False
    if specification.code and event["payload"].get("code", event["payload"].get("response", {}).get("code")) != specification.code:
        return False
    return not specification.effect or event["payload"].get("failure", {}).get("effect") == specification.effect.value


def _safe(check):
    try:
        return bool(check())
    except (KeyError, IndexError, TypeError, ValueError, AttributeError, StopIteration):
        return False


def _trace_integrity(scenario, evidence):
    trace = evidence.trace
    if not trace or [e["sequence_number"] for e in trace] != list(range(1, len(trace) + 1)):
        return False
    times = []
    for event in trace:
        if TraceEvent.model_validate(event).model_dump(mode="json") != event:
            return False
        if event["context"]["session_id"] != evidence.final_state.context.session_id:
            return False
        times.append(datetime.fromisoformat(event["timestamp"]))
    if times != sorted(times) or times[0] < scenario.start_time or times[-1] > datetime.fromisoformat(evidence.final_backend_state["current_time"]):
        return False
    calls = events(trace, "TOOL_CALL")
    if any(not e["attempt_id"] or not e["tool_name"] for e in calls):
        return False
    if len({e["attempt_id"] for e in calls}) != len(calls):
        return False
    results = events(trace, "TOOL_RESULT")
    if len(results) != len(calls):
        return False
    for call in calls:
        candidates = [e for e in results if e["attempt_id"] == call["attempt_id"]]
        if len(candidates) != 1:
            return False
        result = candidates[0]
        decisions = [e for e in trace if e["attempt_id"] == call["attempt_id"] and
                     e["event_type"] in ("TOOL_ALLOWED", "PERMISSION_DENIED", "TOOL_DENIED")]
        if len(decisions) != 1:
            return False
        decision = decisions[0]
        if not call["sequence_number"] < decision["sequence_number"] < result["sequence_number"]:
            return False
        if any(e["context"] != call["context"] or e["tool_name"] != call["tool_name"] for e in (decision, result)):
            return False
        permitted = decision["event_type"] == "TOOL_ALLOWED"
        if result["allowed"] is not permitted or decision["allowed"] is not permitted:
            return False
        if permitted and call["tool_name"] not in TOOL_RESPONSE_TYPES:
            return False
        if not permitted and (response(result)["ok"] or response(result)["commit_status"] not in ("NOT_COMMITTED", "NOT_APPLICABLE")):
            return False
        schema = ToolResponse[TOOL_RESPONSE_TYPES[call["tool_name"]]] if call["tool_name"] in TOOL_RESPONSE_TYPES else ToolResponse
        schema.model_validate(response(result))
    decision_ids = {e["attempt_id"] for e in trace if e["event_type"] in ("TOOL_ALLOWED", "PERMISSION_DENIED", "TOOL_DENIED")}
    return decision_ids == {e["attempt_id"] for e in calls}


def _valid_confirmations(evidence):
    valid = []
    for confirmation in events(evidence.trace, "CONFIRMATION_RECEIVED"):
        before = [e for e in evidence.trace if e["sequence_number"] < confirmation["sequence_number"]]
        users = events(before, "USER_MESSAGE")
        presented = [e for e in events(before, "QUOTE_PRESENTED") if e["context"] == confirmation["context"]]
        if not users or not presented:
            continue
        user, display = users[-1], presented[-1]
        turn = user["payload"]["turn"]
        quote = Quote.model_validate(display["payload"]["quote"])
        if (turn["turn_id"] == confirmation["payload"]["user_turn_id"] and accepts_confirmation(turn["text"]) and
            user["context"] == confirmation["context"] == quote.context.model_dump(mode="json") and
            display["sequence_number"] < user["sequence_number"] and
            confirmation["payload"]["quote_id"] == quote.quote_id and
            confirmation["payload"]["snapshot"] == ConfirmationSnapshot.from_quote(quote).model_dump(mode="json") and
            evidence.final_backend_state["quotes"].get(quote.quote_id) == quote.model_dump(mode="json") and
            quote.created_at <= datetime.fromisoformat(confirmation["timestamp"]) < quote.expires_at):
            valid.append(confirmation)
    return valid


def _transaction_order(evidence, confirmations):
    calls = {e["attempt_id"]: e for e in events(evidence.trace, "TOOL_CALL")}
    for allowed in events(evidence.trace, "TOOL_ALLOWED"):
        call = calls[allowed["attempt_id"]]
        args, context = call["payload"]["tool_input"], call["context"]
        before = [e for e in confirmations if e["context"] == context and e["sequence_number"] < call["sequence_number"]]
        if call["tool_name"] in ("create_booking_hold", "process_payment"):
            quote_id = args["quote_id"] if call["tool_name"] == "create_booking_hold" else (
                evidence.final_backend_state["bookings"][args["booking_id"]]["quote"]["quote_id"])
            if not before or before[-1]["payload"]["quote_id"] != quote_id:
                return False
        elif call["tool_name"] == "issue_ticket":
            observed = [e for e in events(evidence.trace, "TOOL_RESULT") if e["tool_name"] in
                ("process_payment", "get_payment_status") and e["allowed"] is True and
                e["context"] == context and e["sequence_number"] < call["sequence_number"] and
                (response(e).get("data") or {}).get("operation", {}).get("status") == "SUCCESS" and
                response(e)["data"]["operation"]["booking_id"] == args["booking_id"]]
            if not observed:
                return False
    return True


def valid_confirmation_events(evidence: ScenarioEvidence):
    """Reuse the oracle's receipt/provenance policy in the H transaction audit."""
    return _valid_confirmations(evidence)


def _user_evidence(evidence):
    users = events(evidence.trace, "USER_MESSAGE")
    if len(users) != len(evidence.checkpoints):
        return False
    for index, (user, checkpoint) in enumerate(zip(users, evidence.checkpoints)):
        turn = user["payload"]["turn"]
        if turn["turn_number"] != index + 1 or turn["turn_id"] != checkpoint.user_turn_id or turn["turn_number"] != checkpoint.user_turn_number or (
            turn["text"] != checkpoint.action.text or checkpoint.trace_sequence < user["sequence_number"] or
            checkpoint.agent_state["harness"] != checkpoint.state.model_dump(mode="json")
        ):
            return False
        if index + 1 < len(users) and checkpoint.trace_sequence >= users[index + 1]["sequence_number"]:
            return False
    return bool(evidence.checkpoints and evidence.checkpoints[-1].trace_sequence == len(evidence.trace) and
                evidence.final_state == evidence.checkpoints[-1].state and
                evidence.final_backend_state == evidence.checkpoints[-1].backend_state and
                evidence.final_agent_state == evidence.checkpoints[-1].agent_state and
                evidence.final_agent_state["harness"] == evidence.final_state.model_dump(mode="json"))


def _policy_evidence(scenario, evidence):
    """Replay user decisions from public observations, never agent prose/backend."""
    if len(evidence.checkpoints) > scenario.max_user_turns:
        return False
    policy, public_state = ScriptedUserPolicy(scenario), {}
    for checkpoint in evidence.checkpoints:
        if policy.next_action(public_state) != checkpoint.action:
            return False
        public_state = checkpoint.agent_state
    return policy.next_action(public_state) is None and policy.stop_reason == evidence.stop_reason


def _fault_evidence(scenario, evidence):
    actual = evidence.final_backend_state["failure_injection"]
    if actual["rules"] != [rule.model_dump(mode="json") for rule in scenario.failure_rules] or (
        actual["events"] != [e["payload"]["failure"] for e in events(evidence.trace, "FAILURE_INJECTED")]
    ):
        return False
    rules = {r.rule_id: r for r in scenario.failure_rules}
    counts = Counter(e["rule_id"] for e in actual["events"])
    if actual["consumed"] != {r.rule_id: counts[r.rule_id] for r in scenario.failure_rules}:
        return False
    calls = {e["attempt_id"]: e for e in events(evidence.trace, "TOOL_CALL")}
    tools = {"hold": "create_booking_hold", "reprice": "reprice_offer", "payment": "process_payment"}
    for outer in events(evidence.trace, "FAILURE_INJECTED"):
        event = FailureEvent.model_validate(outer["payload"]["failure"])
        rule, call = rules[event.rule_id], calls[outer["attempt_id"]]
        if event.effect != rule.effect or event.trigger != rule.trigger or counts[event.rule_id] > rule.times or (
            outer["tool_name"] != tools[event.trigger.value] or outer["tool_name"] != call["tool_name"] or
            outer["context"] != call["context"] or outer["context"] != event.context.model_dump(mode="json") or
            not call["sequence_number"] < outer["sequence_number"]
        ):
            return False
        for field_name, actual_value in (("offer_id", event.offer_id), ("operation_id", event.operation_id),
            ("session_id", event.context.session_id), ("transaction_intent_id", event.context.transaction_intent_id)):
            if getattr(rule, field_name) is not None and getattr(rule, field_name) != actual_value:
                return False
    return True


def _no_transactions(snapshot):
    return all(not snapshot[name] for name in ("bookings", "payments", "payment_operations", "tickets"))


class ScenarioOracle:
    def evaluate(self, scenario: Scenario, evidence: ScenarioEvidence) -> ScenarioOracleResult:
        try:
            return self._evaluate(scenario, evidence)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError, StopIteration):
            completion = CompletionResult(booking_completed=False, constraint_passed=False, permission_clean=False,
                booking_id=evidence.final_state.booking_id, checks={"valid_evidence": False}, violations=["Malformed scenario evidence"])
            return ScenarioOracleResult(scenario_id=scenario.id, booking_completed=False, expected_terminal_reached=False,
                expected_checkpoint_reached=False, constraint_passed=False, permission_clean=False, scenario_passed=False,
                observed_phase=evidence.final_state.phase, observed_terminal=evidence.final_state.terminal_status,
                completion=completion, checks={"valid_evidence": False}, violations=["valid_evidence"])

    def _evaluate(self, scenario: Scenario, evidence: ScenarioEvidence) -> ScenarioOracleResult:
        state, backend, trace, expected = evidence.final_state, evidence.final_backend_state, evidence.trace, scenario.expectation
        confirmations = []
        try:
            confirmations = _valid_confirmations(evidence)
            authentic = bool(state.confirmation and state.quote and any(
                e["payload"]["user_turn_id"] == state.confirmation.user_turn_id and
                e["payload"]["quote_id"] == state.quote.quote_id and
                e["context"] == state.context.model_dump(mode="json") and
                e["timestamp"] == state.confirmation.confirmed_at.isoformat() for e in confirmations))
            completion = CompletionChecker().check(state, backend, confirmation_authentic=authentic, trace=trace)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            completion = CompletionResult(booking_completed=False, constraint_passed=False, permission_clean=False,
                booking_id=state.booking_id, checks={"valid_evidence": False}, violations=["Malformed scenario evidence"])
        terminal_reached = expected.terminal_status is not None and state.terminal_status == expected.terminal_status
        checkpoint_action = evidence.checkpoints[-1].action.kind if evidence.checkpoints else None
        if checkpoint_action == UserActionKind.APPLY:
            last = evidence.checkpoints[-1]
            applied = [e for e in events(trace, "INPUT_INTERPRETATION_APPLIED") if e["payload"]["actual_user_turn_id"] == last.user_turn_id]
            if len(applied) == 1:
                source = applied[0]["payload"]["source_turn_id"]
                original = next((cp for cp in evidence.checkpoints[:-1] if cp.user_turn_id == source), None)
                checkpoint_action = original.action.kind if original else None
        checkpoint_reached = bool(expected.phase is not None and state.terminal_status is None and state.phase == expected.phase and
                                  checkpoint_action == expected.checkpoint_action)
        denials = events(trace, "PERMISSION_DENIED")
        permission_clean = state.permission_violations == 0 and not denials
        wanted_denials = Counter({(d.tool_name, d.code): d.count for d in expected.expected_denials})
        actual_denials = Counter((e["tool_name"], e["payload"]["code"]) for e in denials)
        checks = {
            "initial_setup": _safe(lambda: backend["seed"] == scenario.backend_seed == evidence.initial_backend_state["seed"] and
                datetime.fromisoformat(evidence.initial_backend_state["current_time"]) == scenario.start_time and
                _no_transactions(evidence.initial_backend_state) and
                evidence.initial_backend_state["failure_injection"]["rules"] == [r.model_dump(mode="json") for r in scenario.failure_rules] and
                all(v == 0 for v in evidence.initial_backend_state["failure_injection"]["consumed"].values())),
            "trace_integrity": _safe(lambda: _trace_integrity(scenario, evidence)),
            "user_ingress_and_adapter_state": _safe(lambda: _user_evidence(evidence)),
            "scripted_user_policy": _safe(lambda: _policy_evidence(scenario, evidence)),
            "failure_provenance": _safe(lambda: _fault_evidence(scenario, evidence)),
            "confirmation_sources": _safe(lambda: len(confirmations) == len(events(trace, "CONFIRMATION_RECEIVED"))),
            "transaction_order": _safe(lambda: _transaction_order(evidence, confirmations)),
            "backend_invariants": _safe(lambda: check_backend_invariants(backend).passed and
                check_backend_invariants(evidence.initial_backend_state).passed and
                all(check_backend_invariants(cp.backend_state).passed for cp in evidence.checkpoints)),
            "expected_outcome": terminal_reached or checkpoint_reached,
            "terminal_evidence": _safe(lambda: not events(trace, "TERMINAL") if state.terminal_status is None else
                events(trace, "TERMINAL")[-1]["payload"]["status"] == state.terminal_status.value and
                events(trace, "TERMINAL")[-1]["context"] == state.context.model_dump(mode="json")),
            "policy_finished": evidence.stop_reason in ("TERMINAL", "EXPECTED_CHECKPOINT"),
            "expected_permissions": actual_denials == wanted_denials,
            "permission_count_consistent": state.permission_violations == len(denials),
            "expected_transaction_counts": _safe(lambda: len(backend["bookings"]) == expected.booking_count and
                sum(p["status"] == "SUCCESS" for p in backend["payments"].values()) == expected.successful_payment_count and
                len(backend["tickets"]) == expected.ticket_count),
            "expected_constraints": _safe(lambda: (expected.phase is not None and expected.phase.value == "WAITING_INPUT" and state.constraints is None) or (
                state.constraints == scenario.expected_constraints and all(b["constraints"] == scenario.expected_constraints.model_dump(mode="json")
                    for b in backend["bookings"].values()))),
            "completion_matches_outcome": completion.booking_completed if expected.terminal_status == TerminalStatus.BOOKED else not completion.booking_completed,
        }
        for i, rule in enumerate(expected.required_events, 1):
            checks[f"required_event_{i}_{rule.event_type}"] = _safe(lambda rule=rule: (
                rule.min_count <= sum(matches(e, rule) for e in trace) and
                (rule.max_count is None or sum(matches(e, rule) for e in trace) <= rule.max_count)))
        for i, rule in enumerate(expected.forbidden_events, 1):
            checks[f"forbidden_event_{i}_{rule.event_type}"] = _safe(lambda rule=rule: not any(matches(e, rule) for e in trace))
        for predicate in expected.predicates:
            checks[predicate.value] = _safe(lambda predicate=predicate: self._predicate(predicate, scenario, evidence, confirmations))
        constraint_passed = completion.constraint_passed and checks["backend_invariants"] and checks["expected_constraints"]
        checks["constraint_passed"] = constraint_passed
        return ScenarioOracleResult(scenario_id=scenario.id, booking_completed=completion.booking_completed,
            expected_terminal_reached=terminal_reached, expected_checkpoint_reached=checkpoint_reached,
            constraint_passed=constraint_passed, permission_clean=permission_clean, scenario_passed=all(checks.values()),
            observed_phase=state.phase, observed_terminal=state.terminal_status, completion=completion,
            checks=checks, violations=[name for name, passed in checks.items() if not passed])

    def _predicate(self, kind, scenario, evidence, confirmations):
        state, backend, trace = evidence.final_state, evidence.final_backend_state, evidence.trace
        calls = {e["attempt_id"]: e for e in events(trace, "TOOL_CALL")}
        booking = backend["bookings"].get(state.booking_id)
        searches = [e for e in events(trace, "TOOL_RESULT", "search_flights") if response(e)["ok"]]
        checkpoints = evidence.checkpoints
        faults = events(trace, "FAILURE_INJECTED")
        if kind == ScenarioPredicate.MISSING:
            draft = evidence.final_agent_state["request_draft"]
            return (evidence.final_agent_state["missing_fields"] == ["destination"] and state.constraints is None and
                draft["origin"] == scenario.initial_constraints.origin and draft["departure_date"] == scenario.initial_constraints.departure_date.isoformat() and
                draft["passenger_count"] == scenario.initial_constraints.passenger_count and _no_transactions(backend))
        if kind == ScenarioPredicate.ALIASES:
            # Business constraints now come from accepted state, not tool args.
            updates = events(trace, "REQUIREMENTS_UPDATED")
            def canonical_before(search):
                call = calls[search["attempt_id"]]
                preceding = [u for u in updates if u["context"] == call["context"] and u["sequence_number"] < call["sequence_number"]]
                return bool(preceding and preceding[-1]["payload"]["constraints"]["origin"] == "SGN" and
                            preceding[-1]["payload"]["constraints"]["destination"] == "HAN" and
                            "constraints" not in call["payload"]["tool_input"])
            return bool(searches and booking and all(canonical_before(e) for e in searches))
        if kind == ScenarioPredicate.PRICE:
            fault = next(e for e in faults if e["payload"]["failure"]["effect"] == "PRICE_CHANGE")
            stale = next(e for e in events(trace, "TOOL_RESULT", "create_booking_hold") if response(e)["code"] == "QUOTE_STALE")
            old = backend["quotes"][calls[stale["attempt_id"]]["payload"]["tool_input"]["quote_id"]]
            new = booking["quote"]
            approval = next(e for e in confirmations if e["payload"]["quote_id"] == new["quote_id"])
            old_approval = next(e for e in confirmations if e["payload"]["quote_id"] == old["quote_id"])
            return (response(stale)["commit_status"] == "NOT_COMMITTED" and old["offer_id"] == new["offer_id"] and
                old["quote_version"] < new["quote_version"] and old["total_price_vnd"] != new["total_price_vnd"] and
                old_approval["sequence_number"] < fault["sequence_number"] < stale["sequence_number"] < approval["sequence_number"] and
                old_approval["payload"]["user_turn_id"] != approval["payload"]["user_turn_id"] and
                all(b["quote"]["quote_id"] != old["quote_id"] for b in backend["bookings"].values()))
        if kind in (ScenarioPredicate.SOLD_OUT, ScenarioPredicate.EXPIRED):
            effect = "SOLD_OUT" if kind == ScenarioPredicate.SOLD_OUT else "OFFER_EXPIRED"
            fault = next(e for e in faults if e["payload"]["failure"]["effect"] == effect)
            failed = next(e for e in events(trace, "TOOL_RESULT", "create_booking_hold") if response(e)["code"] == effect)
            bad_offer = fault["payload"]["failure"]["offer_id"]
            source = [e for e in searches if e["sequence_number"] > failed["sequence_number"] and any(
                d["offer"]["offer_id"] == booking["quote"]["offer_id"] for d in response(e)["data"]["offers"])]
            return bool(response(failed)["commit_status"] == "NOT_COMMITTED" and source and
                booking["quote"]["offer_id"] != bad_offer and
                (kind == ScenarioPredicate.EXPIRED or booking["quote"]["flight_id"] != backend["offers"][bad_offer]["flight_id"]))
        if kind == ScenarioPredicate.DECLINED:
            operations = list(backend["payment_operations"].values())
            return bool(booking and booking["status"] == "HELD" and len(operations) == 1 and
                operations[0]["status"] == "DECLINED" and state.payment_status.value == "DECLINED" and
                not backend["tickets"] and not any(p["status"] == "SUCCESS" for p in backend["payments"].values()))
        if kind == ScenarioPredicate.RECONCILIATION:
            timeout = next(e for e in events(trace, "TOOL_RESULT", "process_payment") if response(e)["code"] == "PAYMENT_TIMEOUT")
            operation = response(timeout)["operation_ref"]
            verified = [e for e in events(trace, "TOOL_RESULT", "get_payment_status") if e["allowed"] is True and response(e)["ok"] and
                response(e)["data"]["operation"]["operation_id"] == operation and response(e)["data"]["operation"]["status"] == "SUCCESS"]
            ticket_calls = events(trace, "TOOL_ALLOWED", "issue_ticket")
            query_calls = events(trace, "TOOL_CALL", "get_payment_status")
            return bool(response(timeout)["commit_status"] == "UNKNOWN" and response(timeout)["data"] is None and
                verified and ticket_calls and query_calls and len(backend["payment_operations"]) == 1 and
                all(e["payload"]["tool_input"]["operation_id"] == operation for e in query_calls) and
                timeout["sequence_number"] < verified[0]["sequence_number"] < ticket_calls[0]["sequence_number"] and
                not any(timeout["sequence_number"] < e["sequence_number"] < verified[0]["sequence_number"]
                    for e in events(trace, "TOOL_CALL", "process_payment")))
        if kind == ScenarioPredicate.CHANGE:
            index = next(i for i, cp in enumerate(checkpoints) if cp.action.kind == UserActionKind.CHANGE)
            old, changed = checkpoints[index - 1], checkpoints[index]
            if changed.state.constraints != scenario.expected_constraints:
                applied = [e for e in events(trace, "INPUT_INTERPRETATION_APPLIED") if e["payload"]["source_turn_id"] == changed.user_turn_id]
                if len(applied) == 1:
                    changed = next(cp for cp in checkpoints if cp.user_turn_id == applied[0]["payload"]["actual_user_turn_id"] and cp.action.kind == UserActionKind.APPLY)
            return bool(index > 0 and old.state.quote and old.state.confirmation is None and _no_transactions(old.backend_state) and
                old.state.constraints == scenario.initial_constraints and changed.state.constraints == scenario.expected_constraints and
                changed.state.context != old.state.context and changed.state.selected_offer_id is None and changed.state.quote is None and
                changed.state.confirmation is None and booking["context"] == changed.state.context.model_dump(mode="json") and
                any(e["payload"]["constraints"] == scenario.expected_constraints.model_dump(mode="json") and
                    e["payload"]["user_turn_id"] == changed.user_turn_id for e in events(trace, "REQUIREMENTS_UPDATED")))
        if kind == ScenarioPredicate.BYPASS:
            cp = next(cp for cp in checkpoints if cp.action.kind == UserActionKind.BYPASS)
            return bool(not accepts_confirmation(cp.action.text) and cp.state.quote and cp.state.confirmation is None and
                cp.state.displayed_quote_id == cp.state.quote.quote_id and _no_transactions(cp.backend_state) and _no_transactions(backend))
        if kind == ScenarioPredicate.REPLAY:
            replay_event = next(iter(events(trace, "TRANSPORT_REPLAY")))
            holds = [e for e in events(trace, "TOOL_RESULT", "create_booking_hold") if e["allowed"] is True and response(e)["ok"]]
            original = next(e for e in holds if e["attempt_id"] == replay_event["payload"]["original_attempt_id"])
            replay = next(e for e in holds if e["sequence_number"] > replay_event["sequence_number"])
            allowed = next(e for e in events(trace, "TOOL_ALLOWED") if e["attempt_id"] == replay["attempt_id"])
            entries = [v for v in backend["idempotency_store"] if v["operation"] == "hold"]
            reserves = [v for v in backend["inventory_events"] if v["event_type"] == "HOLD_RESERVED"]
            return (calls[original["attempt_id"]]["payload"]["tool_input"] == calls[replay["attempt_id"]]["payload"]["tool_input"] and
                original["context"] == replay["context"] and response(original)["data"] == response(replay)["data"] and
                allowed["payload"]["replay"] is True and len(entries) == len(reserves) == len(backend["bookings"]) == 1 and
                entries[0]["key"] == state.operation_keys["create_booking_hold"] == replay_event["payload"]["idempotency_key"] and
                reserves[0]["seats_delta"] == -len(scenario.passengers))
        if kind == ScenarioPredicate.PASSENGERS:
            invalid = next(cp for cp in checkpoints if cp.action.kind == UserActionKind.INVALID_PASSENGERS)
            valid = next(cp for cp in checkpoints if cp.action.kind == UserActionKind.PASSENGERS and cp.user_turn_number > invalid.user_turn_number)
            rejected = next(e for e in events(trace, "PASSENGER_VALIDATION_FAILED") if e["payload"]["user_turn_id"] == invalid.user_turn_id)
            accepted = next(e for e in events(trace, "PASSENGERS_VALIDATED") if e["payload"]["user_turn_id"] == valid.user_turn_id)
            return (invalid.state.quote is None and invalid.state.confirmation is None and _no_transactions(invalid.backend_state) and
                rejected["sequence_number"] < accepted["sequence_number"] < events(trace, "TOOL_ALLOWED", "create_booking_hold")[0]["sequence_number"] and
                booking["passengers"] == [p.model_dump(mode="json") for p in scenario.passengers])
        raise ValueError("Unknown scenario predicate")
