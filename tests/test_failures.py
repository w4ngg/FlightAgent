import pytest
from pydantic import ValidationError

from app.backend import MockFlightBackend
from app.failure_demo import FAILURE_CASES, run_failure_demo
from app.failures import FailureConfig, FailureEffect, FailureRule, FailureTrigger
from app.invariants import check_backend_invariants
from app.models import (
    BookingStatus, CommitStatus, ConfirmationRecord, ConfirmationSnapshot,
    PaymentStatus, SearchConstraints, TransactionContext,
)


def hold(backend, prepared, *, key="hold-1", confirmation=None):
    constraints, passengers, context, _, quote, original = prepared
    return backend.create_booking_hold(quote.quote_id, constraints, passengers, context,
                                       original if confirmation is None else confirmation, key)


def pay(backend, prepared, booking, *, key="pay-1", operation_id="PAYOP-1", amount=None):
    return backend.process_payment(booking.booking_id, operation_id,
                                   booking.total_price_vnd if amount is None else amount,
                                   prepared[2], prepared[-1], key)


def reprice(backend, prepared):
    return backend.reprice_offer(prepared[3].offer.offer_id, prepared[0], prepared[1], prepared[2])


def with_quote(backend, prepared, quote):
    constraints, passengers, context, selected, _, _ = prepared
    confirmation = ConfirmationRecord(context=context, snapshot=ConfirmationSnapshot.from_quote(quote),
                                      user_turn_id="mock-user-turn-2", confirmed_at=backend.clock.now())
    return constraints, passengers, context, selected, quote, confirmation


def rule_for(prepared, effect, trigger=FailureTrigger.HOLD, **values):
    return FailureRule(rule_id="test-rule", trigger=trigger, effect=effect,
                       offer_id=prepared[3].offer.offer_id, **values)


@pytest.mark.parametrize("values", [
    {"times": 0}, {"times": True}, {"new_price": 1.5},
    {"trigger": "payment", "effect": "SOLD_OUT"},
    {"trigger": "hold", "effect": "PAYMENT_DECLINED"},
    {"operation_id": "PAYOP-1"}, {"effect": "SOLD_OUT", "new_price": 1},
])
def test_invalid_rule_contracts_fail_at_configuration(values):
    baseline = {"rule_id": "x", "trigger": "hold", "effect": "PRICE_CHANGE"}
    with pytest.raises(ValidationError):
        FailureRule(**{**baseline, **values})


def test_ambiguous_payment_rules_and_duplicate_ids_are_rejected(backend):
    first = FailureRule(rule_id="a", trigger="payment", effect="PAYMENT_DECLINED")
    second = FailureRule(rule_id="b", trigger="payment", effect="TIMEOUT_UNRESOLVED")
    with pytest.raises(ValueError, match="disjoint"):
        backend.configure_failure(rules=[first, second])
    with pytest.raises(ValueError, match="unique"):
        backend.configure_failure(rules=[first, first])
    with pytest.raises(ValueError, match="either"):
        backend.configure_failure(FailureConfig(), rules=[])


def test_price_config_rejects_an_unused_price():
    with pytest.raises(ValidationError):
        FailureConfig(new_price=2_000_000)


def test_reprice_changes_group_price_once_and_invalidates_old_confirmation(backend, prepare):
    prepared = prepare()
    old = prepared[4]
    backend.configure_failure(FailureConfig(force_price_change=True, new_price=old.unit_price_vnd + 200_000,
                                           target_offer_id=old.offer_id))
    changed = reprice(backend, prepared).data
    assert changed.unit_price_vnd == old.unit_price_vnd + 200_000
    assert changed.total_price_vnd == 3 * changed.unit_price_vnd
    assert changed.quote_version == old.quote_version + 1
    assert hold(backend, prepared).code == "QUOTE_STALE"
    revised = with_quote(backend, prepared, changed)
    assert hold(backend, revised, confirmation=prepared[-1]).code == "CONFIRMATION_REQUIRED"
    assert hold(backend, revised).ok
    state = backend.get_backend_state()
    assert len(state["failure_injection"]["events"]) == 1
    assert state["failure_injection"]["consumed"] == {"config-price": 1}
    assert backend.check_invariants().passed


def test_force_price_without_new_price_has_deterministic_default_delta(backend, prepare):
    prepared = prepare()
    backend.configure_failure(FailureConfig(force_price_change=True))
    first = reprice(backend, prepared).data
    second = reprice(backend, prepared).data
    assert first.unit_price_vnd == prepared[4].unit_price_vnd + 250_000
    assert second.unit_price_vnd == first.unit_price_vnd
    assert len(backend.get_backend_state()["failure_injection"]["events"]) == 1


def test_price_change_at_hold_prevents_commit_then_allows_reconfirmation(backend, prepare):
    prepared = prepare()
    backend.configure_failure(rules=[rule_for(prepared, FailureEffect.PRICE_CHANGE)])
    assert hold(backend, prepared).code == "QUOTE_STALE"
    state = backend.get_backend_state()
    assert state["bookings"] == {} and state["inventory_events"] == []
    updated_quote = reprice(backend, prepared).data
    assert hold(backend, with_quote(backend, prepared, updated_quote)).ok
    assert backend.check_invariants().passed


def test_repriced_total_over_budget_cannot_be_confirmed_into_a_booking(backend, prepare):
    prepared = prepare(2)
    budget = prepared[4].total_price_vnd
    constraints = SearchConstraints(**{**prepared[0].model_dump(), "max_budget_vnd": budget})
    prepared = (constraints, *prepared[1:])
    backend.configure_failure(FailureConfig(force_price_change=True, target_offer_id=prepared[4].offer_id))
    assert reprice(backend, prepared).code == "CONSTRAINT_VIOLATION"
    assert hold(backend, prepared).code == "QUOTE_STALE"
    assert backend.get_backend_state()["bookings"] == {}


@pytest.mark.parametrize("effect, code", [(FailureEffect.SOLD_OUT, "SOLD_OUT"),
                                         (FailureEffect.OFFER_EXPIRED, "OFFER_EXPIRED")])
def test_invalid_selected_offer_never_commits_and_an_alternative_can_be_booked(backend, prepare, effect, code):
    prepared = prepare()
    backend.configure_failure(rules=[rule_for(prepared, effect)])
    denied = hold(backend, prepared)
    assert denied.code == code and denied.commit_status == CommitStatus.NOT_COMMITTED
    assert backend.get_backend_state()["bookings"] == {}
    assert backend.get_offer(prepared[4].offer_id).code == code
    alternatives = backend.search_flights(prepared[0]).data.offers
    assert alternatives and all(item.offer.offer_id != prepared[4].offer_id for item in alternatives)
    if effect == FailureEffect.SOLD_OUT:
        assert all(item.flight.flight_id != prepared[4].flight_id for item in alternatives)
        ledger = backend.get_backend_state()["inventory_events"]
        assert ledger[0]["event_type"] == "EXTERNAL_SOLD_OUT"
        assert ledger[0]["seats_delta"] == -prepared[3].available_seats
    next_offer = alternatives[0]
    quote = backend.reprice_offer(next_offer.offer.offer_id, prepared[0], prepared[1], prepared[2]).data
    alternative = (prepared[0], prepared[1], prepared[2], next_offer, quote, None)
    alternative = with_quote(backend, alternative, quote)
    booking = hold(backend, alternative).data
    assert pay(backend, alternative, booking).ok
    assert backend.issue_ticket(booking.booking_id, prepared[2], "ticket-1").ok
    assert backend.check_invariants().passed


def test_denied_or_invalid_hold_does_not_consume_failure_rules(backend, prepare):
    prepared = prepare()
    backend.configure_failure(rules=[rule_for(prepared, FailureEffect.SOLD_OUT)])
    response = backend.create_booking_hold(prepared[4].quote_id, prepared[0], prepared[1], prepared[2], None, "hold-1")
    assert response.code == "CONFIRMATION_REQUIRED"
    wrong_constraints = SearchConstraints(**{**prepared[0].model_dump(), "max_budget_vnd": 1})
    assert hold(backend, (wrong_constraints, *prepared[1:])).code == "CONSTRAINT_VIOLATION"
    assert backend.get_backend_state()["failure_injection"]["events"] == []
    assert hold(backend, prepared).code == "SOLD_OUT"


def test_rule_times_target_and_reset_are_deterministic(backend, prepare):
    prepared = prepare()
    backend.configure_failure(rules=[rule_for(prepared, FailureEffect.PRICE_CHANGE, FailureTrigger.REPRICE,
                                              times=2, session_id="test-session", transaction_intent_id="intent-1")])
    configured_state = backend.get_backend_state()
    wrong_context = TransactionContext(session_id="other", transaction_intent_id="intent-1")
    assert backend.reprice_offer(prepared[4].offer_id, prepared[0], prepared[1], wrong_context).data.unit_price_vnd == prepared[4].unit_price_vnd
    prices = [reprice(backend, prepared).data.unit_price_vnd for _ in range(3)]
    assert prices == [prepared[4].unit_price_vnd + 250_000, prepared[4].unit_price_vnd + 500_000,
                      prepared[4].unit_price_vnd + 500_000]
    assert backend.get_backend_state()["failure_injection"]["consumed"] == {"test-rule": 2}
    backend.reset()
    reset_state = backend.get_backend_state()
    assert reset_state["offers"] == configured_state["offers"]
    assert reset_state["failure_injection"]["consumed"] == {"test-rule": 0}
    assert reset_state["failure_injection"]["events"] == []


def test_price_change_for_another_session_does_not_change_an_existing_locked_booking(backend, prepare):
    first = prepare(session_id="first-session")
    second = prepare(session_id="second-session")
    backend.configure_failure(rules=[rule_for(second, FailureEffect.PRICE_CHANGE, FailureTrigger.REPRICE,
                                              session_id="second-session")])
    booking = hold(backend, first).data
    new_quote = reprice(backend, second).data
    assert new_quote.total_price_vnd > booking.total_price_vnd
    assert pay(backend, first, booking).ok
    current = backend.get_booking(booking.booking_id).data
    assert current.quote == first[4] and current.total_price_vnd == first[4].total_price_vnd
    assert hold(backend, with_quote(backend, second, new_quote)).ok
    assert backend.check_invariants().passed


def test_sold_out_only_takes_unreserved_seats_and_expiry_restores_the_existing_hold(backend, clock, prepare):
    first = prepare(session_id="first-session")
    second = prepare(session_id="second-session")
    backend.configure_failure(rules=[rule_for(second, FailureEffect.SOLD_OUT, session_id="second-session")])
    booking = hold(backend, first).data
    assert hold(backend, second).code == "SOLD_OUT"
    flight_id = booking.quote.flight_id
    assert backend.get_inventory(flight_id).data.available_seats == 0
    external = backend.get_backend_state()["inventory_events"][-1]
    assert external["seats_delta"] == -(first[3].available_seats - len(first[1]))
    assert backend.check_invariants().passed
    clock.advance(backend.config.hold_ttl_seconds)
    assert backend.sweep_expired_holds() == [booking.booking_id]
    assert backend.get_inventory(flight_id).data.available_seats == len(first[1])
    assert backend.sweep_expired_holds() == []
    assert backend.check_invariants().passed


def test_reset_unknown_episode_restores_every_store_and_rearms_the_same_rule(backend, clock, prepare):
    backend.configure_failure(FailureConfig(payment_mode="timeout", target_operation_id="PAYOP-1"))
    initial = backend.get_backend_state()
    prepared = prepare()
    booking = hold(backend, prepared).data
    assert pay(backend, prepared, booking).code == "PAYMENT_TIMEOUT"
    unknown_state = backend.get_backend_state()
    clock.advance(2000)
    assert MockFlightBackend().get_backend_state()["failure_injection"]["rules"] == []
    backend.reset(seed=42)
    assert backend.get_backend_state() == initial
    prepared = prepare()
    booking = hold(backend, prepared).data
    assert pay(backend, prepared, booking).code == "PAYMENT_TIMEOUT"
    assert backend.get_backend_state() == unknown_state
    assert backend.check_invariants().passed


@pytest.mark.parametrize("mode", ["success", "declined", "timeout", "timeout_but_success"])
def test_payment_outcomes_replay_and_charge_cardinality(backend, prepare, mode):
    prepared = prepare()
    backend.configure_failure(FailureConfig(payment_mode=mode, target_operation_id="PAYOP-1"))
    booking = hold(backend, prepared).data
    response = pay(backend, prepared, booking)
    state = backend.get_backend_state()
    status = backend.get_payment_status("PAYOP-1", prepared[2]).data.operation.status
    expected_status = {"success": PaymentStatus.SUCCESS, "declined": PaymentStatus.DECLINED,
                       "timeout": PaymentStatus.UNKNOWN, "timeout_but_success": PaymentStatus.SUCCESS}[mode]
    assert status == expected_status
    assert len(state["payment_operations"]) == 1
    assert sum(p["status"] == "SUCCESS" for p in state["payments"].values()) == int(mode in ("success", "timeout_but_success"))
    if mode in ("timeout", "timeout_but_success"):
        assert response.code == "PAYMENT_TIMEOUT" and response.commit_status == CommitStatus.UNKNOWN
        assert response.retryable and response.operation_ref == "PAYOP-1" and response.data is None
    elif mode == "declined":
        assert response.code == "PAYMENT_DECLINED" and response.commit_status == CommitStatus.NOT_COMMITTED
        assert response.data.payment.status == PaymentStatus.DECLINED
    replay = pay(backend, prepared, booking)
    assert replay.ok == (mode in ("success", "timeout_but_success"))
    assert pay(backend, prepared, booking, key="other-key", operation_id="PAYOP-2").code == "PAYMENT_ALREADY_STARTED"
    assert pay(backend, prepared, booking, amount=1).code == "IDEMPOTENCY_CONFLICT"
    assert len(backend.get_backend_state()["payment_operations"]) == 1
    assert len(backend.get_backend_state()["failure_injection"]["events"]) == int(mode != "success")
    assert backend.check_invariants().passed


def test_timeout_but_committed_reconciles_then_issues_exactly_n_tickets(backend, clock, prepare):
    prepared = prepare()
    backend.configure_failure(FailureConfig(payment_mode="timeout_but_success"))
    booking = hold(backend, prepared).data
    assert pay(backend, prepared, booking).commit_status == CommitStatus.UNKNOWN
    clock.advance(1000)
    assert backend.sweep_expired_holds() == []
    status = backend.get_payment_status("PAYOP-1", prepared[2])
    assert status.ok and status.commit_status == CommitStatus.NOT_APPLICABLE
    assert status.data.operation.status == PaymentStatus.SUCCESS
    assert status.data.booking.status == BookingStatus.PAID
    first = backend.issue_ticket(booking.booking_id, prepared[2], "ticket-1")
    assert first.ok and len(first.data.tickets) == 3
    assert backend.issue_ticket(booking.booking_id, prepared[2], "ticket-1").data.tickets == first.data.tickets
    assert len(backend.get_backend_state()["payments"]) == 1
    assert backend.check_invariants().passed


def test_unknown_payment_preserves_reservation_and_blocks_ticket_and_new_charge(backend, clock, prepare):
    prepared = prepare()
    backend.configure_failure(FailureConfig(payment_mode="timeout"))
    booking = hold(backend, prepared).data
    assert pay(backend, prepared, booking).code == "PAYMENT_TIMEOUT"
    clock.advance(2000)
    assert backend.sweep_expired_holds() == []
    for _ in range(3):
        assert backend.get_payment_status("PAYOP-1", prepared[2]).data.operation.status == PaymentStatus.UNKNOWN
    assert backend.get_booking(booking.booking_id).data.status == BookingStatus.HELD
    assert backend.get_inventory(prepared[4].flight_id).data.available_seats == prepared[3].available_seats - 3
    assert backend.issue_ticket(booking.booking_id, prepared[2], "ticket-1").code == "PAYMENT_REQUIRED"
    assert pay(backend, prepared, booking, key="pay-2", operation_id="PAYOP-2").code == "PAYMENT_ALREADY_STARTED"
    assert backend.get_backend_state()["payments"] == {}
    assert backend.check_invariants().passed


def test_declined_payment_expires_and_releases_once_without_revival(backend, clock, prepare):
    prepared = prepare()
    backend.configure_failure(FailureConfig(payment_mode="declined"))
    booking = hold(backend, prepared).data
    assert pay(backend, prepared, booking).code == "PAYMENT_DECLINED"
    assert backend.issue_ticket(booking.booking_id, prepared[2], "ticket-1").code == "PAYMENT_REQUIRED"
    clock.advance(600)
    assert backend.sweep_expired_holds() == [booking.booking_id]
    assert backend.sweep_expired_holds() == []
    assert pay(backend, prepared, booking).code == "PAYMENT_DECLINED"
    assert backend.get_booking(booking.booking_id).data.status == BookingStatus.EXPIRED
    assert backend.get_inventory(prepared[4].flight_id).data.available_seats == prepared[3].available_seats
    assert backend.check_invariants().passed


def test_payment_rule_target_does_not_fire_for_another_operation(backend, prepare):
    prepared = prepare()
    backend.configure_failure(FailureConfig(payment_mode="timeout", target_operation_id="PAYOP-target"))
    booking = hold(backend, prepared).data
    assert pay(backend, prepared, booking).ok
    assert backend.get_backend_state()["failure_injection"]["events"] == []


def test_invalid_payment_does_not_consume_rule(backend, prepare):
    prepared = prepare()
    backend.configure_failure(FailureConfig(payment_mode="declined"))
    booking = hold(backend, prepared).data
    assert pay(backend, prepared, booking, amount=1).code == "AMOUNT_MISMATCH"
    assert backend.get_backend_state()["failure_injection"]["events"] == []
    assert pay(backend, prepared, booking).code == "PAYMENT_DECLINED"


def test_configuration_cannot_erase_existing_episode_history(backend, prepare):
    prepared = prepare()
    assert hold(backend, prepared).ok
    with pytest.raises(ValueError, match="reset"):
        backend.configure_failure(FailureConfig(payment_mode="declined"))


def test_backend_invariants_detect_corrupted_inventory_tickets_and_confirmation(backend, prepare):
    prepared = prepare()
    booking = hold(backend, prepared).data
    assert pay(backend, prepared, booking).ok
    assert backend.issue_ticket(booking.booking_id, prepared[2], "ticket-1").ok
    assert backend.check_invariants().passed
    state = backend.get_backend_state()
    state["inventory"][prepared[4].flight_id]["available_seats"] += 1
    assert not check_backend_invariants(state).checks["inventory_ledger"]
    state = backend.get_backend_state()
    state["tickets"].pop(next(iter(state["tickets"])))
    assert not check_backend_invariants(state).checks["booking_state_and_ticket_count"]
    state = backend.get_backend_state()
    state["bookings"][booking.booking_id]["confirmation"]["snapshot"]["total_price_vnd"] += 1
    assert not check_backend_invariants(state).checks["confirmation_snapshot"]
    assert backend.check_invariants().passed  # Inspection mutation never changes the backend.


def test_invariants_reject_releasing_unknown_payment_even_with_a_balanced_ledger(backend, clock, prepare):
    prepared = prepare()
    backend.configure_failure(FailureConfig(payment_mode="timeout"))
    booking = hold(backend, prepared).data
    assert pay(backend, prepared, booking).code == "PAYMENT_TIMEOUT"
    clock.advance(600)
    state = backend.get_backend_state()
    state["bookings"][booking.booking_id]["status"] = "EXPIRED"
    state["inventory"][booking.quote.flight_id]["available_seats"] += len(booking.passengers)
    state["inventory_events"].append({
        "sequence_number": 2, "event_type": "HOLD_RELEASED", "timestamp": clock.now().isoformat(),
        "flight_id": booking.quote.flight_id, "seats_delta": len(booking.passengers),
        "booking_id": booking.booking_id, "rule_id": None,
    })
    report = check_backend_invariants(state)
    assert report.checks["inventory_ledger"] and report.checks["reservation_lifecycle"]
    assert not report.checks["operation_lifecycle"] and not report.passed
    assert backend.check_invariants().passed


@pytest.mark.parametrize("corruption, check", [
    ("duplicate_charge", "no_duplicate_charge"),
    ("payment_amount", "payment_links_and_amounts"),
    ("duplicate_ticket", "ticket_links_and_uniqueness"),
])
def test_invariants_reject_duplicate_or_mismatched_payment_and_ticket(backend, prepare, corruption, check):
    prepared = prepare()
    booking = hold(backend, prepared).data
    assert pay(backend, prepared, booking).ok
    assert backend.issue_ticket(booking.booking_id, prepared[2], "ticket-1").ok
    state = backend.get_backend_state()
    if corruption == "duplicate_charge":
        payment = next(iter(state["payments"].values()))
        state["payments"]["PAY-duplicate"] = {**payment, "payment_id": "PAY-duplicate"}
    elif corruption == "payment_amount":
        next(iter(state["payments"].values()))["amount_vnd"] += 1
    else:
        ticket = next(iter(state["tickets"].values()))
        state["tickets"]["TKT-duplicate"] = {**ticket, "ticket_number": "TKT-duplicate"}
    report = check_backend_invariants(state)
    assert not report.checks[check] and not report.passed
    assert backend.check_invariants().passed


@pytest.mark.parametrize("case", FAILURE_CASES)
def test_scripted_failure_demo_is_deterministic_and_obeys_recovery_order(case):
    first = run_failure_demo(case)
    assert run_failure_demo(case) == first
    expected = "HANDOFF_REQUIRED" if case == "timeout" else "FAILED" if case in ("declined", "hold_expired") else "BOOKED"
    assert first["result"] == expected and first["invariants"]["passed"]
    assert first["ticket_count"] == (3 if expected == "BOOKED" else 0)
    assert first["committed_payments"] == int(expected == "BOOKED")
    if case == "timeout":
        assert first["reconciliation_checks"] == 2
        assert sum(item["tool"] == "process_payment" for item in first["observations"]) == 1
    if case == "timeout_but_success":
        tools = [item["tool"] for item in first["observations"]]
        assert tools.index("process_payment") < tools.index("get_payment_status") < tools.index("issue_ticket")
        assert first["reconciliation_checks"] == 1
