import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, time

import pytest

from app.backend import MockFlightBackend
from app.demo import run_demo
from app.models import (
    BookingStatus, CommitStatus, ConfirmationRecord, ConfirmationSnapshot,
    Passenger, PaymentStatus, SearchConstraints, TransactionContext,
)


def hold(backend, prepared, key="hold-1"):
    constraints, passengers, context, _, quote, confirmation = prepared
    return backend.create_booking_hold(quote.quote_id, constraints, passengers, context, confirmation, key)


def pay(backend, held_data, key="pay-1", operation_id="PAYOP-1"):
    booking, context, confirmation, _ = held_data
    return backend.process_payment(booking.booking_id, operation_id, booking.total_price_vnd, context, confirmation, key)


def test_seed_has_all_routes_airlines_and_fare_inventory_links(backend):
    state = backend.get_backend_state()
    assert len(state["airports"]) == 3
    assert len(state["flights"]) == 36
    assert len(state["offers"]) == 54
    assert len(state["inventory"]) == 36
    assert {(f["origin"], f["destination"]) for f in state["flights"].values()} == {
        (a, b) for a in ("SGN", "HAN", "DAD") for b in ("SGN", "HAN", "DAD") if a != b
    }
    assert len({f["airline"] for f in state["flights"].values()}) == 3
    assert all(offer["flight_id"] in state["inventory"] for offer in state["offers"].values())


def test_reset_reproduces_time_seed_and_clears_all_runtime_stores(backend, clock, prepare):
    initial = backend.get_backend_state()
    assert hold(backend, prepare()).ok
    clock.advance(100)
    backend.reset(seed=42)
    assert backend.get_backend_state() == initial
    other = MockFlightBackend(seed=43).get_backend_state()
    assert other["offers"] != initial["offers"]


@pytest.mark.parametrize("name, expected", [("Sài Gòn", "SGN"), (" TP HCM ", "SGN"), ("Hà Nội", "HAN"),
                                            ("Đà Nẵng", "DAD"), ("sgn", "SGN")])
def test_airport_resolution(backend, name, expected):
    response = backend.resolve_airport(name)
    assert response.ok and response.data.code == expected
    assert response.commit_status == CommitStatus.NOT_APPLICABLE


def test_unknown_airport_and_offer_are_structured_errors(backend):
    assert backend.resolve_airport("Tokyo").code == "UNKNOWN_AIRPORT"
    assert backend.get_offer("INVENTED").code == "OFFER_NOT_FOUND"
    assert backend.get_booking("INVENTED").code == "BOOKING_NOT_FOUND"


def test_search_filters_route_date_group_budget_and_time(backend):
    constraints = SearchConstraints(origin="SGN", destination="HAN", departure_date=date(2026, 10, 10),
                                    passenger_count=2, max_budget_vnd=4_000_000, latest_departure=time(12))
    response = backend.search_flights(constraints)
    assert response.ok and response.data.offers
    for details in response.data.offers:
        assert details.flight.origin == "SGN" and details.flight.destination == "HAN"
        assert details.flight.departure_time.date() == date(2026, 10, 10)
        assert details.flight.departure_time.hour < 12
        assert details.total_price_vnd <= 4_000_000
        assert details.total_price_vnd == 2 * details.offer.unit_price_vnd


def test_departure_boundary_can_be_exclusive(backend):
    common = dict(origin="SGN", destination="HAN", departure_date=date(2026, 10, 10),
                  passenger_count=1, latest_departure=time(8, 10))
    assert backend.search_flights(SearchConstraints(**common)).data.offers
    assert backend.search_flights(SearchConstraints(**common, latest_inclusive=False)).data.offers == []


def test_empty_search_is_success_not_backend_error(backend):
    constraints = SearchConstraints(origin="SGN", destination="DAD", departure_date=date(2026, 11, 10), passenger_count=1)
    response = backend.search_flights(constraints)
    assert response.ok and response.code == "OK" and response.data.offers == []
    assert response.model_dump(mode="json")["data"] == {"offers": []}


def test_past_date_is_rejected(backend):
    constraints = SearchConstraints(origin="SGN", destination="HAN", departure_date=date(2026, 10, 4), passenger_count=1)
    assert backend.search_flights(constraints).code == "INVALID_DATE"


def test_three_passenger_end_to_end_keeps_amount_inventory_and_ticket_links(backend, held):
    booking, context, _, selected = held
    assert booking.status == BookingStatus.HELD
    assert booking.total_price_vnd == 3 * selected.offer.unit_price_vnd
    payment = pay(backend, held)
    assert payment.ok and payment.data.operation.status == PaymentStatus.SUCCESS
    assert payment.data.booking.status == BookingStatus.PAID
    result = backend.issue_ticket(booking.booking_id, context, "ticket-1")
    assert result.ok and result.data.booking.status == BookingStatus.TICKETED
    assert len(result.data.tickets) == 3
    assert {t.passenger_index for t in result.data.tickets} == {0, 1, 2}
    assert {t.passenger_id for t in result.data.tickets} == {p.passenger_id for p in booking.passengers}
    assert all(t.booking_id == booking.booking_id and t.flight_id == selected.flight.flight_id for t in result.data.tickets)
    assert backend.get_inventory(selected.flight.flight_id).data.available_seats == selected.available_seats - 3
    state = backend.get_backend_state()
    assert len(state["bookings"]) == len(state["payments"]) == len(state["payment_operations"]) == 1
    assert len(state["tickets"]) == 3
    status = backend.get_payment_status("PAYOP-1", context)
    assert status.data.booking.status == BookingStatus.TICKETED
    assert status.commit_status == CommitStatus.NOT_APPLICABLE
    json.dumps(state)


def test_missing_confirmation_does_not_reserve_seats(backend, prepare):
    constraints, passengers, context, selected, quote, _ = prepare()
    response = backend.create_booking_hold(quote.quote_id, constraints, passengers, context, None, "hold-1")
    assert not response.ok and response.code == "CONFIRMATION_REQUIRED"
    assert response.commit_status == CommitStatus.NOT_COMMITTED
    assert backend.get_backend_state()["bookings"] == {}
    assert backend.get_inventory(selected.flight.flight_id).data.available_seats == selected.available_seats


def test_confirmation_with_wrong_price_is_rejected(backend, prepare):
    prepared = list(prepare())
    confirmation = prepared[-1]
    wrong_snapshot = ConfirmationSnapshot(**{**confirmation.snapshot.model_dump(), "total_price_vnd": 1})
    prepared[-1] = ConfirmationRecord(**{**confirmation.model_dump(), "snapshot": wrong_snapshot})
    assert hold(backend, prepared).code == "CONFIRMATION_REQUIRED"
    assert backend.get_backend_state()["bookings"] == {}


def test_editing_passenger_with_same_id_invalidates_quote(backend, prepare):
    prepared = list(prepare())
    passengers = list(prepared[1])
    passengers[0] = Passenger(**{**passengers[0].model_dump(), "date_of_birth": date(1991, 1, 1)})
    prepared[1] = passengers
    assert hold(backend, prepared).code == "PASSENGER_CHANGED"
    assert backend.get_backend_state()["bookings"] == {}


def test_quote_expires_at_exact_ttl(backend, clock, prepare):
    prepared = prepare()
    clock.advance(300)
    assert hold(backend, prepared).code == "QUOTE_EXPIRED"
    assert backend.get_backend_state()["bookings"] == {}


def test_reprice_versions_invalidate_old_quote_for_same_intent(backend, prepare):
    prepared = prepare()
    constraints, passengers, context, selected, old_quote, _ = prepared
    new_quote = backend.reprice_offer(selected.offer.offer_id, constraints, passengers, context).data
    assert new_quote.quote_version == old_quote.quote_version + 1
    assert hold(backend, prepared).code == "QUOTE_STALE"


@pytest.mark.parametrize("edit", ["missing_profile", "duplicate_document", "future_birth"])
def test_invalid_passengers_cannot_get_a_quote(backend, prepare, edit):
    constraints, passengers, context, selected, _, _ = prepare()
    if edit == "missing_profile":
        passengers = passengers[:2]
    elif edit == "duplicate_document":
        passengers[1] = Passenger(**{**passengers[1].model_dump(), "document_number": passengers[0].document_number})
    else:
        passengers[0] = Passenger(**{**passengers[0].model_dump(), "date_of_birth": date(2030, 1, 1)})
    response = backend.reprice_offer(selected.offer.offer_id, constraints, passengers, context)
    assert response.code == "INVALID_PASSENGER"


def test_unpaid_booking_cannot_issue_tickets(backend, held):
    booking, context, _, _ = held
    assert backend.issue_ticket(booking.booking_id, context, "ticket-1").code == "PAYMENT_REQUIRED"
    assert backend.get_backend_state()["tickets"] == {}


def test_payment_checks_amount_and_confirmation(backend, held):
    booking, context, confirmation, _ = held
    assert backend.process_payment(booking.booking_id, "PAYOP-1", 1, context, confirmation, "pay-1").code == "AMOUNT_MISMATCH"
    assert backend.process_payment(booking.booking_id, "PAYOP-1", booking.total_price_vnd, context, None, "pay-1").code == "CONFIRMATION_REQUIRED"
    assert backend.get_backend_state()["payments"] == {}
    assert backend.get_booking(booking.booking_id).data.status == BookingStatus.HELD


def test_payment_cannot_target_another_context(backend, held):
    booking, _, confirmation, _ = held
    other_context = TransactionContext(session_id="other-session", transaction_intent_id="other-intent")
    response = backend.process_payment(booking.booking_id, "PAYOP-1", booking.total_price_vnd,
                                       other_context, confirmation, "pay-1")
    assert response.code == "CONTEXT_MISMATCH"
    assert backend.get_backend_state()["payments"] == {}


def test_payment_remains_valid_after_quote_expiry_inside_hold_ttl(backend, clock, held):
    clock.advance(301)
    assert pay(backend, held).ok


def test_expired_hold_restores_inventory_once_and_cannot_be_paid(backend, clock, held):
    booking, _, _, selected = held
    clock.advance(600)
    assert backend.sweep_expired_holds() == [booking.booking_id]
    assert backend.sweep_expired_holds() == []
    assert backend.get_booking(booking.booking_id).data.status == BookingStatus.EXPIRED
    assert backend.get_inventory(selected.flight.flight_id).data.available_seats == selected.available_seats
    assert pay(backend, held).code == "INVALID_BOOKING_STATE"
    assert backend.get_backend_state()["payments"] == {}


def test_paid_booking_does_not_expire_or_release_seats(backend, clock, held):
    booking, _, _, selected = held
    assert pay(backend, held).ok
    clock.advance(1000)
    assert backend.sweep_expired_holds() == []
    assert backend.get_booking(booking.booking_id).data.status == BookingStatus.PAID
    assert backend.get_inventory(selected.flight.flight_id).data.available_seats == selected.available_seats - 3


def test_hold_replay_does_not_reserve_again_or_extend_ttl(backend, clock, prepare):
    prepared = prepare()
    first = hold(backend, prepared)
    clock.advance(100)
    second = hold(backend, prepared)
    assert second.data == first.data
    assert len(backend.get_backend_state()["bookings"]) == 1
    selected = prepared[3]
    assert backend.get_inventory(selected.flight.flight_id).data.available_seats == selected.available_seats - 3
    assert hold(backend, prepared, "hold-other-key").code == "HOLD_ALREADY_CREATED"


def test_same_hold_key_with_changed_payload_conflicts(backend, prepare):
    prepared = prepare()
    assert hold(backend, prepared).ok
    changed = list(prepared)
    passengers = list(changed[1])
    passengers[0] = Passenger(**{**passengers[0].model_dump(), "first_name": "Changed"})
    changed[1] = passengers
    assert hold(backend, changed).code == "IDEMPOTENCY_CONFLICT"
    assert len(backend.get_backend_state()["bookings"]) == 1


def test_payment_replay_and_new_key_do_not_charge_twice(backend, held):
    first = pay(backend, held)
    assert first.ok
    assert pay(backend, held).data.payment.payment_id == first.data.payment.payment_id
    denied = pay(backend, held, "pay-other-key", "PAYOP-2")
    assert denied.code == "PAYMENT_ALREADY_STARTED" and denied.operation_ref == "PAYOP-1"
    assert len(backend.get_backend_state()["payments"]) == 1


def test_ticket_retry_and_historical_replay_never_regress_backend(backend, clock, held):
    booking, context, confirmation, _ = held
    assert pay(backend, held).ok
    first = backend.issue_ticket(booking.booking_id, context, "ticket-1")
    assert backend.issue_ticket(booking.booking_id, context, "ticket-1").data == first.data
    assert backend.issue_ticket(booking.booking_id, context, "ticket-2").data.tickets == first.data.tickets
    clock.advance(1000)
    assert pay(backend, held).ok
    hold_replay = backend.create_booking_hold(booking.quote.quote_id, booking.constraints, booking.passengers,
                                               context, confirmation, "hold-1")
    assert hold_replay.ok
    assert backend.get_booking(booking.booking_id).data.status == BookingStatus.TICKETED
    assert len(backend.get_backend_state()["tickets"]) == 3


def test_returned_models_and_inspection_are_detached_from_backend(backend, prepare):
    prepared = prepare()
    result = hold(backend, prepared)
    booking_id = result.data.booking_id
    result.data.passengers.clear()
    prepared[-1].snapshot.passenger_ids.clear()
    assert len(backend.get_booking(booking_id).data.passengers) == 3
    assert len(backend.get_booking(booking_id).data.confirmation.snapshot.passenger_ids) == 3
    snapshot = backend.get_backend_state()
    snapshot["inventory"][prepared[3].flight.flight_id]["available_seats"] = 0
    assert backend.get_inventory(prepared[3].flight.flight_id).data.available_seats > 0


def test_parallel_holds_do_not_oversell(backend, prepare):
    small = prepare(1)
    capacity = small[3].available_seats
    count = capacity // 2 + 1
    first = prepare(count, session_id="session-a")
    second = prepare(count, session_id="session-b")
    assert first[3].offer.offer_id == second[3].offer.offer_id
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda prepared: hold(backend, prepared), [first, second]))
    assert sum(response.ok for response in responses) == 1
    assert {response.code for response in responses} == {"OK", "SOLD_OUT"}
    assert backend.get_inventory(first[3].flight.flight_id).data.available_seats == capacity - count


def test_demo_is_a_complete_llm_free_booking():
    result = run_demo(passenger_count=3)
    assert result["result"] == "BOOKED"
    assert result["passenger_count"] == len(result["ticket_numbers"]) == 3
    assert result["total_price_vnd"] == 3 * result["unit_price_vnd"]
    assert all(result["checks"].values())
