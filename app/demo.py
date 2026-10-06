"""Manual happy-path driver for milestone A: python -m app.demo."""

import argparse
import json
from datetime import date, timedelta
from typing import TypeVar

from app.backend import MockFlightBackend
from app.config import DEFAULT_SEED
from app.models import (
    BookingStatus, ConfirmationRecord, ConfirmationSnapshot, Passenger,
    PaymentStatus, SearchConstraints, ToolResponse, TransactionContext,
)

DataType = TypeVar("DataType")


def _require_data(response: ToolResponse[DataType]) -> DataType:
    if not response.ok or response.data is None:
        raise RuntimeError(response.model_dump_json())
    return response.data


def run_demo(passenger_count: int = 3, seed: int = DEFAULT_SEED) -> dict:
    backend = MockFlightBackend(seed=seed)
    context = TransactionContext(session_id="demo-session", transaction_intent_id="demo-intent-1")
    origin = _require_data(backend.resolve_airport("Sài Gòn"))
    destination = _require_data(backend.resolve_airport("Hà Nội"))
    constraints = SearchConstraints(
        origin=origin.code, destination=destination.code,
        departure_date=backend.clock.now().date() + timedelta(days=5),
        passenger_count=passenger_count,
    )
    passengers = [Passenger(
        passenger_id=f"MOCK-P{i + 1:03d}", first_name=f"Passenger{i + 1}", last_name="Synthetic",
        date_of_birth=date(1990, 1, 1), document_type="IDENTITY_CARD",
        document_number=f"MOCK-DOC-{i + 1:03d}",
    ) for i in range(passenger_count)]
    offers = _require_data(backend.search_flights(constraints)).offers
    if not offers:
        raise RuntimeError("No seeded offer has enough seats for this group")
    selected = offers[0]
    quote = _require_data(backend.reprice_offer(selected.offer.offer_id, constraints, passengers, context))

    # This is an explicitly scripted mock user confirmation, not LLM inference.
    confirmation = ConfirmationRecord(
        context=context, snapshot=ConfirmationSnapshot.from_quote(quote),
        user_turn_id="mock-user-confirmation-1", confirmed_at=backend.clock.now(),
    )
    held = _require_data(backend.create_booking_hold(
        quote.quote_id, constraints, passengers, context, confirmation, "HOLD-demo-intent-1",
    ))
    payment = _require_data(backend.process_payment(
        held.booking_id, "PAYOP-demo-intent-1", held.total_price_vnd,
        context, confirmation, "PAY-demo-intent-1",
    ))
    ticket_result = _require_data(backend.issue_ticket(held.booking_id, context, "TICKET-demo-intent-1"))
    read_back = _require_data(backend.get_booking(held.booking_id))
    inventory = _require_data(backend.get_inventory(quote.flight_id))
    checks = {
        "booking_ticketed": read_back.status == BookingStatus.TICKETED,
        "payment_success": payment.operation.status == PaymentStatus.SUCCESS,
        "group_price_correct": read_back.total_price_vnd == quote.unit_price_vnd * passenger_count,
        "ticket_count_correct": len(ticket_result.tickets) == passenger_count,
        "inventory_correct": selected.available_seats - inventory.available_seats == passenger_count,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Demo read-back verification failed: {checks}")
    return {
        "result": "BOOKED", "mode": "MOCK_BACKEND_WITHOUT_LLM", "seed": seed,
        "route": f"{read_back.quote.origin} -> {read_back.quote.destination}",
        "departure_date": read_back.quote.departure_date.isoformat(),
        "offer_id": quote.offer_id, "quote_version": quote.quote_version,
        "booking_id": read_back.booking_id, "booking_status": read_back.status.value,
        "passenger_count": passenger_count, "unit_price_vnd": quote.unit_price_vnd,
        "total_price_vnd": read_back.total_price_vnd,
        "payment_operation_id": payment.operation.operation_id,
        "payment_id": payment.payment.payment_id,
        "ticket_numbers": [ticket.ticket_number for ticket in ticket_result.tickets],
        "remaining_seats": inventory.available_seats, "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone A mock booking demo (no LLM/API key)")
    parser.add_argument("--passengers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.passengers < 1:
        parser.error("--passengers must be positive")
    try:
        result = run_demo(args.passengers, args.seed)
    except RuntimeError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
