"""Backend integrity checks, independent of agents and conversational success."""

from collections import Counter
from datetime import datetime

from pydantic import ValidationError

from app.config import VIETNAM_TZ
from app.models import (
    BackendInvariantReport, ConfirmationSnapshot, Flight, SearchConstraints, payload_hash,
)


def check_backend_invariants(state: dict) -> BackendInvariantReport:
    bookings = state["bookings"]
    payments = state["payments"]
    operations = state["payment_operations"]
    tickets = state["tickets"]
    events = state["inventory_events"]
    inventory = state["inventory"]
    checks: dict[str, bool] = {}

    checks["inventory_bounds"] = all(
        0 <= item["available_seats"] <= item["total_seats"] for item in inventory.values()
    )
    delta = Counter()
    for event in events:
        delta[event["flight_id"]] += event["seats_delta"]
    checks["inventory_ledger"] = all(
        item["available_seats"] == item["total_seats"] + delta[flight_id]
        for flight_id, item in inventory.items()
    ) and all(event["flight_id"] in inventory for event in events)

    reservations_valid = True
    group_price_valid = True
    confirmations_valid = True
    constraints_valid = True
    for booking_id, booking in bookings.items():
        quote = booking["quote"]
        passengers = booking["passengers"]
        count = len(passengers)
        reserved = [event for event in events if event["booking_id"] == booking_id and event["event_type"] == "HOLD_RESERVED"]
        released = [event for event in events if event["booking_id"] == booking_id and event["event_type"] == "HOLD_RELEASED"]
        reservations_valid &= (
            len(reserved) == 1 and reserved[0]["seats_delta"] == -count
            and reserved[0]["flight_id"] == quote["flight_id"]
            and (len(released) == 1 and released[0]["seats_delta"] == count
                 and released[0]["flight_id"] == quote["flight_id"]
                 if booking["status"] == "EXPIRED" else len(released) == 0)
        )
        group_price_valid &= (
            count > 0 and count == quote["passenger_count"] == booking["constraints"]["passenger_count"]
            and len({p["passenger_id"] for p in passengers}) == count
            and len({(p["document_type"], p["document_number"]) for p in passengers}) == count
            and quote["passenger_ids"] == [p["passenger_id"] for p in passengers]
            and quote["passenger_payload_hash"] == payload_hash(passengers)
            and booking["total_price_vnd"] == quote["total_price_vnd"] == quote["unit_price_vnd"] * count
            and booking["currency"] == quote["currency"] == "VND"
        )
        confirmation = booking["confirmation"]
        expected_snapshot = {field: quote[field] for field in ConfirmationSnapshot.model_fields}
        confirmations_valid &= (
            confirmation["snapshot"] == expected_snapshot
            and state["quotes"].get(quote["quote_id"]) == quote
            and confirmation["context"] == quote["context"] == booking["context"]
            and bool(confirmation["user_turn_id"].strip())
            and datetime.fromisoformat(quote["created_at"]) <= datetime.fromisoformat(confirmation["confirmed_at"])
            <= datetime.fromisoformat(booking["created_at"]) < datetime.fromisoformat(quote["expires_at"])
        )
        try:
            constraints = SearchConstraints.model_validate(booking["constraints"])
            flight = Flight.model_validate(state["flights"].get(quote["flight_id"]))
            offer = state["offers"].get(quote["offer_id"])
            constraints_valid &= (
                constraints.permits(flight, booking["total_price_vnd"])
                and quote["origin"] == flight.origin and quote["destination"] == flight.destination
                and quote["departure_date"] == flight.departure_time.astimezone(VIETNAM_TZ).date().isoformat()
                and offer is not None and offer["flight_id"] == flight.flight_id
            )
        except (ValidationError, TypeError):
            constraints_valid = False
    checks["reservation_lifecycle"] = bool(reservations_valid) and all(
        event["booking_id"] in bookings if event["event_type"] != "EXTERNAL_SOLD_OUT"
        else event["booking_id"] is None and event["seats_delta"] <= 0
        for event in events
    )
    checks["group_price_and_passengers"] = bool(group_price_valid)
    checks["confirmation_snapshot"] = bool(confirmations_valid)
    checks["booking_constraints"] = bool(constraints_valid)

    payment_links_valid = True
    for payment_id, payment in payments.items():
        booking = bookings.get(payment["booking_id"])
        operation = operations.get(payment["operation_id"])
        payment_links_valid &= bool(
            booking and operation and payment["payment_id"] == payment_id
            and operation["payment_id"] == payment_id
            and operation["booking_id"] == payment["booking_id"]
            and operation["context"] == booking["context"]
            and payment["status"] == operation["status"]
            and payment["amount_vnd"] == operation["amount_vnd"] == booking["total_price_vnd"]
            and payment["currency"] == operation["currency"] == booking["currency"] == "VND"
            and payment["idempotency_key"] == operation["idempotency_key"]
        )
    checks["payment_links_and_amounts"] = bool(payment_links_valid)

    operation_links_valid = True
    for operation_id, operation in operations.items():
        booking = bookings.get(operation["booking_id"])
        payment = payments.get(operation["payment_id"])
        operation_links_valid &= bool(
            booking and operation["operation_id"] == operation_id
            and operation["context"] == booking["context"]
            and operation["amount_vnd"] == booking["total_price_vnd"]
            and operation["currency"] == booking["currency"] == "VND"
            and datetime.fromisoformat(booking["created_at"]) <= datetime.fromisoformat(operation["created_at"])
            < datetime.fromisoformat(booking["hold_expires_at"])
            and (
                payment is None and operation["payment_id"] is None and booking["status"] == "HELD"
                if operation["status"] in ("PENDING", "UNKNOWN")
                else payment is not None and payment["status"] == operation["status"]
                and payment["operation_id"] == operation_id
                and (booking["status"] in ("PAID", "TICKETED") if operation["status"] == "SUCCESS"
                     else operation["status"] == "DECLINED" and booking["status"] in ("HELD", "EXPIRED"))
            )
        )
    operation_counts = Counter(op["booking_id"] for op in operations.values())
    charge_counts = Counter(p["booking_id"] for p in payments.values() if p["status"] == "SUCCESS")
    checks["operation_lifecycle"] = bool(operation_links_valid) and all(n <= 1 for n in operation_counts.values())
    checks["no_duplicate_charge"] = all(n <= 1 for n in charge_counts.values())

    ticket_links_valid = True
    seen = set()
    for ticket_number, ticket in tickets.items():
        booking = bookings.get(ticket["booking_id"])
        identity = ticket["booking_id"], ticket["passenger_index"]
        ticket_links_valid &= bool(
            booking and ticket_number == ticket["ticket_number"] and ticket["status"] == "ISSUED"
            and booking["status"] == "TICKETED" and charge_counts[ticket["booking_id"]] == 1
            and ticket["flight_id"] == booking["quote"]["flight_id"]
            and 0 <= ticket["passenger_index"] < len(booking["passengers"])
            and ticket["passenger_id"] == booking["passengers"][ticket["passenger_index"]]["passenger_id"]
            and identity not in seen
        )
        seen.add(identity)
    ticket_counts = Counter(ticket["booking_id"] for ticket in tickets.values())
    checks["ticket_links_and_uniqueness"] = bool(ticket_links_valid)
    checks["booking_state_and_ticket_count"] = all(
        charge_counts[booking_id] == 1 and ticket_counts[booking_id] == len(booking["passengers"])
        if booking["status"] == "TICKETED"
        else charge_counts[booking_id] == 1 and ticket_counts[booking_id] == 0
        if booking["status"] == "PAID"
        else charge_counts[booking_id] == 0 and ticket_counts[booking_id] == 0
        if booking["status"] in ("HELD", "EXPIRED") else False
        for booking_id, booking in bookings.items()
    )

    return BackendInvariantReport(
        passed=all(checks.values()), checks=checks,
        violations=[f"{name} failed" for name, passed in checks.items() if not passed],
    )
