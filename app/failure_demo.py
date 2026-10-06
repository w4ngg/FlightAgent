"""Scripted, LLM-free recovery examples for milestone B, not an agent harness."""

import argparse
import json
from datetime import date, timedelta
from typing import Literal

from app.backend import MockFlightBackend
from app.config import DEFAULT_SEED
from app.demo import _require_data
from app.failures import FailureConfig, FailureEffect, FailureRule, FailureTrigger
from app.models import ConfirmationRecord, ConfirmationSnapshot, Passenger, SearchConstraints, TransactionContext

FailureCase = Literal["price_change", "sold_out", "offer_expired", "declined", "timeout", "timeout_but_success", "hold_expired"]
FAILURE_CASES = ("price_change", "sold_out", "offer_expired", "declined", "timeout", "timeout_but_success", "hold_expired")


def run_failure_demo(case: FailureCase, passenger_count: int = 3, seed: int = DEFAULT_SEED) -> dict:
    if case not in FAILURE_CASES:
        raise ValueError(f"Unknown failure case: {case}")
    backend = MockFlightBackend(seed=seed)
    context = TransactionContext(session_id="failure-demo-session", transaction_intent_id="intent-1")
    constraints = SearchConstraints(
        origin="SGN", destination="HAN", departure_date=backend.clock.now().date() + timedelta(days=5),
        passenger_count=passenger_count,
    )
    passengers = [Passenger(
        passenger_id=f"MOCK-P{i + 1}", first_name=f"Mock{i + 1}", last_name="Synthetic",
        date_of_birth=date(1990, 1, 1), document_type="IDENTITY_CARD", document_number=f"MOCK-DOC-{i + 1}",
    ) for i in range(passenger_count)]
    observations = []

    def observe(tool, response):
        observations.append({"tool": tool, "code": response.code, "ok": response.ok,
                             "commit_status": response.commit_status.value, "operation_ref": response.operation_ref})
        return response

    def confirm(quote, turn):
        return ConfirmationRecord(context=context, snapshot=ConfirmationSnapshot.from_quote(quote),
                                  user_turn_id=f"mock-user-confirmation-{turn}", confirmed_at=backend.clock.now())

    offers = _require_data(observe("search_flights", backend.search_flights(constraints))).offers
    if not offers:
        raise RuntimeError("No offer has enough seats for this group")
    selected = offers[0]
    quote = _require_data(observe("reprice_offer", backend.reprice_offer(selected.offer.offer_id, constraints, passengers, context)))
    confirmation = confirm(quote, 1)
    if case in ("price_change", "sold_out", "offer_expired"):
        effect = {"price_change": FailureEffect.PRICE_CHANGE, "sold_out": FailureEffect.SOLD_OUT,
                  "offer_expired": FailureEffect.OFFER_EXPIRED}[case]
        backend.configure_failure(rules=[FailureRule(rule_id=f"demo-{case}", trigger=FailureTrigger.HOLD,
                                                     effect=effect, offer_id=selected.offer.offer_id)])
    elif case in ("declined", "timeout", "timeout_but_success"):
        backend.configure_failure(FailureConfig(payment_mode=case, target_operation_id="PAYOP-demo-1"))

    held = observe("create_booking_hold", backend.create_booking_hold(
        quote.quote_id, constraints, passengers, context, confirmation, "hold-intent-1",
    ))
    if not held.ok:
        if case in ("sold_out", "offer_expired"):
            alternatives = _require_data(observe("search_flights", backend.search_flights(constraints))).offers
            if not alternatives:
                raise RuntimeError("No alternative remains after the injected failure")
            selected = alternatives[0]
        elif case != "price_change":
            raise RuntimeError(held.model_dump_json())
        quote = _require_data(observe("reprice_offer", backend.reprice_offer(selected.offer.offer_id, constraints, passengers, context)))
        confirmation = confirm(quote, 2)
        held = observe("create_booking_hold", backend.create_booking_hold(
            quote.quote_id, constraints, passengers, context, confirmation, "hold-intent-1",
        ))
    booking = _require_data(held)
    if case == "hold_expired":
        backend.clock.advance(backend.config.hold_ttl_seconds)
    response = observe("process_payment", backend.process_payment(
        booking.booking_id, "PAYOP-demo-1", booking.total_price_vnd, context, confirmation, "pay-intent-1",
    ))
    result = "FAILED"
    reconciliation_checks = 0
    if response.code == "PAYMENT_TIMEOUT":
        for _ in range(2):
            reconciliation_checks += 1
            response = observe("get_payment_status", backend.get_payment_status("PAYOP-demo-1", context))
            if _require_data(response).operation.status.value in ("SUCCESS", "DECLINED"):
                break
        if response.data.operation.status.value in ("UNKNOWN", "PENDING"):
            result = "HANDOFF_REQUIRED"
    if response.ok and response.data.operation.status.value == "SUCCESS":
        _require_data(observe("issue_ticket", backend.issue_ticket(booking.booking_id, context, "ticket-intent-1")))
        result = "BOOKED"

    report = backend.check_invariants()
    if not report.passed:
        raise RuntimeError(report.model_dump_json())
    state = backend.get_backend_state()
    current_booking = _require_data(backend.get_booking(booking.booking_id))
    return {
        "case": case, "result": result, "mode": "SCRIPTED_MOCK_BACKEND_RECOVERY",
        "booking_id": booking.booking_id, "booking_status": current_booking.status.value,
        "total_price_vnd": current_booking.total_price_vnd,
        "committed_payments": sum(p["status"] == "SUCCESS" for p in state["payments"].values()),
        "ticket_count": len(state["tickets"]), "reconciliation_checks": reconciliation_checks,
        "payment_operation_id": "PAYOP-demo-1" if state["payment_operations"] else None,
        "recommended_action": "Reconcile the existing operation before any new charge" if result == "HANDOFF_REQUIRED" else None,
        "observations": observations, "failure_injection": state["failure_injection"],
        "inventory_events": state["inventory_events"], "invariants": report.model_dump(mode="json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone B deterministic failure/recovery demo (no LLM)")
    parser.add_argument("--case", choices=FAILURE_CASES, default="timeout_but_success")
    parser.add_argument("--passengers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.passengers < 1:
        parser.error("--passengers must be positive")
    try:
        result = run_failure_demo(args.case, args.passengers, args.seed)
    except RuntimeError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
