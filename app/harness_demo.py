"""Scripted host/user + LangChain guarded tools. No LLM or interactive adapter yet."""

import argparse
import json
from datetime import date, timedelta

from app.backend import MockFlightBackend
from app.config import DEFAULT_SEED
from app.failures import FailureConfig, FailureEffect, FailureRule, FailureTrigger
from app.harness.runtime import FlightBookingHarness
from app.models import Passenger, PaymentStatus, SearchConstraints, ToolResponse
from app.tools import TOOL_RESPONSE_TYPES

HARNESS_CASES = ("success", "price_change", "sold_out", "offer_expired", "declined", "timeout", "timeout_but_success", "hold_expired")


def run_harness_demo(case="success", passenger_count=3, seed=DEFAULT_SEED, output_dir=None):
    if case not in HARNESS_CASES:
        raise ValueError(f"Unknown harness case: {case}")
    payment_mode = case if case in ("declined", "timeout", "timeout_but_success") else "success"
    backend = MockFlightBackend(seed=seed, failure_config=FailureConfig(payment_mode=payment_mode))
    harness = FlightBookingHarness(backend, session_id="harness-demo")
    tools = {tool.name: tool for tool in harness.get_tools()}

    def call(name, **arguments):
        response = ToolResponse[TOOL_RESPONSE_TYPES[name]].model_validate(tools[name].invoke(arguments))
        return response

    def require(response):
        if not response.ok or response.data is None:
            raise RuntimeError(response.model_dump_json())
        return response.data

    constraints = SearchConstraints(origin="SGN", destination="HAN", passenger_count=passenger_count,
                                    departure_date=backend.clock.now().date() + timedelta(days=5))
    turn = harness.receive_user_turn(f"Đặt {passenger_count} vé SGN → HAN ngày {constraints.departure_date}")
    harness.set_requirements(turn.turn_id, constraints)
    offers = require(call("search_flights")).offers
    if not offers:
        raise RuntimeError("No offer has enough seats for this group")
    selected = offers[0]
    turn = harness.receive_user_turn(f"Chọn {selected.offer.offer_id}")
    harness.select_offer(turn.turn_id, selected.offer.offer_id)
    passengers = [Passenger(passenger_id=f"MOCK-P{i + 1}", first_name=f"Mock{i + 1}", last_name="Synthetic",
                            date_of_birth=date(1990, 1, 1), document_type="IDENTITY_CARD", document_number=f"MOCK-DOC-{i + 1}")
                  for i in range(passenger_count)]
    turn = harness.receive_user_turn("Cung cấp hồ sơ hành khách giả lập")
    require(harness.set_passengers(turn.turn_id, passengers))

    def quote_and_confirm():
        quote = require(call("reprice_offer", offer_id=selected.offer.offer_id))
        harness.present_quote()  # The trusted host would render this quote in the UI.
        turn = harness.receive_user_turn("Tôi xác nhận")
        harness.confirm_quote(turn.turn_id, quote.quote_id)
        return quote

    quote = quote_and_confirm()
    if case in ("price_change", "sold_out", "offer_expired"):
        effect = {"price_change": FailureEffect.PRICE_CHANGE, "sold_out": FailureEffect.SOLD_OUT,
                  "offer_expired": FailureEffect.OFFER_EXPIRED}[case]
        harness.configure_failure(rules=[FailureRule(rule_id=f"demo-{case}", trigger=FailureTrigger.HOLD,
                                                     effect=effect, offer_id=selected.offer.offer_id)])
    held = call("create_booking_hold", quote_id=quote.quote_id)
    if not held.ok and case in ("price_change", "sold_out", "offer_expired"):
        if case != "price_change":
            alternatives = require(call("search_flights")).offers
            if not alternatives:
                raise RuntimeError("No alternative remains")
            selected = alternatives[0]
            turn = harness.receive_user_turn(f"Chọn chuyến thay thế {selected.offer.offer_id}")
            harness.select_offer(turn.turn_id, selected.offer.offer_id)
        quote = quote_and_confirm()
        held = call("create_booking_hold", quote_id=quote.quote_id)
    booking = require(held)
    if case == "hold_expired":
        backend.clock.advance(backend.config.hold_ttl_seconds)
    response = call("process_payment", booking_id=booking.booking_id, amount_vnd=booking.total_price_vnd)
    if response.code in ("PAYMENT_TIMEOUT", "BACKEND_EXCEPTION"):
        for _ in range(harness.config.max_reconciliation_checks):
            state = harness.get_state()
            if state.terminal_status or not state.payment_requires_reconciliation:
                break
            response = call("get_payment_status", operation_id=state.payment_operation_id)
    state = harness.get_state()
    if state.payment_status == PaymentStatus.SUCCESS and state.terminal_status is None:
        require(call("issue_ticket", booking_id=booking.booking_id))
    elif state.terminal_status is None:
        harness.fail(response.code)
    state = harness.get_state()
    backend_state = harness.get_backend_state()
    result = {
        "mode": "SCRIPTED_USER_HARNESS_WITHOUT_LLM", "case": case,
        "result": state.terminal_status.value, "booking_id": state.booking_id,
        "payment_operation_id": state.payment_operation_id,
        "booking_status": backend_state["bookings"][state.booking_id]["status"],
        "committed_payments": sum(p["status"] == "SUCCESS" for p in backend_state["payments"].values()),
        "ticket_count": len(backend_state["tickets"]), "step_count": state.step_count,
        "reconciliation_checks": state.reconciliation_checks,
        "completion": harness.check_completion().model_dump(mode="json"),
        "handoff": harness.get_handoff().model_dump(mode="json") if harness.get_handoff() else None,
    }
    if output_dir is not None:
        result["artifacts_dir"] = harness.export_artifacts(output_dir)
    return result


def main():
    parser = argparse.ArgumentParser(description="Milestone C guarded harness demo, scripted user/no LLM")
    parser.add_argument("--case", choices=HARNESS_CASES, default="success")
    parser.add_argument("--passengers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", help="Export JSON evidence into a new directory (existing paths are rejected)")
    args = parser.parse_args()
    if args.passengers < 1:
        parser.error("--passengers must be positive")
    try:
        result = run_harness_demo(args.case, args.passengers, args.seed, args.output_dir)
    except (RuntimeError, ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
