"""Authoritative booking completion, separate from scenario/permission success."""

from app.invariants import check_backend_invariants
from app.models import passenger_payload_hash
from app.state import CompletionResult, HarnessState


class CompletionChecker:
    def check(self, state: HarnessState, backend_state: dict, *, confirmation_authentic: bool,
              trace: list[dict]) -> CompletionResult:
        invariants = check_backend_invariants(backend_state)
        booking = backend_state["bookings"].get(state.booking_id)
        checks = {"backend_invariants": invariants.passed, "booking_exists": booking is not None}
        checks["no_pending_requirement_change"] = state.requested_change is None
        checks["context_matches"] = bool(booking and booking["context"] == state.context.model_dump(mode="json"))
        checks["requirements_match"] = bool(booking and state.constraints and
                                            booking["constraints"] == state.constraints.model_dump(mode="json"))
        checks["selection_matches"] = bool(booking and booking["quote"]["offer_id"] == state.selected_offer_id)
        checks["passengers_match"] = bool(booking and state.passengers and
                                          booking["quote"]["passenger_payload_hash"] == passenger_payload_hash(state.passengers))
        checks["confirmation_authentic"] = bool(confirmation_authentic and booking and state.confirmation and
                                                booking["confirmation"] == state.confirmation.model_dump(mode="json"))
        checks["booking_ticketed"] = bool(booking and booking["status"] == "TICKETED")
        payments = [p for p in backend_state["payments"].values()
                    if p["booking_id"] == state.booking_id and p["status"] == "SUCCESS"]
        checks["payment_success"] = bool(booking and len(payments) == 1 and
                                          payments[0]["amount_vnd"] == booking["total_price_vnd"] and payments[0]["currency"] == "VND")
        tickets = [t for t in backend_state["tickets"].values() if t["booking_id"] == state.booking_id]
        checks["n_tickets"] = bool(booking and len(tickets) == len(booking["passengers"]) and
                                    len({t["passenger_id"] for t in tickets}) == len(booking["passengers"]))
        confirmations = [event for event in trace if event["event_type"] == "CONFIRMATION_RECEIVED"
                         and event["context"] == state.context.model_dump(mode="json")
                         and state.confirmation and event["payload"]["user_turn_id"] == state.confirmation.user_turn_id
                         and state.quote and event["payload"]["quote_id"] == state.quote.quote_id]
        current_transaction_ids = {event["attempt_id"] for event in trace if event["event_type"] == "TOOL_CALL"
                                   and event["context"] == state.context.model_dump(mode="json")
                                   and isinstance(event["payload"]["tool_input"], dict)
                                   and (event["tool_name"] == "create_booking_hold" and state.quote and
                                        event["payload"]["tool_input"].get("quote_id") == state.quote.quote_id
                                        or event["tool_name"] == "process_payment" and state.booking_id and
                                        event["payload"]["tool_input"].get("booking_id") == state.booking_id)}
        transaction_attempts = [event for event in trace if event["event_type"] == "TOOL_ALLOWED"
                                and event["attempt_id"] in current_transaction_ids]
        checks["confirmation_before_transactions"] = bool(confirmations and transaction_attempts and
            min(e["sequence_number"] for e in confirmations) < min(e["sequence_number"] for e in transaction_attempts))
        payment_observations = [event for event in trace if event["event_type"] == "TOOL_RESULT"
                                and event["tool_name"] in ("process_payment", "get_payment_status")
                                and event["context"] == state.context.model_dump(mode="json")
                                and event["payload"]["response"].get("data")
                                and event["payload"]["response"]["data"].get("operation", {}).get("status") == "SUCCESS"
                                and event["payload"]["response"]["data"]["operation"]["booking_id"] == state.booking_id]
        ticket_attempts = [event for event in trace if event["event_type"] == "TOOL_ALLOWED"
                           and event["tool_name"] == "issue_ticket"
                           and event["context"] == state.context.model_dump(mode="json")]
        checks["payment_observed_before_ticket"] = bool(payment_observations and ticket_attempts and
                                                         min(e["sequence_number"] for e in payment_observations) <
                                                         min(e["sequence_number"] for e in ticket_attempts))
        business_match = all(checks[name] for name in ("context_matches", "requirements_match", "passengers_match"))
        return CompletionResult(
            booking_completed=all(checks.values()),
            constraint_passed=invariants.passed and (booking is None or business_match),
            permission_clean=state.permission_violations == 0, booking_id=state.booking_id,
            checks=checks, violations=[f"{name} failed" for name, passed in checks.items() if not passed],
        )
