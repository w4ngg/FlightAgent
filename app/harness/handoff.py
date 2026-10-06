"""Bounded stopping and operator packets without pretending to cancel/refund."""

from app.state import HandoffPacket, HarnessConfig, HarnessState
from app.models import PaymentStatus


class HandoffManager:
    def __init__(self, config: HarnessConfig):
        self.config = config

    def reason(self, state: HarnessState) -> str | None:
        if state.step_count > self.config.max_steps:
            return "STEP_BUDGET_EXCEEDED"
        if state.payment_requires_reconciliation and state.reconciliation_checks >= self.config.max_reconciliation_checks:
            return "PAYMENT_UNRESOLVED"
        if state.invalid_passenger_attempts >= self.config.max_invalid_passenger_attempts:
            return "INVALID_PASSENGER_LIMIT"
        if state.permission_violations >= self.config.max_permission_violations:
            return "PERMISSION_VIOLATION_LIMIT"
        if state.general_retry_count >= self.config.max_general_retries:
            return "RETRY_LIMIT"
        if state.replan_count > self.config.max_replans:
            return "REPLAN_LIMIT"
        return None

    def create_packet(self, reason: str, state: HarnessState, backend_state: dict, trace: list[dict]) -> HandoffPacket:
        booking = backend_state["bookings"].get(state.booking_id)
        if booking is None:
            booking = next((b for b in backend_state["bookings"].values()
                            if b["context"] == state.context.model_dump(mode="json")), None)
        operation = backend_state["payment_operations"].get(state.payment_operation_id)
        if operation and operation["status"] in ("UNKNOWN", "PENDING") or (
            state.payment_status in (PaymentStatus.UNKNOWN, PaymentStatus.PENDING)
        ):
            action = "Reconcile the existing payment operation before any new charge; preserve the reservation"
        elif operation and operation["status"] == "SUCCESS":
            action = "Payment has committed; inspect existing tickets and finish ticketing without charging again"
        elif booking:
            action = "Inspect the existing booking and requested change; do not automatically cancel, refund or create a new transaction"
        else:
            action = "Review user requirements and trace before continuing the session"
        return HandoffPacket(
            reason=reason, context=state.context, phase_at_handoff=state.phase, boundary=state.boundary,
            selected_offer_id=state.selected_offer_id, quote_id=state.quote.quote_id if state.quote else None,
            confirmed_by_user_turn_id=state.confirmation.user_turn_id if state.confirmation else None,
            booking_id=booking["booking_id"] if booking else state.booking_id,
            payment_operation_id=state.payment_operation_id, payment_id=operation["payment_id"] if operation else state.payment_id,
            last_observed_payment_status=state.payment_status,
            backend_booking_status=booking["status"] if booking else None,
            backend_operation_status=operation["status"] if operation else None,
            operation_keys=state.operation_keys, requested_change=state.requested_change,
            recommended_action=action,
            trace_summary=[f'{event["sequence_number"]}:{event["event_type"]}:{event["tool_name"] or "-"}' for event in trace[-12:]],
        )
