"""Permission checks distinguish original-result replay from a new commit."""

from datetime import datetime

from app.models import PaymentStatus, TerminalStatus
from app.state import HarnessState, TransactionBoundary
from app.tools import SIDE_EFFECT_TOOLS


class PermissionGuard:
    def authorize(self, name: str, state: HarnessState, backend_state: dict, *,
                  replay: bool, confirmation_authentic: bool) -> str | None:
        if state.terminal_status is not None:
            if state.terminal_status == TerminalStatus.HANDOFF_REQUIRED or not replay:
                return "Session is terminal; no new agent actions are permitted"
        if name == "reprice_offer" and state.boundary == TransactionBoundary.POST_TRANSACTION:
            return "A held transaction uses its locked quote; do not replace it by repricing"
        if name == "get_payment_status":
            return None
        if name not in SIDE_EFFECT_TOOLS:
            return None
        if name == "process_payment" and state.payment_requires_reconciliation:
            return "Reconcile the original payment operation; blind payment retry is forbidden"
        if not confirmation_authentic:
            return "A genuine user turn must confirm the current displayed quote"
        if replay:
            return None  # Payload/source were checked before this guard; no new commit.
        now = datetime.fromisoformat(backend_state["current_time"])
        if name == "create_booking_hold":
            if state.boundary == TransactionBoundary.POST_TRANSACTION:
                return "This intent already has a transaction"
            if state.quote is None or state.quote.expires_at <= now:
                return "Quote expired; reprice, display and confirm the new quote"
        elif name == "process_payment":
            booking = backend_state["bookings"][state.booking_id]
            if booking["status"] != "HELD" or datetime.fromisoformat(booking["hold_expires_at"]) <= now:
                return "Payment requires a live HELD booking"
            if state.payment_operation_id is not None:
                return "A payment operation already exists; reconcile it rather than create another"
        elif name == "issue_ticket":
            if state.payment_status != PaymentStatus.SUCCESS:
                return "Ticketing requires observed payment SUCCESS, not hidden backend state"
            if backend_state["bookings"][state.booking_id]["status"] not in ("PAID", "TICKETED"):
                return "Backend booking is not ready for ticketing"
        return None
