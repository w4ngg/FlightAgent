"""Validate business sources and requirements independently of agent prompts."""

from datetime import datetime

from app.config import VIETNAM_TZ

from app.models import Flight, Passenger, passenger_payload_hash
from app.state import HarnessState


class DataConstraintPolicy:
    def validate_passengers(self, passengers: list[Passenger], count: int, today) -> str | None:
        if len(passengers) != count:
            return "Provide exactly one profile per passenger"
        if len({p.passenger_id for p in passengers}) != count:
            return "Passenger IDs must be unique"
        if len({(p.document_type, p.document_number) for p in passengers}) != count:
            return "Passenger documents must be unique"
        if any(p.date_of_birth > today for p in passengers):
            return "Passenger date of birth cannot be in the future"
        return None

    def validate_tool_input(self, name, args, state: HarnessState, backend_state: dict) -> str | None:
        if state.input_blocked and name != "get_payment_status":
            return "Resolve or explicitly apply the pending user input before business actions"
        if name == "resolve_airport":
            return None
        constraints = state.constraints
        if constraints is None:
            return "Validated user requirements are missing"
        if name == "search_flights":
            return None  # Search has no model-controlled business constraints.
        if name in ("get_offer", "reprice_offer"):
            if args.offer_id not in state.known_offer_ids:
                return "Offer ID must originate from search results for this session"
            offer = backend_state["offers"].get(args.offer_id)
            if offer is None:
                return "Offer does not exist in the backend"
            flight = Flight.model_validate(backend_state["flights"][offer["flight_id"]])
            if not constraints.permits(flight, offer["unit_price_vnd"] * constraints.passenger_count):
                return "Offer violates the user's route/date/budget/airline/time constraints"
            if name == "reprice_offer":
                if state.selected_offer_id != args.offer_id:
                    return "Reprice requires the current selected offer"
                today = datetime.fromisoformat(backend_state["current_time"]).astimezone(VIETNAM_TZ).date()
                problem = self.validate_passengers(state.passengers, constraints.passenger_count, today)
                if problem:
                    return problem
            return None
        if name == "create_booking_hold":
            quote = state.quote
            if quote is None or args.quote_id != quote.quote_id:
                return "Quote ID must match the current backend quote"
            if quote.context != state.context or quote.offer_id != state.selected_offer_id:
                return "Quote must belong to the current selection/session/intent"
            if backend_state["quotes"].get(quote.quote_id) != quote.model_dump(mode="json"):
                return "Quote payload must originate unchanged from the backend"
            if quote.passenger_payload_hash != passenger_payload_hash(state.passengers):
                return "Passenger edits invalidate the quote"
            return None
        if name == "get_payment_status":
            if args.operation_id != state.payment_operation_id:
                return "Query only the payment operation reference allocated for this session/intent"
            operation = backend_state["payment_operations"].get(args.operation_id)
            if operation and operation["context"] != state.context.model_dump(mode="json"):
                return "Payment operation belongs to another session/intent"
            return None
        booking = backend_state["bookings"].get(args.booking_id)
        if args.booking_id != state.booking_id or booking is None:
            return "Booking ID must originate from this session's hold response"
        if booking["context"] != state.context.model_dump(mode="json"):
            return "Booking belongs to another session/intent"
        if booking["constraints"] != constraints.model_dump(mode="json") or (
            booking["quote"]["passenger_payload_hash"] != passenger_payload_hash(state.passengers)
        ):
            return "Current user requirements/passengers do not match the held booking"
        if name == "process_payment" and args.amount_vnd != booking["total_price_vnd"]:
            return "Payment amount must equal the backend booking's locked group total"
        return None
