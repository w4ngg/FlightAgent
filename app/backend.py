"""In-memory transactional flight backend with deterministic failure injection."""

from dataclasses import dataclass
from datetime import timedelta
from threading import RLock
from typing import Any, Sequence, TypeVar

from app.clock import Clock, SimulatedClock, require_aware
from app.config import BackendConfig, DEFAULT_SEED, VIETNAM_TZ
from app.failures import FailureConfig, FailureEffect, FailureInjector, FailureRule, FailureTrigger
from app.mock_data import make_seed_data, normalize_airport_query
from app.models import (
    Airport, BackendInvariantReport, Booking, BookingStatus, CommitStatus, ConfirmationRecord,
    ConfirmationSnapshot, DomainModel, Inventory, InventoryEvent, OfferDetails, Passenger,
    Payment, PaymentOperation, PaymentResult, PaymentStatus, Quote,
    SearchConstraints, SearchResult, Ticket, TicketResult, ToolResponse,
    TransactionContext, passenger_payload_hash, payload_hash,
)

ModelType = TypeVar("ModelType", bound=DomainModel)


def _updated(model: ModelType, **changes: Any) -> ModelType:
    """Validate replacement values instead of using unchecked copy updates."""
    return type(model).model_validate({**model.model_dump(), **changes})


def _ok(
    data: ModelType, *, committed: bool = False, operation_ref: str | None = None
) -> ToolResponse[ModelType]:
    response = ToolResponse[type(data)](
        ok=True, code="OK", data=data, operation_ref=operation_ref,
        commit_status=CommitStatus.COMMITTED if committed else CommitStatus.NOT_APPLICABLE,
    )
    return response.model_copy(deep=True)


def _error(
    code: str, message: str, *, transaction: bool = False, operation_ref: str | None = None,
    retryable: bool = False, commit_status: CommitStatus | None = None, data: DomainModel | None = None,
) -> ToolResponse[Any]:
    return ToolResponse[Any](
        ok=False, code=code, error_message=message, operation_ref=operation_ref,
        retryable=retryable, data=data,
        commit_status=commit_status if commit_status is not None else (
            CommitStatus.NOT_COMMITTED if transaction else CommitStatus.NOT_APPLICABLE
        ),
    ).model_copy(deep=True)


@dataclass(frozen=True)
class _IdempotencyEntry:
    payload_hash: str
    response: ToolResponse[Any]


class MockFlightBackend:
    """Domain authority with isolated stores and atomic transitions per process.

    Callers supply trusted TransactionContext/ConfirmationRecord values. A real
    user-turn confirmation guard belongs to milestone C, not to this backend.
    Inspection methods are for the lab/operator, not additional agent tools.
    """

    def __init__(
        self, seed: int = DEFAULT_SEED, *, clock: Clock | None = None,
        config: BackendConfig | None = None, failure_config: FailureConfig | None = None,
    ) -> None:
        self.clock = clock if clock is not None else SimulatedClock()
        self.config = config if config is not None else BackendConfig()
        self._initial_time = require_aware(self.clock.now())
        self._lock = RLock()
        self._failures = FailureInjector((failure_config or FailureConfig()).to_rules())
        self.reset(seed)

    def reset(self, seed: int = DEFAULT_SEED) -> None:
        """Reset fixtures, IDs, stores and simulated time for a fresh episode."""
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        with self._lock:
            if isinstance(self.clock, SimulatedClock):
                self.clock.reset(self._initial_time)
            self.seed = seed
            self._airports, self._flights, self._inventory, self._offers = make_seed_data(
                seed, require_aware(self.clock.now())
            )
            self._quotes: dict[str, Quote] = {}
            self._quote_versions: dict[tuple[str, str, str], int] = {}
            self._latest_quote_ids: dict[tuple[str, str, str], str] = {}
            self._bookings: dict[str, Booking] = {}
            self._payments: dict[str, Payment] = {}
            self._payment_operations: dict[str, PaymentOperation] = {}
            self._tickets: dict[str, Ticket] = {}
            self._idempotency: dict[tuple[str, str, str], _IdempotencyEntry] = {}
            self._counters: dict[str, int] = {}
            self._inventory_events: list[InventoryEvent] = []
            self._failures.reset()

    def configure_failure(
        self, config: FailureConfig | None = None, *, rules: Sequence[FailureRule] | None = None,
    ) -> None:
        """Configure before an episode's transactions; reset re-arms the same rules."""
        if config is not None and rules is not None:
            raise ValueError("Supply either FailureConfig or explicit rules, not both")
        injector = FailureInjector(rules if rules is not None else (config or FailureConfig()).to_rules())
        with self._lock:
            if self._bookings or self._failures.get_state()["events"]:
                raise ValueError("Configure failures before transactions/injection; reset for a new episode")
            self._failures = injector

    def _inventory_event(
        self, event_type: str, flight_id: str, seats_delta: int, *,
        booking_id: str | None = None, rule_id: str | None = None,
    ) -> None:
        self._inventory_events.append(InventoryEvent(
            sequence_number=len(self._inventory_events) + 1, event_type=event_type,
            timestamp=self.clock.now(), flight_id=flight_id, seats_delta=seats_delta,
            booking_id=booking_id, rule_id=rule_id,
        ))

    def _apply_offer_failures(
        self, trigger: FailureTrigger, offer_id: str, context: TransactionContext,
    ) -> None:
        rules = self._failures.consume(trigger, timestamp=self.clock.now(), context=context, offer_id=offer_id)
        for rule in rules:
            offer = self._offers[offer_id]
            if rule.effect == FailureEffect.PRICE_CHANGE:
                price = rule.new_price if rule.new_price is not None else offer.unit_price_vnd + 250_000
                self._offers[offer_id] = _updated(offer, unit_price_vnd=price)
                for scope in list(self._latest_quote_ids):
                    if scope[2] == offer_id:
                        del self._latest_quote_ids[scope]
            elif rule.effect == FailureEffect.OFFER_EXPIRED:
                self._offers[offer_id] = _updated(offer, expires_at=self.clock.now())
            elif rule.effect == FailureEffect.SOLD_OUT:
                inventory = self._inventory[offer.flight_id]
                self._inventory[offer.flight_id] = _updated(inventory, available_seats=0)
                self._inventory_event("EXTERNAL_SOLD_OUT", offer.flight_id, -inventory.available_seats, rule_id=rule.rule_id)

    def _new_id(self, prefix: str) -> str:
        number = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = number
        return f"{prefix}-{number:04d}"

    @staticmethod
    def _quote_scope(context: TransactionContext, offer_id: str) -> tuple[str, str, str]:
        return context.session_id, context.transaction_intent_id, offer_id

    @staticmethod
    def _key(operation: str, context: TransactionContext, key: str) -> tuple[str, str, str]:
        return operation, context.session_id, key

    def _replay(
        self, operation: str, context: TransactionContext, key: str, request_payload: dict
    ) -> ToolResponse[Any] | None:
        if not isinstance(key, str) or not key.strip() or key != key.strip():
            return _error("INVALID_IDEMPOTENCY_KEY", "A nonempty, trimmed key is required", transaction=True)
        entry = self._idempotency.get(self._key(operation, context, key))
        if entry is None:
            return None
        if entry.payload_hash != payload_hash(request_payload):
            return _error("IDEMPOTENCY_CONFLICT", "This key already belongs to a different payload", transaction=True)
        return entry.response.model_copy(deep=True)

    def _remember(
        self, operation: str, context: TransactionContext, key: str,
        request_payload: dict, response: ToolResponse[Any],
    ) -> None:
        self._idempotency[self._key(operation, context, key)] = _IdempotencyEntry(
            payload_hash(request_payload), response.model_copy(deep=True)
        )

    def _passenger_error(self, passengers: list[Passenger], count: int) -> str | None:
        if len(passengers) != count:
            return "Passenger count must match the number of profiles"
        if len({p.passenger_id for p in passengers}) != count:
            return "Passenger IDs must be unique"
        if len({(p.document_type, p.document_number) for p in passengers}) != count:
            return "Passenger documents must be unique in a booking"
        today = self.clock.now().astimezone(VIETNAM_TZ).date()
        if any(p.date_of_birth > today for p in passengers):
            return "Date of birth cannot be in the future"
        return None

    def _confirmation_error(
        self, confirmation: ConfirmationRecord | None, quote: Quote,
        context: TransactionContext,
    ) -> str | None:
        if confirmation is None:
            return "Explicit confirmation is required"
        if confirmation.context != context or quote.context != context:
            return "Confirmation and quote must belong to the current session/intent"
        if confirmation.snapshot != ConfirmationSnapshot.from_quote(quote):
            return "Confirmation does not match the current quote and passengers"
        if not quote.created_at <= confirmation.confirmed_at <= self.clock.now():
            return "Confirmation time must be between quote creation and the current time"
        return None

    def _sweep_expired(self) -> list[str]:
        expired = []
        now = self.clock.now()
        for booking_id, booking in list(self._bookings.items()):
            if booking.status != BookingStatus.HELD or now < booking.hold_expires_at:
                continue
            if any(
                operation.booking_id == booking_id
                and operation.status in (PaymentStatus.PENDING, PaymentStatus.UNKNOWN)
                for operation in self._payment_operations.values()
            ):
                continue
            inventory = self._inventory[booking.quote.flight_id]
            restored = _updated(inventory, available_seats=inventory.available_seats + len(booking.passengers))
            self._inventory[inventory.flight_id] = restored
            self._bookings[booking_id] = _updated(booking, status=BookingStatus.EXPIRED)
            self._inventory_event("HOLD_RELEASED", inventory.flight_id, len(booking.passengers), booking_id=booking_id)
            expired.append(booking_id)
        return expired

    def sweep_expired_holds(self) -> list[str]:
        with self._lock:
            return self._sweep_expired()

    def resolve_airport(self, query: str) -> ToolResponse[Airport]:
        with self._lock:
            if not isinstance(query, str) or not query.strip():
                return _error("INVALID_AIRPORT", "An airport name or code is required")
            normalized = normalize_airport_query(query)
            for airport in self._airports.values():
                labels = [airport.code, airport.city, airport.name, *airport.aliases]
                if normalized in {normalize_airport_query(label) for label in labels}:
                    return _ok(airport)
            return _error("UNKNOWN_AIRPORT", f"Unsupported airport: {query}")

    def _offer_details(self, offer_id: str, passenger_count: int) -> OfferDetails:
        offer = self._offers[offer_id]
        return OfferDetails(
            offer=offer, flight=self._flights[offer.flight_id],
            available_seats=self._inventory[offer.flight_id].available_seats,
            passenger_count=passenger_count, total_price_vnd=offer.unit_price_vnd * passenger_count,
        )

    def search_flights(self, constraints: SearchConstraints, *, sort_by="price", limit=None) -> ToolResponse[SearchResult]:
        with self._lock:
            if sort_by not in {"price", "departure_time", "duration"} or (
                limit is not None and (type(limit) is not int or not 1 <= limit <= 20)
            ):
                return _error("INVALID_SEARCH_OPTIONS", "Use a supported sort and limit 1..20")
            self._sweep_expired()
            now = self.clock.now()
            if constraints.departure_date < now.astimezone(VIETNAM_TZ).date():
                return _error("INVALID_DATE", "Departure date cannot be in the past")
            found = []
            for offer in self._offers.values():
                details = self._offer_details(offer.offer_id, constraints.passenger_count)
                if offer.expires_at <= now or details.flight.departure_time <= now:
                    continue
                if details.available_seats < constraints.passenger_count:
                    continue
                if constraints.permits(details.flight, details.total_price_vnd):
                    found.append(details)
            def ordering(item):
                return {"price": item.total_price_vnd, "departure_time": item.flight.departure_time,
                        "duration": item.flight.arrival_time - item.flight.departure_time}[sort_by]
            found.sort(key=lambda item: (
                0 if item.flight.airline in constraints.preferred_airlines else 1,
                ordering(item), item.total_price_vnd, item.flight.departure_time, item.offer.offer_id,
            ))
            return _ok(SearchResult(offers=found if limit is None else found[:limit]))

    def get_offer(self, offer_id: str, passenger_count: int = 1) -> ToolResponse[OfferDetails]:
        with self._lock:
            self._sweep_expired()
            if type(passenger_count) is not int or passenger_count < 1:
                return _error("INVALID_PASSENGER_COUNT", "Passenger count must be a positive integer")
            if offer_id not in self._offers:
                return _error("OFFER_NOT_FOUND", f"Unknown offer: {offer_id}")
            details = self._offer_details(offer_id, passenger_count)
            if details.offer.expires_at <= self.clock.now():
                return _error("OFFER_EXPIRED", "The selected offer is no longer available")
            if details.available_seats < passenger_count:
                return _error("SOLD_OUT", "Not enough seats remain for this group")
            return _ok(details)

    def reprice_offer(
        self, offer_id: str, constraints: SearchConstraints, passengers: list[Passenger],
        context: TransactionContext,
    ) -> ToolResponse[Quote]:
        with self._lock:
            problem = self._passenger_error(passengers, constraints.passenger_count)
            if problem:
                return _error("INVALID_PASSENGER", problem)
            found = self.get_offer(offer_id, constraints.passenger_count)
            if not found.ok:
                return found
            details = found.data
            if not constraints.permits(details.flight, details.total_price_vnd):
                return _error("CONSTRAINT_VIOLATION", "Offer does not meet the current request")
            self._apply_offer_failures(FailureTrigger.REPRICE, offer_id, context)
            found = self.get_offer(offer_id, constraints.passenger_count)
            if not found.ok:
                return found
            details = found.data
            if not constraints.permits(details.flight, details.total_price_vnd):
                return _error("CONSTRAINT_VIOLATION", "Repriced offer violates the current request")
            now = self.clock.now()
            scope = self._quote_scope(context, offer_id)
            version = self._quote_versions.get(scope, 0) + 1
            quote = Quote(
                quote_id=self._new_id("QTE"), quote_version=version, context=context,
                offer_id=offer_id, flight_id=details.flight.flight_id,
                origin=details.flight.origin, destination=details.flight.destination,
                departure_date=details.flight.departure_time.astimezone(VIETNAM_TZ).date(),
                passenger_ids=[passenger.passenger_id for passenger in passengers],
                passenger_payload_hash=passenger_payload_hash(passengers),
                passenger_count=constraints.passenger_count, unit_price_vnd=details.offer.unit_price_vnd,
                total_price_vnd=details.total_price_vnd, created_at=now,
                expires_at=min(now + timedelta(seconds=self.config.quote_ttl_seconds), details.offer.expires_at),
            )
            self._quotes[quote.quote_id] = quote.model_copy(deep=True)
            self._quote_versions[scope] = version
            self._latest_quote_ids[scope] = quote.quote_id
            return _ok(quote)

    def create_booking_hold(
        self, quote_id: str, constraints: SearchConstraints, passengers: list[Passenger],
        context: TransactionContext, confirmation: ConfirmationRecord | None,
        idempotency_key: str,
    ) -> ToolResponse[Booking]:
        with self._lock:
            self._sweep_expired()
            request = {
                "context": context.model_dump(mode="json"), "quote_id": quote_id,
                "constraints": constraints.model_dump(mode="json"),
                "passengers": [p.model_dump(mode="json") for p in passengers],
                "confirmation": confirmation.model_dump(mode="json") if confirmation else None,
            }
            replay = self._replay("hold", context, idempotency_key, request)
            if replay is not None:
                return replay
            quote = self._quotes.get(quote_id)
            if quote is None:
                return _error("QUOTE_NOT_FOUND", "Quote must originate from the backend", transaction=True)
            problem = self._confirmation_error(confirmation, quote, context)
            if problem:
                return _error("CONFIRMATION_REQUIRED", problem, transaction=True)
            existing_booking = next((b for b in self._bookings.values() if b.context == context), None)
            if existing_booking is not None:
                return _error("HOLD_ALREADY_CREATED", "This intent already has a booking; replay its original key",
                              transaction=True, operation_ref=existing_booking.booking_id)
            scope = self._quote_scope(context, quote.offer_id)
            if self._latest_quote_ids.get(scope) != quote_id:
                return _error("QUOTE_STALE", "A newer quote has invalidated this confirmation", transaction=True)
            now = self.clock.now()
            if quote.expires_at <= now:
                return _error("QUOTE_EXPIRED", "Reprice and confirm again before holding", transaction=True)
            problem = self._passenger_error(passengers, constraints.passenger_count)
            if problem:
                return _error("INVALID_PASSENGER", problem, transaction=True)
            if quote.passenger_count != constraints.passenger_count or (
                quote.passenger_payload_hash != passenger_payload_hash(passengers)
            ):
                return _error("PASSENGER_CHANGED", "Passenger edits invalidate the quote and confirmation", transaction=True)
            found = self.get_offer(quote.offer_id, constraints.passenger_count)
            if not found.ok:
                return _error(found.code, found.error_message, transaction=True)
            details = found.data
            if details.offer.unit_price_vnd != quote.unit_price_vnd:
                return _error("QUOTE_STALE", "Current offer price differs from the confirmed quote", transaction=True)
            if not constraints.permits(details.flight, details.total_price_vnd):
                return _error("CONSTRAINT_VIOLATION", "Current offer violates the request", transaction=True)
            self._apply_offer_failures(FailureTrigger.HOLD, quote.offer_id, context)
            found = self.get_offer(quote.offer_id, constraints.passenger_count)
            if not found.ok:
                return _error(found.code, found.error_message, transaction=True)
            details = found.data
            if self._latest_quote_ids.get(scope) != quote_id:
                return _error("QUOTE_STALE", "Price change invalidated the confirmed quote", transaction=True)
            if details.offer.unit_price_vnd != quote.unit_price_vnd:
                return _error("QUOTE_STALE", "Current offer price differs from the confirmed quote", transaction=True)
            if not constraints.permits(details.flight, details.total_price_vnd):
                return _error("CONSTRAINT_VIOLATION", "Current offer violates the request", transaction=True)
            booking = Booking(
                booking_id=self._new_id("BKG"), context=context, quote=quote,
                confirmation=confirmation, constraints=constraints, passengers=passengers,
                status=BookingStatus.HELD, total_price_vnd=quote.total_price_vnd,
                hold_expires_at=now + timedelta(seconds=self.config.hold_ttl_seconds),
                idempotency_key=idempotency_key, created_at=now,
            ).model_copy(deep=True)
            inventory = self._inventory[quote.flight_id]
            reduced = _updated(inventory, available_seats=inventory.available_seats - quote.passenger_count)
            self._bookings[booking.booking_id] = booking
            self._inventory[quote.flight_id] = reduced
            self._inventory_event("HOLD_RESERVED", quote.flight_id, -quote.passenger_count, booking_id=booking.booking_id)
            response = _ok(booking, committed=True, operation_ref=booking.booking_id)
            self._remember("hold", context, idempotency_key, request, response)
            return response

    def process_payment(
        self, booking_id: str, operation_id: str, amount_vnd: int,
        context: TransactionContext, confirmation: ConfirmationRecord | None,
        idempotency_key: str,
    ) -> ToolResponse[PaymentResult]:
        """Register one logical operation, then simulate the configured outcome."""
        with self._lock:
            self._sweep_expired()
            if type(amount_vnd) is not int or amount_vnd < 0:
                return _error("INVALID_AMOUNT", "Amount must be an integer VND value", transaction=True)
            if not isinstance(operation_id, str) or not operation_id.strip() or operation_id != operation_id.strip():
                return _error("INVALID_OPERATION_ID", "A stable, nonempty operation ID is required", transaction=True)
            request = {
                "context": context.model_dump(mode="json"), "booking_id": booking_id,
                "operation_id": operation_id, "amount_vnd": amount_vnd, "currency": "VND",
                "confirmation": confirmation.model_dump(mode="json") if confirmation else None,
            }
            replay = self._replay("payment", context, idempotency_key, request)
            if replay is not None:
                return replay
            booking = self._bookings.get(booking_id)
            if booking is None:
                return _error("BOOKING_NOT_FOUND", "Payment requires an existing booking", transaction=True)
            if booking.context != context:
                return _error("CONTEXT_MISMATCH", "Booking belongs to a different session/intent", transaction=True)
            problem = self._confirmation_error(confirmation, booking.quote, context)
            if problem:
                return _error("CONFIRMATION_REQUIRED", problem, transaction=True)
            existing = next((op for op in self._payment_operations.values() if op.booking_id == booking_id), None)
            if existing is not None:
                return _error("PAYMENT_ALREADY_STARTED", "Reconcile the existing payment instead of charging again",
                              transaction=True, operation_ref=existing.operation_id)
            if operation_id in self._payment_operations:
                return _error("OPERATION_CONFLICT", "Operation ID already belongs to another payment", transaction=True)
            if booking.status != BookingStatus.HELD:
                return _error("INVALID_BOOKING_STATE", f"Cannot pay a {booking.status.value} booking", transaction=True)
            if amount_vnd != booking.total_price_vnd:
                return _error("AMOUNT_MISMATCH", "Payment must equal the booking's locked total", transaction=True)
            now = self.clock.now()
            operation = PaymentOperation(
                operation_id=operation_id, booking_id=booking_id, context=context,
                amount_vnd=amount_vnd, status=PaymentStatus.PENDING,
                idempotency_key=idempotency_key, payload_hash=payload_hash(request), created_at=now,
            )
            self._payment_operations[operation_id] = operation
            rules = self._failures.consume(
                FailureTrigger.PAYMENT, timestamp=now, context=context,
                offer_id=booking.quote.offer_id, operation_id=operation_id,
            )
            effect = rules[0].effect if rules else None
            if effect == FailureEffect.TIMEOUT_UNRESOLVED:
                self._payment_operations[operation_id] = _updated(operation, status=PaymentStatus.UNKNOWN)
                response = _error("PAYMENT_TIMEOUT", "Payment result is unknown; reconcile this operation",
                                  transaction=True, retryable=True, commit_status=CommitStatus.UNKNOWN,
                                  operation_ref=operation_id)
                self._remember("payment", context, idempotency_key, request, response)
                return response
            if effect == FailureEffect.PAYMENT_DECLINED:
                payment = Payment(
                    payment_id=self._new_id("PAY"), operation_id=operation_id, booking_id=booking_id,
                    amount_vnd=amount_vnd, status=PaymentStatus.DECLINED,
                    idempotency_key=idempotency_key, created_at=now,
                )
                declined = _updated(operation, status=PaymentStatus.DECLINED, payment_id=payment.payment_id)
                self._payment_operations[operation_id] = declined
                self._payments[payment.payment_id] = payment
                response = _error("PAYMENT_DECLINED", "Payment was declined; no money was charged",
                                  transaction=True, operation_ref=operation_id,
                                  data=PaymentResult(operation=declined, payment=payment, booking=booking))
                self._remember("payment", context, idempotency_key, request, response)
                return response
            payment = Payment(
                payment_id=self._new_id("PAY"), operation_id=operation_id, booking_id=booking_id,
                amount_vnd=amount_vnd, status=PaymentStatus.SUCCESS,
                idempotency_key=idempotency_key, created_at=now,
            )
            committed_operation = _updated(operation, status=PaymentStatus.SUCCESS, payment_id=payment.payment_id)
            paid = _updated(booking, status=BookingStatus.PAID)
            self._payment_operations[operation_id] = committed_operation
            self._payments[payment.payment_id] = payment
            self._bookings[booking_id] = paid
            response = _ok(PaymentResult(operation=committed_operation, payment=payment, booking=paid),
                           committed=True, operation_ref=operation_id)
            self._remember("payment", context, idempotency_key, request, response)
            if effect == FailureEffect.TIMEOUT_BUT_COMMITTED:
                return _error("PAYMENT_TIMEOUT", "Provider response timed out; reconcile before proceeding",
                              transaction=True, retryable=True, commit_status=CommitStatus.UNKNOWN,
                              operation_ref=operation_id)
            return response

    def get_payment_status(
        self, operation_id: str, context: TransactionContext,
    ) -> ToolResponse[PaymentResult]:
        with self._lock:
            self._sweep_expired()
            operation = self._payment_operations.get(operation_id)
            if operation is None:
                return _error("OPERATION_NOT_FOUND", "No payment operation with this reference exists")
            if operation.context != context:
                return _error("CONTEXT_MISMATCH", "Payment belongs to a different session/intent")
            return _ok(PaymentResult(
                operation=operation,
                payment=self._payments.get(operation.payment_id) if operation.payment_id else None,
                booking=self._bookings[operation.booking_id],
            ), operation_ref=operation_id)

    def issue_ticket(
        self, booking_id: str, context: TransactionContext, idempotency_key: str,
    ) -> ToolResponse[TicketResult]:
        with self._lock:
            self._sweep_expired()
            request = {"booking_id": booking_id, "context": context.model_dump(mode="json")}
            replay = self._replay("ticket", context, idempotency_key, request)
            if replay is not None:
                return replay
            booking = self._bookings.get(booking_id)
            if booking is None:
                return _error("BOOKING_NOT_FOUND", "Ticket issuance requires a booking", transaction=True)
            if booking.context != context:
                return _error("CONTEXT_MISMATCH", "Booking belongs to a different session/intent", transaction=True)
            payment = next((p for p in self._payments.values() if p.booking_id == booking_id), None)
            if payment is None or payment.status != PaymentStatus.SUCCESS:
                return _error("PAYMENT_REQUIRED", "Tickets require a verified successful payment", transaction=True)
            if payment.amount_vnd != booking.total_price_vnd or payment.currency != booking.currency:
                return _error("PAYMENT_MISMATCH", "Payment does not cover this booking", transaction=True)
            if booking.status not in (BookingStatus.PAID, BookingStatus.TICKETED):
                return _error("INVALID_BOOKING_STATE", "Booking is not ready for ticketing", transaction=True)
            if booking.status == BookingStatus.TICKETED:
                tickets = sorted(
                    (t for t in self._tickets.values() if t.booking_id == booking_id),
                    key=lambda ticket: ticket.passenger_index,
                )
            else:
                tickets = [Ticket(
                    ticket_number=self._new_id("TKT"), booking_id=booking_id,
                    passenger_id=passenger.passenger_id, passenger_index=index,
                    flight_id=booking.quote.flight_id,
                ) for index, passenger in enumerate(booking.passengers)]
            if len(tickets) != len(booking.passengers):
                return _error("TICKET_INCONSISTENCY", "Ticket count does not match passengers", transaction=True)
            ticketed = _updated(booking, status=BookingStatus.TICKETED)
            self._tickets.update({ticket.ticket_number: ticket for ticket in tickets})
            self._bookings[booking_id] = ticketed
            response = _ok(TicketResult(booking=ticketed, tickets=tickets), committed=True, operation_ref=booking_id)
            self._remember("ticket", context, idempotency_key, request, response)
            return response

    def get_booking(self, booking_id: str) -> ToolResponse[Booking]:
        """Operator/completion inspection; not an extra agent tool."""
        with self._lock:
            self._sweep_expired()
            booking = self._bookings.get(booking_id)
            return _ok(booking) if booking else _error("BOOKING_NOT_FOUND", "Unknown booking")

    def get_inventory(self, flight_id: str) -> ToolResponse[Inventory]:
        with self._lock:
            self._sweep_expired()
            inventory = self._inventory.get(flight_id)
            return _ok(inventory) if inventory else _error("FLIGHT_NOT_FOUND", "Unknown flight")

    def get_backend_state(self) -> dict[str, Any]:
        """Detached JSON-compatible snapshot for tests, CLI and later evaluation."""
        with self._lock:
            self._sweep_expired()
            stores = {
                "airports": self._airports, "flights": self._flights, "offers": self._offers,
                "inventory": self._inventory, "quotes": self._quotes, "bookings": self._bookings,
                "payments": self._payments, "payment_operations": self._payment_operations,
                "tickets": self._tickets,
            }
            return {
                "seed": self.seed, "current_time": self.clock.now().isoformat(),
                **{name: {key: model.model_dump(mode="json") for key, model in store.items()}
                   for name, store in stores.items()},
                "idempotency_store": [
                    {"operation": key[0], "session_id": key[1], "key": key[2],
                     "payload_hash": entry.payload_hash, "result": entry.response.model_dump(mode="json")}
                    for key, entry in self._idempotency.items()
                ],
                "failure_injection": self._failures.get_state(),
                "inventory_events": [event.model_dump(mode="json") for event in self._inventory_events],
            }

    def check_invariants(self) -> BackendInvariantReport:
        """Backend integrity only; agent completion/scenario evaluation comes later."""
        from app.invariants import check_backend_invariants

        with self._lock:
            return check_backend_invariants(self.get_backend_state())
