"""Pattern-independent harness, trusted user ingress and guarded tool execution.

This is not send()/NLP or an agent adapter. CLI/scenario hosts own
the user ingress methods; model-facing tools expose only business proposals.
"""

import json
import unicodedata
from datetime import date, time
from pathlib import Path
from threading import RLock
from typing import Any

from pydantic import BaseModel, ValidationError

from app.backend import MockFlightBackend
from app.config import VIETNAM_TZ
from app.harness.completion import CompletionChecker
from app.harness.constraints import DataConstraintPolicy
from app.harness.handoff import HandoffManager
from app.harness.permissions import PermissionGuard
from app.harness.trace import TraceCollector
from app.models import (
    BookingStatus, CommitStatus, ConfirmationRecord, ConfirmationSnapshot, Passenger,
    PaymentStatus, SearchConstraints, TerminalStatus, ToolResponse, TransactionContext, payload_hash,
)
from app.state import HarnessConfig, HarnessState, SessionPhase, TransactionBoundary, UserTurn
from app.tools import SIDE_EFFECT_TOOLS, TOOL_RESPONSE_TYPES, TOOL_SCHEMAS, create_tools


def _json_value(value):
    def encode(item):
        if isinstance(item, BaseModel):
            return item.model_dump(mode="json")
        if isinstance(item, (date, time)):
            return item.isoformat()
        raise TypeError(f"Unsupported input type: {type(item).__name__}")
    return json.loads(json.dumps(value, default=encode, ensure_ascii=False))


def _error(code, message, *, tool_name=None, operation_ref=None, unknown=False):
    return ToolResponse[Any](
        ok=False, code=code, error_message=message, operation_ref=operation_ref,
        retryable=unknown, commit_status=CommitStatus.UNKNOWN if unknown else (
            CommitStatus.NOT_COMMITTED if tool_name in SIDE_EFFECT_TOOLS else CommitStatus.NOT_APPLICABLE),
    )


def accepts_confirmation(text: str) -> bool:
    normalized = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    normalized = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    normalized = " ".join(normalized.strip(" .!\n\t").split())
    # Deliberately conservative. Ambiguous/compound messages need clarification
    # or an explicit host-side confirm action, not an LLM-created boolean.
    return normalized in {"dong y", "xac nhan", "toi dong y", "toi xac nhan", "yes", "confirm"}


class FlightBookingHarness:
    def __init__(self, backend: MockFlightBackend | None = None, *, session_id="session-1",
                 config: HarnessConfig | None = None):
        self._backend = backend if backend is not None else MockFlightBackend()
        self.config = config if config is not None else HarnessConfig()
        self._lock = RLock()
        self.constraint_policy = DataConstraintPolicy()
        self.permission_guard = PermissionGuard()
        self.completion_checker = CompletionChecker()
        self.handoff_manager = HandoffManager(self.config)
        self._session_id = session_id
        self._initialize()

    def _initialize(self):
        self._intent_number = 1
        self._turn_number = 0
        self._turns: dict[str, UserTurn] = {}
        self._confirmation_receipts: dict[str, ConfirmationRecord] = {}
        self._invalid_passenger_turns: set[str] = set()
        self._committed_requests: dict[str, str] = {}
        self._handoff = None
        self._state = HarnessState(context=TransactionContext(
            session_id=self._session_id, transaction_intent_id="intent-1"))
        self._trace = TraceCollector(self._backend.clock)
        self._initial_backend_state = self._backend.get_backend_state()
        self._fault_events_seen = len(self._initial_backend_state["failure_injection"]["events"])

    def _patch(self, **changes):
        self._state = HarnessState.model_validate({**self._state.model_dump(), **changes})

    def get_state(self) -> HarnessState:
        with self._lock:
            return self._state.model_copy(deep=True)

    def get_backend_state(self) -> dict:
        with self._lock:
            return self._backend.get_backend_state()

    def get_trace(self) -> list[dict]:
        with self._lock:
            return self._trace.get_trace()

    def get_handoff(self):
        with self._lock:
            return self._handoff.model_copy(deep=True) if self._handoff else None

    def get_tools(self):
        return create_tools(self)

    def current_time(self):
        """Expose the clock, not backend stores, to conversation hosts."""
        return self._backend.clock.now()

    def advance_time(self, seconds: int):
        """Explicit lab operator command; never an agent tool or automatic wait."""
        with self._lock:
            from app.clock import SimulatedClock
            if not isinstance(self._backend.clock, SimulatedClock):
                raise ValueError("Only a simulated clock can be advanced")
            self._backend.clock.advance(seconds)
            self._trace.record("TIME_ADVANCED", self._state.context, seconds=seconds,
                               current_time=self._backend.clock.now().isoformat())
            return self._backend.clock.now()

    def record_host_event(self, event, **payload):
        """Conversation audit only. This is not part of the model tool registry."""
        if event not in {"MODEL_OUTPUT", "MODEL_ERROR", "INPUT_REJECTED", "AGENT_RESPONSE", "SESSION_PAUSED", "TRANSPORT_REPLAY",
                         "PLAN_REQUEST", "PLAN_CREATED", "PLAN_INVALIDATED", "PLAN_STEP", "PLAN_STEP_RESULT", "PLAN_PAUSED", "PLAN_REJECTED",
                         "HYBRID_NODE", "HYBRID_PAUSED", "HYBRID_ROUTER_REQUEST", "HYBRID_ROUTER_REJECTED", "HYBRID_DECISION",
                         "HYBRID_INTERPRETATION_APPLIED", "HYBRID_TOOL_RESULT", "HYBRID_RECOVERY",
                         "INPUT_INTERPRETATION", "INPUT_INTERPRETATION_REJECTED", "INPUT_INTERPRETATION_APPLIED",
                         "INPUT_SUSPENDED", "INPUT_EVIDENCE_WARNING", "REQUEST_PROPOSAL_CREATED",
                         "REQUEST_PROPOSAL_UPDATED", "REQUEST_PROPOSAL_DISPLAYED", "REQUEST_PROPOSAL_PRESERVED",
                         "REQUEST_APPLY_REJECTED", "REQUEST_PROPOSAL_KEPT", "REQUEST_PROPOSAL_APPLIED"}:
            raise ValueError("Unsupported conversation event")
        with self._lock:
            self._trace.record(event, self._state.context, **_json_value(payload))

    def record_model_failure(self, exception_type: str, *, diagnostic=None):
        """A failed provider decision consumes the shared retry budget, not a charge retry."""
        with self._lock:
            self.record_host_event("MODEL_ERROR", exception_type=exception_type,
                                   **({"diagnostic": diagnostic} if diagnostic is not None else {}))
            self._patch(general_retry_count=self._state.general_retry_count + 1)
            self._check_budget()

    def reset(self, seed=42):
        """Operator-only fresh episode, never an automatic recovery action."""
        with self._lock:
            self._backend.reset(seed)
            self._initialize()

    def configure_failure(self, config=None, *, rules=None):
        with self._lock:
            self._backend.configure_failure(config, rules=rules)
            self._initial_backend_state["failure_injection"] = self._backend.get_backend_state()["failure_injection"]

    def receive_user_turn(self, text: str) -> UserTurn:
        """Trusted UI/driver ingress. Never expose this method as an LLM tool."""
        with self._lock:
            turn = UserTurn(turn_id=f"{self._session_id}:user-{self._turn_number + 1}",
                            turn_number=self._turn_number + 1, text=text, timestamp=self._backend.clock.now())
            self._turn_number += 1
            self._turns[turn.turn_id] = turn
            self._trace.record("USER_MESSAGE", self._state.context, turn=turn.model_dump(mode="json"))
            return turn.model_copy(deep=True)

    def _user_turn(self, turn_id):
        turn = self._turns.get(turn_id)
        if turn is None or turn.turn_number != self._turn_number:
            raise ValueError("Use the latest genuine user turn from this session")
        return turn

    def _invalidate_quote(self, *, clear_selection=False):
        self._patch(quote=None, confirmation=None, displayed_quote_id=None, displayed_after_turn=None,
                    **({"selected_offer_id": None} if clear_selection else {}))

    def _new_intent(self):
        self._intent_number += 1
        self._patch(context=TransactionContext(session_id=self._session_id,
                    transaction_intent_id=f"intent-{self._intent_number}"), operation_keys={})
        self._committed_requests.clear()
        self._invalidate_quote(clear_selection=True)

    def set_requirements(self, turn_id, constraints: SearchConstraints):
        with self._lock:
            self._user_turn(turn_id)
            if self._state.boundary == TransactionBoundary.POST_TRANSACTION:
                supplied = _json_value(constraints)
                try:
                    supplied = SearchConstraints.model_validate(supplied).model_dump(mode="json")
                except (ValidationError, ValueError, TypeError):
                    pass  # Even an invalid requested change must stop the old transaction.
                if self._state.constraints and supplied == self._state.constraints.model_dump(mode="json"):
                    return
                self._patch(requested_change={"constraints": supplied, "user_turn_id": turn_id})
                self.request_handoff("REQUIREMENTS_CHANGED_POST_TRANSACTION")
                return
            if self._state.terminal_status is not None:
                raise ValueError("Reset explicitly to start a new episode")
            try:
                validated = SearchConstraints.model_validate(_json_value(constraints))
                if validated.departure_date < self._backend.clock.now().astimezone(VIETNAM_TZ).date():
                    raise ValueError("Departure date cannot be in the past")
            except (ValidationError, ValueError, TypeError):
                # Keep the last accepted request/selection for inspection, but
                # never execute it while a proposed replacement is invalid.
                self.block_input(turn_id)
                raise
            if validated == self._state.constraints:
                if self._state.input_blocked:
                    phase = (SessionPhase.WAITING_CONFIRMATION if self._state.quote else
                             SessionPhase.WAITING_PASSENGERS if self._state.selected_offer_id else
                             SessionPhase.WAITING_SELECTION)
                    self._patch(input_blocked=False, phase=phase)
                return
            if self._state.constraints is not None:
                self._new_intent()
            self._patch(constraints=validated, requirements_user_turn_id=turn_id, input_blocked=False,
                        known_offer_ids=[], phase=SessionPhase.WAITING_SELECTION)
            self._trace.record("REQUIREMENTS_UPDATED", self._state.context, user_turn_id=turn_id,
                               constraints=validated.model_dump(mode="json"))

    def block_input(self, turn_id):
        """Suspend old work without destroying inspectable accepted data."""
        with self._lock:
            self._user_turn(turn_id)
            self._patch(input_blocked=True, confirmation=None, displayed_quote_id=None,
                        displayed_after_turn=None, phase=SessionPhase.WAITING_INPUT)
            self.record_host_event("INPUT_SUSPENDED", user_turn_id=turn_id)

    def resume_input(self, turn_id):
        """Host-only: a validated non-request interpretation has been applied."""
        with self._lock:
            self._user_turn(turn_id)
            phase = (SessionPhase.WAITING_CONFIRMATION if self._state.quote else
                     SessionPhase.WAITING_PASSENGERS if self._state.selected_offer_id else
                     SessionPhase.WAITING_SELECTION if self._state.constraints else SessionPhase.WAITING_INPUT)
            self._patch(input_blocked=False, phase=phase)

    def select_offer(self, turn_id, offer_id: str):
        with self._lock:
            self._user_turn(turn_id)
            if offer_id == self._state.selected_offer_id:
                return
            if self._state.boundary == TransactionBoundary.POST_TRANSACTION:
                self._patch(requested_change={"offer_id": offer_id, "user_turn_id": turn_id})
                self.request_handoff("SELECTION_CHANGED_POST_TRANSACTION")
                return
            if self._state.terminal_status:
                raise ValueError("Session is terminal")
            previous_selection = self._state.selected_offer_id
            self._invalidate_quote(clear_selection=True)
            self._patch(phase=SessionPhase.WAITING_SELECTION)
            args = TOOL_SCHEMAS["get_offer"].model_validate({"offer_id": offer_id})
            offer_id = args.offer_id
            problem = self.constraint_policy.validate_tool_input("get_offer", args, self._state, self._backend.get_backend_state())
            if problem:
                raise ValueError(problem)
            details = self._backend.get_offer(offer_id, self._state.constraints.passenger_count)
            if not details.ok:
                raise ValueError(details.error_message)
            if previous_selection:
                self._new_intent()
            self._invalidate_quote()
            self._patch(selected_offer_id=offer_id, phase=SessionPhase.WAITING_PASSENGERS)
            self._trace.record("OFFER_SELECTED", self._state.context, user_turn_id=turn_id, offer_id=offer_id)

    def set_passengers(self, turn_id, passengers):
        with self._lock:
            self._user_turn(turn_id)
            if self._state.constraints is None:
                raise ValueError("Set requirements first")
            if self._state.terminal_status and self._state.boundary == TransactionBoundary.PRE_TRANSACTION:
                raise ValueError("Session is terminal")
            if self._state.boundary == TransactionBoundary.POST_TRANSACTION:
                supplied = _json_value(passengers)
                try:
                    supplied = [Passenger.model_validate(p).model_dump(mode="json") for p in supplied]
                except (ValidationError, ValueError, TypeError):
                    pass
                if supplied == [p.model_dump(mode="json") for p in self._state.passengers]:
                    return ToolResponse[HarnessState](ok=True, code="OK", data=self.get_state())
                self._patch(requested_change={"passengers": supplied, "user_turn_id": turn_id})
                self.request_handoff("PASSENGERS_CHANGED_POST_TRANSACTION")
                return _error("HANDOFF_REQUIRED", "Passenger changes after hold need operator review")
            try:
                if not isinstance(passengers, list):
                    raise ValueError("Passenger profiles must be a list")
                validated = [Passenger.model_validate(_json_value(p)) for p in passengers]
                problem = self.constraint_policy.validate_passengers(
                    validated, self._state.constraints.passenger_count,
                    self._backend.clock.now().astimezone(VIETNAM_TZ).date())
                if problem:
                    raise ValueError(problem)
            except (ValidationError, ValueError, TypeError):
                self._invalidate_quote()
                if turn_id not in self._invalid_passenger_turns:
                    self._invalid_passenger_turns.add(turn_id)
                    self._patch(invalid_passenger_attempts=self._state.invalid_passenger_attempts + 1)
                self._trace.record("PASSENGER_VALIDATION_FAILED", self._state.context, user_turn_id=turn_id)
                self._check_budget()
                return _error("INVALID_PASSENGER", "Provide N valid, unique passenger profiles")
            if validated == self._state.passengers:
                return ToolResponse[HarnessState](ok=True, code="OK", data=self.get_state())
            self._invalidate_quote()
            self._patch(passengers=validated, passengers_user_turn_id=turn_id, invalid_passenger_attempts=0,
                        phase=SessionPhase.WAITING_CONFIRMATION if self._state.selected_offer_id else SessionPhase.WAITING_SELECTION)
            self._trace.record("PASSENGERS_VALIDATED", self._state.context, user_turn_id=turn_id, count=len(validated))
            return ToolResponse[HarnessState](ok=True, code="OK", data=self.get_state())

    def present_quote(self):
        """Host renders this exact quote to the user before asking for approval."""
        with self._lock:
            quote = self._state.quote
            if quote is None or self._state.terminal_status:
                raise ValueError("No current quote to present")
            if self._state.boundary == TransactionBoundary.POST_TRANSACTION:
                raise ValueError("Do not request a replacement confirmation for a held transaction")
            self._patch(displayed_quote_id=quote.quote_id, displayed_after_turn=self._turn_number,
                        confirmation=None, phase=SessionPhase.WAITING_CONFIRMATION)
            self._trace.record("QUOTE_PRESENTED", self._state.context, quote=quote.model_dump(mode="json"))
            return quote.model_copy(deep=True)

    def confirm_quote(self, turn_id, quote_id):
        with self._lock:
            turn = self._user_turn(turn_id)
            quote = self._state.quote
            if self._state.input_blocked or self._state.terminal_status or self._state.boundary == TransactionBoundary.POST_TRANSACTION:
                raise ValueError("No new confirmation is permitted for this transaction")
            if quote is None or quote_id != quote.quote_id or self._state.displayed_quote_id != quote_id:
                raise ValueError("Confirm the current quote that has actually been presented")
            if self._state.displayed_after_turn is None or turn.turn_number <= self._state.displayed_after_turn:
                raise ValueError("Confirmation must come from a new user turn after quote presentation")
            if not accepts_confirmation(turn.text):
                self._patch(confirmation=None, phase=SessionPhase.WAITING_CONFIRMATION)
                raise ValueError("Ambiguous, negative or bypass instructions are not quote confirmation")
            if quote.expires_at <= self._backend.clock.now():
                raise ValueError("Quote expired; reprice and confirm the new quote")
            if turn_id in self._confirmation_receipts:
                existing = self._confirmation_receipts[turn_id]
                if existing.snapshot != ConfirmationSnapshot.from_quote(quote):
                    raise ValueError("A user turn cannot confirm a different quote")
                return existing.model_copy(deep=True)
            record = ConfirmationRecord(context=self._state.context, snapshot=ConfirmationSnapshot.from_quote(quote),
                                        user_turn_id=turn_id, confirmed_at=self._backend.clock.now())
            self._confirmation_receipts[turn_id] = record
            self._patch(confirmation=record, phase=SessionPhase.EXECUTING)
            self._trace.record("CONFIRMATION_RECEIVED", self._state.context, quote_id=quote_id, user_turn_id=turn_id,
                               snapshot=record.snapshot.model_dump(mode="json"))
            return record.model_copy(deep=True)

    def _confirmation_authentic(self):
        record = self._state.confirmation
        return bool(record and record.context == self._state.context and self._state.quote and
                    record.snapshot == ConfirmationSnapshot.from_quote(self._state.quote) and
                    self._confirmation_receipts.get(record.user_turn_id) == record and
                    record.user_turn_id in self._turns and accepts_confirmation(self._turns[record.user_turn_id].text))

    def _key(self, name):
        prefix = {"create_booking_hold": "HOLD", "process_payment": "PAY", "issue_ticket": "TICKET"}[name]
        return f"{prefix}-{payload_hash(self._state.context.model_dump(mode='json'))[:24]}"

    def _operation_id(self):
        return f"PAYOP-{payload_hash(self._state.context.model_dump(mode='json'))[:24]}"

    def _bound_args(self, name, args):
        state = self._state
        if name == "resolve_airport":
            return {"query": args.query}
        if name == "search_flights":
            return {"constraints": state.constraints, "sort_by": args.sort_by, "limit": args.limit}
        if name == "get_offer":
            return {"offer_id": args.offer_id, "passenger_count": state.constraints.passenger_count}
        common = {"context": state.context}
        if name == "reprice_offer":
            return {**common, "offer_id": args.offer_id, "constraints": state.constraints, "passengers": state.passengers}
        if name == "get_payment_status":
            return {**common, "operation_id": args.operation_id}
        common["idempotency_key"] = self._key(name)
        if name == "create_booking_hold":
            return {**common, "quote_id": args.quote_id, "constraints": state.constraints,
                    "passengers": state.passengers, "confirmation": state.confirmation}
        if name == "process_payment":
            return {**common, "booking_id": args.booking_id, "operation_id": self._operation_id(),
                    "amount_vnd": args.amount_vnd, "confirmation": state.confirmation}
        return {**common, "booking_id": args.booking_id}

    def execute_guarded_tool(self, name: str, tool_input: Any) -> ToolResponse:
        with self._lock:
            self._patch(step_count=self._state.step_count + 1)
            attempt_id = f"attempt-{self._state.step_count}"
            try:
                printable = _json_value(tool_input)
            except (TypeError, ValueError):
                printable = {"invalid_input_type": type(tool_input).__name__}
            self._trace.record("TOOL_CALL", self._state.context, tool_name=name, attempt_id=attempt_id,
                               tool_input=printable, phase_before=self._state.phase.value)
            if self._state.step_count > self.config.max_steps:
                self.request_handoff("STEP_BUDGET_EXCEEDED")
                return self._deny(name, attempt_id, "STEP_BUDGET_EXCEEDED", "Episode step budget exhausted", violation=False)
            if name not in TOOL_SCHEMAS:
                return self._deny(name, attempt_id, "UNKNOWN_TOOL", "Tool is not in the eight-tool registry")
            try:
                if not isinstance(tool_input, dict):
                    raise ValueError("Tool input must be an object")
                args = TOOL_SCHEMAS[name].model_validate(_json_value(tool_input))
            except (ValidationError, TypeError, ValueError):
                return self._deny(name, attempt_id, "SCHEMA_ERROR", "Tool arguments do not match the strict schema")
            try:
                snapshot = self._backend.get_backend_state()
                problem = self.constraint_policy.validate_tool_input(name, args, self._state, snapshot)
                if problem:
                    return self._deny(name, attempt_id, "DATA_CONSTRAINT_VIOLATION", problem)
                bound = self._bound_args(name, args)
                request_hash = payload_hash(_json_value(bound))
                replay = self._committed_requests.get(name) == request_hash
                problem = self.permission_guard.authorize(name, self._state, snapshot, replay=replay,
                                                          confirmation_authentic=self._confirmation_authentic())
            except Exception as exc:
                self._trace.record("TOOL_EXCEPTION", self._state.context, tool_name=name, attempt_id=attempt_id,
                                   exception_type=type(exc).__name__, stage="precondition")
                response = self._deny(name, attempt_id, "PRECONDITION_ERROR", "Could not safely validate backend data", violation=False)
                self.request_handoff("BACKEND_INSPECTION_FAILED")
                return response
            if problem:
                return self._deny(name, attempt_id, "PERMISSION_DENIED", problem)
            self._trace.record("TOOL_ALLOWED", self._state.context, tool_name=name, attempt_id=attempt_id,
                               allowed=True, replay=replay)
            if name in SIDE_EFFECT_TOOLS:
                self._patch(operation_keys={**self._state.operation_keys, name: bound["idempotency_key"]})
            if name == "process_payment" and self._state.payment_operation_id is None:
                self._patch(payment_operation_id=bound["operation_id"])
            if name == "get_payment_status" and self._state.payment_requires_reconciliation:
                self._patch(reconciliation_checks=self._state.reconciliation_checks + 1)
            try:
                response = getattr(self._backend, name)(**bound)
                if not isinstance(response, ToolResponse):
                    raise TypeError("Backend returned a malformed tool response")
                response = ToolResponse[TOOL_RESPONSE_TYPES[name]].model_validate(response.model_dump(mode="json"))
                if response.ok and response.data is None:
                    raise ValueError("Successful tool response must contain typed data")
                if response.code == "PAYMENT_DECLINED" and (
                    name != "process_payment" or response.data is None or response.data.operation.status != PaymentStatus.DECLINED
                ):
                    raise ValueError("Declined response must carry a verified declined operation")
            except Exception as exc:
                self._trace.record("TOOL_EXCEPTION", self._state.context, tool_name=name, attempt_id=attempt_id,
                                   exception_type=type(exc).__name__)
                response = _error("BACKEND_EXCEPTION", "Backend execution failed; do not assume a transaction failed",
                                  tool_name=name, operation_ref=self._state.payment_operation_id if name == "process_payment" else
                                  self._state.booking_id or bound.get("idempotency_key"), unknown=name in SIDE_EFFECT_TOOLS)
            if name in SIDE_EFFECT_TOOLS and (response.commit_status in (CommitStatus.COMMITTED, CommitStatus.UNKNOWN)
                                             or response.code == "PAYMENT_DECLINED"):
                self._committed_requests[name] = request_hash
            self._record_faults(name, attempt_id)
            self._trace.record("TOOL_RESULT", self._state.context, tool_name=name, attempt_id=attempt_id,
                               allowed=True, response=response.model_dump(mode="json"))
            self._reduce(name, response)
            if response.code == "BACKEND_EXCEPTION" and name in SIDE_EFFECT_TOOLS and name != "process_payment":
                self._patch(transaction_uncertain=True)
                self.request_handoff("TRANSACTION_RESULT_UNKNOWN")
            if self._state.terminal_status is None:
                self._check_budget()
                try:
                    if self._state.terminal_status is None and not self._backend.check_invariants().passed:
                        self.request_handoff("BACKEND_INCONSISTENT")
                except Exception:
                    self.request_handoff("BACKEND_INSPECTION_FAILED")
            if name == "issue_ticket" and response.ok and self._state.terminal_status is None:
                self.finish()
            return response.model_copy(deep=True)

    def _deny(self, name, attempt_id, code, message, *, violation=True):
        if violation:
            self._patch(permission_violations=self._state.permission_violations + 1)
        response = _error(code, message, tool_name=name, operation_ref=self._state.payment_operation_id)
        self._trace.record("PERMISSION_DENIED" if violation else "TOOL_DENIED", self._state.context,
                           tool_name=name, attempt_id=attempt_id, allowed=False, code=code, reason=message)
        self._trace.record("TOOL_RESULT", self._state.context, tool_name=name, attempt_id=attempt_id,
                           allowed=False, response=response.model_dump(mode="json"))
        self._check_budget()
        return response

    def _record_faults(self, name, attempt_id):
        try:
            events = self._backend.get_backend_state()["failure_injection"]["events"]
        except Exception:
            self.request_handoff("BACKEND_INSPECTION_FAILED")
            return
        for event in events[self._fault_events_seen:]:
            if event["context"] == self._state.context.model_dump(mode="json"):
                self._trace.record("FAILURE_INJECTED", self._state.context, tool_name=name, attempt_id=attempt_id, failure=event)
        self._fault_events_seen = len(events)

    def _booking_status(self, incoming):
        current = self._state.booking_status
        rank = {BookingStatus.HELD: 0, BookingStatus.PAID: 1, BookingStatus.TICKETED: 2, BookingStatus.EXPIRED: 3}
        return incoming if current is None or rank.get(incoming, 0) >= rank.get(current, 0) else current

    def _reduce(self, name, response):
        state = self._state
        data = response.data
        active_phase = SessionPhase.TERMINAL if state.terminal_status else SessionPhase.EXECUTING
        if response.ok:
            if name == "search_flights":
                self._patch(known_offer_ids=sorted(set(state.known_offer_ids) | {d.offer.offer_id for d in data.offers}),
                            phase=state.phase if state.selected_offer_id else SessionPhase.WAITING_SELECTION)
            elif name == "reprice_offer":
                self._invalidate_quote()
                self._patch(quote=data, phase=SessionPhase.WAITING_CONFIRMATION)
            elif name == "create_booking_hold":
                self._patch(booking_id=data.booking_id, booking_status=self._booking_status(data.status), phase=active_phase)
            elif name in ("process_payment", "get_payment_status"):
                self._patch(payment_status=data.operation.status, payment_id=data.operation.payment_id,
                            booking_status=self._booking_status(data.booking.status),
                            payment_requires_reconciliation=data.operation.status in (PaymentStatus.PENDING, PaymentStatus.UNKNOWN),
                            phase=SessionPhase.RECONCILING_PAYMENT if data.operation.status in (PaymentStatus.PENDING, PaymentStatus.UNKNOWN)
                            else active_phase)
                if data.operation.status == PaymentStatus.DECLINED:
                    self._terminal(TerminalStatus.FAILED, "PAYMENT_DECLINED")
            elif name == "issue_ticket":
                self._patch(booking_status=self._booking_status(data.booking.status),
                            ticket_numbers=[ticket.ticket_number for ticket in data.tickets], phase=active_phase)
            return
        if response.code in ("PAYMENT_TIMEOUT", "BACKEND_EXCEPTION") and name == "process_payment":
            self._patch(payment_status=PaymentStatus.UNKNOWN, payment_requires_reconciliation=True,
                        phase=SessionPhase.RECONCILING_PAYMENT)
        elif response.code == "PAYMENT_DECLINED":
            self._patch(payment_status=PaymentStatus.DECLINED, payment_id=data.operation.payment_id)
            self._terminal(TerminalStatus.FAILED, "PAYMENT_DECLINED")
        elif name == "issue_ticket" and state.payment_status == PaymentStatus.SUCCESS:
            self.request_handoff("TICKETING_FAILED_AFTER_PAYMENT")
        elif name != "get_payment_status":
            self._patch(general_retry_count=state.general_retry_count + 1)
            if state.boundary == TransactionBoundary.PRE_TRANSACTION and response.code in (
                "SOLD_OUT", "OFFER_EXPIRED", "QUOTE_EXPIRED", "QUOTE_STALE", "CONSTRAINT_VIOLATION",
            ):
                self._invalidate_quote(clear_selection=response.code in ("SOLD_OUT", "OFFER_EXPIRED"))
                self._patch(phase=SessionPhase.WAITING_SELECTION if self._state.selected_offer_id is None else SessionPhase.WAITING_CONFIRMATION)

    def _terminal(self, status, reason):
        if self._state.terminal_status == status:
            return
        self._patch(terminal_status=status, phase=SessionPhase.TERMINAL)
        self._trace.record("TERMINAL", self._state.context, status=status.value, reason=reason)

    def _check_budget(self):
        if self._state.terminal_status is None:
            reason = self.handoff_manager.reason(self._state)
            if reason:
                self.request_handoff(reason)

    def request_handoff(self, reason="AGENT_CANNOT_CONTINUE"):
        with self._lock:
            if self._handoff is None:
                try:
                    snapshot = self._backend.get_backend_state()
                except Exception:
                    snapshot = {"bookings": {}, "payment_operations": {}}
                self._handoff = self.handoff_manager.create_packet(reason, self._state,
                    snapshot, self.get_trace())
                self._trace.record("HANDOFF", self._state.context, packet=self._handoff.model_dump(mode="json"))
                self._terminal(TerminalStatus.HANDOFF_REQUIRED, reason)
            return self.get_handoff()

    def cancel(self, turn_id):
        with self._lock:
            self._user_turn(turn_id)
            if self._state.boundary == TransactionBoundary.POST_TRANSACTION:
                self._patch(requested_change={"cancel_requested": True, "user_turn_id": turn_id})
                return self.request_handoff("CANCEL_REQUESTED_POST_TRANSACTION")
            if self._state.terminal_status is not None:
                raise ValueError("Session is terminal")
            self._terminal(TerminalStatus.CANCELLED, "USER_STOPPED_BEFORE_TRANSACTION")

    def fail(self, reason="REQUEST_CANNOT_BE_COMPLETED"):
        """End a known non-payment failure; never hide an unresolved/paid obligation."""
        with self._lock:
            if self._state.terminal_status:
                return self.get_state()
            snapshot = self._backend.get_backend_state()
            operation = snapshot["payment_operations"].get(self._state.payment_operation_id)
            if self._state.transaction_uncertain or self._state.payment_requires_reconciliation or (
                operation and operation["status"] in ("PENDING", "UNKNOWN", "SUCCESS")
            ):
                self.request_handoff(reason)
            else:
                self._terminal(TerminalStatus.FAILED, reason)
            return self.get_state()

    def record_model_decision(self, *, node="agent", replan=False):
        with self._lock:
            if self._state.terminal_status:
                return False
            self._patch(step_count=self._state.step_count + 1,
                        replan_count=self._state.replan_count + int(replan))
            self._trace.record("REPLAN" if replan else "LLM_CALL", self._state.context, node=node)
            self._check_budget()
            return self._state.terminal_status is None

    def check_completion(self):
        with self._lock:
            return self.completion_checker.check(self._state, self._backend.get_backend_state(),
                                                confirmation_authentic=self._confirmation_authentic(), trace=self.get_trace())

    def finish(self):
        with self._lock:
            report = self.check_completion()
            self._trace.record("COMPLETION_CHECK", self._state.context, report=report.model_dump(mode="json"))
            if report.booking_completed and self._state.terminal_status is None:
                self._terminal(TerminalStatus.BOOKED, "BACKEND_VERIFIED")
            elif self._state.booking_status == BookingStatus.TICKETED and not report.booking_completed:
                self.request_handoff("COMPLETION_CHECK_FAILED")
            return report

    def export_artifacts(self, output_dir):
        """Write inspectable snapshots into a NEW directory; never restore from them."""
        with self._lock:
            directory = Path(output_dir)
            directory.mkdir(parents=True, exist_ok=False)
            backend_state = self._backend.get_backend_state()
            state = self.get_state()
            trace = self.get_trace()
            completion = self.completion_checker.check(state, backend_state,
                confirmation_authentic=self._confirmation_authentic(), trace=trace)
            artifacts = {
                "initial_state.json": self._initial_backend_state,
                "backend_state.json": backend_state,
                "agent_state.json": state.model_dump(mode="json"),
                "trace.json": trace,
                "completion.json": completion.model_dump(mode="json"),
                "handoff.json": self._handoff.model_dump(mode="json") if self._handoff else None,
            }
            for filename, value in artifacts.items():
                (directory / filename).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return str(directory.resolve())
