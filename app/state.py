"""Harness-owned snapshots. Agents only receive detached copies of these values."""

from enum import Enum
from typing import Any

from pydantic import AwareDatetime, Field

from app.models import (
    BookingStatus, ConfirmationRecord, DomainModel, NonemptyString, Passenger,
    PaymentStatus, PositiveInt, Quote, SearchConstraints, TerminalStatus, TransactionContext,
)


class SessionPhase(str, Enum):
    WAITING_INPUT = "WAITING_INPUT"
    WAITING_SELECTION = "WAITING_SELECTION"
    WAITING_PASSENGERS = "WAITING_PASSENGERS"
    WAITING_CONFIRMATION = "WAITING_CONFIRMATION"
    RECONCILING_PAYMENT = "RECONCILING_PAYMENT"
    EXECUTING = "EXECUTING"
    TERMINAL = "TERMINAL"


class TransactionBoundary(str, Enum):
    PRE_TRANSACTION = "PRE_TRANSACTION"
    POST_TRANSACTION = "POST_TRANSACTION"


class HarnessConfig(DomainModel):
    max_steps: PositiveInt = 25
    max_general_retries: PositiveInt = 10
    max_invalid_passenger_attempts: PositiveInt = 3
    max_reconciliation_checks: PositiveInt = 2
    max_replans: PositiveInt = 3
    max_permission_violations: PositiveInt = 3


class UserTurn(DomainModel):
    turn_id: NonemptyString
    turn_number: PositiveInt
    text: NonemptyString
    timestamp: AwareDatetime


class HarnessState(DomainModel):
    context: TransactionContext
    phase: SessionPhase = SessionPhase.WAITING_INPUT
    terminal_status: TerminalStatus | None = None
    constraints: SearchConstraints | None = None
    input_blocked: bool = False  # Pending/invalid request must not reuse old consent.
    passengers: list[Passenger] = Field(default_factory=list)
    requirements_user_turn_id: str | None = None
    passengers_user_turn_id: str | None = None
    known_offer_ids: list[str] = Field(default_factory=list)
    selected_offer_id: str | None = None
    quote: Quote | None = None
    confirmation: ConfirmationRecord | None = None
    displayed_quote_id: str | None = None
    displayed_after_turn: int | None = None
    booking_id: str | None = None
    booking_status: BookingStatus | None = None
    payment_operation_id: str | None = None
    payment_id: str | None = None
    payment_status: PaymentStatus | None = None  # Last OBSERVED status, not hidden backend state.
    payment_requires_reconciliation: bool = False
    transaction_uncertain: bool = False
    ticket_numbers: list[str] = Field(default_factory=list)
    operation_keys: dict[str, str] = Field(default_factory=dict)
    step_count: int = 0
    general_retry_count: int = 0
    invalid_passenger_attempts: int = 0
    reconciliation_checks: int = 0
    replan_count: int = 0
    permission_violations: int = 0
    requested_change: dict[str, Any] | None = None

    @property
    def boundary(self) -> TransactionBoundary:
        return TransactionBoundary.POST_TRANSACTION if (
            self.booking_id or self.payment_operation_id or self.transaction_uncertain
        ) else TransactionBoundary.PRE_TRANSACTION


class CompletionResult(DomainModel):
    booking_completed: bool
    constraint_passed: bool
    permission_clean: bool
    booking_id: str | None
    checks: dict[str, bool]
    violations: list[str]


class HandoffPacket(DomainModel):
    reason: NonemptyString
    context: TransactionContext
    phase_at_handoff: SessionPhase
    boundary: TransactionBoundary
    terminal_status: TerminalStatus = TerminalStatus.HANDOFF_REQUIRED
    selected_offer_id: str | None
    quote_id: str | None
    confirmed_by_user_turn_id: str | None
    booking_id: str | None
    payment_operation_id: str | None
    payment_id: str | None
    last_observed_payment_status: PaymentStatus | None
    backend_booking_status: str | None
    backend_operation_status: str | None
    operation_keys: dict[str, str]
    requested_change: dict[str, Any] | None
    recommended_action: NonemptyString
    trace_summary: list[str]
