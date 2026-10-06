"""Serializable scenario/user-policy/oracle contracts, independent of agents."""

from enum import Enum
from typing import Annotated, Any, Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from app.config import DEFAULT_START_TIME
from app.config import VIETNAM_TZ
from app.failures import FailureEffect, FailureRule
from app.models import DomainModel, NonemptyString, Passenger, PositiveInt, SearchConstraints, TerminalStatus
from app.state import CompletionResult, HarnessConfig, HarnessState, SessionPhase
from app.tools import TOOL_SCHEMAS


NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]


class PolicyKind(str, Enum):
    STANDARD = "standard"
    MISSING_DESTINATION = "missing_destination_checkpoint"
    CHANGE_DESTINATION = "change_destination_before_transaction"
    BYPASS_CONFIRMATION = "bypass_confirmation_checkpoint"
    INVALID_PASSENGER = "invalid_passenger_then_correct"


class UserActionKind(str, Enum):
    INITIAL = "initial_request"
    SELECT = "select_offer"
    PASSENGERS = "valid_passengers"
    INVALID_PASSENGERS = "invalid_passengers"
    CONFIRM = "confirm_current_quote"
    CHANGE = "change_requirements"
    BYPASS = "request_bypass"
    RESUME = "resume"
    APPLY = "apply_input_interpretation"


class ScenarioPredicate(str, Enum):
    ALIASES = "canonical_alias_route"
    PRICE = "price_reconfirmation"
    SOLD_OUT = "sold_out_recovery"
    EXPIRED = "expired_offer_recovery"
    DECLINED = "declined_without_tickets"
    RECONCILIATION = "timeout_reconciliation"
    CHANGE = "destination_change"
    BYPASS = "bypass_not_confirmation"
    REPLAY = "idempotent_hold_replay"
    PASSENGERS = "invalid_passenger_recovery"
    MISSING = "missing_destination_checkpoint"


class EventExpectation(DomainModel):
    event_type: NonemptyString
    tool_name: str | None = None
    code: str | None = None
    effect: FailureEffect | None = None
    min_count: NonnegativeInt = 1
    max_count: NonnegativeInt | None = None

    @model_validator(mode="after")
    def validate_filter(self) -> Self:
        if self.tool_name is not None and self.tool_name not in TOOL_SCHEMAS:
            raise ValueError("Event tool filter must name a guarded tool")
        if self.max_count is not None and self.max_count < self.min_count:
            raise ValueError("Event count range is reversed")
        if self.effect and self.event_type != "FAILURE_INJECTED":
            raise ValueError("Failure effect filter requires FAILURE_INJECTED")
        return self


class ExpectedDenial(DomainModel):
    tool_name: NonemptyString
    code: NonemptyString
    count: PositiveInt = 1


class ScenarioExpectation(DomainModel):
    terminal_status: TerminalStatus | None = TerminalStatus.BOOKED
    phase: SessionPhase | None = None
    checkpoint_action: UserActionKind | None = None
    booking_count: NonnegativeInt = 1
    successful_payment_count: NonnegativeInt = 1
    ticket_count: NonnegativeInt = 2
    required_events: list[EventExpectation] = Field(default_factory=list)
    forbidden_events: list[EventExpectation] = Field(default_factory=list)
    expected_denials: list[ExpectedDenial] = Field(default_factory=list)
    predicates: list[ScenarioPredicate] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if (self.terminal_status is None) == (self.phase is None):
            raise ValueError("Choose exactly one terminal or phase checkpoint expectation")
        if self.phase is not None and (self.phase == SessionPhase.TERMINAL or self.checkpoint_action is None):
            raise ValueError("A checkpoint needs a waiting/active phase and an action marker")
        if self.terminal_status is not None and self.checkpoint_action is not None:
            raise ValueError("Terminal expectation cannot specify a checkpoint marker")
        denial_keys = [(d.tool_name, d.code) for d in self.expected_denials]
        if len(set(denial_keys)) != len(denial_keys):
            raise ValueError("Expected denial filters must be unique")
        if len(set(self.predicates)) != len(self.predicates):
            raise ValueError("Scenario predicates must be unique")
        return self


class ScriptedUserConfig(DomainModel):
    kind: PolicyKind = PolicyKind.STANDARD
    offer_strategy: Literal["lowest_total_then_departure_then_id"] = "lowest_total_then_departure_then_id"
    # Explicit lab user consent: confirm EACH displayed valid current quote,
    # not a blanket confirmation field supplied to an agent or backend.
    confirmation_text: Literal["đồng ý", "xác nhận", "confirm"] = "đồng ý"
    change_message: NonemptyString | None = None
    bypass_message: NonemptyString | None = None
    invalid_passengers: list[dict[str, Any]] | None = None

    @model_validator(mode="after")
    def validate_branch(self) -> Self:
        for kind, field_name in ((PolicyKind.CHANGE_DESTINATION, "change_message"),
                                 (PolicyKind.BYPASS_CONFIRMATION, "bypass_message"),
                                 (PolicyKind.INVALID_PASSENGER, "invalid_passengers")):
            if (self.kind == kind) != (getattr(self, field_name) is not None):
                raise ValueError(f"{field_name} belongs only to {kind.value}")
        return self


class Scenario(DomainModel):
    id: Annotated[str, Field(pattern=r"^S\d{2}$")]
    name: NonemptyString
    description: NonemptyString
    initial_user_message: NonemptyString
    backend_seed: Annotated[int, Field(strict=True)] = 42
    start_time: AwareDatetime = DEFAULT_START_TIME
    harness_config: HarnessConfig = Field(default_factory=HarnessConfig)
    initial_constraints: SearchConstraints
    expected_constraints: SearchConstraints
    passengers: list[Passenger]
    user_policy: ScriptedUserConfig = Field(default_factory=ScriptedUserConfig)
    failure_rules: list[FailureRule] = Field(default_factory=list)
    duplicate_hold_delivery: bool = False
    max_user_turns: PositiveInt = 12
    expectation: ScenarioExpectation

    @model_validator(mode="after")
    def validate_setup(self) -> Self:
        if self.initial_constraints.passenger_count != self.expected_constraints.passenger_count:
            raise ValueError("The current user policy preserves group size")
        if len(self.passengers) != self.expected_constraints.passenger_count:
            raise ValueError("Scenario must supply N synthetic passenger profiles")
        if len({p.passenger_id for p in self.passengers}) != len(self.passengers) or len({
            (p.document_type, p.document_number) for p in self.passengers
        }) != len(self.passengers):
            raise ValueError("Scenario passenger IDs/documents must be unique")
        today = self.start_time.astimezone(VIETNAM_TZ).date()
        if any(p.date_of_birth > today for p in self.passengers):
            raise ValueError("Scenario passenger DOB is in the future")
        if self.initial_constraints.departure_date < today or self.expected_constraints.departure_date < today:
            raise ValueError("Scenario departure cannot precede its clock")
        if len({r.rule_id for r in self.failure_rules}) != len(self.failure_rules):
            raise ValueError("Failure rule IDs must be unique")
        if self.expectation.terminal_status == TerminalStatus.BOOKED and (
            self.expectation.booking_count != 1 or self.expectation.successful_payment_count != 1 or
            self.expectation.ticket_count != len(self.passengers)
        ):
            raise ValueError("BOOKED expectation requires one booking/charge and N tickets")
        return self


class UserAction(DomainModel):
    kind: UserActionKind
    text: NonemptyString


class ScenarioCheckpoint(DomainModel):
    action: UserAction
    user_turn_id: NonemptyString
    user_turn_number: PositiveInt
    trace_sequence: PositiveInt
    state: HarnessState
    backend_state: dict[str, Any]
    # Same public adapter contract used by future patterns; no NLP text grading.
    agent_state: dict[str, Any]


class ScenarioEvidence(DomainModel):
    initial_backend_state: dict[str, Any]
    final_backend_state: dict[str, Any]
    final_state: HarnessState
    final_agent_state: dict[str, Any]
    trace: list[dict[str, Any]]
    checkpoints: list[ScenarioCheckpoint]
    handoff: dict[str, Any] | None = None
    stop_reason: Literal["TERMINAL", "EXPECTED_CHECKPOINT", "NO_ELIGIBLE_OFFER", "USER_TURN_LIMIT", "AGENT_ERROR"]


class ScenarioOracleResult(DomainModel):
    scenario_id: str
    booking_completed: bool
    expected_terminal_reached: bool
    expected_checkpoint_reached: bool
    constraint_passed: bool
    permission_clean: bool
    scenario_passed: bool
    observed_phase: SessionPhase
    observed_terminal: TerminalStatus | None
    completion: CompletionResult
    checks: dict[str, bool]
    violations: list[str]
