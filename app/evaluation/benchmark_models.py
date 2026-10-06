"""Versioned benchmark contracts; missing measurements are not zero."""

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import Field, model_validator

from app.evaluation.contracts import NonnegativeInt
from app.agents.input_review import InputReviewConfig
from app.models import DomainModel, NonemptyString, PositiveInt, TerminalStatus
from app.state import SessionPhase


Pattern = Literal["react", "plan_execute", "hybrid"]
Milliseconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
USD = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]


class BenchmarkModelConfig(DomainModel):
    provider: Literal["fake", "gemini"] = "fake"
    model_name: NonemptyString | None = None
    temperature: Annotated[float, Field(ge=0, le=2, allow_inf_nan=False)] | None = None

    @model_validator(mode="after")
    def valid_mode(self):
        if self.provider == "gemini" and self.model_name is None:
            raise ValueError("Choose an explicit model for a Gemini benchmark")
        if self.provider == "fake" and (self.model_name is not None or self.temperature is not None):
            raise ValueError("Fake mode has pattern-specific test doubles, not a shared LLM/temperature")
        return self


class TokenPricing(DomainModel):
    """User-supplied rates, not a fetched price or a billing statement."""
    model_name: NonemptyString
    service_tier: Literal["standard", "priority", "flex"] = "standard"
    input_usd_per_million: USD
    output_usd_per_million: USD
    cached_input_usd_per_million: USD | None = None


class BenchmarkConfig(DomainModel):
    patterns: list[Pattern] = Field(default_factory=lambda: ["react", "plan_execute", "hybrid"], min_length=1)
    scenario_ids: list[Annotated[str, Field(pattern=r"^S\d{2}$")]] = Field(
        default_factory=lambda: [f"S{i:02}" for i in range(1, 13)], min_length=1)
    repetitions: Annotated[int, Field(strict=True, ge=1, le=1000)] = 3
    model: BenchmarkModelConfig = Field(default_factory=BenchmarkModelConfig)
    input_review: InputReviewConfig = Field(default_factory=InputReviewConfig)
    backend_seed: Annotated[int, Field(strict=True)] | None = None
    pricing: TokenPricing | None = None
    allow_paid_calls: bool = False
    fail_fast: bool = False

    @model_validator(mode="after")
    def unique_and_comparable(self):
        if len(set(self.patterns)) != len(self.patterns) or len(set(self.scenario_ids)) != len(self.scenario_ids):
            raise ValueError("Patterns/scenario IDs must be unique")
        if self.pricing and (self.model.provider != "gemini" or self.pricing.model_name != self.model.model_name):
            raise ValueError("Pricing must match the configured real model exactly")
        return self

    @property
    def planned_episodes(self):
        return len(self.patterns) * len(self.scenario_ids) * self.repetitions


class ModelCallMeasurement(DomainModel):
    call_index: PositiveInt
    duration_ms: Milliseconds
    returned: bool
    exception_type: str | None = None
    served_model: str | None = None
    service_tier: str | None = None
    input_tokens: NonnegativeInt | None = None
    output_tokens: NonnegativeInt | None = None
    total_tokens: NonnegativeInt | None = None
    cached_input_tokens: NonnegativeInt | None = None
    cache_write_tokens: NonnegativeInt | None = None
    reasoning_tokens: NonnegativeInt | None = None


class UnsafeTransactionFinding(DomainModel):
    kind: Literal["hold", "payment", "ticket_batch"]
    reference: NonemptyString
    reasons: list[NonemptyString] = Field(min_length=1)


class EpisodeResult(DomainModel):
    schema_version: Literal[1] = 1
    episode_id: NonemptyString
    agent_type: Pattern
    scenario_id: str
    run_index: PositiveInt
    backend_seed: int
    provider: Literal["fake", "gemini"]
    model_name: str | None
    temperature: float | None
    simulated: bool
    status: Literal["evaluated", "setup_error", "driver_error", "interrupted"]
    exception_type: str | None = None
    expected_booking: bool
    booking_completed: bool = False
    scenario_passed: bool = False
    constraint_passed: bool = False
    backend_consistent: bool | None = None
    permission_clean: bool | None = None
    terminal_status: TerminalStatus | None = None
    phase: SessionPhase | None = None
    stop_reason: str | None = None
    environment_hash: str | None = None
    model_decisions: NonnegativeInt | None = None
    model_calls: NonnegativeInt
    llm_calls: NonnegativeInt
    model_error_count: NonnegativeInt | None = None
    tool_attempts: NonnegativeInt | None = None
    executed_tool_calls: NonnegativeInt | None = None
    permission_violations: NonnegativeInt | None = None
    illegal_side_effect_attempts: NonnegativeInt | None = None
    guard_denials: NonnegativeInt | None = None
    committed_side_effects: NonnegativeInt | None = None
    booking_count: NonnegativeInt | None = None
    successful_payment_count: NonnegativeInt | None = None
    ticket_count: NonnegativeInt | None = None
    unsafe_transactions: NonnegativeInt | None = None
    unsafe_findings: list[UnsafeTransactionFinding] = Field(default_factory=list)
    expected_minimum_tool_calls: NonnegativeInt | None = None
    excess_tool_calls: NonnegativeInt | None = None
    plan_rejections: NonnegativeInt | None = None
    blocked_plan_steps: NonnegativeInt | None = None
    replan_count: NonnegativeInt | None = None
    reconciliation_checks: NonnegativeInt | None = None
    transport_replays: NonnegativeInt | None = None
    recovery_opportunities: NonnegativeInt | None = None
    recovered_failures: NonnegativeInt | None = None
    recovery_success: bool | None = None
    recovery_outcomes: list[dict] = Field(default_factory=list)
    handoff_count: NonnegativeInt | None = None
    user_turns: NonnegativeInt | None = None
    duration_ms: Milliseconds
    agent_active_ms: Milliseconds
    model_duration_ms: Milliseconds
    usage_reported_calls: NonnegativeInt
    usage_complete: bool
    observed_input_tokens: NonnegativeInt
    observed_output_tokens: NonnegativeInt
    input_tokens: NonnegativeInt | None = None
    output_tokens: NonnegativeInt | None = None
    total_tokens: NonnegativeInt | None = None
    cached_input_tokens: NonnegativeInt | None = None
    cache_write_tokens: NonnegativeInt | None = None
    reasoning_tokens: NonnegativeInt | None = None
    estimated_cost_usd: USD | None = None
    observed_estimated_cost_usd: USD | None = None
    costed_calls: NonnegativeInt
    model_measurements: list[ModelCallMeasurement]
    oracle_violations: list[str] = Field(default_factory=list)
    measurement_warnings: list[str] = Field(default_factory=list)
    artifact_path: str
