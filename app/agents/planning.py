"""Untrusted planner proposals: actions/reasons, never business IDs or consent."""

from enum import Enum
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from app.models import DomainModel


MAX_PLAN_STEPS = 24
PLANNER_CONTEXT_MARKER = "\nCURRENT_PLANNING_JSON:\n"


class PlanAction(str, Enum):
    REQUEST_INPUT = "REQUEST_INPUT"
    RESOLVE_AIRPORT = "RESOLVE_AIRPORT"
    SEARCH_FLIGHTS = "SEARCH_FLIGHTS"
    REQUEST_SELECTION = "REQUEST_SELECTION"
    GET_OFFER = "GET_OFFER"
    REQUEST_PASSENGER = "REQUEST_PASSENGER"
    REPRICE = "REPRICE"
    REQUEST_CONFIRMATION = "REQUEST_CONFIRMATION"
    CREATE_HOLD = "CREATE_HOLD"
    PAY = "PAY"
    CHECK_PAYMENT = "CHECK_PAYMENT"
    ISSUE_TICKET = "ISSUE_TICKET"


class PlanStep(DomainModel):
    action: PlanAction
    reason: Annotated[str, Field(min_length=1, max_length=512)]
    # Resolve a STATE field, not an arbitrary query or invented airport.
    endpoint: Literal["origin", "destination"] | None

    @model_validator(mode="after")
    def validate_step(self) -> Self:
        if not self.reason.strip():
            raise ValueError("A step needs a short action rationale")
        if (self.action == PlanAction.RESOLVE_AIRPORT) != (self.endpoint is not None):
            raise ValueError("Only RESOLVE_AIRPORT requires an endpoint field")
        return self


class BookingPlan(DomainModel):
    steps: Annotated[list[PlanStep], Field(min_length=1, max_length=MAX_PLAN_STEPS)]
