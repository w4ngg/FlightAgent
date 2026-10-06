"""Read-only conversation decisions. No transaction authority or tool arguments."""

from datetime import date
import re
from typing import Literal

from pydantic import Field, model_validator

from app.agents.input import AIRPORT_ALIASES, normalize
from app.models import DomainModel


ROUTER_CONTEXT_MARKER = "\nCURRENT_ROUTING_JSON:\n"
RequestFieldName = Literal["origin", "destination", "departure_date", "passenger_count", "max_budget_vnd"]
Clarification = Literal["ROUTE", "DATE", "COUNT", "SELECTION", "REQUEST", "UNSUPPORTED_INPUT"]


class GroundedRequestField(DomainModel):
    field: RequestFieldName
    value: str = Field(min_length=1, max_length=80)
    evidence: str = Field(min_length=1, max_length=160)


class ConversationDecision(DomainModel):
    action: Literal["CONTINUE", "CLARIFY", "PROPOSE_REQUEST", "PROPOSE_SELECTION"]
    request_fields: list[GroundedRequestField] = Field(max_length=5)
    selection_index: int | None = Field(ge=1, le=1000)
    selection_evidence: str | None = Field(min_length=1, max_length=160)
    clarification: Clarification | None

    @model_validator(mode="after")
    def coherent_decision(self):
        if len({f.field for f in self.request_fields}) != len(self.request_fields):
            raise ValueError("Duplicate request fields")
        if bool(self.request_fields) != (self.action == "PROPOSE_REQUEST"):
            raise ValueError("Only request proposals contain request fields")
        selecting = self.action == "PROPOSE_SELECTION"
        if (self.selection_index is not None) != selecting or (self.selection_evidence is not None) != selecting:
            raise ValueError("Only selection proposals contain a grounded index")
        if (self.clarification is not None) != (self.action == "CLARIFY"):
            raise ValueError("Only clarification decisions contain a question category")
        return self


NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "mot": 1, "hai": 2, "ba": 3, "bon": 4, "nam": 5,
                "sau": 6, "bay": 7, "tam": 8, "chin": 9, "muoi": 10,
                "first": 1, "second": 2, "third": 3}


def has_literal_span(text, evidence):
    """Do not ground SGN in NOTSGN, or the index 1 in option 21."""
    return bool(evidence and evidence == evidence.strip() and
                re.search(r"(?<!\w)" + re.escape(evidence) + r"(?!\w)", text))


def grounded_number(text):
    value = normalize(text)
    if value.isascii() and value.isdecimal():
        return int(value)
    if value in NUMBER_WORDS:
        return NUMBER_WORDS[value]
    raise ValueError("Unsupported numeric evidence")


def grounded_request_fields(decision: ConversationDecision, actual_text: str) -> dict:
    """Canonicalize literal spans from the actual turn, never model guesses.

    Mapping a span to a semantic slot remains a proposal: a second actual user
    turn must approve the displayed interpretation before trusted ingress.
    Optional time/airline constraints still use the shared local/JSON grammar.
    """
    result = {}
    for item in decision.request_fields:
        if not has_literal_span(actual_text, item.evidence):
            raise ValueError("Evidence must be a literal span of the current user turn")
        if item.field in ("origin", "destination"):
            value = AIRPORT_ALIASES.get(normalize(item.evidence))
            if value is None or item.value != value:
                raise ValueError("Unsupported or invented airport")
        elif item.field == "departure_date":
            if "/" in item.evidence:
                day, month, year = map(int, item.evidence.split("/"))
                value = date(year, month, day).isoformat()
            else:
                value = date.fromisoformat(item.evidence).isoformat()
            if item.value != value:
                raise ValueError("Invented date")
        else:
            value = grounded_number(item.evidence)
            if item.value != str(value):
                raise ValueError("Invented number")
            if item.field == "max_budget_vnd" and not item.evidence.strip().isdecimal():
                raise ValueError("Budget proposal requires literal integer VND")
        result[item.field] = value
    return result
