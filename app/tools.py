"""Eight LangChain tools. Every invocation uses the same guarded dispatcher."""

from typing import Annotated, Any, Callable, Literal

from langchain_core.tools import BaseTool
from pydantic import Field, PrivateAttr

from app.models import (
    Airport, Booking, DomainModel, MoneyVND, NonemptyString, OfferDetails,
    PaymentResult, Quote, SearchConstraints, SearchResult, TicketResult,
)


class ResolveArgs(DomainModel):
    query: NonemptyString


class SearchArgs(DomainModel):
    sort_by: Literal["price", "departure_time", "duration"] = "price"
    limit: Annotated[int, Field(strict=True, ge=1, le=20)] = 5


class OfferArgs(DomainModel):
    offer_id: NonemptyString


class HoldArgs(DomainModel):
    quote_id: NonemptyString


class PaymentArgs(DomainModel):
    booking_id: NonemptyString
    amount_vnd: MoneyVND


class StatusArgs(DomainModel):
    operation_id: NonemptyString


class TicketArgs(DomainModel):
    booking_id: NonemptyString


TOOL_SCHEMAS = {
    "resolve_airport": ResolveArgs, "search_flights": SearchArgs, "get_offer": OfferArgs,
    "reprice_offer": OfferArgs, "create_booking_hold": HoldArgs,
    "process_payment": PaymentArgs, "get_payment_status": StatusArgs, "issue_ticket": TicketArgs,
}
TOOL_DESCRIPTIONS = {
    "resolve_airport": "Resolve a supported Vietnamese city or airport alias to SGN, HAN or DAD.",
    "search_flights": "Search using constraints bound by the harness. Supply only sort_by and limit (1..20). User airline preferences rank first; never supply or relax constraints.",
    "get_offer": "Inspect an offer ID previously returned by search for this session.",
    "reprice_offer": "Reprice the selected offer using validated passengers; a new quote requires new user confirmation.",
    "create_booking_hold": "Hold the current quote only after genuine user confirmation; harness supplies keys/context.",
    "process_payment": "Pay the held booking's exact total once. After UNKNOWN, query status; never retry a charge blindly.",
    "get_payment_status": "Reconcile the existing payment operation reference; no new charge or operation is created.",
    "issue_ticket": "Issue one ticket per passenger only after payment SUCCESS has been observed.",
}
SIDE_EFFECT_TOOLS = frozenset(("create_booking_hold", "process_payment", "issue_ticket"))
TOOL_RESPONSE_TYPES = {
    "resolve_airport": Airport, "search_flights": SearchResult, "get_offer": OfferDetails,
    "reprice_offer": Quote, "create_booking_hold": Booking, "process_payment": PaymentResult,
    "get_payment_status": PaymentResult, "issue_ticket": TicketResult,
}


class GuardedTool(BaseTool):
    _dispatch: Callable = PrivateAttr()

    def __init__(self, name: str, dispatch: Callable):
        super().__init__(name=name, description=TOOL_DESCRIPTIONS[name], args_schema=TOOL_SCHEMAS[name])
        self._dispatch = dispatch

    def _to_args_and_kwargs(self, tool_input, tool_call_id):
        # Defer parsing to the boundary so schema denials are traced too. Keep
        # the full input: never drop forged extra fields or no-argument input.
        return (tool_input,), {}

    def _run(self, tool_input: Any, run_manager=None) -> dict:
        return self._dispatch(self.name, tool_input).model_dump(mode="json")


def create_tools(harness) -> list[BaseTool]:
    """Expose no raw backend function, user ingress, confirmation or state setter."""
    return [GuardedTool(name, harness.execute_guarded_tool) for name in TOOL_SCHEMAS]
