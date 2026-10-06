"""Offline structured-planner test double. Not evidence of LLM reasoning."""

import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from app.agents.planning import BookingPlan, PlanAction, PlanStep, PLANNER_CONTEXT_MARKER


class OfflinePlannerModel(BaseChatModel):
    @property
    def _llm_type(self):
        return "flight-lab-offline-planner-test-double"

    def with_structured_output(self, schema, *, method="json_schema", strict=True, include_raw=False, **kwargs):
        if schema is not BookingPlan or method != "json_schema" or strict is not True or kwargs:
            raise ValueError("Offline planner implements the BookingPlan structured-output contract only")

        def parse(messages):
            raw = self.invoke(messages)
            try:
                parsed = schema.model_validate_json(raw.text)
            except ValueError as exc:
                if include_raw:
                    return {"raw": raw, "parsed": None, "parsing_error": exc}
                raise
            return {"raw": raw, "parsed": parsed, "parsing_error": None} if include_raw else parsed

        return RunnableLambda(parse)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        public = json.loads(messages[0].content.split(PLANNER_CONTEXT_MARKER, 1)[1])["public_state"]
        state = public["harness"]
        actions = []
        if state["booking_id"]:
            if state["payment_requires_reconciliation"]:
                actions = ["CHECK_PAYMENT", "ISSUE_TICKET"]
            elif state["payment_status"] == "SUCCESS":
                actions = ["ISSUE_TICKET"]
            else:
                actions = ["PAY", "CHECK_PAYMENT", "ISSUE_TICKET"]
        else:
            if not state["constraints"]:
                actions.append("REQUEST_INPUT")
            if not state["selected_offer_id"]:
                if public["offers"] is None:
                    actions.append("SEARCH_FLIGHTS")
                actions.append("REQUEST_SELECTION")
            if not state["selected_offer_id"] or state["selected_offer_id"] not in public["inspected_offers"]:
                actions.append("GET_OFFER")
            if not public["passengers_valid"]:
                actions.append("REQUEST_PASSENGER")
            quote = state["quote"]
            if not quote or quote["expires_at"] <= public["current_time"]:
                actions.append("REPRICE")
            actions.extend(["REQUEST_CONFIRMATION", "CREATE_HOLD", "PAY", "CHECK_PAYMENT", "ISSUE_TICKET"])
        plan = BookingPlan(steps=[PlanStep(action=PlanAction(a), reason=f"Complete {a.lower()} from current observed state",
                                          endpoint=None) for a in actions])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=plan.model_dump_json()))])
