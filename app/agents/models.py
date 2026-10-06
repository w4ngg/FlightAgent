"""Pattern model factories; fake mode is explicitly not an LLM."""

import json
import os
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


CONTEXT_MARKER = "\nCURRENT_SESSION_JSON:\n"


class OfflineBookingModel(BaseChatModel):
    """A deterministic test double reading ONLY public state/observations.

    This exercises the same graph/tools as real mode, but is not evidence of
    LLM reasoning, NLP quality or a comparison of agent patterns.
    """
    @property
    def _llm_type(self):
        return "flight-lab-offline-test-double"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        context = json.loads(messages[0].content.split(CONTEXT_MARKER, 1)[1])
        state = context["harness"]
        name, args = None, {}
        if not state["terminal_status"] and not context["input_errors"] and state["constraints"]:
            if not state["selected_offer_id"]:
                if context["offers"] is None:
                    name, args = "search_flights", {}
            elif state["selected_offer_id"] not in context["inspected_offers"]:
                name, args = "get_offer", {"offer_id": state["selected_offer_id"]}
            elif context["passengers_valid"]:
                quote = state["quote"]
                if not state["booking_id"]:
                    if not quote or quote["expires_at"] <= context["current_time"]:
                        name, args = "reprice_offer", {"offer_id": state["selected_offer_id"]}
                    elif state["confirmation"]:
                        name, args = "create_booking_hold", {"quote_id": quote["quote_id"]}
                elif state["payment_requires_reconciliation"]:
                    name, args = "get_payment_status", {"operation_id": state["payment_operation_id"]}
                elif state["payment_status"] == "SUCCESS":
                    name, args = "issue_ticket", {"booking_id": state["booking_id"]}
                elif not state["payment_operation_id"]:
                    name, args = "process_payment", {"booking_id": state["booking_id"], "amount_vnd": quote["total_price_vnd"]}
        message = AIMessage(content="" if name else "Cần thông tin từ người dùng.", tool_calls=[{
            "name": name, "args": args, "id": f"offline-call-{state['step_count']}", "type": "tool_call",
        }] if name else [])
        return ChatResult(generations=[ChatGeneration(message=message)])


def create_model(provider: str, model_name: str | None = None, *, temperature: float | None = None):
    if provider == "fake":
        return OfflineBookingModel()
    if provider != "gemini":
        raise ValueError("Provider must be fake or gemini; OpenAI has been replaced")
    # Explicit project .env only; no ancestor search and no secret printing.
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    name = model_name or os.environ.get("MODEL_NAME")
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key or not name:
        raise ValueError("Gemini mode requires GEMINI_API_KEY/GOOGLE_API_KEY and MODEL_NAME (or --model)")
    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() in {"true", "1"}:
        raise ValueError("This lab uses Gemini Developer API; disable GOOGLE_GENAI_USE_VERTEXAI")
    from langchain_google_genai import ChatGoogleGenerativeAI
    # No hidden provider retries. Each failed decision has a visible harness
    # budget; actual tool calls are always executed locally by the harness.
    # The SDK defaults to 0.7. Explicit None omits sampling from the API request
    # so an unspecified temperature uses the selected hosted model's defaults.
    settings = {"temperature": temperature}
    # In the pinned Google integration, max_retries is TOTAL attempts: 1 means
    # one HTTP attempt; 0 can select SDK defaults and enable hidden retries.
    return ChatGoogleGenerativeAI(model=name, api_key=key, vertexai=False, timeout=30, max_retries=1, **settings)


def create_planner_model(provider: str, model_name: str | None = None, *, temperature: float | None = None):
    if provider == "fake":
        from app.agents.planner_model import OfflinePlannerModel
        return OfflinePlannerModel()
    return create_model(provider, model_name, temperature=temperature) if temperature is not None else create_model(provider, model_name)


def create_router_model(provider: str, model_name: str | None = None, *, temperature: float | None = None):
    if provider == "fake":
        from app.agents.router_model import OfflineRouterModel
        return OfflineRouterModel()
    return create_model(provider, model_name, temperature=temperature) if temperature is not None else create_model(provider, model_name)
