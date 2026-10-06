"""Offline conversation router; deliberately not an NLP model or an LLM."""

import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from app.agents.routing import ConversationDecision, ROUTER_CONTEXT_MARKER


class OfflineRouterModel(BaseChatModel):
    @property
    def _llm_type(self):
        return "flight-lab-offline-router-test-double"

    def with_structured_output(self, schema, *, method="json_schema", strict=True, include_raw=False, **kwargs):
        if schema is not ConversationDecision or method != "json_schema" or strict is not True or kwargs:
            raise ValueError("Offline router implements only the ConversationDecision contract")

        def parse(messages):
            raw = self.invoke(messages)
            parsed = schema.model_validate_json(raw.text)
            return {"raw": raw, "parsed": parsed, "parsing_error": None} if include_raw else parsed

        return RunnableLambda(parse)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        context = json.loads(messages[0].content.split(ROUTER_CONTEXT_MARKER, 1)[1])
        recognized = context["local_input_recognized"]
        decision = ConversationDecision(action="CONTINUE" if recognized else "CLARIFY", request_fields=[],
            selection_index=None, selection_evidence=None, clarification=None if recognized else "UNSUPPORTED_INPUT")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=decision.model_dump_json()))])
