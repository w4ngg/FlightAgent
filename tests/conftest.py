from datetime import date

import pytest

from app.backend import MockFlightBackend
from app.clock import SimulatedClock
from app.models import ConfirmationRecord, ConfirmationSnapshot, Passenger, SearchConstraints, TransactionContext


@pytest.fixture
def clock():
    return SimulatedClock()


@pytest.fixture
def backend(clock):
    return MockFlightBackend(clock=clock)


@pytest.fixture
def prepare(backend):
    def build(count=3, *, session_id="test-session", intent_id="intent-1", **constraints_overrides):
        constraints = SearchConstraints(
            origin="SGN", destination="HAN", departure_date=date(2026, 10, 10),
            passenger_count=count, **constraints_overrides,
        )
        passengers = [Passenger(
            passenger_id=f"P{i + 1}", first_name=f"Mock{i + 1}", last_name="Synthetic",
            date_of_birth=date(1990, 1, i + 1), document_type="IDENTITY_CARD",
            document_number=f"MOCK-DOC-{i + 1}",
        ) for i in range(count)]
        context = TransactionContext(session_id=session_id, transaction_intent_id=intent_id)
        result = backend.search_flights(constraints)
        assert result.ok and result.data.offers
        selected = result.data.offers[0]
        quoted = backend.reprice_offer(selected.offer.offer_id, constraints, passengers, context)
        assert quoted.ok
        quote = quoted.data
        confirmation = ConfirmationRecord(
            context=context, snapshot=ConfirmationSnapshot.from_quote(quote),
            user_turn_id="mock-user-turn-1", confirmed_at=backend.clock.now(),
        )
        return constraints, passengers, context, selected, quote, confirmation
    return build


@pytest.fixture
def held(backend, prepare):
    constraints, passengers, context, selected, quote, confirmation = prepare()
    response = backend.create_booking_hold(quote.quote_id, constraints, passengers, context, confirmation, "hold-1")
    assert response.ok
    return response.data, context, confirmation, selected


@pytest.fixture
def gemini_model(monkeypatch):
    """Real SDK with EVERY sync/async HTTP request intercepted locally."""
    import httpx
    from langchain_google_genai import ChatGoogleGenerativeAI

    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "false")
    models = []

    def build(handle, name="gemini-3-mock-model", **settings):
        settings = {"temperature": None, **settings}
        model = ChatGoogleGenerativeAI(
            model=name, api_key="test-fake-key", vertexai=False, timeout=30, max_retries=1,
            client_args={"transport": httpx.MockTransport(handle), "trust_env": False}, **settings,
        )
        models.append(model)
        return model

    yield build
    for model in models:
        model.client.close()


@pytest.fixture
def gemini_response():
    """Native generateContent fixture, with binary thought-signature replay."""
    def build(message, *, finish="STOP", model="gemini-3-mock-model"):
        calls = getattr(message, "tool_calls", [])
        parts = [{"functionCall": {"name": c["name"], "args": c["args"]},
                  "thoughtSignature": "c2lnbmF0dXJl"} for c in calls]
        if not parts:
            parts = [{"text": message if isinstance(message, str) else message.text,
                      "thoughtSignature": "c2lnbmF0dXJl"}]
        return {"candidates": [{"content": {"role": "model", "parts": parts},
                                "finishReason": finish, "safetyRatings": []}],
                "modelVersion": model,
                "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 25,
                                  "thoughtsTokenCount": 5, "totalTokenCount": 130,
                                  "cachedContentTokenCount": 20}}
    return build


@pytest.fixture
def gemini_prompt():
    def extract(request, requests):
        import json
        assert request.url.path.endswith(":generateContent")
        payload = json.loads(request.content)
        requests.append(payload)
        return payload["systemInstruction"]["parts"][0]["text"]
    return extract
