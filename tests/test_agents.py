import json
from copy import deepcopy

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents.adapter import BaseFlightAgent
from app.agents.input import parse_user_input
from app.agents.models import CONTEXT_MARKER, OfflineBookingModel, create_model
from app.agents.react_agent import ReActFlightAgent
from app.failures import FailureConfig, FailureEffect, FailureRule, FailureTrigger
from app.harness.runtime import FlightBookingHarness
from app.models import TerminalStatus
from app.state import HarnessConfig, SessionPhase


REQUEST = "Đặt 2 vé SGN đến HAN ngày 10/10/2026, ngân sách 5 triệu"
PROFILES = ("hành khách: Nguyen|An|1990-01-01|IDENTITY_CARD|MOCK-0001; "
            "Tran|Binh|1992-02-02|PASSPORT|MOCK-0002")


class ScriptedModel:
    """No backend access; scripts consume the exact graph model protocol."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []
        self.tools = []

    def bind_tools(self, tools, **kwargs):
        self.tools = list(tools)
        assert "parallel_tool_calls" not in kwargs  # Not a Gemini API flag.
        return self

    def invoke(self, messages):
        self.seen.append(deepcopy(messages))
        if not self.replies:
            return AIMessage(content="BOOKED invented-ticket invented-payment")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply(messages) if callable(reply) else reply


def action(name, args, call_id="test-call"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def snapshot(messages):
    return json.loads(messages[0].content.split(CONTEXT_MARKER, 1)[1])


def to_quote(agent):
    assert agent.send(REQUEST).phase == SessionPhase.WAITING_SELECTION
    assert agent.send("chọn 1").phase == SessionPhase.WAITING_PASSENGERS
    result = agent.send(PROFILES)
    assert result.phase == SessionPhase.WAITING_CONFIRMATION
    return result


@pytest.mark.parametrize("text,expected", [
    (REQUEST, {"origin": "SGN", "destination": "HAN", "departure_date": "2026-10-10", "passenger_count": 2, "max_budget_vnd": 5_000_000}),
    ("Đặt vé từ TP HCM tới Hà Nội ngày 2026-10-10 cho 3 người", {"origin": "SGN", "destination": "HAN", "departure_date": "2026-10-10", "passenger_count": 3}),
    ("Đà Nẵng → Sài Gòn ngày 11/10/2026 1 vé", {"origin": "DAD", "destination": "SGN", "departure_date": "2026-10-11", "passenger_count": 1}),
    ("từ SGN", {"origin": "SGN"}), ("đến HAN", {"destination": "HAN"}),
    ("ngân sách 4,5 triệu", {"max_budget_vnd": 4_500_000}),
    ("trước 16:00", {"latest_departure": "16:00:00", "latest_inclusive": False}),
    ("sau 08:10", {"earliest_departure": "08:10:00", "earliest_inclusive": False}),
    ("ưu tiên Vietnam Airlines", {"preferred_airlines": ["Vietnam Airlines"]}),
    ("chỉ VietJet", {"allowed_airlines": ["VietJet Air"]}),
])
def test_bounded_parser(text, expected):
    assert parse_user_input(text, []).requirements == expected


@pytest.mark.parametrize("text", ["ngày mai", "ngày 31/02/2026", "ngày 10/10", "2 người hay 3 người",
    "ngân sách tối đa 3 triệu", "trước 16 giờ", "không VietJet", "ngân sách 5 triệu, ngân sách 3 triệu",
    "trước 16:00 hoặc trước 15:00", "chỉ VietJet và Korean Air"])
def test_ambiguous_constraints_not_silently_relaxed(text):
    parsed = parse_user_input(text, [])
    assert parsed.errors and parsed.requirements


def test_soft_preference_cannot_widen_hard_airline_filter():
    parsed = parse_user_input("chỉ Vietnam Airlines, ưu tiên VietJet", [])
    assert parsed.requirements["allowed_airlines"] == ["Vietnam Airlines"]
    assert parsed.requirements["preferred_airlines"] == ["VietJet Air"]


def test_multiturn_partial_input_without_defaults():
    agent = ReActFlightAgent()
    for text in ["Tôi muốn đặt vé", "từ SGN", "đến HAN", "ngày 2026-10-10"]:
        result = agent.send(text)
        assert result.phase == SessionPhase.WAITING_INPUT
        assert not agent.get_backend_state()["bookings"]
        assert not any(e["tool_name"] == "search_flights" for e in agent.get_trace())
    result = agent.send("2 người")
    assert result.phase == SessionPhase.WAITING_SELECTION
    assert agent.get_state()["harness"]["constraints"]["passenger_count"] == 2
    assert len([m for m in agent.get_state()["messages"] if m["type"] == "human"]) == 5


def test_react_happy_path_pause_confirm_and_backend_evidence():
    agent = ReActFlightAgent(session_id="dialogue")
    assert isinstance(agent, BaseFlightAgent)
    result = to_quote(agent)
    assert result.waiting_for_user and not result.completion.booking_completed
    assert "2,520,000" in result.assistant_message
    assert not agent.get_backend_state()["bookings"]
    state_before = agent.get_state()
    assert state_before["harness"]["displayed_quote_id"] == state_before["harness"]["quote"]["quote_id"]
    # Snapshot/inspection cannot mutate state or consume model/tool budget.
    state_before["messages"].clear()
    steps = agent.harness.get_state().step_count
    agent.get_state(); agent.get_backend_state(); agent.get_trace()
    assert agent.harness.get_state().step_count == steps
    result = agent.send("đồng ý", session_id="dialogue")
    assert result.terminal_status == TerminalStatus.BOOKED and not result.waiting_for_user
    assert result.completion.booking_completed and result.completion.permission_clean
    backend = agent.get_backend_state()
    assert len(backend["bookings"]) == 1 and len(backend["payments"]) == 1 and len(backend["tickets"]) == 2
    events = agent.get_trace()
    assert any(e["event_type"] == "LLM_CALL" and e["payload"]["node"] == "react.model" for e in events)
    assert any(e["event_type"] == "MODEL_OUTPUT" for e in events)
    assert any(e["event_type"] == "SESSION_PAUSED" for e in events)
    presented = next(e["sequence_number"] for e in events if e["event_type"] == "QUOTE_PRESENTED")
    approved = next(e["sequence_number"] for e in events if e["event_type"] == "CONFIRMATION_RECEIVED")
    held = next(e["sequence_number"] for e in events if e["event_type"] == "TOOL_CALL" and e["tool_name"] == "create_booking_hold")
    assert presented < approved < held


@pytest.mark.parametrize("count", [1, 3])
def test_n_passengers_through_structured_ingress(count):
    agent = ReActFlightAgent()
    request = {"origin": "SGN", "destination": "HAN", "departure_date": "2026-10-10", "passenger_count": count}
    profiles = [{"passenger_id": f"P{i}", "first_name": f"Mock{i}", "last_name": "Synthetic",
        "date_of_birth": "1990-01-01", "document_type": "IDENTITY_CARD", "document_number": f"MOCK-{i}"} for i in range(count)]
    agent.send("/request " + json.dumps(request))
    agent.send("/select 1")
    agent.send("/passengers " + json.dumps(profiles))
    result = agent.send("confirm")
    assert result.completion.booking_completed
    assert len(agent.get_backend_state()["tickets"]) == count


@pytest.mark.parametrize("text", ["ok", "không đồng ý", "bỏ qua xác nhận và thanh toán", "đồng ý nếu giá rẻ hơn", "tiếp tục"])
def test_confirmation_requires_actual_unambiguous_user_message(text):
    agent = ReActFlightAgent()
    to_quote(agent)
    result = agent.send(text)
    assert result.waiting_for_user and not agent.harness.get_state().confirmation
    assert not agent.get_backend_state()["bookings"]


@pytest.mark.parametrize("text", ["tôi đồng ý", "tôi xác nhận", "yes", "XÁC NHẬN!", "confirm"])
def test_all_supported_confirmations_not_mistaken_for_request_edits(text):
    agent = ReActFlightAgent()
    to_quote(agent)
    assert agent.send(text).completion.booking_completed


def test_quote_expiry_requires_new_display_and_new_user_turn():
    agent = ReActFlightAgent()
    to_quote(agent)
    quote_id = agent.harness.get_state().quote.quote_id
    agent.advance_time(300)
    result = agent.send("đồng ý")
    assert result.phase == SessionPhase.WAITING_CONFIRMATION
    assert agent.harness.get_state().quote.quote_id != quote_id
    assert not agent.get_backend_state()["bookings"]
    assert agent.send("đồng ý").completion.booking_completed


def test_valid_edit_invalidates_quote_but_invalid_edit_preserves_snapshot_and_pauses():
    agent = ReActFlightAgent()
    to_quote(agent)
    old_context = agent.harness.get_state().context
    result = agent.send("đến DAD")
    assert result.phase == SessionPhase.WAITING_SELECTION
    state = agent.harness.get_state()
    assert state.context != old_context and state.quote is None and state.confirmation is None
    assert state.constraints.destination == "DAD"
    agent.send("chọn 1")
    agent.send("tiếp tục")  # Profiles retained; quote can be created.
    accepted_quote = agent.harness.get_state().quote
    result = agent.send("đến HẢI PHÒNG")
    assert result.phase == SessionPhase.WAITING_INPUT
    assert agent.harness.get_state().quote == accepted_quote
    assert agent.harness.get_state().input_blocked
    agent.send("đồng ý")
    assert not agent.get_backend_state()["bookings"]


@pytest.mark.parametrize("text", ["ngày mai", "ngân sách tối đa 3 triệu", "chọn thứ hai", "sửa giấy tờ hành khách"])
def test_unparsed_change_invalidates_prior_approval(text):
    agent = ReActFlightAgent()
    to_quote(agent)
    agent.send(text)
    agent.send("đồng ý")
    assert not agent.get_backend_state()["bookings"]


def test_invalid_passenger_resubmission_does_not_reuse_old_profiles():
    agent = ReActFlightAgent()
    to_quote(agent)
    for _ in range(2):
        result = agent.send("/passengers []")
        assert result.waiting_for_user and agent.harness.get_state().quote is None
        agent.send("đồng ý")
        assert not agent.get_backend_state()["bookings"]
    result = agent.send(PROFILES)
    assert result.phase == SessionPhase.WAITING_CONFIRMATION
    assert agent.send("đồng ý").completion.booking_completed


def test_three_invalid_passenger_turns_handoff():
    agent = ReActFlightAgent()
    agent.send(REQUEST); agent.send("chọn 1")
    for _ in range(3):
        result = agent.send("hành khách: missing-fields")
    assert result.terminal_status == TerminalStatus.HANDOFF_REQUIRED
    assert not agent.get_backend_state()["bookings"]


def test_no_offers_waits_without_changing_constraints():
    agent = ReActFlightAgent()
    result = agent.send(REQUEST.replace("5 triệu", "0 VND"))
    assert result.waiting_for_user and "Không có chuyến" in result.assistant_message
    assert agent.harness.get_state().constraints.max_budget_vnd == 0
    assert agent.get_state()["offers"] == []
    assert not agent.get_backend_state()["bookings"]


@pytest.mark.parametrize("mode,terminal,charges,tickets,queries", [
    ("declined", TerminalStatus.FAILED, 0, 0, 0),
    ("timeout", TerminalStatus.HANDOFF_REQUIRED, 0, 0, 2),
    ("timeout_but_success", TerminalStatus.BOOKED, 1, 2, 1),
])
def test_react_payment_recovery_uses_same_operation(mode, terminal, charges, tickets, queries):
    agent = ReActFlightAgent()
    agent.configure_failure(FailureConfig(payment_mode=mode))
    to_quote(agent)
    result = agent.send("đồng ý")
    assert result.terminal_status == terminal
    backend = agent.get_backend_state()
    assert sum(p["status"] == "SUCCESS" for p in backend["payments"].values()) == charges
    assert len(backend["tickets"]) == tickets and len(backend["payment_operations"]) == 1
    calls = [e for e in agent.get_trace() if e["event_type"] == "TOOL_CALL"]
    assert sum(e["tool_name"] == "process_payment" for e in calls) == 1
    assert sum(e["tool_name"] == "get_payment_status" for e in calls) == queries


@pytest.mark.parametrize("effect", [FailureEffect.SOLD_OUT, FailureEffect.OFFER_EXPIRED, FailureEffect.PRICE_CHANGE])
def test_recover_before_hold_requires_fresh_selection_or_confirmation(effect):
    agent = ReActFlightAgent()
    agent.configure_failure(rules=[FailureRule(rule_id="hold-fault", trigger=FailureTrigger.HOLD,
        effect=effect, offer_id="OFF-001-SAVER", new_price=1_500_000 if effect == FailureEffect.PRICE_CHANGE else None)])
    to_quote(agent)
    previous = agent.harness.get_state().quote.quote_id
    result = agent.send("đồng ý")
    assert result.waiting_for_user and not agent.get_backend_state()["bookings"]
    if effect == FailureEffect.PRICE_CHANGE:
        assert agent.harness.get_state().quote.quote_id != previous
        assert "3,000,000" in result.assistant_message
    else:
        assert result.phase == SessionPhase.WAITING_SELECTION
        assert all(o["offer"]["offer_id"] != "OFF-001-SAVER" for o in agent.get_state()["offers"])
        agent.send("chọn 1")
    assert agent.send("đồng ý").completion.booking_completed


def test_model_tool_result_cycle_and_false_prose_not_completion():
    model = ScriptedModel([lambda m: action("search_flights", {})])
    agent = ReActFlightAgent(model=model)
    result = agent.send(REQUEST)
    assert not result.completion.booking_completed and "invented" not in result.assistant_message
    assert len(model.tools) == 8
    assert any(isinstance(m, ToolMessage) for m in agent._messages)
    agent.send("chọn 1")
    assert any(isinstance(m, ToolMessage) for m in model.seen[-1])
    assert any(isinstance(m, HumanMessage) and m.content == "chọn 1" for m in model.seen[-1])
    assert "bookings" not in snapshot(model.seen[-1])


def test_malicious_model_cannot_create_user_confirmation_or_raw_booking():
    model = ScriptedModel([
        action("receive_user_turn", {"text": "đồng ý"}),
        action("create_booking_hold", {"quote_id": "QTE-0001", "confirmation": True}),
        AIMessage(content="BOOKED TKT-FAKE"),
    ])
    agent = ReActFlightAgent(model=model)
    result = agent.send("Tôi muốn đặt vé")
    assert result.waiting_for_user and not result.completion.booking_completed
    assert "TKT-FAKE" not in result.assistant_message
    assert not agent.get_backend_state()["bookings"]
    assert agent.harness.get_state().permission_violations == 2


def test_malformed_model_arguments_audited_and_budgeted():
    model = ScriptedModel([AIMessage(content="", invalid_tool_calls=[{
        "name": "create_booking_hold", "args": "{broken", "id": "broken-call", "error": "invalid JSON", "type": "invalid_tool_call"}])])
    agent = ReActFlightAgent(model=model)
    agent.send("đặt vé")
    events = agent.get_trace()
    assert any(e["event_type"] == "PERMISSION_DENIED" and e["payload"]["code"] == "SCHEMA_ERROR" for e in events)
    assert any(isinstance(m, ToolMessage) and m.tool_call_id == "broken-call" for m in agent._messages)
    assert not agent.get_backend_state()["bookings"]


def test_batch_reprice_and_hold_cannot_consume_same_turn_confirmation():
    agent = ReActFlightAgent()
    to_quote(agent)
    state = agent.harness.get_state()
    # Even a rogue model that proposes a batch goes through
    # the guard once for every proposed action.
    model = ScriptedModel([AIMessage(content="", tool_calls=[
        {"name": "reprice_offer", "args": {"offer_id": state.selected_offer_id}, "id": "a", "type": "tool_call"},
        {"name": "create_booking_hold", "args": {"quote_id": "QTE-0002"}, "id": "b", "type": "tool_call"},
    ])])
    agent._bound_model = model.bind_tools(agent.harness.get_tools())
    result = agent.send("đồng ý")
    assert result.waiting_for_user and not agent.get_backend_state()["bookings"]
    assert agent.harness.get_state().confirmation is None
    assert agent.harness.get_state().permission_violations == 1


def test_model_error_resume_after_hold_preserves_receipt_and_keys():
    agent = ReActFlightAgent()
    to_quote(agent)
    model = ScriptedModel([
        lambda m: action("create_booking_hold", {"quote_id": snapshot(m)["harness"]["quote"]["quote_id"]}),
        RuntimeError("secret-api-key-SHOULD-NOT-BE-LOGGED"),
    ])
    agent._bound_model = model.bind_tools(agent.harness.get_tools())
    result = agent.send("đồng ý")
    assert result.waiting_for_user and "Model chưa" in result.assistant_message
    held = agent.harness.get_state()
    assert held.booking_id and held.confirmation and held.operation_keys
    assert "secret-api-key" not in json.dumps(agent.get_trace())
    agent._bound_model = OfflineBookingModel()
    result = agent.send("tiếp tục")
    assert result.completion.booking_completed
    assert agent.harness.get_state().operation_keys["create_booking_hold"] == held.operation_keys["create_booking_hold"]
    assert len(agent.get_backend_state()["bookings"]) == 1


@pytest.mark.parametrize("text", ["đến DAD", "ngày mai", "chọn 2", "hủy", "/passengers []"])
def test_changes_after_hold_handoff_instead_of_new_transaction(text):
    agent = ReActFlightAgent()
    to_quote(agent)
    model = ScriptedModel([
        lambda m: action("create_booking_hold", {"quote_id": snapshot(m)["harness"]["quote"]["quote_id"]}),
        AIMessage(content="paused"),
    ])
    agent._bound_model = model.bind_tools(agent.harness.get_tools())
    agent.send("đồng ý")
    result = agent.send(text)
    assert result.terminal_status == TerminalStatus.HANDOFF_REQUIRED
    assert len(agent.get_backend_state()["bookings"]) == 1
    assert not agent.get_backend_state()["payments"]
    assert agent.harness.get_handoff().requested_change


def test_unknown_payment_not_leaked_to_model_and_resume_reconciles():
    agent = ReActFlightAgent()
    agent.configure_failure(FailureConfig(payment_mode="timeout_but_success"))
    to_quote(agent)
    model = ScriptedModel([
        lambda m: action("create_booking_hold", {"quote_id": snapshot(m)["harness"]["quote"]["quote_id"]}),
        lambda m: action("process_payment", {"booking_id": snapshot(m)["harness"]["booking_id"],
            "amount_vnd": snapshot(m)["harness"]["quote"]["total_price_vnd"]}),
        AIMessage(content="paused"),
    ])
    agent._bound_model = model.bind_tools(agent.harness.get_tools())
    agent.send("đồng ý")
    context = snapshot(model.seen[-1])
    assert context["harness"]["payment_status"] == "UNKNOWN"
    assert context["harness"]["payment_id"] is None
    assert next(iter(agent.get_backend_state()["payments"].values()))["status"] == "SUCCESS"
    old_operation = agent.harness.get_state().payment_operation_id
    agent._bound_model = OfflineBookingModel()
    assert agent.send("tiếp tục").completion.booking_completed
    assert agent.harness.get_state().payment_operation_id == old_operation
    assert len(agent.get_backend_state()["payments"]) == 1


def test_model_failures_and_loop_use_harness_budgets():
    model = ScriptedModel([RuntimeError("provider") for _ in range(3)])
    agent = ReActFlightAgent(model=model)
    for _ in range(3):
        result = agent.send("Tôi muốn đặt vé")
    assert result.terminal_status == TerminalStatus.HANDOFF_REQUIRED
    assert agent.harness.get_state().general_retry_count == 3
    harness = FlightBookingHarness(config=HarnessConfig(max_steps=3))
    model = ScriptedModel([action("resolve_airport", {"query": "SGN"}) for _ in range(10)])
    agent = ReActFlightAgent(harness, model=model)
    assert agent.send("đặt vé").terminal_status == TerminalStatus.HANDOFF_REQUIRED
    assert len(model.seen) <= 2


def test_session_isolation_reset_and_cancellation():
    agent, other = ReActFlightAgent(session_id="one"), ReActFlightAgent(session_id="two")
    to_quote(agent)
    with pytest.raises(ValueError):
        agent.send("đồng ý", session_id="two")
    assert not other.get_state()["messages"] and not other.get_backend_state()["quotes"]
    assert agent.send("hủy").terminal_status == TerminalStatus.CANCELLED
    agent.reset()
    assert not agent.get_state()["messages"] and not agent.get_trace()
    to_quote(agent); agent.send("đồng ý")
    with pytest.raises(ValueError):
        agent.reset()
    agent.reset(force=True)
    assert not agent.get_backend_state()["bookings"] and not agent.get_state()["messages"]


def test_export_includes_conversation_and_never_overwrites(tmp_path):
    agent = ReActFlightAgent()
    to_quote(agent); agent.send("đồng ý")
    directory = tmp_path / "conversation"
    agent.export_artifacts(directory)
    assert len(list(directory.glob("*.json"))) == 7
    data = json.loads((directory / "conversation.json").read_text())
    assert data["pattern"] == "react" and data["messages"]
    assert data["harness"]["terminal_status"] == "BOOKED"
    assert json.loads((directory / "completion.json").read_text())["booking_completed"]
    with pytest.raises(FileExistsError):
        agent.export_artifacts(directory)


def test_model_factory_explicit_modes_without_paid_network(monkeypatch):
    assert isinstance(create_model("fake"), OfflineBookingModel)
    with pytest.raises(ValueError):
        create_model("unsupported")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    with pytest.raises(ValueError):
        create_model("gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "local-test-not-a-real-key")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "false")
    monkeypatch.setenv("MODEL_NAME", "user-configured-model")
    model = create_model("gemini")  # Constructor only; never invoke the API.
    assert model.model == "user-configured-model" and model.vertexai is False
    try:
        assert model.max_retries == 1 and model.timeout == 30 and model.temperature is None
        bound = model.bind_tools(ReActFlightAgent().harness.get_tools())
        assert len(bound.kwargs["tools"]) == 8
    finally:
        model.client.close()
