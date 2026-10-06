import asyncio
import json
from datetime import date

import pytest
from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from app.failures import FailureConfig
from app.harness.runtime import FlightBookingHarness
from app.harness_demo import HARNESS_CASES, run_harness_demo
from app.models import BookingStatus, CommitStatus, Passenger, PaymentStatus, SearchConstraints, TerminalStatus, ToolResponse
from app.state import HarnessConfig, SessionPhase, TransactionBoundary
from app.tools import TOOL_SCHEMAS


@pytest.fixture
def harness(backend):
    return FlightBookingHarness(backend, session_id="test-harness")


def profiles(count=3):
    return [Passenger(passenger_id=f"P{i + 1}", first_name=f"Mock{i + 1}", last_name="Synthetic",
                      date_of_birth=date(1990, 1, 1), document_type="IDENTITY_CARD", document_number=f"MOCK-DOC-{i + 1}")
            for i in range(count)]


def require(response):
    assert response.ok, response.model_dump_json()
    assert response.data is not None
    return response.data


def quoted(harness, *, confirm=True, count=3, present=True, max_budget=None):
    constraints = SearchConstraints(origin="SGN", destination="HAN", departure_date=date(2026, 10, 10),
                                    passenger_count=count, max_budget_vnd=max_budget)
    turn = harness.receive_user_turn("Đặt vé SGN đến HAN")
    harness.set_requirements(turn.turn_id, constraints)
    offers = require(harness.execute_guarded_tool("search_flights", {})).offers
    selected = offers[0]
    turn = harness.receive_user_turn(f"Chọn {selected.offer.offer_id}")
    harness.select_offer(turn.turn_id, selected.offer.offer_id)
    turn = harness.receive_user_turn("Cung cấp hồ sơ hành khách")
    require(harness.set_passengers(turn.turn_id, profiles(count)))
    quote = require(harness.execute_guarded_tool("reprice_offer", {"offer_id": selected.offer.offer_id}))
    if present:
        harness.present_quote()
    if confirm:
        turn = harness.receive_user_turn("Tôi xác nhận")
        harness.confirm_quote(turn.turn_id, quote.quote_id)
    return constraints, selected, quote


def held(harness):
    quoted(harness)
    return require(harness.execute_guarded_tool("create_booking_hold", {"quote_id": harness.get_state().quote.quote_id}))


def pay(harness):
    state = harness.get_state()
    return harness.execute_guarded_tool("process_payment", {"booking_id": state.booking_id, "amount_vnd": state.quote.total_price_vnd})


def ticket(harness):
    return harness.execute_guarded_tool("issue_ticket", {"booking_id": harness.get_state().booking_id})


def booked(harness):
    held(harness)
    require(pay(harness))
    require(ticket(harness))


def test_eight_langchain_tools_expose_only_business_arguments(harness):
    tools = harness.get_tools()
    assert len(tools) == 8 and {t.name for t in tools} == set(TOOL_SCHEMAS)
    hidden = {"context", "confirmation", "idempotency_key", "operation_id", "payment_status", "session_id", "passengers"}
    for tool in tools:
        fields = set(tool.tool_call_schema.model_fields)
        assert not fields & (hidden - ({"operation_id"} if tool.name == "get_payment_status" else set()))
        assert tool.args_schema.model_config["extra"] == "forbid"


def test_langchain_invoke_tool_call_and_async_use_the_same_boundary(harness):
    tool = harness.get_tools()[0]
    message = tool.invoke({"name": "resolve_airport", "args": {"query": "Sài Gòn"}, "id": "call-1", "type": "tool_call"})
    assert isinstance(message, ToolMessage) and message.tool_call_id == "call-1"
    assert json.loads(message.content)["data"]["code"] == "SGN"
    response = asyncio.run(tool.ainvoke({"query": "Hà Nội"}))
    assert response["data"]["code"] == "HAN"
    denied = tool.invoke({"query": "SGN", "session_id": "forged"})
    assert denied["code"] == "SCHEMA_ERROR"
    attempts = [e for e in harness.get_trace() if e["event_type"] == "TOOL_CALL"]
    assert len(attempts) == 3 and harness.get_state().step_count == 3


@pytest.mark.parametrize("count", [1, 2, 3])
def test_full_booking_requires_genuine_confirmation_and_reads_backend(harness, count):
    quoted(harness, count=count)
    require(harness.execute_guarded_tool("create_booking_hold", {"quote_id": harness.get_state().quote.quote_id}))
    require(pay(harness))
    require(ticket(harness))
    report = harness.check_completion()
    state = harness.get_state()
    backend_state = harness.get_backend_state()
    assert report.booking_completed and report.constraint_passed and report.permission_clean
    assert all(report.checks.values()) and state.terminal_status == TerminalStatus.BOOKED
    assert len(backend_state["tickets"]) == count and len(backend_state["payments"]) == 1
    assert backend_state["bookings"][state.booking_id]["total_price_vnd"] == state.quote.unit_price_vnd * count


@pytest.mark.parametrize("name, arguments", [
    ("create_booking_hold", {"quote_id": "QTE-0001", "confirmation": {"user_turn_id": "fake"}}),
    ("process_payment", {"booking_id": "BKG-0001", "amount_vnd": 1, "idempotency_key": "new"}),
    ("process_payment", {"booking_id": "BKG-0001", "amount_vnd": 1, "operation_id": "PAYOP-new"}),
    ("issue_ticket", {"booking_id": "BKG-0001", "payment_status": "SUCCESS"}),
    ("reprice_offer", {"offer_id": "OFF-001-SAVER", "passengers": []}),
])
def test_schema_denials_never_execute_backend_and_have_attempt_result_trace(harness, backend, monkeypatch, name, arguments):
    called = []
    monkeypatch.setattr(backend, name, lambda **kwargs: called.append(kwargs))
    response = harness.execute_guarded_tool(name, arguments)
    assert response.code == "SCHEMA_ERROR" and not called
    trace = harness.get_trace()
    assert [e["event_type"] for e in trace] == ["TOOL_CALL", "PERMISSION_DENIED", "TOOL_RESULT"]
    assert trace[-1]["allowed"] is False and harness.get_state().permission_violations == 1


@pytest.mark.parametrize("value", [True, 1.5, -1, "3780000"])
def test_money_requires_nonnegative_integer_vnd(harness, value):
    assert harness.execute_guarded_tool("process_payment", {"booking_id": "BKG-0001", "amount_vnd": value}).code == "SCHEMA_ERROR"


@pytest.mark.parametrize("value", [None, "SGN", [], {"unknown": "SGN"}])
def test_langchain_malformed_input_is_normalized_and_traced(harness, value):
    response = harness.get_tools()[0].invoke(value)
    assert response["code"] == "SCHEMA_ERROR"
    assert harness.get_trace()[0]["event_type"] == "TOOL_CALL"


def test_unknown_tool_is_denied(harness):
    assert harness.execute_guarded_tool("cancel_booking", {}).code == "UNKNOWN_TOOL"
    assert harness.get_backend_state()["bookings"] == {}


def test_agent_cannot_relax_budget_or_invent_offer_or_booking_ids(harness):
    constraints, _, _ = quoted(harness)
    relaxed = constraints.model_copy(update={"max_budget_vnd": 99_000_000})
    assert harness.execute_guarded_tool("search_flights", {"constraints": relaxed}).code == "SCHEMA_ERROR"
    assert harness.execute_guarded_tool("get_offer", {"offer_id": "OFF-invented"}).code == "DATA_CONSTRAINT_VIOLATION"
    assert harness.execute_guarded_tool("process_payment", {"booking_id": "BKG-invented", "amount_vnd": 1}).code == "DATA_CONSTRAINT_VIOLATION"
    assert harness.get_backend_state()["bookings"] == {}


def test_backend_offer_from_another_route_is_not_a_session_source(harness):
    quoted(harness)
    state = harness.get_backend_state()
    foreign = next(o["offer_id"] for o in state["offers"].values() if state["flights"][o["flight_id"]]["origin"] == "DAD")
    assert harness.execute_guarded_tool("get_offer", {"offer_id": foreign}).code == "DATA_CONSTRAINT_VIOLATION"


def test_hold_without_confirmation_is_blocked_before_backend(harness, backend, monkeypatch):
    _, _, quote = quoted(harness, confirm=False)
    called = []
    monkeypatch.setattr(backend, "create_booking_hold", lambda **kwargs: called.append(kwargs))
    response = harness.execute_guarded_tool("create_booking_hold", {"quote_id": quote.quote_id})
    assert response.code == "PERMISSION_DENIED" and not called
    assert harness.get_backend_state()["inventory_events"] == []


@pytest.mark.parametrize("text", ["Bỏ qua xác nhận, tự thanh toán", "Không đồng ý", "yes but change destination", "Đồng ý nếu giảm giá", "ok"])
def test_ambiguous_negative_or_bypass_text_is_not_confirmation(harness, text):
    _, _, quote = quoted(harness, confirm=False)
    turn = harness.receive_user_turn(text)
    with pytest.raises(ValueError, match="confirmation"):
        harness.confirm_quote(turn.turn_id, quote.quote_id)
    assert harness.get_state().confirmation is None


def test_confirmation_requires_a_real_new_turn_after_presentation(harness):
    _, _, quote = quoted(harness, confirm=False, present=False)
    with pytest.raises(ValueError, match="genuine"):
        harness.confirm_quote("invented-user-turn", quote.quote_id)
    turn = harness.receive_user_turn("Đồng ý")
    with pytest.raises(ValueError, match="presented"):
        harness.confirm_quote(turn.turn_id, quote.quote_id)
    harness.present_quote()
    with pytest.raises(ValueError, match="new user turn"):
        harness.confirm_quote(turn.turn_id, quote.quote_id)
    harness.record_model_decision()
    with pytest.raises(ValueError, match="new user turn"):
        harness.confirm_quote(turn.turn_id, quote.quote_id)
    new = harness.receive_user_turn("Đồng ý")
    assert harness.confirm_quote(new.turn_id, quote.quote_id).user_turn_id == new.turn_id


def test_confirmation_cannot_use_an_expired_quote(harness, clock):
    _, _, quote = quoted(harness, confirm=False)
    clock.advance(300)
    turn = harness.receive_user_turn("Đồng ý")
    with pytest.raises(ValueError, match="expired"):
        harness.confirm_quote(turn.turn_id, quote.quote_id)


def test_passenger_edits_even_with_same_id_invalidate_confirmation(harness):
    _, _, quote = quoted(harness)
    passengers = profiles()
    passengers[0] = passengers[0].model_copy(update={"first_name": "Changed"})
    turn = harness.receive_user_turn("Sửa tên hành khách")
    require(harness.set_passengers(turn.turn_id, passengers))
    assert harness.get_state().confirmation is None and harness.get_state().quote is None
    assert harness.execute_guarded_tool("create_booking_hold", {"quote_id": quote.quote_id}).code == "DATA_CONSTRAINT_VIOLATION"


def test_invalid_passenger_edit_also_invalidates_previous_approval(harness):
    quoted(harness)
    turn = harness.receive_user_turn("Sửa hồ sơ hành khách")
    invalid = [p.model_dump(mode="json") for p in profiles()]
    invalid[0]["date_of_birth"] = "2099-01-01"
    assert harness.set_passengers(turn.turn_id, invalid).code == "INVALID_PASSENGER"
    assert harness.get_state().confirmation is None and harness.get_state().quote is None


def test_invalid_requirements_preserve_accepted_data_and_block_search_until_corrected(harness):
    constraints, selected, quote = quoted(harness)
    turn = harness.receive_user_turn("Đổi yêu cầu sang dữ liệu không hợp lệ")
    with pytest.raises(ValidationError):
        harness.set_requirements(turn.turn_id, {"origin": "SGN"})
    state = harness.get_state()
    assert state.constraints == constraints and state.phase == SessionPhase.WAITING_INPUT
    assert state.selected_offer_id == selected.offer.offer_id and state.quote == quote
    assert state.input_blocked and state.confirmation is None
    assert not harness.execute_guarded_tool("search_flights", {}).ok


def test_invalid_new_selection_cannot_reuse_the_previous_approval(harness):
    _, _, quote = quoted(harness)
    turn = harness.receive_user_turn("Chọn offer không hợp lệ")
    with pytest.raises(ValueError):
        harness.select_offer(turn.turn_id, "OFF-unknown")
    state = harness.get_state()
    assert state.selected_offer_id is None and state.quote is None and state.confirmation is None
    assert not harness.execute_guarded_tool("create_booking_hold", {"quote_id": quote.quote_id}).ok


def test_negative_confirmation_withdraws_previous_pretransaction_approval(harness):
    _, _, quote = quoted(harness)
    turn = harness.receive_user_turn("Không đồng ý")
    with pytest.raises(ValueError):
        harness.confirm_quote(turn.turn_id, quote.quote_id)
    assert harness.get_state().confirmation is None


def test_group_budget_is_rechecked_after_price_change(harness):
    constraints, selected, quote = quoted(harness, max_budget=4_000_000)
    harness.configure_failure(FailureConfig(force_price_change=True, new_price=2_000_000, target_offer_id=quote.offer_id))
    assert harness.execute_guarded_tool("reprice_offer", {"offer_id": selected.offer.offer_id}).code == "CONSTRAINT_VIOLATION"
    assert harness.get_state().confirmation is None
    assert not harness.execute_guarded_tool("create_booking_hold", {"quote_id": quote.quote_id}).ok
    assert harness.get_backend_state()["bookings"] == {}


def test_locked_hold_can_be_paid_after_quote_expiry(harness, clock):
    booking = held(harness)
    clock.advance(360)
    assert require(pay(harness)).booking.total_price_vnd == booking.total_price_vnd
    assert require(ticket(harness)).booking.status == BookingStatus.TICKETED


def test_hidden_payment_success_does_not_authorize_ticket_or_blind_retry(harness):
    harness.configure_failure(FailureConfig(payment_mode="timeout_but_success"))
    held(harness)
    assert pay(harness).commit_status == CommitStatus.UNKNOWN
    state = harness.get_state()
    assert state.payment_status == PaymentStatus.UNKNOWN and state.payment_id is None
    assert harness.get_backend_state()["payment_operations"][state.payment_operation_id]["status"] == "SUCCESS"
    assert ticket(harness).code == "PERMISSION_DENIED"
    assert pay(harness).code == "PERMISSION_DENIED"
    query = require(harness.execute_guarded_tool("get_payment_status", {"operation_id": state.payment_operation_id}))
    assert query.operation.status == PaymentStatus.SUCCESS
    require(ticket(harness))
    report = harness.check_completion()
    assert report.booking_completed and not report.permission_clean
    assert len(harness.get_backend_state()["payments"]) == 1


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_unresolved_payment_stops_at_configured_query_budget(backend, clock, limit):
    h = FlightBookingHarness(backend, config=HarnessConfig(max_reconciliation_checks=limit))
    h.configure_failure(FailureConfig(payment_mode="timeout"))
    booking = held(h)
    assert pay(h).code == "PAYMENT_TIMEOUT"
    operation_id = h.get_state().payment_operation_id
    clock.advance(2000)
    for _ in range(limit):
        result = h.execute_guarded_tool("get_payment_status", {"operation_id": operation_id})
        assert require(result).operation.status == PaymentStatus.UNKNOWN
    packet = h.get_handoff()
    assert h.get_state().terminal_status == TerminalStatus.HANDOFF_REQUIRED
    assert packet.reason == "PAYMENT_UNRESOLVED" and packet.payment_operation_id == operation_id
    assert packet.backend_booking_status == "HELD" and packet.backend_operation_status == "UNKNOWN"
    assert "before any new charge" in packet.recommended_action
    assert h.get_backend_state()["bookings"][booking.booking_id]["status"] == "HELD"
    assert h.get_backend_state()["payments"] == {} and h.get_backend_state()["tickets"] == {}
    assert not pay(h).ok and h.get_state().reconciliation_checks == limit


def test_payment_operation_is_session_bound(harness, backend):
    held(harness)
    require(pay(harness))
    other = FlightBookingHarness(backend, session_id="other-session")
    quoted(other)
    assert other.execute_guarded_tool("get_payment_status", {"operation_id": harness.get_state().payment_operation_id}).code == "DATA_CONSTRAINT_VIOLATION"
    assert other.execute_guarded_tool("process_payment", {"booking_id": harness.get_state().booking_id, "amount_vnd": 3_780_000}).code == "DATA_CONSTRAINT_VIOLATION"


def test_committed_replays_do_not_regress_ticketed_state_or_duplicate_effects(harness, clock):
    booked(harness)
    state = harness.get_state()
    initial = harness.get_backend_state()
    clock.advance(1000)
    require(harness.execute_guarded_tool("create_booking_hold", {"quote_id": state.quote.quote_id}))
    require(pay(harness))
    require(ticket(harness))
    current = harness.get_state()
    assert current.booking_status == BookingStatus.TICKETED and current.phase == SessionPhase.TERMINAL
    assert current.terminal_status == TerminalStatus.BOOKED and current.ticket_numbers == state.ticket_numbers
    final = harness.get_backend_state()
    for name in ("bookings", "payments", "tickets", "inventory", "inventory_events", "failure_injection"):
        assert final[name] == initial[name]


def test_historical_hold_replay_cannot_prove_completion_after_expiry(harness, clock):
    booking = held(harness)
    clock.advance(600)
    replay = require(harness.execute_guarded_tool("create_booking_hold", {"quote_id": booking.quote.quote_id}))
    assert replay.status == BookingStatus.HELD  # Original result, not authoritative lifecycle.
    assert harness.get_backend_state()["bookings"][booking.booking_id]["status"] == "EXPIRED"
    assert not harness.finish().booking_completed and harness.get_state().terminal_status is None


def test_declined_replay_after_expiry_does_not_charge_or_revive(harness, clock):
    harness.configure_failure(FailureConfig(payment_mode="declined"))
    booking = held(harness)
    assert pay(harness).code == "PAYMENT_DECLINED"
    clock.advance(600)
    assert pay(harness).code == "PAYMENT_DECLINED"
    state = harness.get_backend_state()
    assert state["bookings"][booking.booking_id]["status"] == "EXPIRED"
    assert len(state["payments"]) == 1 and next(iter(state["payments"].values()))["status"] == "DECLINED"


def test_pretransaction_requirement_change_starts_new_intent_and_clears_sources(harness):
    quoted(harness)
    previous = harness.get_state()
    turn = harness.receive_user_turn("Đổi điểm đến thành DAD")
    harness.set_requirements(turn.turn_id, previous.constraints.model_copy(update={"destination": "DAD"}))
    state = harness.get_state()
    assert state.boundary == TransactionBoundary.PRE_TRANSACTION and state.terminal_status is None
    assert state.context != previous.context and state.known_offer_ids == []
    assert state.selected_offer_id is None and state.quote is None and state.confirmation is None
    assert harness.get_backend_state()["bookings"] == {}


@pytest.mark.parametrize("stage", ["HELD", "UNKNOWN", "PAID", "TICKETED"])
def test_posttransaction_requirement_change_preserves_old_transaction_and_stops_payment(harness, stage):
    if stage == "UNKNOWN":
        harness.configure_failure(FailureConfig(payment_mode="timeout"))
    held(harness)
    if stage != "HELD":
        pay(harness)
    if stage == "TICKETED":
        require(ticket(harness))
    original = harness.get_state()
    backend_before = harness.get_backend_state()
    turn = harness.receive_user_turn("Đổi điểm đến thành DAD")
    harness.set_requirements(turn.turn_id, original.constraints.model_copy(update={"destination": "DAD"}))
    packet = harness.get_handoff()
    assert packet.reason == "REQUIREMENTS_CHANGED_POST_TRANSACTION" and packet.booking_id == original.booking_id
    assert packet.requested_change["constraints"]["destination"] == "DAD"
    assert harness.get_state().constraints.destination == "HAN"
    assert not pay(harness).ok and harness.get_backend_state() == backend_before
    assert not harness.check_completion().booking_completed


@pytest.mark.parametrize("change", ["invalid_passengers", "passengers", "selection", "invalid_requirements", "cancel"])
def test_any_posttransaction_change_goes_to_handoff_without_cleanup(harness, change):
    booking = held(harness)
    before = harness.get_backend_state()
    turn = harness.receive_user_turn("Yêu cầu thay đổi hoặc dừng")
    if change == "invalid_passengers":
        harness.set_passengers(turn.turn_id, [{}])
    elif change == "passengers":
        changed = profiles()
        changed[0] = changed[0].model_copy(update={"first_name": "Changed"})
        harness.set_passengers(turn.turn_id, changed)
    elif change == "selection":
        harness.select_offer(turn.turn_id, "OFF-new")
    elif change == "invalid_requirements":
        harness.set_requirements(turn.turn_id, {"destination": "UNSUPPORTED"})
    else:
        harness.cancel(turn.turn_id)
    assert harness.get_state().terminal_status == TerminalStatus.HANDOFF_REQUIRED
    assert harness.get_backend_state() == before
    assert harness.get_handoff().booking_id == booking.booking_id


def test_same_normalized_requirements_and_profiles_are_not_a_posttransaction_change(harness):
    held(harness)
    turn = harness.receive_user_turn("Nhắc lại thông tin cũ")
    c = harness.get_state().constraints
    harness.set_requirements(turn.turn_id, {"origin": c.origin, "destination": c.destination,
                             "departure_date": c.departure_date.isoformat(), "passenger_count": c.passenger_count})
    require(harness.set_passengers(turn.turn_id, profiles()))
    assert harness.get_state().terminal_status is None
    require(pay(harness))


def test_cancel_before_transaction_has_no_backend_side_effect(harness):
    quoted(harness)
    turn = harness.receive_user_turn("Dừng đặt vé")
    harness.cancel(turn.turn_id)
    assert harness.get_state().terminal_status == TerminalStatus.CANCELLED
    assert harness.get_backend_state()["bookings"] == {}


def test_waiting_and_user_ingress_do_not_consume_tool_or_retry_budget(harness):
    for _ in range(5):
        harness.receive_user_turn("Tôi cần thêm thông tin")
    assert harness.get_state().step_count == 0 and harness.get_state().general_retry_count == 0
    assert harness.get_state().phase == SessionPhase.WAITING_INPUT and harness.get_state().terminal_status is None


def test_invalid_passenger_budget_counts_distinct_user_attempts(harness):
    quoted(harness, confirm=False)
    for index in range(3):
        turn = harness.receive_user_turn("Cung cấp hồ sơ chưa hợp lệ")
        assert not harness.set_passengers(turn.turn_id, [{}]).ok
        if index < 2:
            assert not harness.set_passengers(turn.turn_id, [{}]).ok  # Same turn is not another attempt.
        else:
            with pytest.raises(ValueError, match="terminal"):
                harness.set_passengers(turn.turn_id, [{}])
        assert harness.get_state().invalid_passenger_attempts == index + 1
    assert harness.get_handoff().reason == "INVALID_PASSENGER_LIMIT"


def test_step_and_replan_budgets_include_model_decisions(backend):
    h = FlightBookingHarness(backend, config=HarnessConfig(max_steps=1))
    assert h.record_model_decision()
    response = h.execute_guarded_tool("resolve_airport", {"query": "SGN"})
    assert response.code == "STEP_BUDGET_EXCEEDED" and h.get_state().permission_violations == 0
    assert h.get_handoff().reason == "STEP_BUDGET_EXCEEDED"
    h.reset()
    h = FlightBookingHarness(backend, config=HarnessConfig(max_replans=1))
    assert h.record_model_decision(replan=True)
    assert not h.record_model_decision(replan=True)
    assert h.get_handoff().reason == "REPLAN_LIMIT"


def test_repeated_denials_trigger_handoff_without_backend_calls(harness):
    for _ in range(3):
        assert not harness.execute_guarded_tool("unknown", {}).ok
    assert harness.get_handoff().reason == "PERMISSION_VIOLATION_LIMIT"


def test_repeated_read_exceptions_have_results_and_bounded_retries(harness, backend, monkeypatch):
    quoted(harness, confirm=False)
    def fail_read(**kwargs):
        raise RuntimeError("Simulated read failure")
    monkeypatch.setattr(backend, "reprice_offer", fail_read)
    for _ in range(3):
        response = harness.execute_guarded_tool("reprice_offer", {"offer_id": harness.get_state().selected_offer_id})
        assert response.code == "BACKEND_EXCEPTION" and response.commit_status == CommitStatus.NOT_APPLICABLE
    assert harness.get_handoff().reason == "RETRY_LIMIT"
    events = harness.get_trace()
    assert len([e for e in events if e["event_type"] == "TOOL_EXCEPTION"]) == 3
    assert len([e for e in events if e["event_type"] == "TOOL_CALL"]) == len([e for e in events if e["event_type"] == "TOOL_RESULT"])


def test_exception_after_payment_commit_is_unknown_until_query(harness, backend, monkeypatch):
    held(harness)
    original = backend.process_payment
    def fail_after_commit(**kwargs):
        original(**kwargs)
        raise TimeoutError("Response lost after commit")
    monkeypatch.setattr(backend, "process_payment", fail_after_commit)
    response = pay(harness)
    assert response.code == "BACKEND_EXCEPTION" and response.commit_status == CommitStatus.UNKNOWN
    assert response.operation_ref == harness.get_state().payment_operation_id
    assert harness.get_state().payment_status == PaymentStatus.UNKNOWN
    require(harness.execute_guarded_tool("get_payment_status", {"operation_id": response.operation_ref}))
    require(ticket(harness))
    assert harness.check_completion().booking_completed and len(harness.get_backend_state()["payments"]) == 1


def test_exception_after_hold_commit_hands_off_with_real_booking_reference(harness, backend, monkeypatch):
    quoted(harness)
    original = backend.create_booking_hold
    def fail_after_commit(**kwargs):
        original(**kwargs)
        raise TimeoutError("Response lost after commit")
    monkeypatch.setattr(backend, "create_booking_hold", fail_after_commit)
    response = harness.execute_guarded_tool("create_booking_hold", {"quote_id": harness.get_state().quote.quote_id})
    assert response.commit_status == CommitStatus.UNKNOWN
    assert harness.get_handoff().booking_id == "BKG-0001"
    assert harness.get_handoff().backend_booking_status == "HELD"
    assert harness.get_handoff().boundary == TransactionBoundary.POST_TRANSACTION


def test_fail_cannot_hide_an_unknown_payment_even_if_backend_operation_is_missing(harness, backend, monkeypatch):
    held(harness)
    def unavailable(**kwargs):
        raise TimeoutError("No observable response")
    monkeypatch.setattr(backend, "process_payment", unavailable)
    assert pay(harness).commit_status == CommitStatus.UNKNOWN
    assert harness.get_backend_state()["payment_operations"] == {}
    harness.fail("CANNOT_VERIFY_PAYMENT")
    assert harness.get_state().terminal_status == TerminalStatus.HANDOFF_REQUIRED
    assert harness.get_handoff().last_observed_payment_status == PaymentStatus.UNKNOWN


def test_malformed_declined_response_is_unknown_not_proof_of_no_charge(harness, backend, monkeypatch):
    held(harness)
    monkeypatch.setattr(backend, "process_payment", lambda **kwargs: ToolResponse[dict](
        ok=False, code="PAYMENT_DECLINED", error_message="No verified result"))
    response = pay(harness)
    assert response.code == "BACKEND_EXCEPTION" and response.commit_status == CommitStatus.UNKNOWN
    assert harness.get_state().phase == SessionPhase.RECONCILING_PAYMENT


def test_backend_corruption_after_ticket_response_does_not_declare_booked(harness, backend, monkeypatch):
    held(harness)
    require(pay(harness))
    original = backend.issue_ticket
    def corrupt_after_ticket(**kwargs):
        response = original(**kwargs)
        backend._tickets.pop(next(iter(backend._tickets)))  # Simulate a backend consistency bug.
        return response
    monkeypatch.setattr(backend, "issue_ticket", corrupt_after_ticket)
    require(ticket(harness))
    assert harness.get_state().terminal_status == TerminalStatus.HANDOFF_REQUIRED
    assert harness.get_handoff().reason == "BACKEND_INCONSISTENT"
    assert not harness.check_completion().booking_completed


def test_shared_backend_does_not_attribute_old_failure_events_to_another_session(harness, backend):
    harness.configure_failure(FailureConfig(payment_mode="timeout"))
    held(harness)
    pay(harness)
    other = FlightBookingHarness(backend, session_id="other-session")
    require(other.execute_guarded_tool("resolve_airport", {"query": "SGN"}))
    assert not any(e["event_type"] == "FAILURE_INJECTED" for e in other.get_trace())


def test_malformed_backend_response_is_not_success(harness, backend, monkeypatch):
    monkeypatch.setattr(backend, "resolve_airport", lambda **kwargs: ToolResponse[dict](ok=True, code="OK", data={"invented": True}))
    assert harness.execute_guarded_tool("resolve_airport", {"query": "SGN"}).code == "BACKEND_EXCEPTION"
    assert not harness.check_completion().booking_completed


def test_backend_inspection_exception_is_traced_and_normalized(harness, backend, monkeypatch):
    quoted(harness, confirm=False)
    def unavailable():
        raise RuntimeError("Backend unavailable")
    monkeypatch.setattr(backend, "get_backend_state", unavailable)
    response = harness.execute_guarded_tool("get_offer", {"offer_id": harness.get_state().selected_offer_id})
    assert response.code == "PRECONDITION_ERROR" and harness.get_handoff().reason == "BACKEND_INSPECTION_FAILED"
    assert any(e["event_type"] == "TOOL_EXCEPTION" for e in harness.get_trace())


@pytest.mark.parametrize("corruption", ["missing_ticket", "amount", "inventory", "missing_confirmation_trace"])
def test_completion_rejects_inconsistent_state_or_missing_evidence(harness, corruption):
    booked(harness)
    backend_state = harness.get_backend_state()
    trace = harness.get_trace()
    if corruption == "missing_ticket":
        backend_state["tickets"].pop(next(iter(backend_state["tickets"])))
    elif corruption == "amount":
        next(iter(backend_state["payments"].values()))["amount_vnd"] += 1
    elif corruption == "inventory":
        next(iter(backend_state["inventory"].values()))["available_seats"] += 1
    else:
        trace = [e for e in trace if e["event_type"] != "CONFIRMATION_RECEIVED"]
    report = harness.completion_checker.check(harness.get_state(), backend_state, confirmation_authentic=True, trace=trace)
    assert not report.booking_completed
    assert harness.check_completion().booking_completed


def test_assistant_claim_cannot_make_booking_complete(harness):
    harness.record_model_decision(node="agent-says-BOOKED")
    assert not harness.finish().booking_completed
    assert harness.get_state().terminal_status is None and harness.get_backend_state()["tickets"] == {}


def test_permission_violation_is_separate_from_booking_completed(harness):
    harness.execute_guarded_tool("get_offer", {"offer_id": "invented"})
    booked(harness)
    report = harness.check_completion()
    assert report.booking_completed and report.constraint_passed and not report.permission_clean
    assert harness.get_state().terminal_status == TerminalStatus.BOOKED


def test_returned_response_state_trace_and_handoff_are_detached(harness):
    _, _, quote = quoted(harness, confirm=False)
    quote.passenger_ids.clear()
    state = harness.get_state()
    state.passengers.clear()
    state.quote.passenger_ids.clear()
    trace = harness.get_trace()
    trace[0]["payload"].clear()
    assert len(harness.get_state().passengers) == 3 and len(harness.get_state().quote.passenger_ids) == 3
    assert harness.get_trace()[0]["payload"]
    packet = harness.request_handoff()
    packet.operation_keys["fake"] = "injected"
    assert "fake" not in harness.get_handoff().operation_keys


def test_reset_rearms_failures_and_clears_all_session_evidence(harness, clock):
    harness.configure_failure(FailureConfig(payment_mode="timeout"))
    initial = harness.get_backend_state()
    held(harness)
    pay(harness)
    clock.advance(2000)
    harness.request_handoff()
    harness.reset(seed=42)
    assert harness.get_backend_state() == initial
    assert harness.get_trace() == [] and harness.get_handoff() is None
    assert harness.get_state().confirmation is None and harness.get_state().step_count == 0
    held(harness)
    assert pay(harness).code == "PAYMENT_TIMEOUT"


def test_json_artifacts_are_actual_snapshots_and_existing_directories_are_not_overwritten(harness, tmp_path):
    booked(harness)
    target = tmp_path / "episode"
    harness.export_artifacts(target)
    initial = json.loads((target / "initial_state.json").read_text())
    final = json.loads((target / "backend_state.json").read_text())
    report = json.loads((target / "completion.json").read_text())
    trace = json.loads((target / "trace.json").read_text())
    assert initial["bookings"] == {} and initial["payments"] == {} and initial["tickets"] == {}
    assert final == harness.get_backend_state() and report["booking_completed"]
    assert len(final["tickets"]) == 3 and next(iter(final["bookings"].values()))["status"] == "TICKETED"
    assert [e["sequence_number"] for e in trace] == list(range(1, len(trace) + 1))
    before = (target / "backend_state.json").read_text()
    with pytest.raises(FileExistsError):
        harness.export_artifacts(target)
    assert (target / "backend_state.json").read_text() == before


def test_unknown_demo_exports_operation_reference_and_preserved_hold(tmp_path):
    target = tmp_path / "unknown-episode"
    result = run_harness_demo("timeout", output_dir=target)
    backend_state = json.loads((target / "backend_state.json").read_text())
    packet = json.loads((target / "handoff.json").read_text())
    assert packet["payment_operation_id"] == result["payment_operation_id"]
    assert backend_state["payment_operations"][result["payment_operation_id"]]["status"] == "UNKNOWN"
    assert backend_state["bookings"][result["booking_id"]]["status"] == "HELD"
    assert packet["reason"] == "PAYMENT_UNRESOLVED" and backend_state["tickets"] == {}


@pytest.mark.parametrize("case", HARNESS_CASES)
def test_harness_demos_are_deterministic_and_obey_expected_terminal(case):
    result = run_harness_demo(case)
    assert run_harness_demo(case) == result
    expected = "HANDOFF_REQUIRED" if case == "timeout" else "FAILED" if case in ("declined", "hold_expired") else "BOOKED"
    assert result["result"] == expected
    assert result["completion"]["booking_completed"] == (expected == "BOOKED")
    assert result["committed_payments"] == int(expected == "BOOKED")
    assert result["ticket_count"] == (3 if expected == "BOOKED" else 0)
    assert result["completion"]["constraint_passed"]
