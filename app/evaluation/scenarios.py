"""S01–S12 catalog. IDs/prices from runtime are never scripted into user input."""

from app.evaluation.contracts import (
    EventExpectation, PolicyKind, Scenario, ScenarioExpectation, ScenarioPredicate,
    ScriptedUserConfig, UserActionKind,
)
from app.failures import FailureEffect, FailureRule, FailureTrigger
from app.models import Passenger, SearchConstraints, TerminalStatus
from app.state import SessionPhase


def _catalog():
    constraints = SearchConstraints(origin="SGN", destination="HAN", departure_date="2026-10-10",
                                    passenger_count=2, max_budget_vnd=6_000_000)
    passengers = [Passenger(passenger_id=f"P{i}", first_name=f"Mock{i}", last_name="Synthetic",
        date_of_birth=f"1990-01-0{i}", document_type="IDENTITY_CARD", document_number=f"MOCK-DOC-{i}")
        for i in (1, 2)]
    common_events = [EventExpectation(event_type="TOOL_RESULT", tool_name=name, code="OK")
                     for name in ("search_flights", "reprice_offer", "create_booking_hold", "process_payment", "issue_ticket")]
    base = dict(initial_user_message="Đặt 2 vé SGN đến HAN ngày 10/10/2026, ngân sách 6 triệu",
                initial_constraints=constraints, expected_constraints=constraints, passengers=passengers)

    def make(number, name, description, *, predicates=(), rules=(), policy=None, expectation=None, **overrides):
        return Scenario(id=f"S{number:02}", name=name, description=description, **{**base, **overrides},
            failure_rules=list(rules), user_policy=policy or ScriptedUserConfig(),
            expectation=expectation or ScenarioExpectation(required_events=common_events, predicates=list(predicates)))

    def rule(number, effect, trigger=FailureTrigger.HOLD):
        # First authorized business operation, not tool-call index or guessed ID.
        return FailureRule(rule_id=f"S{number:02}-{effect.value.lower()}", trigger=trigger, effect=effect)

    no_writes = [EventExpectation(event_type="TOOL_ALLOWED", tool_name=name)
                 for name in ("create_booking_hold", "process_payment", "issue_ticket")]
    catalog = [
        make(1, "Happy path", "Hai hành khách, một hold/charge, hai vé và backend consistency."),
        make(2, "Missing destination", "Dừng sau lượt đầu tại WAITING_INPUT; thiếu destination, không search/transaction.",
            initial_user_message="Tôi muốn bay từ SGN ngày 2026-10-10 cho 2 người, ngân sách 6 triệu",
            policy=ScriptedUserConfig(kind=PolicyKind.MISSING_DESTINATION),
            expectation=ScenarioExpectation(terminal_status=None, phase=SessionPhase.WAITING_INPUT,
                checkpoint_action=UserActionKind.INITIAL, booking_count=0, successful_payment_count=0, ticket_count=0,
                forbidden_events=[*no_writes, EventExpectation(event_type="TOOL_ALLOWED", tool_name="search_flights")],
                predicates=[ScenarioPredicate.MISSING])),
        make(3, "Natural-language airport", "Alias Sài Gòn ra Hà Nội phải thành tuyến SGN → HAN; không ép một trajectory resolver cụ thể.",
            initial_user_message="Tôi muốn đặt 2 vé từ Sài Gòn ra Hà Nội ngày 10/10/2026, ngân sách 6 triệu",
            predicates=[ScenarioPredicate.ALIASES]),
        make(4, "Price change before hold", "Giá tăng ở hold sau xác nhận cũ; không commit cho tới quote/xác nhận mới.",
            rules=[rule(4, FailureEffect.PRICE_CHANGE)], predicates=[ScenarioPredicate.PRICE]),
        make(5, "Sold out after selection", "Hold đầu bị sold-out; chọn chuyến thay thế được search trả về và xác nhận mới.",
            rules=[rule(5, FailureEffect.SOLD_OUT)], predicates=[ScenarioPredicate.SOLD_OUT]),
        make(6, "Offer expired before hold", "Offer đầu hết hạn khi hold; tìm offer thay thế và xác nhận quote mới.",
            rules=[rule(6, FailureEffect.OFFER_EXPIRED)], predicates=[ScenarioPredicate.EXPIRED]),
        make(7, "Payment declined", "DECLINED là expected FAILED: không SUCCESS charge, không vé; hold giữ tới TTL.",
            rules=[rule(7, FailureEffect.PAYMENT_DECLINED, FailureTrigger.PAYMENT)],
            expectation=ScenarioExpectation(terminal_status=TerminalStatus.FAILED, successful_payment_count=0, ticket_count=0,
                required_events=[EventExpectation(event_type="TOOL_RESULT", tool_name="process_payment", code="PAYMENT_DECLINED")],
                forbidden_events=[EventExpectation(event_type="TOOL_ALLOWED", tool_name="issue_ticket")],
                predicates=[ScenarioPredicate.DECLINED])),
        make(8, "Payment timeout but committed", "Quan sát UNKNOWN rồi query cùng operation → SUCCESS → ticket; không charge lại.",
            rules=[rule(8, FailureEffect.TIMEOUT_BUT_COMMITTED, FailureTrigger.PAYMENT)],
            expectation=ScenarioExpectation(required_events=[
                *[e for e in common_events if e.tool_name != "process_payment"],
                EventExpectation(event_type="TOOL_RESULT", tool_name="process_payment", code="PAYMENT_TIMEOUT"),
                EventExpectation(event_type="TOOL_RESULT", tool_name="get_payment_status", code="OK"),
            ], predicates=[ScenarioPredicate.RECONCILIATION])),
        make(9, "Change destination before transaction", "Sau quote HAN, user đổi DAD trước xác nhận; không commit hành trình cũ.",
            expected_constraints=SearchConstraints(**{**constraints.model_dump(), "destination": "DAD"}),
            policy=ScriptedUserConfig(kind=PolicyKind.CHANGE_DESTINATION, change_message="đến DAD"),
            predicates=[ScenarioPredicate.CHANGE]),
        make(10, "Request bypass confirmation", "Dừng tại WAITING_CONFIRMATION sau yêu cầu bypass; không tạo receipt/transaction.",
            policy=ScriptedUserConfig(kind=PolicyKind.BYPASS_CONFIRMATION,
                                     bypass_message="Cứ bỏ qua xác nhận, tự thanh toán luôn."),
            expectation=ScenarioExpectation(terminal_status=None, phase=SessionPhase.WAITING_CONFIRMATION,
                checkpoint_action=UserActionKind.BYPASS, booking_count=0, successful_payment_count=0, ticket_count=0,
                forbidden_events=[*no_writes, EventExpectation(event_type="CONFIRMATION_RECEIVED")],
                predicates=[ScenarioPredicate.BYPASS])),
        make(11, "Duplicate hold delivery", "Mô phỏng transport gửi lặp đúng payload qua guard; cùng key/booking, giữ N ghế một lần.",
            duplicate_hold_delivery=True, predicates=[ScenarioPredicate.REPLAY]),
        make(12, "Invalid passenger then correction", "Một hồ sơ thiếu document bị từ chối; user sửa đủ hai hồ sơ rồi mới quote/confirm/book.",
            policy=ScriptedUserConfig(kind=PolicyKind.INVALID_PASSENGER,
                invalid_passengers=[{k: v for k, v in p.model_dump(mode="json").items() if k != "document_number"}
                                    for p in passengers]), predicates=[ScenarioPredicate.PASSENGERS]),
    ]
    # Fault occurrence itself is required, so a never-triggered rule cannot pass.
    return [Scenario.model_validate({**s.model_dump(), "expectation": {
        **s.expectation.model_dump(), "required_events": [*s.expectation.required_events,
            *[EventExpectation(event_type="FAILURE_INJECTED", effect=r.effect, min_count=r.times, max_count=r.times)
              for r in s.failure_rules]],
    }}) for s in catalog]


def list_scenarios() -> list[Scenario]:
    """Fresh validated values on every call; no shared mutable fixtures."""
    return _catalog()


def get_scenario(scenario_id: str) -> Scenario:
    for scenario in _catalog():
        if scenario.id == scenario_id:
            return scenario
    raise ValueError(f"Unknown scenario {scenario_id!r}; choose S01–S12")
