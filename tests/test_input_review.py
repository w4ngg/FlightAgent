"""Multi-turn review acceptance. Scripted ingress/HTTP only, never hosted APIs."""

import io
import json
from copy import deepcopy

import httpx
import pytest
from langchain_core.messages import AIMessage, SystemMessage

from app.agents.factory import create_agent
from app.agents.input_review import InputReviewConfig, RequestProposal
from app.agents.interpretation import InputDecision, SearchPatch
from app.agents.models import OfflineBookingModel
from app.cli import run_cli
from app.models import payload_hash
from test_input_interpretation import ScriptedInput, JSON_REQUEST, PROFILE


PATTERNS = ["react", "plan_execute", "hybrid"]
FULL_TEXT = "đặt 3 vé từ SGN đến HAN ngày 10/10/2026, budget 6 triệu"
FIX_TEXT = "tôi đặt lộn chỉ đặt 1 vé budget y nguyên"


def decision(patch, spans=None, clear=None):
    return InputDecision(intent="UPDATE_SEARCH", search_patch=SearchPatch(**patch) if patch else None,
        clear_fields=clear or [], evidence=[{"field": k, "text": v} for k, v in (spans or {}).items()])


def full(date_span="10/10/2026"):
    return decision({"origin": "SGN", "destination": "HAN", "departure_date": "2026-10-10",
        "passenger_count": 3, "max_budget_vnd": 6_000_000}, {"origin": "SGN", "destination": "HAN",
        "departure_date": date_span, "passenger_count": "3 vé", "max_budget_vnd": "6 triệu"})


def agent_for(pattern="react", *inputs, mode="strict"):
    return create_agent(pattern, input_model=ScriptedInput(*inputs),
                        input_review_config=InputReviewConfig(evidence_mode=mode))


def calls(agent):
    return [e for e in agent.get_trace() if e["event_type"] == "TOOL_CALL"]


@pytest.mark.parametrize("pattern", PATTERNS)
def test_followup_preserves_full_snapshot_and_field_turns_then_commits_same_hash(pattern):
    agent = agent_for(pattern, full(), decision({"passenger_count": 1}, {"passenger_count": "1 vé"}))
    agent.send(FULL_TEXT)
    old = agent.get_state()["pending_interpretation"]
    assert old["status"] == "READY" and old["apply_enabled"]
    assert agent.get_state()["request_draft"] == {} and not calls(agent)
    agent.send(FIX_TEXT)
    state = agent.get_state()
    p = state["pending_interpretation"]
    assert p["revision"] > old["revision"] and p["snapshot_hash"] != old["snapshot_hash"]
    assert p["snapshot"] == p["preview"]
    assert p["snapshot"]["origin"] == "SGN" and p["snapshot"]["destination"] == "HAN"
    assert p["snapshot"]["departure_date"] == "2026-10-10"
    assert p["snapshot"]["passenger_count"] == 1 and p["snapshot"]["max_budget_vnd"] == 6_000_000
    assert not state["working_missing_fields"] and state["harness"]["constraints"] is None
    for key in ("origin", "destination", "departure_date", "max_budget_vnd"):
        assert p["field_sources"][key] == old["field_sources"][key]
    assert p["field_sources"]["passenger_count"]["source_turn_id"].endswith("user-2")
    assert not calls(agent)
    agent.send("áp dụng")
    state = agent.get_state()
    assert state["pending_interpretation"] is None
    assert payload_hash(state["request_draft"]) == p["snapshot_hash"]
    assert payload_hash(state["harness"]["constraints"]) == p["snapshot_hash"]
    receipt = state["last_request_receipt"]
    assert receipt["snapshot_hash"] == p["snapshot_hash"] and len(receipt["source_turn_ids"]) == 2
    assert receipt["applied_by_turn_id"].endswith("user-3")
    assert state["accepted_field_sources"] == p["field_sources"]
    assert state["harness"]["confirmation"] is None and not agent.get_backend_state()["bookings"]
    assert calls(agent)[0]["tool_name"] == "search_flights"


@pytest.mark.parametrize("pattern", PATTERNS)
def test_accumulate_partial_requests_and_partial_apply_without_search(pattern):
    agent = agent_for(pattern, decision({"origin": "SGN"}, {"origin": "SGN"}),
        decision({"destination": "HAN"}, {"destination": "HAN"}),
        decision({"departure_date": "2026-10-10"}, {"departure_date": "10/10/2026"}),
        decision({"passenger_count": 2}, {"passenger_count": "2 người"}))
    agent.send("từ SGN"); agent.send("đến HAN"); agent.send("áp dụng")
    state = agent.get_state()
    assert state["request_draft"]["origin"] == "SGN" and state["request_draft"]["destination"] == "HAN"
    assert state["harness"]["constraints"] is None and state["harness"]["input_blocked"]
    assert not calls(agent)
    agent.send("ngày 10/10/2026"); agent.send("2 người")
    assert not agent.get_state()["working_missing_fields"] and not calls(agent)
    agent.send("áp dụng")
    assert agent.harness.get_state().constraints.passenger_count == 2


@pytest.mark.parametrize("pattern", PATTERNS)
def test_warn_accepts_iso_value_with_vietnamese_evidence_and_never_spends_retry_budget(pattern):
    text = FULL_TEXT.replace("10/10/2026", "10 tháng 10 năm 2026")
    agent = agent_for(pattern, full("10 tháng 10 năm 2026"), mode="warn")
    result = agent.send(text)
    p = agent.get_state()["pending_interpretation"]
    source = p["field_sources"]["departure_date"]
    assert source["literal_status"] == "VERIFIED" and source["semantic_status"] == "UNSUPPORTED_FORMAT"
    assert p["apply_enabled"] and "CẢNH BÁO" in result.assistant_message
    assert p["warnings"][0]["field"] == "departure_date"
    assert agent.harness.get_state().general_retry_count == 0 and not calls(agent)
    assert agent.harness.get_state().step_count == 1
    assert agent.get_state()["last_model_error"] is None
    assert not any(e["event_type"] == "MODEL_ERROR" for e in agent.get_trace())
    agent.send("áp dụng")
    assert agent.harness.get_state().constraints.departure_date.isoformat() == "2026-10-10"
    assert agent.get_state()["accepted_field_sources"]["departure_date"] == source


def test_strict_keeps_rejecting_unsupported_evidence():
    agent = agent_for("react", full("10 tháng 10 năm 2026"))
    agent.send(FULL_TEXT.replace("10/10/2026", "10 tháng 10 năm 2026"))
    assert agent.get_state()["pending_interpretation"] is None
    assert agent.harness.get_state().general_retry_count == 1 and not calls(agent)


@pytest.mark.parametrize("spans,reason", [({}, "EVIDENCE_MISSING"),
    ({"passenger_count": "2 người"}, "EVIDENCE_NOT_IN_CURRENT_TURN"),
    ({"passenger_count": "1 vé"}, "EVIDENCE_VALUE_MISMATCH")])
def test_warn_records_unverified_sources_honestly_and_fix_replaces_active_warning(spans, reason):
    agent = agent_for("react", decision({"passenger_count": 2}, spans),
        decision({"passenger_count": 1}, {"passenger_count": "1 vé"}), mode="warn")
    agent.send("1 vé")
    p = agent.get_state()["pending_interpretation"]
    source = p["field_sources"]["passenger_count"]
    assert reason in source["reason_codes"] and len(p["warnings"]) == 1
    if reason != "EVIDENCE_VALUE_MISMATCH":
        assert source["literal_span"] is None and source["literal_status"] != "VERIFIED"
    agent.send("chỉ 1 vé")
    assert not agent.get_state()["pending_interpretation"]["warnings"]
    assert any(e["event_type"] == "INPUT_EVIDENCE_WARNING" for e in agent.get_trace())


@pytest.mark.parametrize("mode", ["strict", "warn"])
@pytest.mark.parametrize("bad", [
    {"departure_date": "2026-13-10"}, {"departure_date": "2026-01-01"},
    {"departure_date": "10/10/2026"}, {"origin": "DND"}, {"passenger_count": True},
    {"passenger_count": -1}, {"max_budget_vnd": -1}, {"max_budget_vnd": True},
    {"origin": "HAN", "destination": "HAN"},
    {"earliest_departure": "18:00:00", "latest_departure": "08:00:00"},
    {"earliest_departure": "08:00+07:00"},
])
def test_warn_never_weakens_schema_or_domain_checks(mode, bad):
    value = {"intent": "UPDATE_SEARCH", "search_patch": bad, "evidence": []}
    agent = agent_for("react", value, mode=mode)
    agent.send("request change")
    assert agent.get_state()["pending_interpretation"] is None
    assert agent.get_state()["request_draft"] == {} and not calls(agent)
    assert agent.harness.get_state().input_blocked


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("bad", [RuntimeError("do-not-echo-secret"),
    {"intent": "UPDATE_SEARCH", "search_patch": {"passenger_count": True}},
    decision({"departure_date": "2026-13-10"}, {"departure_date": "2026-13-10"}),
    InputDecision(intent="CLARIFY", clarification="REQUEST"),
])
def test_failure_preserves_proposal_locks_apply_and_keep_requires_fresh_user_apply(pattern, bad):
    agent = agent_for(pattern, full(), bad)
    agent.send(FULL_TEXT)
    original = agent.get_state()["pending_interpretation"]
    agent.send("follow-up not accepted")
    kept = agent.get_state()["pending_interpretation"]
    assert kept["status"] == "NEEDS_REVIEW" and not kept["apply_enabled"]
    for key in ("snapshot", "snapshot_hash", "field_sources"):
        assert kept[key] == original[key]
    assert not calls(agent)
    agent.send("áp dụng")
    assert not agent.get_state()["request_draft"] and not calls(agent)
    agent.send("/proposal show")
    assert not agent.get_state()["pending_interpretation"]["apply_enabled"]
    agent.send("/proposal keep")
    p = agent.get_state()["pending_interpretation"]
    assert p["status"] == "READY" and p["apply_enabled"] and p["revision"] > original["revision"]
    assert p["snapshot_hash"] == original["snapshot_hash"] and not calls(agent)
    assert p["latest_source_turn_id"] == original["latest_source_turn_id"]
    agent.send("áp dụng")
    assert agent.harness.get_state().constraints.passenger_count == 3
    assert "do-not-echo-secret" not in json.dumps(agent.get_trace())


def test_corrected_patch_merges_into_preserved_proposal_after_failure():
    agent = agent_for("react", full(), RuntimeError("failed"),
        decision({"passenger_count": 1}, {"passenger_count": "1 vé"}))
    agent.send(FULL_TEXT); agent.send("bad follow-up"); agent.send(FIX_TEXT)
    p = agent.get_state()["pending_interpretation"]
    assert p["status"] == "READY" and p["snapshot"]["max_budget_vnd"] == 6_000_000
    agent.send("áp dụng")
    assert agent.harness.get_state().constraints.passenger_count == 1


def test_optional_omission_null_and_explicit_clear_preserve_correct_sources():
    agent = agent_for("react", full(), decision({"preferred_airlines": ["Vietnam Airlines"]},
        {"preferred_airlines": "ưu tiên Vietnam Airlines"}),
        decision({}, {"clear.max_budget_vnd": "bỏ budget"}, clear=["max_budget_vnd"]))
    agent.send(FULL_TEXT); agent.send("ưu tiên Vietnam Airlines")
    p = agent.get_state()["pending_interpretation"]
    assert p["snapshot"]["max_budget_vnd"] == 6_000_000 and p["snapshot"]["allowed_airlines"] == []
    old = p["field_sources"]["origin"]
    agent.send("bỏ budget")
    p = agent.get_state()["pending_interpretation"]
    assert p["snapshot"]["max_budget_vnd"] is None and p["field_sources"]["origin"] == old
    assert p["field_sources"]["max_budget_vnd"]["source_turn_id"].endswith("user-3")
    agent.send("áp dụng")
    assert agent.harness.get_state().constraints.max_budget_vnd is None


@pytest.mark.parametrize("change", ["basis", "display", "time"])
def test_stale_display_and_clock_changes_cannot_commit_and_do_not_destroy_proposal(change):
    agent = agent_for("react", full())
    agent.send(FULL_TEXT)
    original = agent.get_state()["pending_interpretation"]
    if change == "basis":
        agent._draft["passenger_count"] = 9
    elif change == "display":
        agent._pending_interpretation = agent._pending_interpretation.model_copy(update={"displayed_hash": "wrong"})
    else:
        agent.advance_time(6 * 86400)
    agent.send("áp dụng")
    p = agent.get_state()["pending_interpretation"]
    assert p["snapshot_hash"] == original["snapshot_hash"] and not p["apply_enabled"]
    assert agent.harness.get_state().constraints is None and not calls(agent)
    agent.send("/proposal keep")
    if change != "display":
        assert not agent.get_state()["pending_interpretation"]["apply_enabled"]


def test_commit_failure_and_public_copy_mutation_do_not_change_reviewed_snapshot(monkeypatch):
    agent = agent_for("react", full())
    agent.send(FULL_TEXT)
    state = agent.get_state()
    expected = deepcopy(state["pending_interpretation"])
    state["pending_interpretation"]["snapshot"]["passenger_count"] = 999
    state["pending_interpretation"]["field_sources"]["origin"]["source_turn_id"] = "forged"
    assert agent.get_state()["pending_interpretation"] == expected
    monkeypatch.setattr(agent.harness, "set_requirements", lambda *a: (_ for _ in ()).throw(ValueError("commit rejected")))
    agent.send("áp dụng")
    assert agent.get_state()["request_draft"] == {}
    assert agent.get_state()["pending_interpretation"]["snapshot_hash"] == expected["snapshot_hash"]
    assert not any(e["event_type"] == "REQUEST_PROPOSAL_APPLIED" for e in agent.get_trace())


@pytest.mark.parametrize("native", ["/request {broken", "chọn 1", "/passengers []"])
def test_native_failure_does_not_destroy_or_bypass_pending_request(native):
    agent = agent_for("react", full())
    agent.send(FULL_TEXT)
    old = agent.get_state()["pending_interpretation"]["snapshot_hash"]
    agent.send(native)
    p = agent.get_state()["pending_interpretation"]
    assert p["snapshot_hash"] == old and p["status"] == "NEEDS_REVIEW" and not calls(agent)
    agent.send("/proposal keep")
    assert agent.get_state()["pending_interpretation"]["apply_enabled"]


@pytest.mark.parametrize("kind", ["selection", "passenger", "cancel"])
def test_warn_is_search_only_and_never_relaxes_other_ingress(kind):
    value = {"selection": InputDecision(intent="SELECT_OFFER", selection_index=1,
        evidence=[{"field": "selection_index", "text": "2"}]),
        "passenger": InputDecision(intent="SUBMIT_PASSENGERS", passengers=[{"first_name": "Invented"}]),
        "cancel": InputDecision(intent="CANCEL", evidence=[{"field": "cancel", "text": "hủy"}])}[kind]
    agent = agent_for("react", value, mode="warn")
    agent.send(JSON_REQUEST)
    agent.send("a different current turn")
    assert agent.harness.get_state().selected_offer_id is None
    assert not agent.harness.get_state().passengers and not agent.harness.get_state().terminal_status
    assert agent.harness.get_state().input_blocked


def test_read_only_question_and_cli_show_preserve_locked_proposal_without_steps():
    agent = agent_for("react", full(), RuntimeError("bad"), InputDecision(intent="ASK_INFO", topic="REQUEST"))
    agent.send(FULL_TEXT); agent.send("invalid follow-up")
    before = agent.get_state()["pending_interpretation"]
    agent.send("yêu cầu hiện tại là gì?")
    assert agent.get_state()["pending_interpretation"] == before and not calls(agent)
    step = agent.harness.get_state().step_count
    messages = len(agent.get_state()["messages"])
    output = io.StringIO()
    run_cli(agent, io.StringIO("/proposal show\n/quit\n"), output)
    assert agent.harness.get_state().step_count == step and len(agent.get_state()["messages"]) == messages
    assert "NEEDS_REVIEW" in output.getvalue()


def test_warn_survives_export_and_reset_but_no_proposal_receipts_survive_reset(tmp_path):
    agent = agent_for("react", full("10 tháng 10 năm 2026"), mode="warn")
    agent.send(FULL_TEXT.replace("10/10/2026", "10 tháng 10 năm 2026"))
    directory = agent.export_artifacts(tmp_path / "review")
    state = json.loads((tmp_path / "review" / "conversation.json").read_text())
    assert directory and state["input_evidence_mode"] == "warn" and state["pending_interpretation"]["warnings"]
    agent.reset()
    state = agent.get_state()
    assert state["input_evidence_mode"] == "warn" and state["pending_interpretation"] is None
    assert state["accepted_request_revision"] == 0 and state["last_request_receipt"] is None


@pytest.mark.parametrize("pattern", PATTERNS)
def test_sdk_mock_transport_receives_history_and_merged_pending_context(pattern, gemini_model, gemini_response, gemini_prompt):
    seen, requests, decisions = [], [], [full("10 tháng 10 năm 2026"), decision({"passenger_count": 1}, {"passenger_count": "1 vé"})]
    def handler(request):
        prompt = gemini_prompt(request, requests)
        seen.append(prompt)
        if prompt.startswith("Interpret THIS user turn"):
            return httpx.Response(200, json=gemini_response(decisions.pop(0).model_dump_json()))
        if pattern == "react":
            result = OfflineBookingModel().invoke([SystemMessage(content=prompt)])
        elif pattern == "plan_execute":
            from app.agents.planner_model import OfflinePlannerModel
            result = OfflinePlannerModel().invoke([SystemMessage(content=prompt)])
        else:
            raise AssertionError("Hosted Hybrid does not invoke the legacy router")
        return httpx.Response(200, json=gemini_response(result))
    agent = create_agent(pattern, model=gemini_model(handler), input_review_config=InputReviewConfig(evidence_mode="warn"))
    agent.send(FULL_TEXT.replace("10/10/2026", "10 tháng 10 năm 2026")); agent.send(FIX_TEXT)
    assert len(seen) == 2 and '"max_budget_vnd": 6000000' in seen[1]
    assert len(requests[1]["contents"]) > len(requests[0]["contents"])
    p = agent.get_state()["pending_interpretation"]
    agent.send("áp dụng")
    assert payload_hash(agent.get_state()["harness"]["constraints"]) == p["snapshot_hash"]


def test_null_is_no_change_and_airline_clear_has_its_own_source():
    agent = agent_for("react", full(), decision({"allowed_airlines": ["Vietnam Airlines"]},
        {"allowed_airlines": "chỉ Vietnam Airlines"}),
        decision({"passenger_count": 1, "max_budget_vnd": None, "destination": None}, {"passenger_count": "1 vé"}),
        decision({}, {"clear.allowed_airlines": "bỏ giới hạn hãng"}, clear=["allowed_airlines"]))
    agent.send(FULL_TEXT); agent.send("chỉ Vietnam Airlines")
    before = agent.get_state()["pending_interpretation"]
    agent.send("1 vé budget y nguyên")
    p = agent.get_state()["pending_interpretation"]
    assert p["snapshot"]["max_budget_vnd"] == 6_000_000 and p["snapshot"]["destination"] == "HAN"
    assert p["field_sources"]["max_budget_vnd"] == before["field_sources"]["max_budget_vnd"]
    assert p["field_sources"]["destination"] == before["field_sources"]["destination"]
    agent.send("bỏ giới hạn hãng")
    p = agent.get_state()["pending_interpretation"]
    assert p["snapshot"]["allowed_airlines"] == []
    assert p["field_sources"]["allowed_airlines"]["source_turn_id"].endswith("user-4")
    assert p["field_sources"]["allowed_airlines"]["semantic_status"] == "NOT_APPLICABLE"


def test_inherited_warning_keeps_original_source_on_count_only_followup():
    agent = agent_for("react", full("10 tháng 10 năm 2026"),
        decision({"passenger_count": 1}, {"passenger_count": "1 vé"}), mode="warn")
    agent.send(FULL_TEXT.replace("10/10/2026", "10 tháng 10 năm 2026"))
    before = agent.get_state()["pending_interpretation"]
    agent.send(FIX_TEXT)
    p = agent.get_state()["pending_interpretation"]
    assert p["warnings"] == before["warnings"]
    assert p["field_sources"]["departure_date"] == before["field_sources"]["departure_date"]
    assert p["warnings"][0]["source_turn_id"].endswith("user-1")
    assert agent.harness.get_state().general_retry_count == 0


def test_extra_evidence_is_bounded_warning_and_duplicate_evidence_still_rejects():
    agent = agent_for("react", decision({"origin": "SGN"}, {"origin": "SGN", "arbitrary-secret-path": "SGN"}),
        {"intent": "UPDATE_SEARCH", "search_patch": {"passenger_count": 1}, "evidence": [
            {"field": "passenger_count", "text": "1"}, {"field": "passenger_count", "text": "1"}]}, mode="warn")
    agent.send("từ SGN")
    p = agent.get_state()["pending_interpretation"]
    assert p["warnings"] == [{"field": "evidence", "reason_code": "EVIDENCE_EXTRA_FIELD",
                               "source_turn_id": p["source_turn_id"]}]
    assert "arbitrary-secret-path" not in json.dumps(p["field_sources"])
    agent.send("1 vé")
    kept = agent.get_state()["pending_interpretation"]
    assert kept["status"] == "NEEDS_REVIEW" and kept["snapshot_hash"] == p["snapshot_hash"]


@pytest.mark.parametrize("pattern", PATTERNS)
def test_native_replacement_commits_only_after_validation_and_has_native_sources(pattern):
    agent = agent_for(pattern, full())
    agent.send(FULL_TEXT)
    agent.send(JSON_REQUEST)
    state = agent.get_state()
    assert state["pending_interpretation"] is None and state["harness"]["constraints"]["destination"] == "DAD"
    assert state["harness"]["constraints"]["passenger_count"] == 1
    assert state["last_request_receipt"] is None
    assert state["accepted_field_sources"]["destination"]["source_kind"] == "USER_COMMAND"
    assert state["accepted_field_sources"]["destination"]["source_turn_id"].endswith("user-2")
    assert state["accepted_field_sources"]["preferred_airlines"]["source_kind"] == "HOST_DEFAULT"


def test_invalid_native_replacement_keeps_accepted_data_sources_and_proposal():
    agent = agent_for("react", full())
    agent.send(JSON_REQUEST)
    accepted = agent.get_state()
    agent.send(FULL_TEXT)
    p = agent.get_state()["pending_interpretation"]
    agent.send(JSON_REQUEST.replace('"passenger_count":1', '"passenger_count":true'))
    state = agent.get_state()
    assert state["request_draft"] == accepted["request_draft"]
    assert state["accepted_field_sources"] == accepted["accepted_field_sources"]
    assert state["harness"]["constraints"] == accepted["harness"]["constraints"]
    assert state["pending_interpretation"]["snapshot_hash"] == p["snapshot_hash"]
    assert state["pending_interpretation"]["status"] == "NEEDS_REVIEW"


def test_native_commit_does_not_keep_old_review_receipt():
    agent = agent_for("react", full())
    agent.send(FULL_TEXT); agent.send("áp dụng")
    assert agent.get_state()["last_request_receipt"]
    agent.send(JSON_REQUEST)
    assert agent.get_state()["last_request_receipt"] is None


@pytest.mark.parametrize("pattern", PATTERNS)
def test_keep_never_resets_shared_budget_or_resurrects_terminal_session(pattern):
    agent = agent_for(pattern, full(), RuntimeError("bad"), RuntimeError("bad"), RuntimeError("bad"))
    agent.send(FULL_TEXT); agent.send("bad input")
    count = agent.harness.get_state().general_retry_count
    agent.send("/proposal keep")
    assert agent.harness.get_state().general_retry_count == count
    agent.send("bad input"); agent.send("bad input")
    before = agent.get_state()
    assert before["harness"]["terminal_status"] == "HANDOFF_REQUIRED"
    agent.send("/proposal keep"); agent.send("áp dụng"); agent.send("tiếp tục")
    after = agent.get_state()
    assert after["pending_interpretation"] == before["pending_interpretation"]
    assert after["harness"]["general_retry_count"] == before["harness"]["general_retry_count"]
    assert after["harness"]["terminal_status"] == "HANDOFF_REQUIRED" and not calls(agent)
    assert not after["request_draft"]


@pytest.mark.parametrize("mode", ["strict", "warn"])
@pytest.mark.parametrize("kind", ["change", "keep", "apply", "failure"])
def test_review_controls_after_hold_preserve_keys_and_never_start_another_transaction(mode, kind):
    from test_agents import ScriptedModel, action, snapshot
    value = RuntimeError("provider-secret") if kind == "failure" else decision({"passenger_count": 2})
    agent = agent_for("react", value, mode=mode)
    agent.send(JSON_REQUEST); agent.send("/select 1"); agent.send(PROFILE)
    agent._bound_model = ScriptedModel([
        lambda m: action("create_booking_hold", {"quote_id": snapshot(m)["harness"]["quote"]["quote_id"]}),
        AIMessage(content="pause after hold"),
    ])
    agent.send("đồng ý")
    before = agent.harness.get_state()
    count = len(calls(agent))
    assert before.booking_id and not agent.get_backend_state()["payments"]
    agent.send({"change": "đổi thành 2 người", "keep": "/proposal keep", "apply": "áp dụng",
                "failure": "thay yêu cầu"}[kind])
    after = agent.harness.get_state()
    assert after.booking_id == before.booking_id and after.operation_keys == before.operation_keys
    assert after.constraints == before.constraints and after.context == before.context
    assert len(calls(agent)) == count and len(agent.get_backend_state()["bookings"]) == 1
    assert not agent.get_backend_state()["payments"] and agent.get_state()["pending_interpretation"] is None
    if kind in {"change", "failure"}:
        assert after.terminal_status.value == "HANDOFF_REQUIRED"


@pytest.mark.parametrize("status", [400, 503])
def test_sdk_api_error_preserves_previous_warn_proposal_without_retry_or_apply(status, gemini_model, gemini_response):
    requests = []
    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, json=gemini_response(full().model_dump_json()))
        return httpx.Response(status, json={"error": {"code": status, "status": "INVALID_ARGUMENT" if status == 400 else "UNAVAILABLE",
                                                     "message": "Provider error"}})
    agent = create_agent("react", model=gemini_model(handler), input_review_config=InputReviewConfig(evidence_mode="warn"))
    agent.send(FULL_TEXT)
    p = agent.get_state()["pending_interpretation"]
    agent.send(FIX_TEXT)
    state = agent.get_state()
    assert state["pending_interpretation"]["snapshot_hash"] == p["snapshot_hash"]
    assert state["pending_interpretation"]["status"] == "NEEDS_REVIEW"
    assert state["last_model_error"]["http_status"] == status and state["last_model_error"]["evidence_mode"] == "warn"
    assert len(requests) == 2 and state["harness"]["general_retry_count"] == 1
    agent.send("áp dụng")
    assert not agent.get_state()["request_draft"] and not calls(agent) and len(requests) == 2


def test_review_config_is_strict_frozen_and_not_model_controlled():
    from pydantic import ValidationError
    assert InputReviewConfig().evidence_mode == "strict"
    with pytest.raises(ValidationError):
        InputReviewConfig(evidence_mode="skip")
    with pytest.raises(ValidationError):
        InputReviewConfig(evidence_mode="warn", disable_schema=True)
    config = InputReviewConfig(evidence_mode="warn")
    with pytest.raises(ValidationError):
        config.evidence_mode = "strict"
    with pytest.raises(ValidationError):
        create_agent("react", input_review_config={"evidence_mode": "skip"})


@pytest.mark.parametrize("kind", ["json", "truncated", "raw_calls", "coherence"])
def test_sdk_invalid_followup_keeps_last_valid_proposal_locked(kind, gemini_model, gemini_response):
    requests = []
    bad = {
        "json": gemini_response("not-json-private-response"),
        "truncated": gemini_response(full().model_dump_json(), finish="MAX_TOKENS"),
        "raw_calls": gemini_response(AIMessage(content="", tool_calls=[{
            "id": "unexpected-ingress-tool", "name": "search_flights", "args": {}}])),
        "coherence": gemini_response(json.dumps({"intent": "ASK_INFO", "topic": "REQUEST",
                                                 "search_patch": {"passenger_count": 1}})),
    }[kind]
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=gemini_response(full().model_dump_json()) if len(requests) == 1 else bad)
    agent = create_agent("react", model=gemini_model(handler), input_review_config=InputReviewConfig(evidence_mode="warn"))
    agent.send(FULL_TEXT)
    p = agent.get_state()["pending_interpretation"]
    agent.send(FIX_TEXT)
    state = agent.get_state()
    for key in ("snapshot", "snapshot_hash", "field_sources"):
        assert state["pending_interpretation"][key] == p[key]
    assert state["pending_interpretation"]["status"] == "NEEDS_REVIEW"
    assert not state["pending_interpretation"]["apply_enabled"]
    assert state["harness"]["general_retry_count"] == 1 and not calls(agent)
    agent.send("áp dụng")
    assert not agent.get_state()["request_draft"] and len(requests) == 2


def test_proposal_payload_is_immutable_even_through_internal_properties():
    from pydantic import ValidationError
    agent = agent_for("react", full())
    agent.send(FULL_TEXT)
    p = agent._pending_interpretation
    assert isinstance(p, RequestProposal)
    original = p.public()
    snapshot, sources, basis = p.snapshot, p.sources, p.basis
    snapshot["allowed_airlines"].append("Vietnam Airlines")
    sources["origin"]["source_turn_id"] = "forged"
    basis["draft"]["origin"] = "HAN"
    assert p.public() == original
    with pytest.raises(ValidationError):
        p.snapshot_json = "{}"
