"""H acceptance cases. Added code-only; execution is intentionally deferred."""

from decimal import Decimal

import pytest
from langchain_core.messages import AIMessage

from app.agents.react_agent import ReActFlightAgent
from app.evaluation.benchmark_models import BenchmarkConfig, BenchmarkModelConfig, ModelCallMeasurement, TokenPricing
from app.evaluation.episode import run_scenario
from app.evaluation.metrics import EpisodeEvaluator, estimate_call_cost, usage_metrics
from app.evaluation.oracle import events
from app.evaluation.scenarios import get_scenario, list_scenarios
from app.evaluation.summary import distribution, summarize_group
from app.evaluation.telemetry import MeasuredModel, measurement_fields
from app.evaluation.transaction_audit import audit_transactions


@pytest.fixture(scope="module")
def catalog_runs():
    return {s.id: run_scenario(s, lambda h: ReActFlightAgent(h)) for s in list_scenarios()}


def evaluate(run):
    return EpisodeEvaluator().evaluate(run, agent_type="react", run_index=1,
        model=BenchmarkModelConfig(), calls=[], duration_ms=0, agent_active_ms=0)


@pytest.mark.parametrize("scenario_id", [f"S{i:02}" for i in range(1, 13)])
def test_catalog_metrics_keep_scenario_and_booking_success_separate(catalog_runs, scenario_id):
    run, result = catalog_runs[scenario_id], evaluate(catalog_runs[scenario_id])
    assert result.scenario_passed and result.constraint_passed and result.backend_consistent
    assert result.booking_completed == (scenario_id not in {"S02", "S07", "S10"})
    assert result.permission_violations == result.illegal_side_effect_attempts == result.unsafe_transactions == 0
    assert result.tool_attempts == len(events(run.evidence.trace, "TOOL_CALL"))
    assert result.executed_tool_calls == len(events(run.evidence.trace, "TOOL_ALLOWED"))
    assert result.excess_tool_calls == 0
    assert result.llm_calls == 0 and result.input_tokens is None and result.estimated_cost_usd is None


def test_expected_checkpoint_and_decline_do_not_depress_task_completion(catalog_runs):
    summary = summarize_group([evaluate(catalog_runs[s]) for s in ("S01", "S02", "S07", "S10")])
    assert summary["scenario_pass_rate"] == summary["task_completion_rate"] == 1
    assert summary["denominators"]["task_completion_expected_booked"] == 1
    assert summary["denominators"]["scenario_pass_rate"] == 4
    assert summary["recovery_rate"] is None


@pytest.mark.parametrize("scenario_id", ["S04", "S05", "S06", "S08", "S12"])
def test_only_observed_recoverable_opportunities_have_a_denominator(catalog_runs, scenario_id):
    result = evaluate(catalog_runs[scenario_id])
    assert result.recovery_opportunities == result.recovered_failures == 1 and result.recovery_success
    assert summarize_group([result])["recovery_rate"] == 1
    assert result.recovery_outcomes[0]["source"] == ("scripted_user_correction" if scenario_id == "S12" else "injected_failure")


def test_replay_counts_attempts_but_not_duplicate_commits(catalog_runs):
    result = evaluate(catalog_runs["S11"])
    assert result.tool_attempts == 7 and result.transport_replays == 1
    assert result.committed_side_effects == 3 and result.booking_count == result.successful_payment_count == 1
    assert result.ticket_count == 2 and not result.unsafe_findings
    assert evaluate(catalog_runs["S07"]).committed_side_effects == 2  # A declined operation is still recorded.


def test_evaluator_recomputes_evidence_instead_of_trusting_cached_pass(catalog_runs):
    broken = catalog_runs["S01"].model_copy(deep=True)
    broken.evidence.final_backend_state["tickets"].clear()
    assert broken.result.scenario_passed  # The cached flag is deliberately stale.
    result = evaluate(broken)
    assert not result.scenario_passed and not result.booking_completed
    assert result.expected_minimum_tool_calls is None and result.excess_tool_calls is None
    assert result.unsafe_transactions is None and result.measurement_warnings


def test_missing_trace_is_unknown_not_zero(catalog_runs):
    broken = catalog_runs["S01"].model_copy(deep=True)
    broken.evidence.trace.clear()
    result = evaluate(broken)
    assert not result.scenario_passed and result.tool_attempts is None and result.permission_violations is None
    summary = summarize_group([result])
    assert summary["permission_violation_rate"] is None and summary["unsafe_transaction_count"] is None


def test_audit_counts_distinct_committed_actions_once_across_checkpoints(catalog_runs):
    evidence = catalog_runs["S01"].evidence.model_copy(deep=True)
    booking = next(iter(evidence.final_backend_state["bookings"].values()))
    booking["confirmation"]["snapshot"]["total_price_vnd"] += 1
    findings = audit_transactions(evidence)
    assert {f.kind for f in findings} == {"hold", "payment", "ticket_batch"}
    assert len({(f.kind, f.reference) for f in findings}) == len(findings) == 3


def test_guard_denial_is_not_an_unsafe_commit():
    class IllegalModel:
        def bind_tools(self, tools, **kwargs):
            return self
        def invoke(self, messages):
            return AIMessage(content="", tool_calls=[{"name": "process_payment", "args": {
                "booking_id": "invented", "amount_vnd": 1}, "id": "bad-call", "type": "tool_call"}])
    run = run_scenario(get_scenario("S01"), lambda h: ReActFlightAgent(h, model=IllegalModel()))
    result = evaluate(run)
    assert result.illegal_side_effect_attempts > 0 and result.permission_violations > 0
    assert result.committed_side_effects == result.unsafe_transactions == 0
    assert summarize_group([result])["permission_violation_rate"] == result.illegal_side_effect_attempts


def message_with_usage():
    return AIMessage(content="", response_metadata={"model_name": "mock-model", "service_tier": "standard"},
        usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
            "input_token_details": {"cache_read": 40}, "output_token_details": {"reasoning": 3}})


def measurement(**overrides):
    return ModelCallMeasurement(call_index=1, duration_ms=2, returned=True,
                                **{**measurement_fields(message_with_usage()), **overrides})


def pricing(**overrides):
    return TokenPricing(**{"model_name": "mock-model", "input_usd_per_million": "2", "output_usd_per_million": "8",
        "cached_input_usd_per_million": ".5", **overrides})


def test_token_cost_partitions_input_and_never_double_counts_reasoning():
    call = measurement()
    assert estimate_call_cost(call, pricing()) == Decimal("0.000300")
    assert estimate_call_cost(call, pricing(service_tier="flex")) is None
    assert estimate_call_cost(call, None) is None
    assert estimate_call_cost(measurement(cached_input_tokens=None), pricing()) is None
    assert estimate_call_cost(measurement(cache_write_tokens=10), pricing()) is None
    assert estimate_call_cost(measurement(service_tier=None), pricing()) is None


def test_incomplete_provider_usage_and_fake_tokens_are_not_invented():
    model = BenchmarkModelConfig(provider="gemini", model_name="mock-model")
    failed = ModelCallMeasurement(call_index=2, duration_ms=4, returned=False, exception_type="TimeoutError")
    result = usage_metrics([measurement(), failed], model, pricing())
    assert result["model_calls"] == result["llm_calls"] == 2
    assert not result["usage_complete"] and result["input_tokens"] is None and result["estimated_cost_usd"] is None
    assert result["observed_input_tokens"] == 100 and result["observed_estimated_cost_usd"] == Decimal("0.000300")
    assert usage_metrics([], model, None)["estimated_cost_usd"] == 0  # No invoke, known zero.
    fake = usage_metrics([measurement()], BenchmarkModelConfig(), None)
    assert fake["model_calls"] == 1 and fake["llm_calls"] == 0 and fake["input_tokens"] is None


def test_measurement_wraps_actual_structured_invoke_even_with_a_parse_error():
    raw = message_with_usage()
    packet = {"raw": raw, "parsed": None, "parsing_error": ValueError("not logged")}
    class Runnable:
        def with_structured_output(self, *args, **kwargs):
            return self
        def invoke(self, messages):
            return packet
    measured = MeasuredModel(Runnable())
    assert measured.with_structured_output(object, include_raw=True).invoke([]) is packet
    assert measured.calls[0].input_tokens == 100 and measured.calls[0].duration_ms >= 0
    detached = measured.calls
    detached.clear()
    assert len(measured.calls) == 1


def test_provider_exceptions_are_counted_and_redacted_without_retry():
    class Runnable:
        def bind_tools(self, *args, **kwargs):
            return self
        def invoke(self, messages):
            raise RuntimeError("secret provider message")
    measured = MeasuredModel(Runnable())
    with pytest.raises(RuntimeError):
        measured.bind_tools([]).invoke([])
    assert len(measured.calls) == 1 and measured.calls[0].exception_type == "RuntimeError"
    assert not measured.calls[0].returned and "secret provider message" not in measured.calls[0].model_dump_json()


def test_inconsistent_usage_is_unknown_and_distribution_handles_missing_samples():
    raw = message_with_usage()
    raw.usage_metadata["total_tokens"] = 999
    assert "input_tokens" not in measurement_fields(raw)
    assert distribution([None])["count"] == 0
    stats = distribution([1, None, 3])
    assert stats["count"] == 2 and stats["mean"] == stats["p50"] == 2 and stats["stdev"] > 0


@pytest.mark.parametrize("kwargs", [{"repetitions": 0}, {"repetitions": True}, {"patterns": ["react", "react"]},
    {"scenario_ids": ["S01", "S01"]}, {"pricing": pricing()}])
def test_invalid_benchmark_contracts_fail_before_any_episode(kwargs):
    with pytest.raises(ValueError):
        BenchmarkConfig(**kwargs)
