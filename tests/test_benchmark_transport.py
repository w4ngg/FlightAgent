"""H Gemini SDK acceptance code. All HTTP intercepted; not executed."""

import httpx
import pytest
from langchain_core.messages import SystemMessage

from app.agents.factory import create_pattern_model
from app.agents.interpretation import INPUT_CONTEXT_MARKER, InputDecision, SearchPatch, FieldEvidence
from app.evaluation.benchmark_models import BenchmarkConfig, BenchmarkModelConfig, TokenPricing
from app.evaluation.runner import ExperimentRunner


@pytest.mark.parametrize("pattern", ["react", "plan_execute", "hybrid"])
@pytest.mark.parametrize("scenario_id", ["S01", "S08"])
def test_runner_counts_sdk_invokes_and_native_usage_without_inventing_tier(
    pattern, scenario_id, tmp_path, gemini_model, gemini_response, gemini_prompt
):
    requests, offline = [], create_pattern_model(pattern, "fake")

    def handle(request):
        prompt = gemini_prompt(request, requests)
        if INPUT_CONTEXT_MARKER in prompt:
            # Shared ingress precedes each pattern; review/apply is driven by
            # scripted user policy, not an invented quote confirmation.
            message = InputDecision(intent="UPDATE_SEARCH", search_patch=SearchPatch(
                origin="SGN", destination="HAN", departure_date="2026-10-10", passenger_count=2, max_budget_vnd=6_000_000),
                evidence=[FieldEvidence(field=k, text=v) for k, v in {
                    "origin":"SGN", "destination":"HAN", "departure_date":"10/10/2026",
                    "passenger_count":"2", "max_budget_vnd":"6 triệu"}.items()]).model_dump_json()
        else:
            message = offline.invoke([SystemMessage(content=prompt)])
        return httpx.Response(200, json=gemini_response(message))

    config = BenchmarkConfig(patterns=[pattern], scenario_ids=[scenario_id], repetitions=1,
        model=BenchmarkModelConfig(provider="gemini", model_name="gemini-3-mock-model"), allow_paid_calls=True,
        pricing=TokenPricing(model_name="gemini-3-mock-model", input_usd_per_million="2", output_usd_per_million="8",
                             cached_input_usd_per_million=".5"))
    runner = ExperimentRunner(config, model_factory=lambda pattern, settings: gemini_model(handle, settings.model_name))
    summary = runner.run_all(tmp_path / "sdk")
    result = runner.results[0]
    assert summary["all_recorded_scenarios_passed"] and result.scenario_passed
    assert result.model_calls == result.llm_calls == result.usage_reported_calls == len(requests) > 0
    assert result.usage_complete and result.input_tokens == 100 * len(requests)
    assert result.output_tokens == 30 * len(requests)  # Includes 5 thinking tokens.
    assert result.cached_input_tokens == 20 * len(requests) and result.reasoning_tokens == 5 * len(requests)
    assert result.cache_write_tokens is None
    # Pinned LangChain Gemini response metadata does not report billed tier.
    assert result.estimated_cost_usd is None and result.costed_calls == 0
    assert all(c.served_model == "gemini-3-mock-model" and c.service_tier is None for c in result.model_measurements)
    assert result.recovery_opportunities == (1 if scenario_id == "S08" else 0)
    if pattern == "react":
        assert all(sum(len(t["functionDeclarations"]) for t in p.get("tools", [])) == 8 for p in requests if p.get("tools"))
    else:
        assert all(not p.get("tools") and p["generationConfig"]["responseMimeType"] == "application/json" for p in requests)


def test_factory_forwards_temperature_without_changing_sdk_defaults(monkeypatch):
    from app.agents.models import create_model
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-fake-key")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "false")
    model = create_model("gemini", "gemini-3-mock-model", temperature=0)
    try:
        assert model.temperature == 0 and model.model == "gemini-3-mock-model"
        assert model.vertexai is False and model.max_retries == 1 and model.timeout == 30
    finally:
        model.client.close()


def test_runner_rejects_model_or_temperature_fallback_without_any_invoke(tmp_path):
    class ChangedModel:
        model = "other-model"
        temperature = 1
        def bind_tools(self, *args, **kwargs):
            pytest.fail("Fallback must be rejected before binding or invoking")
    config = BenchmarkConfig(patterns=["react"], scenario_ids=["S01"], repetitions=1,
        model=BenchmarkModelConfig(provider="gemini", model_name="requested-model", temperature=0), allow_paid_calls=True)
    runner = ExperimentRunner(config, model_factory=lambda *args: ChangedModel())
    runner.run_all(tmp_path / "fallback")
    assert runner.results[0].status == "setup_error" and runner.results[0].llm_calls == 0
