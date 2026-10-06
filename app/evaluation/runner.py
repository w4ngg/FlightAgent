"""Sequential, rotating-order experiment runner with incremental local evidence."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from app.agents.factory import create_agent, create_pattern_model
from app.agents.interpretation import INPUT_CONTRACT_VERSION
from app.agents.input_review import REVIEW_PROTOCOL_VERSION
from app.evaluation.benchmark_models import BenchmarkConfig, EpisodeResult
from app.evaluation.episode import run_scenario
from app.evaluation.metrics import EpisodeEvaluator, usage_metrics
from app.evaluation.scenarios import get_scenario
from app.evaluation.summary import summarize_benchmark
from app.evaluation.telemetry import MeasuredModel
from app.models import TerminalStatus, payload_hash


def write_new_json(path, value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    with Path(path).open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")


def _default_model_factory(pattern, settings):
    return create_pattern_model(pattern, settings.provider, settings.model_name, temperature=settings.temperature)


class ExperimentRunner:
    def __init__(self, config: BenchmarkConfig | None = None, *, model_factory=None):
        self.config = BenchmarkConfig.model_validate_json((config or BenchmarkConfig()).model_dump_json())
        self.model_factory = model_factory or _default_model_factory
        self.scenarios = {}
        for scenario_id in self.config.scenario_ids:
            scenario = get_scenario(scenario_id)
            if self.config.backend_seed is not None:
                scenario = type(scenario).model_validate({**scenario.model_dump(), "backend_seed": self.config.backend_seed})
            self.scenarios[scenario_id] = scenario
        self.results = []

    def plan(self):
        episodes = []
        for scenario_id in self.config.scenario_ids:
            for run_index in range(1, self.config.repetitions + 1):
                patterns = self.config.patterns
                shift = (run_index - 1) % len(patterns)
                for pattern in patterns[shift:] + patterns[:shift]:
                    episodes.append({"episode_id": f"{pattern}_{scenario_id}_run{run_index:03}", "agent_type": pattern,
                        "scenario_id": scenario_id, "run_index": run_index, "backend_seed": self.scenarios[scenario_id].backend_seed})
        return {"schema_version": 1, "config": self.config.model_dump(mode="json"), "planned_episodes": len(episodes),
            "scheduling": "scenario/run with deterministically rotating pattern order", "episodes": episodes,
            "scenarios": {key: s.model_dump(mode="json") for key, s in self.scenarios.items()},
            "scenario_hashes": {key: payload_hash(s.model_dump(mode="json")) for key, s in self.scenarios.items()},
            "input_contract": {"version": INPUT_CONTRACT_VERSION,
                               "review_protocol": REVIEW_PROTOCOL_VERSION,
                               "evidence_mode": self.config.input_review.evidence_mode,
                               "mode": "shared_llm_readback_apply" if self.config.model.provider == "gemini" else "offline_grammar",
                               "interpretation_is_quote_consent": False, "search_constraints_source": "harness_state"},
            "model_defaults": {"api": "gemini_developer_generate_content" if self.config.model.provider == "gemini" else None,
                               "timeout_seconds": 30, "provider_http_attempts": 1, "provider_max_retries": 1,
                               "temperature": self.config.model.temperature, "temperature_omitted_when_null": True},
            "simulated": self.config.model.provider == "fake"}

    def run_episode(self, agent_type, scenario, run_index, *, output_dir):
        if self.config.model.provider == "gemini" and not self.config.allow_paid_calls:
            raise ValueError("Gemini benchmark requires explicit allow_paid_calls=True")
        if agent_type not in self.config.patterns or scenario.id not in self.scenarios or scenario != self.scenarios[scenario.id]:
            raise ValueError("Episode must use the configured shared scenario and pattern")
        if type(run_index) is not int or not 1 <= run_index <= self.config.repetitions:
            raise ValueError("Run index is outside configured repetitions")
        episode_id = f"{agent_type}_{scenario.id}_run{run_index:03}"
        directory = Path(output_dir) / episode_id
        directory.mkdir(parents=True, exist_ok=False)  # Before any paid call, never overwrite.
        context, turns = {}, []
        started = perf_counter()
        measured, run, error, status = None, None, None, "evaluated"

        def factory(harness):
            context["harness"] = harness
            context["initial"] = harness.get_backend_state()
            nonlocal measured
            raw = self.model_factory(agent_type, self.config.model.model_copy(deep=True))
            if self.config.model.provider == "gemini" and (
                getattr(raw, "model", None) != self.config.model.model_name or
                self.config.model.temperature is not None and getattr(raw, "temperature", None) != self.config.model.temperature
            ):
                raise ValueError("Provider changed the requested shared model/temperature; no fallback allowed")
            measured = MeasuredModel(raw)
            agent = create_agent(agent_type, harness, model=measured, input_review_config=self.config.input_review)
            context["agent"] = agent
            return agent

        try:
            run = run_scenario(scenario, factory, session_id=f"benchmark-{episode_id}", on_turn_duration=turns.append)
        except KeyboardInterrupt:
            status, error = "interrupted", "KeyboardInterrupt"
        except Exception as exc:
            status, error = ("driver_error" if "agent" in context else "setup_error"), type(exc).__name__
        duration = max(0.0, (perf_counter() - started) * 1000)
        calls = measured.calls if measured is not None else []
        if run is not None:
            # Reuse the exact E exporter without letting it recreate our directory.
            evidence_dir = directory / "evidence"
            run.export_artifacts(evidence_dir)
            result = EpisodeEvaluator().evaluate(run, agent_type=agent_type, run_index=run_index,
                model=self.config.model, calls=calls, duration_ms=duration, agent_active_ms=sum(turns),
                pricing=self.config.pricing, artifact_path=f"episodes/{episode_id}")
        else:
            # Never reset/replay to make a broken episode look like an evaluated one.
            write_new_json(directory / "error.json", {"status": status, "exception_type": error})
            write_new_json(directory / "scenario.json", scenario)
            harness = context.get("harness")
            if harness is not None:
                for filename, reader in (("initial_state", lambda: context["initial"]), ("backend_state", harness.get_backend_state),
                    ("agent_state", harness.get_state), ("trace", harness.get_trace), ("handoff", harness.get_handoff)):
                    try:
                        value = reader()
                    except Exception as exc:
                        value = {"inspection_error_type": type(exc).__name__}
                    write_new_json(directory / f"{filename}.json", value)
            result = EpisodeResult(episode_id=episode_id, agent_type=agent_type, scenario_id=scenario.id, run_index=run_index,
                backend_seed=scenario.backend_seed, provider=self.config.model.provider, model_name=self.config.model.model_name,
                temperature=self.config.model.temperature, simulated=self.config.model.provider == "fake", status=status, exception_type=error,
                expected_booking=scenario.expectation.terminal_status == TerminalStatus.BOOKED,
                duration_ms=duration, agent_active_ms=sum(turns), artifact_path=f"episodes/{episode_id}",
                measurement_warnings=["EPISODE_NOT_EVALUATED"],
                **usage_metrics(calls, self.config.model, self.config.pricing))
        write_new_json(directory / "episode_result.json", result)
        return result

    def summarize(self, *, status="completed"):
        return summarize_benchmark(self.results, self.config, status=status)

    def run_all(self, output_dir):
        if self.results:
            raise ValueError("Use a new runner for a new batch; no automatic continuation/reset")
        if self.config.model.provider == "gemini" and not self.config.allow_paid_calls:
            raise ValueError("Gemini benchmark requires explicit allow_paid_calls=True")
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=False)
        manifest = {**self.plan(), "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "validation": "runtime output; implementation acceptance is not implied"}
        write_new_json(directory / "benchmark_manifest.json", manifest)
        fields = [name for name in EpisodeResult.model_fields if name not in
                  {"unsafe_findings", "recovery_outcomes", "model_measurements", "oracle_violations", "measurement_warnings"}]
        status = "completed"
        try:
            with (directory / "benchmark_results.csv").open("x", encoding="utf-8", newline="") as csv_file, (
                directory / "benchmark_results.jsonl").open("x", encoding="utf-8") as jsonl:
                writer = csv.DictWriter(csv_file, fieldnames=fields)
                writer.writeheader()
                csv_file.flush()
                for item in manifest["episodes"]:
                    result = self.run_episode(item["agent_type"], self.scenarios[item["scenario_id"]], item["run_index"],
                                              output_dir=directory / "episodes")
                    self.results.append(result.model_copy(deep=True))
                    data = result.model_dump(mode="json")
                    writer.writerow({field: data[field] for field in fields})  # null -> blank, not zero.
                    jsonl.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
                    csv_file.flush(), jsonl.flush()
                    if result.status == "interrupted":
                        status = "interrupted"
                        break
                    if self.config.fail_fast and not result.scenario_passed:
                        status = "stopped_on_failure"
                        break
        except KeyboardInterrupt:
            status = "interrupted"
        except BaseException:
            status = "failed"
            raise  # Stop before further paid calls if evidence cannot be saved.
        finally:
            write_new_json(directory / "benchmark_summary.json", self.summarize(status=status))
        return self.summarize(status=status)
