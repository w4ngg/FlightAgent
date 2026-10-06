"""Runner/CLI acceptance code; do not execute until validation is requested."""

import csv
import json

import pytest

from app.agents.factory import create_pattern_model
from app.benchmark import main
from app.evaluation.benchmark_models import BenchmarkConfig, BenchmarkModelConfig
from app.evaluation.environment import ScenarioHarness
from app.evaluation.runner import ExperimentRunner


def forbidden(*args, **kwargs):
    pytest.fail("This path must not initialize a model, execute a batch or load credentials")


def test_default_plan_has_108_episodes_without_constructing_models():
    runner = ExperimentRunner(model_factory=forbidden)
    plan = runner.plan()
    assert plan["planned_episodes"] == 108 and len({e["episode_id"] for e in plan["episodes"]}) == 108
    assert plan["simulated"] and runner.results == []
    assert [e["agent_type"] for e in plan["episodes"][:9]] == [
        "react", "plan_execute", "hybrid", "plan_execute", "hybrid", "react", "hybrid", "react", "plan_execute"]


def test_all_patterns_are_fresh_and_export_incremental_reproducible_evidence(tmp_path):
    models = []
    def factory(pattern, settings):
        model = create_pattern_model(pattern, settings.provider)
        models.append(model)
        return model
    runner = ExperimentRunner(BenchmarkConfig(scenario_ids=["S01", "S02"], repetitions=2), model_factory=factory)
    directory = tmp_path / "batch"
    summary = runner.run_all(directory)
    assert summary["complete"] and summary["environment_comparable_for_recorded_pairings"]
    assert summary["all_recorded_scenarios_passed"] and summary["overall"]["task_completion_rate"] == 1
    assert summary["overall"]["denominators"]["task_completion_expected_booked"] == 6
    assert len(models) == len({id(m) for m in models}) == 12
    rows = [json.loads(line) for line in (directory / "benchmark_results.jsonl").read_text().splitlines()]
    with (directory / "benchmark_results.csv").open(newline="") as file:
        table = list(csv.DictReader(file))
    assert len(rows) == len(table) == len(runner.results) == 12
    assert json.loads((directory / "benchmark_summary.json").read_text()) == summary
    assert all(r["simulated"] and r["llm_calls"] == 0 and r["estimated_cost_usd"] is None for r in rows)
    assert all(r["estimated_cost_usd"] == "" for r in table)
    for row in rows:
        episode = directory / row["artifact_path"]
        assert json.loads((episode / "episode_result.json").read_text()) == row
        assert len(list((episode / "evidence").glob("*.json"))) == 11
        assert json.loads((episode / "evidence" / "initial_state.json").read_text())["bookings"] == {}
        trace = json.loads((episode / "evidence" / "trace.json").read_text())
        assert {e["context"]["session_id"] for e in trace} == {"benchmark-" + row["episode_id"]}


def test_existing_directory_is_preserved_before_model_factory(tmp_path):
    directory = tmp_path / "existing"
    directory.mkdir()
    marker = directory / "user.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        ExperimentRunner(model_factory=forbidden).run_all(directory)
    assert marker.read_text() == "keep" and list(directory.iterdir()) == [marker]


def test_setup_error_is_recorded_in_denominators_and_redacts_message(tmp_path):
    def factory(*args):
        raise RuntimeError("secret-provider-credential")
    runner = ExperimentRunner(BenchmarkConfig(patterns=["react"], scenario_ids=["S01"], repetitions=2), model_factory=factory)
    summary = runner.run_all(tmp_path / "failed")
    assert summary["complete"] and summary["recorded_episodes"] == 2
    assert summary["overall"]["task_completion_rate"] == summary["overall"]["scenario_pass_rate"] == 0
    assert summary["overall"]["permission_violation_rate"] is None
    assert all(r.status == "setup_error" and r.exception_type == "RuntimeError" for r in runner.results)
    assert "secret-provider-credential" not in json.dumps([r.model_dump(mode="json") for r in runner.results])


def test_fail_fast_preserves_partial_batch_without_silent_retry(tmp_path):
    counter = []
    def factory(*args):
        counter.append(1)
        raise RuntimeError("private")
    runner = ExperimentRunner(BenchmarkConfig(patterns=["react"], scenario_ids=["S01"], repetitions=3, fail_fast=True),
                              model_factory=factory)
    summary = runner.run_all(tmp_path / "partial")
    assert len(counter) == summary["recorded_episodes"] == 1
    assert summary["status"] == "stopped_on_failure" and not summary["complete"]


def test_driver_error_after_factory_keeps_partial_backend_trace(monkeypatch, tmp_path):
    def broken_driver(scenario, factory, *, session_id, on_turn_duration):
        agent = factory(ScenarioHarness(scenario, session_id=session_id))
        agent.send(scenario.initial_user_message)
        raise RuntimeError("secret-driver-error")
    monkeypatch.setattr("app.evaluation.runner.run_scenario", broken_driver)
    runner = ExperimentRunner(BenchmarkConfig(patterns=["react"], scenario_ids=["S01"], repetitions=1))
    runner.run_all(tmp_path / "driver")
    row = runner.results[0]
    episode = tmp_path / "driver" / row.artifact_path
    assert row.status == "driver_error" and not row.scenario_passed and row.tool_attempts is None
    assert json.loads((episode / "trace.json").read_text())
    assert json.loads((episode / "backend_state.json").read_text())["bookings"] == {}
    assert "secret-driver-error" not in (episode / "error.json").read_text()


def test_keyboard_interrupt_retains_current_episode_and_stops_batch(tmp_path):
    class InterruptedModel:
        def bind_tools(self, *args, **kwargs):
            return self
        def invoke(self, messages):
            raise KeyboardInterrupt
    runner = ExperimentRunner(BenchmarkConfig(patterns=["react"], scenario_ids=["S01"], repetitions=3),
        model_factory=lambda *args: InterruptedModel())
    summary = runner.run_all(tmp_path / "interrupt")
    assert summary["status"] == "interrupted" and summary["recorded_episodes"] == 1 and not summary["complete"]
    row = runner.results[0]
    assert row.status == "interrupted" and row.model_calls == 1 and row.model_measurements[0].exception_type == "KeyboardInterrupt"
    assert (tmp_path / "interrupt" / row.artifact_path / "backend_state.json").is_file()


def test_real_calls_need_explicit_authority_before_directory_or_factory(tmp_path):
    config = BenchmarkConfig(model=BenchmarkModelConfig(provider="gemini", model_name="mock-model"))
    runner = ExperimentRunner(config, model_factory=forbidden)
    with pytest.raises(ValueError, match="allow_paid_calls"):
        runner.run_all(tmp_path / "paid")
    assert not (tmp_path / "paid").exists()


def test_list_and_dry_run_never_load_env_initialize_models_or_write(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("app.benchmark.load_dotenv", forbidden)
    monkeypatch.setattr("app.evaluation.runner.create_pattern_model", forbidden)
    monkeypatch.setattr("app.evaluation.runner.ExperimentRunner.run_all", forbidden)
    monkeypatch.setenv("FLIGHT_AGENT_PROVIDER", "gemini")
    assert main(["--list"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 12
    assert main(["--dry-run", "--output-dir", str(tmp_path / "not-created")]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["simulated"] and plan["planned_episodes"] == 108 and not (tmp_path / "not-created").exists()
    assert main(["--dry-run", "--provider", "gemini", "--model", "explicit-model"]) == 0
    assert json.loads(capsys.readouterr().out)["config"]["model"]["model_name"] == "explicit-model"


@pytest.mark.parametrize("args", [["--dry-run", "--repetitions", "0"], ["--dry-run", "--patterns", "react", "react"],
    ["--dry-run", "--scenarios", "S99"], ["--dry-run", "--provider", "fake", "--model", "wrong"],
    ["--provider", "gemini", "--model", "mock", "--output-dir", "not-created"], []])
def test_invalid_cli_is_rejected_before_any_batch(monkeypatch, args):
    monkeypatch.setattr("app.evaluation.runner.ExperimentRunner.run_all", forbidden)
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2


def test_cli_benchmark_delegates_before_interactive_environment_setup(monkeypatch):
    from app.cli import main as cli_main
    monkeypatch.setattr("app.cli.load_dotenv", forbidden)
    seen = []
    monkeypatch.setattr("app.benchmark.main", lambda args: seen.append(args) or 17)
    assert cli_main(["--benchmark", "--dry-run"]) == 17 and seen == [["--dry-run"]]


def test_cli_exit_code_tracks_scenario_pass_not_booking_completion(tmp_path, capsys):
    assert main(["--patterns", "hybrid", "--scenarios", "S07", "--repetitions", "1",
                 "--output-dir", str(tmp_path / "declined")]) == 0
    output = capsys.readouterr()
    summary = json.loads(output.out)
    assert "SIMULATED" in output.err and summary["overall"]["task_completion_rate"] is None
    assert summary["overall"]["scenario_pass_rate"] == 1
