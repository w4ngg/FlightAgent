import json
import subprocess
import sys

import pytest

from app.agents.adapter import BaseFlightAgent
from app.agents.factory import create_agent, create_pattern_model
from app.agents.plan_execute_agent import PlanExecuteFlightAgent
from app.agents.planner_model import OfflinePlannerModel
from app.cli import main as cli_main
from app.evaluation.episode import run_scenario
from app.evaluation.scenarios import get_scenario
from app.harness.runtime import FlightBookingHarness
from app.scenario_demo import main as scenario_main


INPUT = ("Đặt 2 vé SGN đến HAN ngày 10/10/2026\nchọn 1\n"
    "hành khách: Nguyen|An|1990-01-01|IDENTITY_CARD|MOCK-0001; Tran|Binh|1992-02-02|PASSPORT|MOCK-0002\nđồng ý\n")


@pytest.mark.parametrize("pattern", ["react", "plan_execute"])
def test_common_factory_preserves_harness_and_interchangeable_adapter_contract(pattern):
    harness = FlightBookingHarness(session_id="factory-session")
    agent = create_agent(pattern, harness, model=create_pattern_model(pattern, "fake"))
    assert isinstance(agent, BaseFlightAgent) and agent.name == pattern and agent.harness is harness
    assert agent.get_state()["harness"]["context"]["session_id"] == "factory-session"
    if pattern == "plan_execute":
        assert isinstance(agent.model, OfflinePlannerModel)
    with pytest.raises(ValueError):
        create_agent("unknown", harness)
    with pytest.raises(ValueError):
        create_pattern_model("unknown")


def test_each_pattern_receives_identical_scenario_environment_and_policy_contract():
    scenario = get_scenario("S04")
    runs = [run_scenario(scenario, lambda h, pattern=p: create_agent(pattern, h)) for p in ("react", "plan_execute")]
    assert runs[0].evidence.initial_backend_state == runs[1].evidence.initial_backend_state
    assert all(run.result.scenario_passed for run in runs)
    assert all(run.scenario == scenario for run in runs)
    assert all(run.evidence.final_state.permission_violations == 0 for run in runs)


@pytest.mark.parametrize("mode,terminal", [("success", "BOOKED"), ("timeout", "HANDOFF_REQUIRED"),
                                          ("timeout_but_success", "BOOKED"), ("declined", "FAILED")])
def test_interactive_plan_execute_cli_selects_pattern_and_exports_plan_state(mode, terminal, tmp_path):
    directory = tmp_path / mode
    result = subprocess.run([sys.executable, "-m", "app.cli", "--pattern", "plan_execute", "--provider", "fake",
        "--payment-mode", mode, "--output-dir", str(directory)], input=INPUT + "/state\n/trace\n/quit\n",
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "plan_execute" in result.stdout and terminal in result.stdout and "KHÔNG phải LLM" in result.stdout
    assert "PLAN_CREATED" in result.stdout and '"current_step"' in result.stdout
    public = json.loads((directory / "conversation.json").read_text())
    assert public["pattern"] == "plan_execute" and public["plan"]["steps"]
    assert public["harness"]["terminal_status"] == terminal
    assert len(list(directory.glob("*.json"))) == 7
    assert json.loads((directory / "completion.json").read_text())["booking_completed"] is (terminal == "BOOKED")


@pytest.mark.parametrize("scenario_id,completed", [("S01", True), ("S02", False), ("S04", True), ("S05", True),
                                                 ("S07", False), ("S08", True), ("S09", True), ("S10", False), ("S11", True)])
def test_single_scenario_demo_pattern_selection_and_eleven_file_export(scenario_id, completed, tmp_path):
    directory = tmp_path / scenario_id
    result = subprocess.run([sys.executable, "-m", "app.scenario_demo", "--pattern", "plan_execute",
        "--scenario", scenario_id, "--output-dir", str(directory)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "Offline plan_execute" in result.stdout and "NOT an LLM benchmark" in result.stdout
    report = json.loads((directory / "scenario_result.json").read_text())
    assert report["scenario_passed"] and report["booking_completed"] is completed
    assert len(list(directory.glob("*.json"))) == 11
    plan = json.loads((directory / "conversation.json").read_text())["plan"]
    assert plan is None if scenario_id == "S02" else plan["steps"]
    initial_plan = json.loads((directory / "checkpoints.json").read_text())[0]["agent_state"]["plan"]
    assert initial_plan is None if scenario_id == "S02" else initial_plan
    assert json.loads((directory / "backend_state.json").read_text())["seed"] == 42


@pytest.mark.parametrize("entry", [cli_main, scenario_main])
def test_live_provider_configuration_missing_key_fails_without_api_or_key_echo(entry, monkeypatch, capsys):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("app.cli.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    with pytest.raises(SystemExit) as error:
        entry(["--pattern", "plan_execute", "--provider", "gemini"])
    assert error.value.code == 2 and "GEMINI_API_KEY" in capsys.readouterr().err


def test_listing_with_live_flags_never_loads_credentials_or_constructs_provider(monkeypatch, capsys):
    monkeypatch.setattr("app.scenario_demo.create_pattern_model", lambda *a, **k: pytest.fail("Listing cannot initialize provider"))
    assert scenario_main(["--list", "--pattern", "plan_execute", "--provider", "gemini"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 12


def test_default_scenario_provider_remains_fake_even_with_live_environment(monkeypatch, capsys):
    monkeypatch.setenv("FLIGHT_AGENT_PROVIDER", "gemini")
    assert scenario_main(["--scenario", "S02", "--pattern", "plan_execute"]) == 0
    assert "Offline plan_execute" in capsys.readouterr().out


def test_cli_initializes_selected_planner_model_and_shared_adapter(monkeypatch):
    captured = []
    def fake_factory(provider, model_name):
        captured.append((provider, model_name))
        return OfflinePlannerModel()
    monkeypatch.setattr("app.cli.create_planner_model", fake_factory)
    monkeypatch.setattr("app.cli.run_cli", lambda agent: captured.append(agent))
    assert cli_main(["--pattern", "plan_execute", "--provider", "gemini", "--model", "user-choice"]) == 0
    assert captured[0] == ("gemini", "user-choice")
    assert isinstance(captured[1], PlanExecuteFlightAgent)


def test_scenario_demo_respects_provider_model_flags_without_real_api(monkeypatch, capsys):
    captured = []
    def fake_factory(pattern, provider, model_name):
        captured.append((pattern, provider, model_name))
        return OfflinePlannerModel()
    monkeypatch.setattr("app.scenario_demo.create_pattern_model", fake_factory)
    assert scenario_main(["--scenario", "S01", "--pattern", "plan_execute", "--provider", "gemini", "--model", "user-choice"]) == 0
    assert captured == [("plan_execute", "gemini", "user-choice")]
    text = capsys.readouterr().out
    assert "real API calls/cost" in text and '"scenario_passed": true' in text


@pytest.mark.parametrize("entry", [cli_main, scenario_main])
def test_unknown_pattern_is_rejected_not_silently_replaced(entry, capsys):
    with pytest.raises(SystemExit) as error:
        entry(["--pattern", "unknown"])
    assert error.value.code == 2 and "invalid choice" in capsys.readouterr().err


def test_plan_export_refuses_overwriting_existing_directory(tmp_path):
    directory = tmp_path / "existing"
    directory.mkdir()
    marker = directory / "marker.json"
    marker.write_text("original user evidence")
    agent = PlanExecuteFlightAgent()
    agent.send(INPUT.splitlines()[0])
    with pytest.raises(FileExistsError):
        agent.export_artifacts(directory)
    assert marker.read_text() == "original user evidence" and len(list(directory.iterdir())) == 1
