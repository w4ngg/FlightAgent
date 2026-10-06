import json
import subprocess
import sys

import pytest

from app.agents.adapter import BaseFlightAgent
from app.agents.factory import create_agent, create_pattern_model
from app.agents.hybrid_agent import HybridFlightAgent
from app.agents.router_model import OfflineRouterModel
from app.cli import main as cli_main
from app.evaluation.episode import run_scenario
from app.evaluation.scenarios import get_scenario
from app.harness.runtime import FlightBookingHarness
from app.scenario_demo import main as scenario_main


INPUT = ("Đặt 2 vé SGN đến HAN ngày 10/10/2026\nchọn 1\n"
    "hành khách: Nguyen|An|1990-01-01|IDENTITY_CARD|MOCK-0001; Tran|Binh|1992-02-02|PASSPORT|MOCK-0002\nđồng ý\n")


@pytest.mark.parametrize("pattern", ["react", "plan_execute", "hybrid"])
def test_factory_all_three_patterns_share_contract_harness_and_identical_environment(pattern):
    harness = FlightBookingHarness(session_id="factory-session")
    agent = create_agent(pattern, harness, model=create_pattern_model(pattern))
    assert isinstance(agent, BaseFlightAgent) and agent.harness is harness and agent.name == pattern
    if pattern == "hybrid":
        assert isinstance(agent, HybridFlightAgent) and isinstance(agent.model, OfflineRouterModel)
    run = run_scenario(get_scenario("S04"), lambda h: create_agent(pattern, h))
    reference = run_scenario(get_scenario("S04"), lambda h: create_agent("react", h))
    assert run.evidence.initial_backend_state == reference.evidence.initial_backend_state
    assert run.scenario == reference.scenario and run.result.scenario_passed
    assert run.evidence.final_state.permission_violations == 0


@pytest.mark.parametrize("mode,terminal", [("success", "BOOKED"), ("timeout", "HANDOFF_REQUIRED"),
                                          ("timeout_but_success", "BOOKED"), ("declined", "FAILED")])
def test_hybrid_interactive_cli_exports_observed_node_state_and_verified_outcome(mode, terminal, tmp_path):
    directory = tmp_path / mode
    result = subprocess.run([sys.executable, "-m", "app.cli", "--pattern", "hybrid", "--provider", "fake",
        "--payment-mode", mode, "--output-dir", str(directory)], input=INPUT + "/state\n/trace\n/quit\n",
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "hybrid" in result.stdout and terminal in result.stdout and "KHÔNG phải LLM" in result.stdout
    assert "HYBRID_NODE" in result.stdout and '"pending_input"' in result.stdout
    state = json.loads((directory / "conversation.json").read_text())
    assert state["pattern"] == "hybrid" and state["harness"]["terminal_status"] == terminal
    assert len(list(directory.glob("*.json"))) == 7
    assert json.loads((directory / "completion.json").read_text())["booking_completed"] is (terminal == "BOOKED")


@pytest.mark.parametrize("scenario_id,completed", [("S01", True), ("S02", False), ("S04", True),
    ("S07", False), ("S08", True), ("S10", False), ("S11", True), ("S12", True)])
def test_hybrid_scenario_cli_exports_eleven_files_and_checkpoint_evidence(scenario_id, completed, tmp_path):
    directory = tmp_path / scenario_id
    result = subprocess.run([sys.executable, "-m", "app.scenario_demo", "--pattern", "hybrid", "--scenario", scenario_id,
        "--output-dir", str(directory)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "Offline hybrid" in result.stdout and "NOT an LLM benchmark" in result.stdout
    report = json.loads((directory / "scenario_result.json").read_text())
    assert report["scenario_passed"] and report["booking_completed"] is completed
    assert len(list(directory.glob("*.json"))) == 11
    assert json.loads((directory / "conversation.json").read_text())["pattern"] == "hybrid"
    assert "node" in json.loads((directory / "checkpoints.json").read_text())[0]["agent_state"]


@pytest.mark.parametrize("entry", [cli_main, scenario_main])
def test_hybrid_live_missing_credentials_fails_without_network_or_echo(entry, monkeypatch, capsys):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("app.cli.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    with pytest.raises(SystemExit) as error:
        entry(["--pattern", "hybrid", "--provider", "gemini"])
    assert error.value.code == 2 and "GEMINI_API_KEY" in capsys.readouterr().err


def test_live_listing_does_not_initialize_model_and_scenario_default_ignores_live_env(monkeypatch, capsys):
    monkeypatch.setenv("FLIGHT_AGENT_PROVIDER", "gemini")
    assert scenario_main(["--scenario", "S02", "--pattern", "hybrid"]) == 0
    assert "Offline hybrid" in capsys.readouterr().out
    monkeypatch.setattr("app.scenario_demo.create_pattern_model", lambda *a, **k: pytest.fail("No model initialization for listing"))
    assert scenario_main(["--list", "--pattern", "hybrid", "--provider", "gemini"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 12


def test_cli_selects_correct_router_factory_preserving_provider_and_explicit_model(monkeypatch):
    captured = []
    def fake_factory(provider, model_name):
        captured.append((provider, model_name))
        return OfflineRouterModel()
    monkeypatch.setattr("app.cli.create_router_model", fake_factory)
    monkeypatch.setattr("app.cli.run_cli", lambda agent: captured.append(agent))
    assert cli_main(["--pattern", "hybrid", "--provider", "gemini", "--model", "user-choice"]) == 0
    assert captured[0] == ("gemini", "user-choice") and isinstance(captured[1], HybridFlightAgent)


def test_demo_selects_hybrid_provider_flags_without_live_api(monkeypatch, capsys):
    captured = []
    def fake_factory(pattern, provider, model_name):
        captured.append((pattern, provider, model_name))
        return OfflineRouterModel()
    monkeypatch.setattr("app.scenario_demo.create_pattern_model", fake_factory)
    assert scenario_main(["--scenario", "S01", "--pattern", "hybrid", "--provider", "gemini", "--model", "user-choice"]) == 0
    assert captured == [("hybrid", "gemini", "user-choice")]
    assert "real API calls/cost" in capsys.readouterr().out


def test_hybrid_export_refuses_existing_directory_and_preserves_user_evidence(tmp_path):
    directory = tmp_path / "existing"
    directory.mkdir()
    marker = directory / "marker.json"
    marker.write_text("original user evidence")
    agent = HybridFlightAgent()
    agent.send(INPUT.splitlines()[0])
    with pytest.raises(FileExistsError):
        agent.export_artifacts(directory)
    assert marker.read_text() == "original user evidence" and len(list(directory.iterdir())) == 1
