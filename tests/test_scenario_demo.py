import json
import subprocess
import sys

import pytest
from langchain_core.messages import AIMessage

from app.agents.react_agent import ReActFlightAgent
from app.evaluation.episode import run_scenario
from app.evaluation.scenarios import get_scenario
from app.scenario_demo import main


def test_list_and_show_catalog_do_not_execute_agent(monkeypatch, capsys):
    monkeypatch.setattr("app.scenario_demo.run_scenario", lambda *a, **k: pytest.fail("Listing must not invoke an agent"))
    assert main(["--list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 12 and lines[0].startswith("S01:") and lines[-1].startswith("S12:")
    assert "WAITING_INPUT" in lines[1] and "FAILED" in lines[6]
    assert main(["--show", "S08"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["failure_rules"][0]["effect"] == "TIMEOUT_BUT_COMMITTED"
    assert data["user_policy"]["confirmation_text"] == "đồng ý"


@pytest.mark.parametrize("scenario_id,terminal,completed", [
    ("S01", "BOOKED", True), ("S02", None, False), ("S04", "BOOKED", True),
    ("S07", "FAILED", False), ("S08", "BOOKED", True), ("S10", None, False), ("S11", "BOOKED", True),
])
def test_module_demo_exports_offline_evidence_and_returns_scenario_result(scenario_id, terminal, completed, tmp_path):
    output_dir = tmp_path / scenario_id
    result = subprocess.run([sys.executable, "-m", "app.scenario_demo", "--scenario", scenario_id,
                             "--output-dir", str(output_dir)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "NOT an LLM benchmark" in result.stdout
    report = json.loads((output_dir / "scenario_result.json").read_text())
    assert report["scenario_passed"] and report["booking_completed"] is completed
    assert report["observed_terminal"] == terminal
    assert len(list(output_dir.glob("*.json"))) == 11
    assert json.loads((output_dir / "trace.json").read_text())
    assert json.loads((output_dir / "scripted_user.json").read_text())


def test_demo_never_overwrites_existing_evidence_directory(tmp_path, capsys):
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    marker = output_dir / "marker.json"
    marker.write_text("user evidence")
    assert main(["--scenario", "S02", "--output-dir", str(output_dir)]) == 1
    assert "exist" in capsys.readouterr().err.lower()
    assert marker.read_text() == "user evidence" and len(list(output_dir.iterdir())) == 1


def test_unknown_scenario_is_rejected_before_any_episode(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--scenario", "S99"])
    assert error.value.code == 2 and "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize("after_ingress", [False, True])
def test_driver_preserves_evidence_of_adapter_error_without_forcing_outcome(after_ingress):
    class BrokenAdapter(ReActFlightAgent):
        def send(self, message, session_id=None):
            if after_ingress:
                super().send(message, session_id=session_id)
            raise RuntimeError("private provider diagnostic")
    run = run_scenario(get_scenario("S01"), BrokenAdapter)
    assert run.evidence.stop_reason == "AGENT_ERROR" and not run.result.scenario_passed
    assert len(run.evidence.checkpoints) == int(after_ingress)
    assert run.evidence.final_state.terminal_status is None and run.evidence.handoff is None
    assert "private provider diagnostic" not in run.model_dump_json()


def test_stalled_agent_claim_does_not_reset_or_force_success():
    class ClaimOnlyModel:
        def bind_tools(self, tools, **kwargs):
            return self
        def invoke(self, messages):
            return AIMessage(content="BOOKED, I paid and issued tickets!")
    run = run_scenario(get_scenario("S01"), lambda h: ReActFlightAgent(h, model=ClaimOnlyModel()))
    assert run.evidence.stop_reason == "USER_TURN_LIMIT" and not run.result.scenario_passed
    assert not run.result.booking_completed and not run.evidence.final_backend_state["bookings"]
    assert len(run.evidence.checkpoints) == run.scenario.max_user_turns
    assert run.evidence.final_state.terminal_status is None and run.evidence.handoff is None


def test_factory_must_return_common_adapter():
    with pytest.raises(TypeError, match="BaseFlightAgent"):
        run_scenario(get_scenario("S01"), lambda h: object())
