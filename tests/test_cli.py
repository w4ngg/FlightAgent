import io
import json
import subprocess
import sys

import pytest

from app.agents.react_agent import ReActFlightAgent
from app.cli import main, run_cli


INPUT = ("Đặt 2 vé SGN đến HAN ngày 10/10/2026\nchọn 1\n"
    "hành khách: Nguyen|An|1990-01-01|IDENTITY_CARD|MOCK-0001; Tran|Binh|1992-02-02|PASSPORT|MOCK-0002\nđồng ý\n")


def test_cli_multi_turn_and_inspection_commands(tmp_path):
    agent = ReActFlightAgent()
    output = io.StringIO()
    directory = tmp_path / "cli-evidence"
    commands = INPUT + f"/state\n/completion\n/handoff\n/trace\n/export {directory}\n/quit\n"
    run_cli(agent, io.StringIO(commands), output)
    text = output.getvalue()
    assert "BOOKED:" in text and '"booking_completed": true' in text
    assert '"pattern": "react"' in text and '"event_type": "MODEL_OUTPUT"' in text
    assert json.loads((directory / "completion.json").read_text())["booking_completed"]
    assert len(json.loads((directory / "backend_state.json").read_text())["tickets"]) == 2


def test_cli_operator_commands_do_not_consume_steps():
    agent, output = ReActFlightAgent(), io.StringIO()
    run_cli(agent, io.StringIO("/state\n/backend\n/trace\n/completion\n/handoff\n/help\n/advance 60\n/failure timeout\n/quit\n"), output)
    assert agent.harness.get_state().step_count == 0
    assert not agent.get_state()["messages"]
    assert agent.harness.current_time().minute == 1
    rules = agent.get_backend_state()["failure_injection"]["rules"]
    assert len(rules) == 1 and rules[0]["effect"] == "TIMEOUT_UNRESOLVED"


def test_cli_reset_requires_explicit_acknowledgement_after_transaction():
    agent, output = ReActFlightAgent(), io.StringIO()
    run_cli(agent, io.StringIO(INPUT + "/reset\n/quit\n"), output)
    assert "erase a transaction" in output.getvalue()
    assert len(agent.get_backend_state()["bookings"]) == 1
    run_cli(agent, io.StringIO("/reset confirm\n/quit\n"), output)
    assert not agent.get_backend_state()["bookings"] and not agent.get_state()["messages"]


@pytest.mark.parametrize("command", ["/bad", "/advance -1", "/advance abc", "/failure other", "/export", "/reset nope"])
def test_cli_bad_command_does_not_crash_or_call_model(command):
    agent, output = ReActFlightAgent(), io.StringIO()
    run_cli(agent, io.StringIO(command + "\n/quit\n"), output)
    assert "Lỗi:" in output.getvalue() or "không hợp lệ" in output.getvalue()
    assert agent.harness.get_state().step_count == 0


def test_cli_preserves_existing_export(tmp_path):
    directory = tmp_path / "existing"
    directory.mkdir()
    marker = directory / "marker.json"
    marker.write_text("user evidence")
    output = io.StringIO()
    run_cli(ReActFlightAgent(), io.StringIO(f"/export {directory}\n/quit\n"), output)
    assert "Lỗi:" in output.getvalue()
    assert marker.read_text() == "user evidence"


@pytest.mark.parametrize("mode,terminal", [("success", "BOOKED"), ("timeout", "HANDOFF_REQUIRED"), ("timeout_but_success", "BOOKED")])
def test_module_cli_runs_offline_and_exports(mode, terminal, tmp_path):
    directory = tmp_path / mode
    result = subprocess.run([sys.executable, "-m", "app.cli", "--provider", "fake", "--payment-mode", mode,
                             "--output-dir", str(directory)], input=INPUT + "/quit\n", text=True,
                            capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "KHÔNG phải LLM" in result.stdout and terminal in result.stdout
    state = json.loads((directory / "agent_state.json").read_text())
    assert state["terminal_status"] == terminal
    assert (directory / "conversation.json").exists()


def test_cli_no_key_fails_clearly_without_reading_user_env(monkeypatch, capsys):
    monkeypatch.setattr("app.cli.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    with pytest.raises(SystemExit) as error:
        main(["--provider", "gemini"])
    assert error.value.code != 0 and "GEMINI_API_KEY" in capsys.readouterr().err


@pytest.mark.parametrize("pattern", ["react", "plan_execute", "hybrid"])
@pytest.mark.parametrize("mode", ["strict", "warn"])
def test_cli_review_mode_is_explicit_and_passed_to_each_pattern(pattern, mode, monkeypatch):
    from app.agents.factory import create_agent
    captured = []
    def capture(*args, **kwargs):
        agent = create_agent(*args, **kwargs)
        captured.append(agent)
        return agent
    monkeypatch.setattr("app.cli.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("app.cli.create_agent", capture)
    monkeypatch.setattr("app.cli.run_cli", lambda agent: None)
    arguments = ["--provider", "fake", "--pattern", pattern]
    if mode == "warn":
        arguments += ["--input-evidence-mode", mode]
    assert main(arguments) == 0
    assert captured[0].get_state()["input_evidence_mode"] == mode
