"""Small single-episode exerciser for catalog acceptance; no benchmark metrics."""

import json
from pathlib import Path
from time import perf_counter
from typing import Callable

from app.agents.adapter import BaseFlightAgent
from app.evaluation.contracts import (
    Scenario, ScenarioCheckpoint, ScenarioEvidence, ScenarioOracleResult,
)
from app.evaluation.environment import ScenarioHarness
from app.evaluation.oracle import ScenarioOracle, events
from app.evaluation.user_policy import ScriptedUserPolicy
from app.models import DomainModel


class ScenarioRun(DomainModel):
    scenario: Scenario
    agent_name: str
    evidence: ScenarioEvidence
    result: ScenarioOracleResult

    def export_artifacts(self, output_dir) -> str:
        """Freeze the inspected episode, never overwrite an existing directory."""
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=False)
        evidence = self.evidence
        artifacts = {
            "initial_state.json": evidence.initial_backend_state,
            "backend_state.json": evidence.final_backend_state,
            "agent_state.json": evidence.final_state.model_dump(mode="json"),
            "trace.json": evidence.trace,
            "completion.json": self.result.completion.model_dump(mode="json"),
            "handoff.json": evidence.handoff,
            "conversation.json": evidence.final_agent_state,
            "scenario.json": self.scenario.model_dump(mode="json"),
            "scripted_user.json": [{"action": cp.action.model_dump(mode="json"), "user_turn_id": cp.user_turn_id,
                                     "user_turn_number": cp.user_turn_number} for cp in evidence.checkpoints],
            "checkpoints.json": [cp.model_dump(mode="json") for cp in evidence.checkpoints],
            "scenario_result.json": self.result.model_dump(mode="json"),
        }
        for filename, value in artifacts.items():
            (directory / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return str(directory.resolve())


def run_scenario(scenario: Scenario, agent_factory: Callable[[ScenarioHarness], BaseFlightAgent], *, session_id=None,
                 on_turn_duration: Callable[[float], None] | None = None) -> ScenarioRun:
    """Construct ONE fresh environment and invoke any conforming agent factory.

    User policy only sees public adapter state; oracle sees authoritative state.
    No automatic reset, raw transaction, fabricated receipt or hidden-status
    input is used to get a failing agent to pass. Exhaustion remains a failure.
    """
    scenario = Scenario.model_validate_json(scenario.model_dump_json())
    harness = ScenarioHarness(scenario, session_id=session_id or f"scenario-{scenario.id}")
    initial = harness.get_backend_state()
    agent = agent_factory(harness)
    if not isinstance(agent, BaseFlightAgent):
        raise TypeError("Agent factory must return the common BaseFlightAgent adapter")
    policy, checkpoints = ScriptedUserPolicy(scenario), []
    stop_reason = None
    for _ in range(scenario.max_user_turns):
        action = policy.next_action(agent.get_state())
        if action is None:
            stop_reason = policy.stop_reason
            break
        failed = False
        started = perf_counter() if on_turn_duration is not None else None
        try:
            agent.send(action.text, session_id=harness.get_state().context.session_id)
        except Exception:
            # Keep diagnostic evidence; do not invent an expected terminal or
            # echo provider exception messages that could contain credentials.
            failed = True
        finally:
            if on_turn_duration is not None:
                on_turn_duration(max(0.0, (perf_counter() - started) * 1000))
        trace = harness.get_trace()
        users = events(trace, "USER_MESSAGE")
        # A broken adapter that never registered the user remains a failed run.
        if not users or len(users) <= len(checkpoints):
            stop_reason = "AGENT_ERROR"
            break
        turn = users[-1]["payload"]["turn"]
        checkpoints.append(ScenarioCheckpoint(action=action, user_turn_id=turn["turn_id"], user_turn_number=turn["turn_number"],
            trace_sequence=len(trace), state=harness.get_state(), backend_state=harness.get_backend_state(), agent_state=agent.get_state()))
        if failed:
            stop_reason = "AGENT_ERROR"
            break
    if stop_reason is None:
        # Check a terminal/checkpoint reached on the final allowed turn before
        # reporting exhaustion. No extra user/model turn is executed here.
        next_action = policy.next_action(agent.get_state())
        stop_reason = policy.stop_reason if next_action is None else "USER_TURN_LIMIT"
    packet = harness.get_handoff()
    evidence = ScenarioEvidence(initial_backend_state=initial, final_backend_state=harness.get_backend_state(),
        final_state=harness.get_state(), final_agent_state=agent.get_state(), trace=harness.get_trace(), checkpoints=checkpoints,
        handoff=packet.model_dump(mode="json") if packet else None, stop_reason=stop_reason)
    return ScenarioRun(scenario=scenario, agent_name=agent.name, evidence=evidence,
                       result=ScenarioOracle().evaluate(scenario, evidence))
