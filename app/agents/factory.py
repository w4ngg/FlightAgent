"""One composition point for interactive and single-scenario agent selection."""

from app.agents.models import create_model, create_planner_model, create_router_model
from app.agents.hybrid_agent import HybridFlightAgent
from app.agents.plan_execute_agent import PlanExecuteFlightAgent
from app.agents.react_agent import ReActFlightAgent


AGENT_TYPES = {"react": ReActFlightAgent, "plan_execute": PlanExecuteFlightAgent, "hybrid": HybridFlightAgent}


def create_agent(pattern="react", harness=None, *, model=None, input_model=None, session_id="session-1", input_review_config=None):
    if pattern not in AGENT_TYPES:
        raise ValueError("Unknown agent pattern")
    return AGENT_TYPES[pattern](harness, model=model, input_model=input_model, session_id=session_id,
                              input_review_config=input_review_config)


def create_pattern_model(pattern, provider="fake", model_name=None, *, temperature=None):
    if pattern not in AGENT_TYPES:
        raise ValueError("Unknown agent pattern")
    factory = {"react": create_model, "plan_execute": create_planner_model, "hybrid": create_router_model}[pattern]
    return factory(provider, model_name, temperature=temperature) if temperature is not None else factory(provider, model_name)
