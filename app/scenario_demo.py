"""One catalog episode with a selected pattern/provider. Not H's benchmark."""

import argparse
import json
import sys

from app.agents.factory import AGENT_TYPES, create_agent, create_pattern_model
from app.evaluation.episode import run_scenario
from app.evaluation.scenarios import get_scenario, list_scenarios


def main(argv=None):
    parser = argparse.ArgumentParser(description="S01–S12 catalog and single-episode agent demo")
    parser.add_argument("--list", action="store_true", help="Show IDs, policies and terminal/checkpoint expectations")
    parser.add_argument("--show", choices=[s.id for s in list_scenarios()], help="Print a machine-readable definition")
    parser.add_argument("--scenario", choices=[s.id for s in list_scenarios()], default="S01")
    parser.add_argument("--pattern", choices=tuple(AGENT_TYPES), default="react")
    # Default remains offline even if a project .env contains real credentials.
    parser.add_argument("--provider", choices=("fake", "gemini"), default="fake")
    parser.add_argument("--model", help="Gemini model supporting this pattern; otherwise MODEL_NAME")
    parser.add_argument("--output-dir", help="Export 11 JSON evidence files into a NEW directory")
    args = parser.parse_args(argv)
    if args.list:
        for scenario in list_scenarios():
            expected = scenario.expectation
            print(f"{scenario.id}: {scenario.name} | {expected.terminal_status.value if expected.terminal_status else expected.phase.value}")
        return 0
    if args.show:
        print(get_scenario(args.show).model_dump_json(indent=2))
        return 0
    try:
        model = create_pattern_model(args.pattern, args.provider, args.model)
        print(f"Offline {args.pattern} test double + mock backend; NOT an LLM benchmark." if args.provider == "fake" else
              f"Gemini {args.pattern} + mock backend; real API calls/cost, synthetic profiles only; NOT an aggregate benchmark.")
        run = run_scenario(get_scenario(args.scenario), lambda harness: create_agent(args.pattern, harness, model=model))
    except ValueError:
        parser.error("Không khởi tạo được model/agent. Gemini cần GEMINI_API_KEY/GOOGLE_API_KEY và MODEL_NAME/--model phù hợp; dùng Developer API, không Vertex AI. fake không cần key.")
    print(json.dumps(run.result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    if args.output_dir:
        try:
            print("Artifacts: " + run.export_artifacts(args.output_dir))
        except OSError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    return 0 if run.result.scenario_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
