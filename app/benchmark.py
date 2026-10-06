"""Explicit batch entry point; listing/dry-run never construct a model."""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from app.agents.factory import AGENT_TYPES
from app.agents.input_review import InputReviewConfig
from app.evaluation.benchmark_models import BenchmarkConfig, BenchmarkModelConfig, TokenPricing
from app.evaluation.runner import ExperimentRunner
from app.evaluation.scenarios import list_scenarios


def main(argv=None):
    catalog = list_scenarios()
    parser = argparse.ArgumentParser(description="Repeated flight-agent experiments; fake is NOT an LLM benchmark")
    parser.add_argument("--patterns", choices=tuple(AGENT_TYPES), nargs="+", default=list(AGENT_TYPES))
    parser.add_argument("--scenarios", choices=[s.id for s in catalog], nargs="+", default=[s.id for s in catalog])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--provider", choices=("fake", "gemini"), default="fake",
                        help="Always defaults to fake, irrespective of .env")
    parser.add_argument("--model", help="One explicit Gemini model for all patterns; otherwise MODEL_NAME")
    parser.add_argument("--temperature", type=float, help="Omitted by default; unsupported settings fail, no fallback")
    parser.add_argument("--input-evidence-mode", choices=("strict", "warn"), default="strict")
    parser.add_argument("--seed", type=int, help="Override the shared backend seed, not hosted-model reasoning")
    parser.add_argument("--output-dir", help="A NEW directory; never overwrite or implicitly resume")
    parser.add_argument("--pricing-file", type=Path, help="Optional user-supplied model/tier USD-per-million JSON rates")
    parser.add_argument("--allow-paid-calls", action="store_true", help="Required to actually run Gemini, not dry-run")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after first non-PASS episode, preserving results")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="Show catalog without constructing a model or writing files")
    mode.add_argument("--dry-run", action="store_true", help="Print the experiment plan only, without model/API/file writes")
    args = parser.parse_args(argv)
    if args.list:
        for scenario in catalog:
            expected = scenario.expectation.terminal_status or scenario.expectation.phase
            print(f"{scenario.id}: {scenario.name} — expected {expected.value}")
        return 0
    if not args.dry_run:
        if not args.output_dir:
            parser.error("Chạy batch cần --output-dir trỏ tới thư mục MỚI.")
        if args.provider == "gemini" and not args.allow_paid_calls:
            parser.error("Gemini có thể phát sinh phí; cần explicit --allow-paid-calls. Dùng --dry-run để chỉ xem plan.")
    if args.provider == "gemini" and not args.dry_run:
        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
        if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
            parser.error("Gemini cần GEMINI_API_KEY/GOOGLE_API_KEY; không khởi tạo model hoặc gọi API.")
    # Freeze the chosen ID once; never silently switch to a different model.
    name = args.model or (os.environ.get("MODEL_NAME") if args.provider == "gemini" else None)
    try:
        pricing = TokenPricing.model_validate_json(args.pricing_file.read_text(encoding="utf-8")) if args.pricing_file else None
        config = BenchmarkConfig(patterns=args.patterns, scenario_ids=args.scenarios, repetitions=args.repetitions,
            backend_seed=args.seed, model=BenchmarkModelConfig(provider=args.provider, model_name=name, temperature=args.temperature),
            pricing=pricing, allow_paid_calls=args.allow_paid_calls, fail_fast=args.fail_fast,
            input_review=InputReviewConfig(evidence_mode=args.input_evidence_mode))
        runner = ExperimentRunner(config)
    except (OSError, ValueError) as exc:
        # Pydantic/SDK errors can include input values, so do not echo their text.
        parser.error(f"Cấu hình benchmark không hợp lệ ({type(exc).__name__}). Kiểm tra model, unique patterns/scenarios, repetitions 1–1000 và pricing.")
    if args.dry_run:
        print(json.dumps(runner.plan(), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    label = ("SIMULATED — deterministic test doubles, NOT an LLM quality/cost benchmark" if config.model.provider == "fake" else
             "GEMINI — LLM thật; hồ sơ giả/trace được gửi tới provider, có thể phát sinh phí")
    print(f"{label}; planned episodes: {config.planned_episodes}", file=sys.stderr, flush=True)
    try:
        summary = runner.run_all(args.output_dir)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Dừng batch ({type(exc).__name__}); không retry/reset hoặc overwrite evidence.", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    if summary["status"] == "interrupted":
        return 130
    return 0 if (summary["complete"] and summary["all_recorded_scenarios_passed"] and
                 summary["environment_comparable_for_recorded_pairings"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
