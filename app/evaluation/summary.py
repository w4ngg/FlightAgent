"""Aggregate raw episodes with explicit denominators and missing coverage."""

from collections import Counter, defaultdict
from decimal import Decimal
from statistics import mean, stdev


def distribution(values):
    values = sorted(float(v) for v in values if v is not None)
    if not values:
        return {"count": 0, "mean": None, "stdev": None, "min": None, "max": None, "p50": None, "p95": None}
    def percentile(fraction):
        position = (len(values) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)
    return {"count": len(values), "mean": mean(values), "stdev": stdev(values) if len(values) > 1 else 0.0,
            "min": values[0], "max": values[-1], "p50": percentile(.5), "p95": percentile(.95)}


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize_group(rows):
    total = len(rows)
    expected = [r for r in rows if r.expected_booking]
    permission = [r for r in rows if r.permission_violations is not None and r.illegal_side_effect_attempts is not None]
    unsafe = [r for r in rows if r.unsafe_transactions is not None]
    recovery = [r for r in rows if r.recovery_opportunities is not None and r.recovered_failures is not None]
    handoff = [r for r in rows if r.handoff_count is not None]
    backend = [r for r in rows if r.backend_consistent is not None]
    cost = [r for r in rows if r.estimated_cost_usd is not None]
    opportunities = sum(r.recovery_opportunities for r in recovery)
    recovered = sum(r.recovered_failures for r in recovery)
    return {
        "episodes": total, "status_counts": dict(Counter(r.status for r in rows)),
        "scenario_pass_rate": _rate(sum(r.scenario_passed for r in rows), total),
        "task_completion_rate": _rate(sum(r.booking_completed for r in expected), len(expected)),
        "constraint_pass_rate": _rate(sum(r.constraint_passed for r in rows), total),
        "backend_consistency_rate": _rate(sum(r.backend_consistent for r in backend), total) if len(backend) == total else None,
        "permission_violation_rate": _rate(sum(r.illegal_side_effect_attempts for r in permission), total) if len(permission) == total else None,
        "permission_violation_episode_rate": _rate(sum(r.permission_violations > 0 for r in permission), total) if len(permission) == total else None,
        "observed_permission_violations": sum(r.permission_violations for r in permission),
        "unsafe_transaction_count": sum(r.unsafe_transactions for r in unsafe) if len(unsafe) == total else None,
        "observed_unsafe_findings": sum(len(r.unsafe_findings) for r in rows),
        "recovery_rate": _rate(recovered, opportunities) if len(recovery) == total else None,
        "recovery_opportunities": opportunities, "recovered_failures": recovered,
        "handoff_rate": _rate(sum(r.handoff_count > 0 for r in handoff), total) if len(handoff) == total else None,
        "denominators": {"scenario_pass_rate": total, "task_completion_expected_booked": len(expected),
            "constraint_pass_rate": total, "permission_observed_episodes": len(permission),
            "unsafe_observed_episodes": len(unsafe), "backend_observed_episodes": len(backend),
            "recovery_observed_episodes": len(recovery), "recovery_eligible_observed_opportunities": opportunities,
            "handoff_observed_episodes": len(handoff), "complete_cost_episodes": len(cost)},
        "distributions": {field: distribution([getattr(r, field) for r in rows]) for field in (
            "model_decisions", "model_calls", "llm_calls", "tool_attempts", "executed_tool_calls", "committed_side_effects",
            "excess_tool_calls", "replan_count", "plan_rejections", "blocked_plan_steps", "duration_ms", "agent_active_ms",
            "model_duration_ms", "input_tokens", "output_tokens", "total_tokens", "estimated_cost_usd")},
        "estimated_cost_usd": str(sum((r.estimated_cost_usd for r in cost), Decimal(0))) if len(cost) == total and total else None,
        "observed_estimated_cost_usd": str(sum((r.observed_estimated_cost_usd or Decimal(0) for r in rows), Decimal(0)))
            if any(r.observed_estimated_cost_usd is not None for r in rows) else None,
        "warnings": sorted({w for r in rows for w in r.measurement_warnings}),
    }


def summarize_benchmark(rows, config, *, status="completed"):
    rows = list(rows)
    groups = defaultdict(list)
    pairings = defaultdict(list)
    for row in rows:
        groups[row.agent_type, row.scenario_id].append(row)
        pairings[row.scenario_id, row.run_index].append(row)
    comparable = bool(rows) and all(len(group) == len(config.patterns) and
        {r.agent_type for r in group} == set(config.patterns) and all(r.environment_hash for r in group) and
        len({r.environment_hash for r in group}) == 1 for group in pairings.values())
    return {"schema_version": 1, "status": status, "provider": config.model.provider,
        "simulated": config.model.provider == "fake", "planned_episodes": config.planned_episodes, "recorded_episodes": len(rows),
        "complete": status == "completed" and len(rows) == config.planned_episodes,
        "environment_comparable_for_recorded_pairings": comparable,
        "all_recorded_scenarios_passed": bool(rows) and all(r.scenario_passed for r in rows),
        "metric_definitions": {
            "task_completion_rate": "BOOKED completions / episodes expecting BOOKED; checkpoints/decline excluded",
            "scenario_pass_rate": "independent oracle PASS / all recorded episodes, including setup/driver failures",
            "permission_violation_rate": "illegal SIDE-EFFECT attempts / episodes; mean count, can exceed 1",
            "permission_violation_episode_rate": "episodes with ANY guard policy violation / episodes",
            "tool_attempts": "all guarded attempts, including denial/backend failure/replay",
            "executed_tool_calls": "TOOL_ALLOWED attempts, not necessarily successful commits",
            "committed_side_effects": "unique holds + payment operations (including decline/UNKNOWN) + ticket batches",
            "unsafe_transactions": "distinct committed holds/SUCCESS charges/ticket batches with audited safety findings",
            "recovery_rate": "eligible observed recoverable catalog faults/corrections reaching safe expected BOOKED / opportunities; conservative all-or-none per episode",
            "excess_tool_calls": "max(0, attempts - catalog recovery-path minimum); successful catalog only; distinct endpoint resolutions allowed",
            "duration_ms": "wall time for setup, scripted episode and oracle; excludes artifact persistence",
            "agent_active_ms": "sum of actual agent.send durations; excludes scripted policy/evidence/export/human think-time",
            "llm_calls": "actual model invokes for Gemini; fake has zero LLM calls and separate model_calls",
            "cost": "token-only estimate using supplied model/tier rates and reported cache reads; missing tier/usage is unknown, excludes cache storage and other billed features",
            "missing_coverage": "rates/count totals requiring unknown measurements are null; observed lower bounds are separate",
        },
        "overall": summarize_group(rows),
        "by_pattern": {p: summarize_group([r for r in rows if r.agent_type == p]) for p in config.patterns},
        "by_pattern_scenario": {f"{p}/{s}": summarize_group(groups[p, s]) for p in config.patterns for s in config.scenario_ids},
        "limitations": ["Fake results verify software paths, not LLM quality/cost/latency superiority.",
            "Scripted users and bounded parsing/proposals do not evaluate free-form conversation quality.",
            "Seed repeats the environment, not hosted-model reasoning. Timing includes cache/rate/network effects.",
            "Partial batches and unobserved metrics must not be treated as a complete comparison."]}
