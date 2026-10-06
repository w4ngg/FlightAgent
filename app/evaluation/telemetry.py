"""Measure actual invoke calls without changing prompts, outputs or retries."""

from copy import deepcopy
from time import perf_counter

from langchain_core.messages import AIMessage

from app.evaluation.benchmark_models import ModelCallMeasurement


def _count(value):
    return value if type(value) is int and value >= 0 else None


def _detail(details, suffix):
    if not isinstance(details, dict):
        return None
    values = [v for k, v in details.items() if k == suffix or k in (f"priority_{suffix}", f"flex_{suffix}")]
    return sum(values) if values and all(_count(v) is not None for v in values) else None


def measurement_fields(output):
    raw = output.get("raw") if isinstance(output, dict) else output
    if not isinstance(raw, AIMessage):
        return {}
    metadata = raw.response_metadata
    result = {"served_model": metadata.get("model_name") or metadata.get("model"), "service_tier": metadata.get("service_tier")}
    result = {k: v for k, v in result.items() if isinstance(v, str)}
    usage = raw.usage_metadata
    if not isinstance(usage, dict):
        return result
    input_, output_, total = (_count(usage.get(k)) for k in ("input_tokens", "output_tokens", "total_tokens"))
    if input_ is None or output_ is None or total != input_ + output_:
        return result  # Invalid metadata is unknown, never silently coerced.
    cached = _detail(usage.get("input_token_details"), "cache_read")
    written = _detail(usage.get("input_token_details"), "cache_creation")
    reasoning = _detail(usage.get("output_token_details"), "reasoning")
    if cached is not None and cached > input_ or written is not None and written > input_ or (
        cached is not None and written is not None and cached + written > input_
    ) or reasoning is not None and reasoning > output_:
        return result
    return {**result, "input_tokens": input_, "output_tokens": output_, "total_tokens": total,
            "cached_input_tokens": cached, "cache_write_tokens": written, "reasoning_tokens": reasoning}


class _MeasuredRunnable:
    def __init__(self, runnable, collector):
        self.runnable, self.collector = runnable, collector

    def invoke(self, *args, **kwargs):
        started, output, error = perf_counter(), None, None
        try:
            output = self.runnable.invoke(*args, **kwargs)
            return output
        except BaseException as exc:
            error = type(exc).__name__  # Never serialize provider exception text.
            raise
        finally:
            duration_ms = max(0.0, (perf_counter() - started) * 1000)
            try:
                fields = measurement_fields(output)
            except (ValueError, TypeError, AttributeError, KeyError):
                fields = {}  # Bad telemetry must not replace the actual model output/exception.
            self.collector.append(ModelCallMeasurement(call_index=len(self.collector) + 1,
                duration_ms=duration_ms, returned=error is None, exception_type=error, **fields))


class MeasuredModel:
    """Composition proxy for the two model interfaces used by A–G.

    A new proxy/underlying model is created for each episode. No measurements
    are inferred from simulated timestamps or model-decision budget events.
    """
    def __init__(self, model):
        self.model, self._calls = model, []

    @property
    def calls(self):
        return deepcopy(self._calls)

    def bind_tools(self, *args, **kwargs):
        return _MeasuredRunnable(self.model.bind_tools(*args, **kwargs), self._calls)

    def with_structured_output(self, *args, **kwargs):
        return _MeasuredRunnable(self.model.with_structured_output(*args, **kwargs), self._calls)
