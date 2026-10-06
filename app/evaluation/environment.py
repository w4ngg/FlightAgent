"""Identical scenario environment for all patterns; no business rules copied."""

from copy import deepcopy
from threading import RLock

from app.backend import MockFlightBackend
from app.clock import SimulatedClock
from app.evaluation.contracts import Scenario
from app.harness.runtime import FlightBookingHarness


class ScenarioHarness(FlightBookingHarness):
    """Optional one-shot duplicate delivery at the guarded transport boundary.

    S11 tests infrastructure replay, not whether an LLM spontaneously retries.
    Both deliveries use the original C boundary, same arguments/context/key.
    """
    def __init__(self, scenario: Scenario, *, session_id: str):
        backend = MockFlightBackend(scenario.backend_seed, clock=SimulatedClock(scenario.start_time))
        super().__init__(backend, session_id=session_id, config=scenario.harness_config)
        self.configure_failure(rules=scenario.failure_rules)
        self._duplicate_hold_delivery = scenario.duplicate_hold_delivery
        self._replay_delivered = False
        self._delivery_lock = RLock()

    def execute_guarded_tool(self, name, tool_input):
        with self._delivery_lock:
            original = super().execute_guarded_tool(name, tool_input)
            if name == "create_booking_hold" and original.ok and self._duplicate_hold_delivery and not self._replay_delivered:
                self._replay_delivered = True
                original_result = next(e for e in reversed(self.get_trace()) if
                    e["event_type"] == "TOOL_RESULT" and e["tool_name"] == name)
                self.record_host_event("TRANSPORT_REPLAY", tool_name=name,
                    original_attempt_id=original_result["attempt_id"], original_booking_id=original.data.booking_id,
                    idempotency_key=self.get_state().operation_keys[name])
                # The replay is NOT a raw backend call and cannot get fresh keys.
                return super().execute_guarded_tool(name, deepcopy(tool_input))
            return original

    def reset(self, seed=42):
        super().reset(seed)
        self._replay_delivered = False
