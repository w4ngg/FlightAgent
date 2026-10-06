"""Scripted synthetic user; observes only public adapter state, never backend."""

import json
from datetime import datetime

from app.evaluation.contracts import PolicyKind, Scenario, UserAction, UserActionKind
from app.models import OfferDetails, SearchConstraints
from app.state import HarnessState, SessionPhase


class ScriptedUserPolicy:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario.model_copy(deep=True)
        self._started = False
        self._branch_sent = False
        self._last_action = None
        self._last_non_apply_action = None
        self.stop_reason = None

    def _emit(self, *, kind, text):
        self._last_action = kind
        if kind != UserActionKind.APPLY:
            self._last_non_apply_action = kind
        return UserAction(kind=kind, text=text)

    def next_action(self, public_state: dict) -> UserAction | None:
        if not self._started:
            self._started = True
            return self._emit(kind=UserActionKind.INITIAL, text=self.scenario.initial_user_message)
        state = HarnessState.model_validate(public_state["harness"])
        if state.terminal_status:
            self.stop_reason = "TERMINAL"
            return None
        config = self.scenario.user_policy
        pending = public_state.get("pending_interpretation")
        if pending:
            if pending.get("status", "READY") != "READY" or not pending.get("apply_enabled", True):
                self.stop_reason = "INTERPRETATION_MISMATCH"
                return None
            # Review readback against the scripted user's OWN intended request,
            # never blindly approve model output or a business quote.
            expected = (self.scenario.expected_constraints if self._branch_sent and config.kind == PolicyKind.CHANGE_DESTINATION
                        else self.scenario.initial_constraints).model_dump(mode="json")
            preview = pending.get("preview", {})
            matches = pending.get("intent") == "UPDATE_SEARCH" and preview and all(
                key in expected and expected[key] == value for key, value in preview.items())
            if config.kind == PolicyKind.MISSING_DESTINATION:
                matches = matches and "destination" not in preview and all(key in preview for key in ("origin", "departure_date", "passenger_count"))
            else:
                try:
                    matches = matches and SearchConstraints.model_validate(preview).model_dump(mode="json") == expected
                except ValueError:
                    matches = False
            if not matches:
                self.stop_reason = "INTERPRETATION_MISMATCH"
                return None
            return self._emit(kind=UserActionKind.APPLY, text="áp dụng")
        if self.scenario.expectation.phase is not None and self._last_non_apply_action == self.scenario.expectation.checkpoint_action:
            self.stop_reason = "EXPECTED_CHECKPOINT"
            return None
        if state.phase == SessionPhase.WAITING_SELECTION:
            raw = public_state.get("offers")
            if raw is not None:
                offers = [OfferDetails.model_validate(o) for o in raw]
                eligible = [o for o in offers if state.constraints and
                    state.constraints.permits(o.flight, o.total_price_vnd) and
                    o.available_seats >= state.constraints.passenger_count]
                if not eligible:
                    self.stop_reason = "NO_ELIGIBLE_OFFER"
                    return None
                selected = min(eligible, key=lambda o: (o.total_price_vnd, o.flight.departure_time, o.offer.offer_id))
                return self._emit(kind=UserActionKind.SELECT, text="chọn " + selected.offer.offer_id)
        if state.phase == SessionPhase.WAITING_PASSENGERS or (
            state.selected_offer_id and not public_state.get("passengers_valid", False)
        ):
            if config.kind == PolicyKind.INVALID_PASSENGER and not self._branch_sent:
                self._branch_sent = True
                return self._emit(kind=UserActionKind.INVALID_PASSENGERS,
                                  text="/passengers " + json.dumps(config.invalid_passengers, ensure_ascii=False))
            return self._emit(kind=UserActionKind.PASSENGERS, text="/passengers " + json.dumps(
                [p.model_dump(mode="json") for p in self.scenario.passengers], ensure_ascii=False))
        if state.phase == SessionPhase.WAITING_CONFIRMATION and state.quote and (
            state.displayed_quote_id == state.quote.quote_id and
            state.quote.expires_at > datetime.fromisoformat(public_state["current_time"])
        ):
            if not self._branch_sent:
                if config.kind == PolicyKind.CHANGE_DESTINATION:
                    self._branch_sent = True
                    return self._emit(kind=UserActionKind.CHANGE, text=config.change_message)
                if config.kind == PolicyKind.BYPASS_CONFIRMATION:
                    self._branch_sent = True
                    return self._emit(kind=UserActionKind.BYPASS, text=config.bypass_message)
            return self._emit(kind=UserActionKind.CONFIRM, text=config.confirmation_text)
        # Waiting/UNKNOWN is resumed through a genuine user turn, not direct
        # tool execution, invented confirmation or a hidden-status shortcut.
        return self._emit(kind=UserActionKind.RESUME, text="tiếp tục")
