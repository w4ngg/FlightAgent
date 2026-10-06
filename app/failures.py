"""Deterministic, scoped failure rules. Backend calls consume under its lock."""

from enum import Enum
from typing import Literal, Self, Sequence

from pydantic import AwareDatetime, model_validator

from app.models import DomainModel, MoneyVND, NonemptyString, PositiveInt, TransactionContext


class FailureTrigger(str, Enum):
    REPRICE = "reprice"
    HOLD = "hold"
    PAYMENT = "payment"


class FailureEffect(str, Enum):
    PRICE_CHANGE = "PRICE_CHANGE"
    SOLD_OUT = "SOLD_OUT"
    OFFER_EXPIRED = "OFFER_EXPIRED"
    PAYMENT_DECLINED = "PAYMENT_DECLINED"
    TIMEOUT_UNRESOLVED = "TIMEOUT_UNRESOLVED"
    TIMEOUT_BUT_COMMITTED = "TIMEOUT_BUT_COMMITTED"


class FailureRule(DomainModel):
    rule_id: NonemptyString
    trigger: FailureTrigger
    effect: FailureEffect
    times: PositiveInt = 1
    offer_id: NonemptyString | None = None
    operation_id: NonemptyString | None = None
    session_id: NonemptyString | None = None
    transaction_intent_id: NonemptyString | None = None
    new_price: MoneyVND | None = None  # Unit price for one passenger, not group total.

    @model_validator(mode="after")
    def validate_effect(self) -> Self:
        payment_effects = {
            FailureEffect.PAYMENT_DECLINED, FailureEffect.TIMEOUT_UNRESOLVED,
            FailureEffect.TIMEOUT_BUT_COMMITTED,
        }
        if (self.effect in payment_effects) != (self.trigger == FailureTrigger.PAYMENT):
            raise ValueError("Payment effects require the payment trigger; offer effects require reprice/hold")
        if self.operation_id is not None and self.trigger != FailureTrigger.PAYMENT:
            raise ValueError("operation_id targets payment operations only")
        if self.new_price is not None and self.effect != FailureEffect.PRICE_CHANGE:
            raise ValueError("new_price applies only to PRICE_CHANGE")
        return self


class FailureConfig(DomainModel):
    force_price_change: bool = False
    new_price: MoneyVND | None = None
    force_sold_out: bool = False
    force_offer_expired: bool = False
    payment_mode: Literal["success", "declined", "timeout", "timeout_but_success"] = "success"
    target_offer_id: NonemptyString | None = None
    target_operation_id: NonemptyString | None = None

    @model_validator(mode="after")
    def validate_price_config(self) -> Self:
        if self.new_price is not None and not self.force_price_change:
            raise ValueError("new_price requires force_price_change")
        return self

    def to_rules(self) -> list[FailureRule]:
        rules = []
        for enabled, rule_id, trigger, effect in (
            (self.force_price_change, "config-price", FailureTrigger.REPRICE, FailureEffect.PRICE_CHANGE),
            (self.force_sold_out, "config-sold-out", FailureTrigger.HOLD, FailureEffect.SOLD_OUT),
            (self.force_offer_expired, "config-expired", FailureTrigger.HOLD, FailureEffect.OFFER_EXPIRED),
        ):
            if enabled:
                rules.append(FailureRule(
                    rule_id=rule_id, trigger=trigger, effect=effect, offer_id=self.target_offer_id,
                    new_price=self.new_price if effect == FailureEffect.PRICE_CHANGE else None,
                ))
        payment_effects = {
            "declined": FailureEffect.PAYMENT_DECLINED,
            "timeout": FailureEffect.TIMEOUT_UNRESOLVED,
            "timeout_but_success": FailureEffect.TIMEOUT_BUT_COMMITTED,
        }
        if self.payment_mode != "success":
            rules.append(FailureRule(
                rule_id="config-payment", trigger=FailureTrigger.PAYMENT,
                effect=payment_effects[self.payment_mode], offer_id=self.target_offer_id,
                operation_id=self.target_operation_id,
            ))
        return rules


class FailureEvent(DomainModel):
    sequence_number: PositiveInt
    event_type: Literal["FAILURE_INJECTED"] = "FAILURE_INJECTED"
    timestamp: AwareDatetime
    rule_id: NonemptyString
    trigger: FailureTrigger
    effect: FailureEffect
    offer_id: str | None
    operation_id: str | None
    context: TransactionContext


class FailureInjector:
    def __init__(self, rules: Sequence[FailureRule] = ()):
        self._rules = [FailureRule.model_validate(rule.model_dump()) for rule in rules]
        if len({rule.rule_id for rule in self._rules}) != len(self._rules):
            raise ValueError("Failure rule IDs must be unique")
        payment_rules = [rule for rule in self._rules if rule.trigger == FailureTrigger.PAYMENT]
        for index, first in enumerate(payment_rules):
            for second in payment_rules[index + 1:]:
                # Reject ambiguous outcomes for an operation at configuration time.
                scopes = ("offer_id", "operation_id", "session_id", "transaction_intent_id")
                overlap = all(
                    getattr(first, field) is None or getattr(second, field) is None
                    or getattr(first, field) == getattr(second, field)
                    for field in scopes
                )
                if overlap:
                    raise ValueError("Payment failure rules must have disjoint targets")
        self.reset()

    def reset(self) -> None:
        self._consumed = {rule.rule_id: 0 for rule in self._rules}
        self._events: list[FailureEvent] = []

    def consume(
        self, trigger: FailureTrigger, *, timestamp, context: TransactionContext,
        offer_id: str | None = None, operation_id: str | None = None,
    ) -> list[FailureRule]:
        matched = []
        for rule in self._rules:
            if rule.trigger != trigger or self._consumed[rule.rule_id] >= rule.times:
                continue
            actual = {"offer_id": offer_id, "operation_id": operation_id,
                      "session_id": context.session_id, "transaction_intent_id": context.transaction_intent_id}
            if any(getattr(rule, field) is not None and getattr(rule, field) != value for field, value in actual.items()):
                continue
            self._consumed[rule.rule_id] += 1
            self._events.append(FailureEvent(
                sequence_number=len(self._events) + 1, timestamp=timestamp,
                rule_id=rule.rule_id, trigger=trigger, effect=rule.effect,
                offer_id=offer_id, operation_id=operation_id, context=context,
            ))
            matched.append(rule.model_copy(deep=True))
        return matched

    def get_state(self) -> dict:
        return {
            "rules": [rule.model_dump(mode="json") for rule in self._rules],
            "consumed": dict(self._consumed),
            "events": [event.model_dump(mode="json") for event in self._events],
        }
