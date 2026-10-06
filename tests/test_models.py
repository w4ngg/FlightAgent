from datetime import date, time

import pytest
from pydantic import ValidationError

from app.clock import SimulatedClock
from app.config import BackendConfig
from app.models import FlightOffer, SearchConstraints, ToolResponse


@pytest.mark.parametrize("amount", [1.5, "2000000", True, -1])
def test_money_is_an_integer_not_a_coerced_value(backend, amount):
    offer = next(iter(backend.get_backend_state()["offers"].values()))
    with pytest.raises(ValidationError):
        FlightOffer.model_validate({**offer, "unit_price_vnd": amount})


@pytest.mark.parametrize("count", [0, -1, True, "3", 1.5])
def test_passenger_count_is_positive_and_strict(count):
    with pytest.raises(ValidationError):
        SearchConstraints(origin="SGN", destination="HAN", departure_date=date(2026, 10, 10), passenger_count=count)


def test_constraints_normalize_airport_codes_and_reject_empty_windows():
    constraints = SearchConstraints(origin=" sgn ", destination="han", departure_date="2026-10-10", passenger_count=1)
    assert constraints.origin == "SGN" and constraints.destination == "HAN"
    with pytest.raises(ValidationError):
        SearchConstraints(**{**constraints.model_dump(), "latest_departure": time(8),
                             "earliest_departure": time(9)})
    with pytest.raises(ValidationError):
        SearchConstraints(**{**constraints.model_dump(), "latest_departure": time(8),
                             "earliest_departure": time(8), "latest_inclusive": False})


def test_response_envelope_rejects_contradictory_success():
    with pytest.raises(ValidationError):
        ToolResponse[dict](ok=True, code="SOLD_OUT", data={})
    with pytest.raises(ValidationError):
        ToolResponse[dict](ok=False, code="OK", error_message="wrong")


def test_clock_rejects_naive_time_and_backward_or_fractional_advance():
    from datetime import datetime

    with pytest.raises(ValueError):
        SimulatedClock(datetime(2026, 10, 5))
    clock = SimulatedClock()
    for seconds in (-1, 1.5, True):
        with pytest.raises(ValueError):
            clock.advance(seconds)
    start = clock.now()
    clock.advance(360)
    assert (clock.now() - start).total_seconds() == 360
    clock.reset()
    assert clock.now() == start


def test_ttl_config_rejects_zero():
    with pytest.raises(ValidationError):
        BackendConfig(hold_ttl_seconds=0)
