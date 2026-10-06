"""Synthetic fixtures: 36 flights and 54 offers, generated with a local RNG."""

from datetime import datetime, timedelta
from random import Random

from app.config import VIETNAM_TZ
from app.models import Airline, Airport, Flight, FlightOffer, Inventory


def normalize_airport_query(value: str) -> str:
    import unicodedata

    value = value.strip().casefold().replace("đ", "d")
    value = "".join(
        char for char in unicodedata.normalize("NFD", value)
        if not unicodedata.combining(char)
    )
    return " ".join(value.split())


def make_seed_data(seed: int, reference_time: datetime) -> tuple[
    dict[str, Airport], dict[str, Flight], dict[str, Inventory], dict[str, FlightOffer]
]:
    rng = Random(seed)
    airports = {
        "SGN": Airport(code="SGN", city="Ho Chi Minh City", name="Tan Son Nhat",
                       aliases=["Sài Gòn", "Ho Chi Minh", "TP HCM", "TP.HCM", "Hồ Chí Minh"]),
        "HAN": Airport(code="HAN", city="Ha Noi", name="Noi Bai", aliases=["Hà Nội", "Hanoi"]),
        "DAD": Airport(code="DAD", city="Da Nang", name="Da Nang", aliases=["Đà Nẵng", "Danang"]),
    }
    flights: dict[str, Flight] = {}
    inventory: dict[str, Inventory] = {}
    offers: dict[str, FlightOffer] = {}
    airlines = list(Airline)
    reference_date = reference_time.astimezone(VIETNAM_TZ).date()
    routes = [(origin, destination) for origin in airports for destination in airports if origin != destination]
    for day_offset in (5, 6, 7):
        departure_date = reference_date + timedelta(days=day_offset)
        for route_index, (origin, destination) in enumerate(routes):
            for hour, minute in ((8, 10), (15, 40)):
                number = len(flights) + 1
                flight_id = f"FLT-{number:03d}"
                departure = datetime.combine(departure_date, datetime.min.time(), VIETNAM_TZ)
                departure += timedelta(hours=hour, minutes=minute)
                flight = Flight(
                    flight_id=flight_id, airline=airlines[(number - 1) % len(airlines)],
                    origin=origin, destination=destination, departure_time=departure,
                    arrival_time=departure + timedelta(minutes=85 + 15 * (route_index % 3)),
                    aircraft="MOCK-A320",
                )
                capacity = rng.randint(8, 24)
                flights[flight_id] = flight
                inventory[flight_id] = Inventory(flight_id=flight_id, total_seats=capacity, available_seats=capacity)
                unit_price = rng.randrange(120, 281) * 10_000
                fares = [("Saver", unit_price, 15, False, False)]
                if number % 2 == 0:
                    fares.append(("Flex", unit_price + 350_000, 20, True, True))
                for fare_class, price, baggage, refundable, changeable in fares:
                    offer_id = f"OFF-{number:03d}-{fare_class.upper()}"
                    offers[offer_id] = FlightOffer(
                        offer_id=offer_id, flight_id=flight_id, fare_class=fare_class,
                        unit_price_vnd=price, checked_baggage_kg=baggage,
                        refundable=refundable, changeable=changeable,
                        expires_at=departure - timedelta(hours=1),
                    )
    return airports, flights, inventory, offers
