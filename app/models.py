"""Validated domain values; backend owns transitions between these snapshots."""

import hashlib
import json
from datetime import date, time
from enum import Enum
from typing import Annotated, Generic, Literal, Self, TypeVar

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config import VIETNAM_TZ

AirportCode = Literal["SGN", "HAN", "DAD"]
MoneyVND = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonemptyString = Annotated[str, Field(min_length=1)]


class DomainModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class Airline(str, Enum):
    VIETNAM_AIRLINES = "Vietnam Airlines"
    VIETJET = "VietJet Air"
    BAMBOO = "Bamboo Airways"


class BookingStatus(str, Enum):
    HELD = "HELD"
    PAID = "PAID"
    TICKETED = "TICKETED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class PaymentStatus(str, Enum):
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    DECLINED = "DECLINED"
    UNKNOWN = "UNKNOWN"


class TerminalStatus(str, Enum):
    BOOKED = "BOOKED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    HANDOFF_REQUIRED = "HANDOFF_REQUIRED"


class CommitStatus(str, Enum):
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NOT_COMMITTED = "NOT_COMMITTED"
    COMMITTED = "COMMITTED"
    UNKNOWN = "UNKNOWN"


class Airport(DomainModel):
    code: AirportCode
    city: NonemptyString
    name: NonemptyString
    country: Literal["VN"] = "VN"
    aliases: list[str] = Field(default_factory=list)


class Flight(DomainModel):
    flight_id: NonemptyString
    airline: Airline
    origin: AirportCode
    destination: AirportCode
    departure_time: AwareDatetime
    arrival_time: AwareDatetime
    aircraft: NonemptyString

    @model_validator(mode="after")
    def validate_route_and_times(self) -> Self:
        if self.origin == self.destination:
            raise ValueError("Origin and destination must differ")
        if self.arrival_time <= self.departure_time:
            raise ValueError("Arrival must be later than departure")
        return self


class Inventory(DomainModel):
    flight_id: NonemptyString
    cabin_class: Literal["Economy"] = "Economy"
    total_seats: PositiveInt
    available_seats: Annotated[int, Field(strict=True, ge=0)]

    @model_validator(mode="after")
    def validate_capacity(self) -> Self:
        if self.available_seats > self.total_seats:
            raise ValueError("Available seats cannot exceed total capacity")
        return self


class FlightOffer(DomainModel):
    offer_id: NonemptyString
    flight_id: NonemptyString
    cabin_class: Literal["Economy"] = "Economy"
    fare_class: Literal["Saver", "Flex"]
    unit_price_vnd: MoneyVND
    currency: Literal["VND"] = "VND"
    checked_baggage_kg: Annotated[int, Field(strict=True, ge=0)]
    refundable: bool
    changeable: bool
    expires_at: AwareDatetime


class Passenger(DomainModel):
    passenger_id: NonemptyString
    first_name: NonemptyString
    last_name: NonemptyString
    date_of_birth: date
    document_type: Literal["IDENTITY_CARD", "PASSPORT"]
    document_number: Annotated[str, Field(min_length=4, max_length=32, pattern=r"^[A-Za-z0-9-]+$")]

    @field_validator("first_name", "last_name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("document_number")
    @classmethod
    def normalize_document(cls, value: str) -> str:
        return value.upper()


class SearchConstraints(DomainModel):
    origin: AirportCode
    destination: AirportCode
    departure_date: date
    passenger_count: PositiveInt
    max_budget_vnd: MoneyVND | None = None
    earliest_departure: time | None = None
    latest_departure: time | None = None
    earliest_inclusive: bool = True
    latest_inclusive: bool = True
    preferred_airlines: list[Airline] = Field(default_factory=list)
    allowed_airlines: list[Airline] = Field(default_factory=list)

    @field_validator("origin", "destination", mode="before")
    @classmethod
    def uppercase_code(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("earliest_departure", "latest_departure")
    @classmethod
    def require_local_time(cls, value: time | None) -> time | None:
        if value is not None and value.tzinfo is not None:
            raise ValueError("Departure windows use local Vietnam time without tzinfo")
        return value

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if self.origin == self.destination:
            raise ValueError("Origin and destination must differ")
        if self.earliest_departure is not None and self.latest_departure is not None:
            if self.earliest_departure > self.latest_departure:
                raise ValueError("Departure window is reversed")
            if self.earliest_departure == self.latest_departure and not (
                self.earliest_inclusive and self.latest_inclusive
            ):
                raise ValueError("Departure window is empty")
        return self

    def permits(self, flight: Flight, total_price_vnd: int) -> bool:
        local_departure = flight.departure_time.astimezone(VIETNAM_TZ)
        departure = local_departure.time().replace(tzinfo=None)
        if (flight.origin, flight.destination, local_departure.date()) != (
            self.origin, self.destination, self.departure_date
        ):
            return False
        if self.max_budget_vnd is not None and total_price_vnd > self.max_budget_vnd:
            return False
        if self.allowed_airlines and flight.airline not in self.allowed_airlines:
            return False
        if self.earliest_departure is not None:
            if departure < self.earliest_departure or (
                departure == self.earliest_departure and not self.earliest_inclusive
            ):
                return False
        if self.latest_departure is not None:
            if departure > self.latest_departure or (
                departure == self.latest_departure and not self.latest_inclusive
            ):
                return False
        return True


class TransactionContext(DomainModel):
    session_id: NonemptyString
    transaction_intent_id: NonemptyString


def payload_hash(value: object) -> str:
    """Hash JSON-compatible business data, independent of dictionary key order."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def passenger_payload_hash(passengers: list[Passenger]) -> str:
    return payload_hash([passenger.model_dump(mode="json") for passenger in passengers])


class Quote(DomainModel):
    quote_id: NonemptyString
    quote_version: PositiveInt
    context: TransactionContext
    offer_id: NonemptyString
    flight_id: NonemptyString
    origin: AirportCode
    destination: AirportCode
    departure_date: date
    passenger_ids: list[str]
    passenger_payload_hash: NonemptyString
    passenger_count: PositiveInt
    unit_price_vnd: MoneyVND
    total_price_vnd: MoneyVND
    currency: Literal["VND"] = "VND"
    created_at: AwareDatetime
    expires_at: AwareDatetime

    @model_validator(mode="after")
    def validate_quote(self) -> Self:
        if len(self.passenger_ids) != self.passenger_count:
            raise ValueError("Quote must identify every passenger")
        if self.total_price_vnd != self.unit_price_vnd * self.passenger_count:
            raise ValueError("Quote total must equal unit price times passenger count")
        if self.expires_at <= self.created_at:
            raise ValueError("Quote expiry must be later than creation")
        return self


class ConfirmationSnapshot(DomainModel):
    offer_id: NonemptyString
    quote_version: PositiveInt
    origin: AirportCode
    destination: AirportCode
    departure_date: date
    passenger_ids: list[str]
    passenger_payload_hash: NonemptyString
    passenger_count: PositiveInt
    total_price_vnd: MoneyVND
    currency: Literal["VND"] = "VND"

    @classmethod
    def from_quote(cls, quote: Quote) -> Self:
        return cls.model_validate(quote.model_dump(include=set(cls.model_fields)))


class ConfirmationRecord(DomainModel):
    context: TransactionContext
    snapshot: ConfirmationSnapshot
    user_turn_id: NonemptyString
    confirmed_at: AwareDatetime


class Booking(DomainModel):
    booking_id: NonemptyString
    context: TransactionContext
    quote: Quote
    confirmation: ConfirmationRecord
    constraints: SearchConstraints
    passengers: list[Passenger]
    status: BookingStatus
    total_price_vnd: MoneyVND
    currency: Literal["VND"] = "VND"
    hold_expires_at: AwareDatetime
    idempotency_key: NonemptyString
    created_at: AwareDatetime


class PaymentOperation(DomainModel):
    operation_id: NonemptyString
    booking_id: NonemptyString
    context: TransactionContext
    amount_vnd: MoneyVND
    currency: Literal["VND"] = "VND"
    status: PaymentStatus
    idempotency_key: NonemptyString
    payload_hash: NonemptyString
    payment_id: str | None = None
    created_at: AwareDatetime


class Payment(DomainModel):
    payment_id: NonemptyString
    operation_id: NonemptyString
    booking_id: NonemptyString
    amount_vnd: MoneyVND
    currency: Literal["VND"] = "VND"
    status: PaymentStatus
    idempotency_key: NonemptyString
    created_at: AwareDatetime


class Ticket(DomainModel):
    ticket_number: NonemptyString
    booking_id: NonemptyString
    passenger_id: NonemptyString
    passenger_index: Annotated[int, Field(strict=True, ge=0)]
    flight_id: NonemptyString
    status: Literal["ISSUED"] = "ISSUED"


class OfferDetails(DomainModel):
    flight: Flight
    offer: FlightOffer
    available_seats: Annotated[int, Field(strict=True, ge=0)]
    passenger_count: PositiveInt
    total_price_vnd: MoneyVND


class SearchResult(DomainModel):
    offers: list[OfferDetails]


class PaymentResult(DomainModel):
    operation: PaymentOperation
    payment: Payment | None
    booking: Booking


class TicketResult(DomainModel):
    booking: Booking
    tickets: list[Ticket]


class InventoryEvent(DomainModel):
    sequence_number: PositiveInt
    event_type: Literal["HOLD_RESERVED", "HOLD_RELEASED", "EXTERNAL_SOLD_OUT"]
    timestamp: AwareDatetime
    flight_id: NonemptyString
    seats_delta: Annotated[int, Field(strict=True)]
    booking_id: str | None = None
    rule_id: str | None = None


class BackendInvariantReport(DomainModel):
    passed: bool
    checks: dict[str, bool]
    violations: list[str]


ResponseData = TypeVar("ResponseData")


class ToolResponse(DomainModel, Generic[ResponseData]):
    ok: bool
    code: str
    retryable: bool = False
    commit_status: CommitStatus = CommitStatus.NOT_APPLICABLE
    operation_ref: str | None = None
    data: ResponseData | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_envelope(self) -> Self:
        if self.ok and (self.code != "OK" or self.error_message is not None):
            raise ValueError("Successful responses must have code OK and no error")
        if not self.ok and (self.code == "OK" or not self.error_message):
            raise ValueError("Errors must have a non-OK code and a message")
        return self
