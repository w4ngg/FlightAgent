"""Host-owned search review. Immutable payloads, delta provenance, no consent."""

import json
import re
from copy import deepcopy
from datetime import date, time
from typing import Literal

from app.agents.diagnostics import GroundingError
from app.agents.input import AIRPORT_ALIASES, missing_fields, normalize
from app.agents.interpretation import canonical_date, canonical_money, canonical_passenger_count
from app.agents.interpretation import SearchPatch
from app.agents.routing import has_literal_span
from app.models import DomainModel, SearchConstraints, payload_hash


REVIEW_PROTOCOL_VERSION = "merged_request_snapshot_review_v1"
DEFAULTS = {
    "max_budget_vnd": None, "earliest_departure": None, "latest_departure": None,
    "earliest_inclusive": True, "latest_inclusive": True,
    "preferred_airlines": [], "allowed_airlines": [],
}


class InputReviewConfig(DomainModel):
    evidence_mode: Literal["strict", "warn"] = "strict"


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class RequestProposal(DomainModel):
    """JSON strings prevent mutation through nested dicts or public copies."""
    proposal_id: str
    revision: int
    snapshot_json: str
    sources_json: str
    basis_json: str
    extra_warnings_json: str = "[]"
    latest_source_turn_id: str
    base_request_revision: int
    status: Literal["READY", "NEEDS_REVIEW", "STALE"] = "READY"
    displayed_revision: int | None = None
    displayed_hash: str | None = None
    displayed_at_turn: int | None = None
    last_rejected_turn_json: str | None = None

    @property
    def snapshot(self):
        return json.loads(self.snapshot_json)

    @property
    def sources(self):
        return json.loads(self.sources_json)

    @property
    def basis(self):
        return json.loads(self.basis_json)

    @property
    def snapshot_hash(self):
        return payload_hash(self.snapshot)

    @property
    def warnings(self):
        return [{"field": field, "reason_code": reason, "source_turn_id": source["source_turn_id"]}
                for field, source in self.sources.items() for reason in source["reason_codes"]] + json.loads(self.extra_warnings_json)

    def public(self):
        return {
            "intent": "UPDATE_SEARCH", "proposal_id": self.proposal_id, "revision": self.revision,
            "snapshot": self.snapshot, "preview": self.snapshot, "snapshot_hash": self.snapshot_hash,
            "basis_hash": payload_hash(self.basis), "base_request_revision": self.base_request_revision,
            "field_sources": self.sources, "warnings": self.warnings, "status": self.status,
            "source_turn_id": self.latest_source_turn_id, "latest_source_turn_id": self.latest_source_turn_id,
            "displayed_revision": self.displayed_revision, "displayed_hash": self.displayed_hash,
            "displayed_at_turn": self.displayed_at_turn,
            "apply_enabled": self.status == "READY" and self.displayed_revision == self.revision
                and self.displayed_hash == self.snapshot_hash,
            "last_rejected_turn": json.loads(self.last_rejected_turn_json) if self.last_rejected_turn_json else None,
        }


def normalize_snapshot(candidate, today):
    """Validate partial as well as complete data, independently of evidence."""
    candidate = deepcopy(candidate)
    # Native JSON supports case-insensitive codes, as SearchConstraints does.
    for key in ("origin", "destination"):
        if isinstance(candidate.get(key), str):
            candidate[key] = candidate[key].strip().upper()
    patch = SearchPatch.model_validate(candidate).model_dump(mode="json", exclude_none=True)
    result = {**deepcopy(DEFAULTS), **patch}
    if result.get("departure_date") is not None:
        value = result["departure_date"]
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
            raise ValueError("Departure date must be canonical ISO")
        if date.fromisoformat(value) < today:
            raise ValueError("Past departure date")
    for key in ("earliest_departure", "latest_departure"):
        if result[key] is not None:
            value = result[key]
            if not re.fullmatch(r"[0-9]{2}:[0-9]{2}(?::[0-9]{2})?", value):
                raise ValueError("Time must be local canonical time")
            result[key] = time.fromisoformat(value).isoformat()
    if result.get("origin") and result.get("origin") == result.get("destination"):
        raise ValueError("Equal route endpoints")
    start, end = result["earliest_departure"], result["latest_departure"]
    if start is not None and end is not None and (start > end or start == end and not (
            result["earliest_inclusive"] and result["latest_inclusive"])):
        raise ValueError("Invalid departure window")
    if not missing_fields(result):
        return SearchConstraints.model_validate(result).model_dump(mode="json")
    return result


def search_delta(decision):
    result = decision.search_patch.model_dump(mode="json", exclude_none=True) if decision.search_patch else {}
    for field in decision.clear_fields:
        result[field] = [] if field.endswith("airlines") else None
    return result


def _matches(field, source, value):
    if field in {"origin", "destination"}:
        return AIRPORT_ALIASES.get(normalize(source)) == value
    if field == "departure_date":
        return canonical_date(source) == value
    if field == "passenger_count":
        return canonical_passenger_count(source) == value
    if field == "max_budget_vnd":
        return canonical_money(source) == value
    if field in {"earliest_departure", "latest_departure"}:
        match = re.search(r"\b(\d{1,2}):([0-9]{2})(?::([0-9]{2}))?\b", source)
        if not match:
            raise ValueError("Unsupported evidence time")
        return time(int(match[1]), int(match[2]), int(match[3] or 0)) == time.fromisoformat(value)
    if field in {"earliest_inclusive", "latest_inclusive"}:
        cue = normalize(source)
        exclusive = bool(re.search(r"\b(truoc|sau|before|after)\b", cue))
        inclusive = bool(re.search(r"(tu|den|khong muon hon|khong som hon|at or|inclusive)", cue))
        if not (exclusive or inclusive):
            raise ValueError("Unsupported boundary evidence")
        return value == (inclusive and not exclusive)
    if field.endswith("airlines"):
        labels = {"Vietnam Airlines": ("vietnam airlines",), "VietJet Air": ("vietjet", "vietjet air"),
                  "Bamboo Airways": ("bamboo", "bamboo airways")}
        return bool(value) and len(set(value)) == len(value) and all(
            any(re.search(r"\b" + alias + r"\b", normalize(source)) for alias in labels[name]) for name in value)
    raise ValueError("Unknown evidence field")


def audit_search(decision, turn, delta, mode, revision):
    evidence = {item.field: item.text for item in decision.evidence}
    sources, expected, extras = {}, set(), []
    for field, value in delta.items():
        key = "clear." + field if field in decision.clear_fields else field
        expected.add(key)
        claim = evidence.get(key)
        source = {
            "source_turn_id": turn.turn_id, "source_kind": "USER_LLM_PATCH",
            "claimed_evidence": claim, "literal_status": "MISSING", "literal_span": None,
            "semantic_status": "NOT_APPLICABLE", "reason_codes": [], "introduced_in_revision": revision,
        }
        if not claim:
            source["reason_codes"] = ["EVIDENCE_MISSING"]
        elif not has_literal_span(turn.text, claim):
            source.update(literal_status="NOT_IN_CURRENT_TURN", reason_codes=["EVIDENCE_NOT_IN_CURRENT_TURN"])
        else:
            match = re.search(r"(?<!\w)" + re.escape(claim) + r"(?!\w)", turn.text)
            source.update(literal_status="VERIFIED", literal_span=[match.start(), match.end()])
            if field not in decision.clear_fields:
                try:
                    matches = _matches(field, claim, value)
                except ValueError:
                    source.update(semantic_status="UNSUPPORTED_FORMAT", reason_codes=["EVIDENCE_FORMAT_UNSUPPORTED"])
                else:
                    source["semantic_status"] = "VERIFIED" if matches else "VALUE_MISMATCH"
                    if not matches:
                        source["reason_codes"] = ["EVIDENCE_VALUE_MISMATCH"]
        if source["reason_codes"] and mode == "strict":
            raise GroundingError(field, source["reason_codes"][0])
        sources[field] = source
    if evidence.keys() - expected:
        if mode == "strict":
            raise GroundingError("evidence", "EVIDENCE_EXTRA_FIELD")
        # Do not echo arbitrary model-supplied paths into diagnostics.
        extras.append({"field": "evidence", "reason_code": "EVIDENCE_EXTRA_FIELD", "source_turn_id": turn.turn_id})
    return sources, extras


def source_record(kind, turn_id=None, revision=0):
    return {"source_turn_id": turn_id, "source_kind": kind, "claimed_evidence": None,
            "literal_status": "NOT_APPLICABLE", "literal_span": None,
            "semantic_status": "NOT_APPLICABLE", "reason_codes": [], "introduced_in_revision": revision}


def merge_sources(snapshot, base_sources, updates):
    return {field: deepcopy(updates.get(field, base_sources.get(field, source_record("HOST_DEFAULT"))))
            for field in snapshot}
