"""Shared user/session adapter, independent of how an agent chooses tools."""

import json
import re
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from threading import RLock
from typing import Protocol, runtime_checkable

from langchain_core.messages import AIMessage, HumanMessage, messages_to_dict

from app.agents.input import FIELD_LABELS, missing_fields, parse_user_input, request_errors
from app.agents.input import ParsedInput, normalize
from app.agents.diagnostics import failure_diagnostic, model_secrets
from app.agents.interpretation import InputInterpreter, QUESTIONS, default_input_model, grounded_input
from app.agents.input_review import (
    InputReviewConfig, RequestProposal, REVIEW_PROTOCOL_VERSION, audit_search, encode,
    merge_sources, normalize_snapshot, search_delta, source_record,
)
from app.config import VIETNAM_TZ
from app.harness.runtime import FlightBookingHarness
from app.models import DomainModel, OfferDetails, Passenger, SearchConstraints, TerminalStatus, payload_hash
from app.state import CompletionResult, SessionPhase, TransactionBoundary


class AgentTurnResult(DomainModel):
    assistant_message: str
    session_id: str
    phase: SessionPhase
    terminal_status: TerminalStatus | None
    waiting_for_user: bool
    completion: CompletionResult


@runtime_checkable
class BaseFlightAgent(Protocol):
    """The same ingress/inspection contract serves all three agent patterns."""
    name: str

    def send(self, message: str, session_id: str | None = None) -> AgentTurnResult: ...
    def reset(self, seed: int = 42, *, force: bool = False): ...
    def configure_failure(self, config=None, *, rules=None): ...
    def get_state(self) -> dict: ...
    def get_backend_state(self) -> dict: ...
    def get_trace(self) -> list[dict]: ...
    def get_handoff(self): ...
    def check_completion(self) -> CompletionResult: ...
    def export_artifacts(self, output_dir) -> str: ...


class ConversationAdapter:
    """Host-only parsing, quote display, receipt ingress and inspectable memory.

    One adapter owns one session. send() never resets backend/messages/keys.
    Pause/resume is in-process, not checkpoint restoration from JSON.
    """
    name = "base"

    def __init__(self, harness: FlightBookingHarness, *, input_review_config=None):
        self.harness = harness
        supplied = input_review_config or InputReviewConfig()
        self._review_config = InputReviewConfig.model_validate(
            supplied.model_dump() if isinstance(supplied, InputReviewConfig) else supplied)
        self._lock = RLock()
        self._input_interpreter = None
        self._clear_memory()

    def _configure_input(self, model, input_model=None):
        # False explicitly isolates business-model transport tests. CLI Gemini
        # always uses the shared hosted interpreter; fake uses local ingress.
        chosen = None if input_model is False else input_model if input_model is not None else default_input_model(model)
        self._provider_secrets = model_secrets(model, chosen)
        self._input_interpreter = InputInterpreter(chosen) if chosen is not None else None

    def _clear_memory(self):
        self._messages = []
        self._draft = {}
        self._accepted_sources = {}
        self._request_revision = self._proposal_revision = 0
        self._last_request_receipt = None
        self._offers: list[OfferDetails] | None = None
        self._inspected_offers = set()
        self._passengers_blocked = False
        self._input_errors = []
        self._request_error = None
        self._notice = None
        self._model_error = False
        self._last_model_error = None
        self._last_tool_result = None
        self._pending_interpretation = None
        self._input_question = None
        self._query_topic = None
        self._blocked_scope = None

    def reset(self, seed=42, *, force=False):
        with self._lock:
            if self.harness.get_state().boundary == TransactionBoundary.POST_TRANSACTION and not force:
                raise ValueError("Reset would erase a transaction; explicitly use force=True or /reset confirm")
            self.harness.reset(seed)
            self._clear_memory()

    def configure_failure(self, config=None, *, rules=None):
        with self._lock:
            self.harness.configure_failure(config, rules=rules)

    def _passengers_valid(self):
        state = self.harness.get_state()
        return bool(state.constraints and not self._passengers_blocked and not
            self.harness.constraint_policy.validate_passengers(state.passengers, state.constraints.passenger_count,
                                                              self.harness.current_time().astimezone(VIETNAM_TZ).date()))

    def _public_context(self):
        """No backend payment secrets: observations, user data and clock only."""
        pending = self._pending_public()
        working = pending["snapshot"] if isinstance(self._pending_interpretation, RequestProposal) else self._draft
        return {
            "harness": self.harness.get_state().model_dump(mode="json"),
            "request_draft": deepcopy(self._draft), "missing_fields": missing_fields(self._draft),
            "input_errors": self._input_errors + ([self._request_error] if self._request_error else []),
            "offers": [o.model_dump(mode="json") for o in self._offers] if self._offers is not None else None,
            "inspected_offers": sorted(self._inspected_offers), "passengers_valid": self._passengers_valid(),
            "current_time": self.harness.current_time().isoformat(),
            "input_mode": "llm" if self._input_interpreter else "offline",
            "pending_interpretation": pending,
            "input_evidence_mode": self._review_config.evidence_mode,
            "review_protocol": REVIEW_PROTOCOL_VERSION,
            "accepted_request_revision": self._request_revision,
            "accepted_missing_fields": missing_fields(self._draft),
            "working_missing_fields": missing_fields(working),
            "accepted_field_sources": deepcopy(self._accepted_sources),
            "last_request_receipt": deepcopy(self._last_request_receipt),
        }

    def _pending_public(self):
        p = self._pending_interpretation
        return p.public() if isinstance(p, RequestProposal) else deepcopy(p)

    def _review_today(self):
        return self.harness.current_time().astimezone(VIETNAM_TZ).date()

    def _preserve_request(self, turn, reason, *, stale=False):
        p = self._pending_interpretation
        if isinstance(p, RequestProposal):
            self._pending_interpretation = p.model_copy(update={
                "status": "STALE" if stale or p.status == "STALE" else "NEEDS_REVIEW", "displayed_revision": None,
                "displayed_hash": None, "displayed_at_turn": None,
                "last_rejected_turn_json": encode({"turn_id": turn.turn_id, "reason_code": reason}),
            })
            self.harness.record_host_event("REQUEST_PROPOSAL_PRESERVED", proposal_id=p.proposal_id,
                revision=p.revision, snapshot_hash=p.snapshot_hash, rejected_turn_id=turn.turn_id, reason_code=reason)

    def _publish_request(self, turn, snapshot, sources, extras, basis, *, kept=False):
        previous = self._pending_interpretation
        self._proposal_revision += 1
        self._pending_interpretation = RequestProposal(
            proposal_id=f"{self.harness.get_state().context.session_id}:request-{self._proposal_revision}",
            revision=self._proposal_revision, snapshot_json=encode(snapshot), sources_json=encode(sources),
            basis_json=encode(basis), extra_warnings_json=encode(extras),
            latest_source_turn_id=previous.latest_source_turn_id if kept else turn.turn_id,
            base_request_revision=self._request_revision,
            last_rejected_turn_json=previous.last_rejected_turn_json if isinstance(previous, RequestProposal) else None)
        self.harness.block_input(turn.turn_id)
        self._blocked_scope = "request"
        self._request_error = None
        p = self._pending_interpretation
        event = "REQUEST_PROPOSAL_KEPT" if kept else (
            "REQUEST_PROPOSAL_UPDATED" if isinstance(previous, RequestProposal) else "REQUEST_PROPOSAL_CREATED")
        self.harness.record_host_event(event, proposal_id=p.proposal_id, revision=p.revision,
            snapshot_hash=p.snapshot_hash, source_turn_id=turn.turn_id, evidence_mode=self._review_config.evidence_mode,
            review_protocol=REVIEW_PROTOCOL_VERSION)
        for warning in p.warnings:
            self.harness.record_host_event("INPUT_EVIDENCE_WARNING", proposal_id=p.proposal_id,
                revision=p.revision, snapshot_hash=p.snapshot_hash, evidence_mode=self._review_config.evidence_mode,
                review_protocol=REVIEW_PROTOCOL_VERSION, **warning)

    def _prepare_request(self, turn, decision):
        if self.harness.get_state().boundary == TransactionBoundary.POST_TRANSACTION:
            self.harness.request_handoff("INTERPRETED_CHANGE_POST_TRANSACTION")
            return
        p = self._pending_interpretation
        basis = self._interpretation_basis()
        if isinstance(p, RequestProposal):
            if p.status == "STALE" or p.basis != basis:
                self._preserve_request(turn, "STALE_BASIS", stale=True)
                raise ValueError("Stale request proposal")
            base, old_sources = p.snapshot, p.sources
        else:
            base, old_sources = self._draft, self._accepted_sources
        delta = search_delta(decision)
        snapshot = normalize_snapshot({**base, **delta}, self._review_today())
        updates, extras = audit_search(decision, turn, delta, self._review_config.evidence_mode,
                                      self._proposal_revision + 1)
        self._publish_request(turn, snapshot, merge_sources(snapshot, old_sources, updates), extras, basis)

    def _keep_request(self, turn):
        p = self._pending_interpretation
        if not isinstance(p, RequestProposal):
            return ParsedInput(errors=["Không có proposal yêu cầu để giữ lại."])
        if self.harness.get_state().boundary == TransactionBoundary.POST_TRANSACTION:
            self.harness.request_handoff("INTERPRETED_CHANGE_POST_TRANSACTION")
            return ParsedInput()
        if p.status == "STALE" or p.basis != self._interpretation_basis():
            self._preserve_request(turn, "STALE_BASIS", stale=True)
            return ParsedInput(errors=["Proposal đã cũ; hãy gửi lại yêu cầu."])
        try:
            snapshot = normalize_snapshot(p.snapshot, self._review_today())
        except (ValueError, TypeError):
            self._preserve_request(turn, "REQUEST_DOMAIN_INVALID")
            return ParsedInput(errors=["Proposal không còn hợp lệ; hãy sửa yêu cầu."])
        self._publish_request(turn, snapshot, p.sources, json.loads(p.extra_warnings_json), p.basis, kept=True)
        return ParsedInput()

    def _commit_request(self, turn, snapshot, sources):
        """Validate/stage first. No accepted mutation on rejected input."""
        if self.harness.get_state().boundary == TransactionBoundary.POST_TRANSACTION:
            self.harness.set_requirements(turn.turn_id, snapshot)
            return False
        validated = normalize_snapshot(snapshot, self._review_today())
        if not missing_fields(validated):
            self.harness.set_requirements(turn.turn_id, validated)
            if self.harness.get_state().terminal_status:
                return False
        else:
            self.harness.block_input(turn.turn_id)
        self._draft, self._accepted_sources = deepcopy(validated), deepcopy(sources)
        self._request_revision += 1
        # A native replacement must not retain a receipt for another request.
        # Reviewed apply records its new receipt only after this commit succeeds.
        self._last_request_receipt = None
        self._request_error = None
        self._blocked_scope = "request" if missing_fields(validated) else None
        return True

    def _apply_request(self, turn, p):
        reason = None
        if p.status != "READY":
            reason = "REVIEW_REQUIRED"
        elif p.basis != self._interpretation_basis():
            reason = "STALE_BASIS"
        elif p.displayed_revision != p.revision or p.displayed_hash != p.snapshot_hash or (
                p.displayed_at_turn is None or turn.turn_number <= p.displayed_at_turn):
            reason = "DISPLAY_BINDING_INVALID"
        if reason:
            self._preserve_request(turn, reason, stale=reason == "STALE_BASIS" or p.status == "STALE")
            self.harness.record_host_event("REQUEST_APPLY_REJECTED", proposal_id=p.proposal_id,
                                           revision=p.revision, reason_code=reason)
            return ParsedInput(errors=["Proposal đã cũ hoặc cần xem lại; sửa yêu cầu hoặc /proposal keep rồi xác nhận bản đọc lại mới."])
        try:
            snapshot = normalize_snapshot(p.snapshot, self._review_today())
            if payload_hash(snapshot) != p.snapshot_hash:
                raise ValueError("Snapshot changed")
            before = self.harness.get_state().constraints
            if not self._commit_request(turn, snapshot, p.sources):
                return ParsedInput()
        except (ValueError, TypeError):
            self._preserve_request(turn, "REQUEST_COMMIT_REJECTED")
            self.harness.record_host_event("REQUEST_APPLY_REJECTED", proposal_id=p.proposal_id,
                                           revision=p.revision, reason_code="REQUEST_COMMIT_REJECTED")
            return ParsedInput(errors=["Không áp dụng được proposal; dữ liệu đã chấp nhận được giữ. Hãy sửa hoặc xem lại."])
        if self.harness.get_state().constraints != before:
            self._offers, self._inspected_offers = None, set()
        receipt = {"proposal_id": p.proposal_id, "revision": p.revision, "snapshot_hash": p.snapshot_hash,
            "source_turn_ids": sorted({s["source_turn_id"] for s in p.sources.values() if s["source_turn_id"]}),
            "applied_by_turn_id": turn.turn_id, "evidence_mode": self._review_config.evidence_mode,
            "review_protocol": REVIEW_PROTOCOL_VERSION}
        self._last_request_receipt = receipt
        self.harness.record_host_event("REQUEST_PROPOSAL_APPLIED", **receipt)
        self.harness.record_host_event("INPUT_INTERPRETATION_APPLIED", actual_user_turn_id=turn.turn_id,
            source_turn_id=p.latest_source_turn_id, source_turn_ids=receipt["source_turn_ids"], intent="UPDATE_SEARCH",
            snapshot_hash=p.snapshot_hash, revision=p.revision)
        self._pending_interpretation = None
        return ParsedInput()

    def get_state(self):
        with self._lock:
            return deepcopy({"pattern": self.name, **self._public_context(),
                             "last_model_error": self._last_model_error,
                             "messages": messages_to_dict(self._messages)})

    def _record_model_error(self, exc, *, node, stage):
        # Diagnostics are operator-visible, not fed back as model instructions.
        private = [m.content for m in self._messages if isinstance(m, HumanMessage) and isinstance(m.content, str)]
        private += [value for p in self.harness.get_state().passengers for value in
                    (p.first_name, p.last_name, p.document_number)]
        diagnostic = failure_diagnostic(exc, node=node, stage=stage,
                                        secrets=getattr(self, "_provider_secrets", ()), private_values=private)
        if node == "conversation.input":
            diagnostic["schema_hash"] = self._input_interpreter.schema_hash
            diagnostic["structured_output_method"] = "json_schema"
            diagnostic["evidence_mode"] = self._review_config.evidence_mode
            diagnostic["review_protocol"] = REVIEW_PROTOCOL_VERSION
        self._last_model_error = diagnostic
        self.harness.record_model_failure(type(exc).__name__, diagnostic=diagnostic)
        self._model_error = True
        return diagnostic

    def get_backend_state(self):
        return self.harness.get_backend_state()

    def get_trace(self):
        return self.harness.get_trace()

    def get_handoff(self):
        return self.harness.get_handoff()

    def check_completion(self):
        return self.harness.check_completion()

    def advance_time(self, seconds):
        with self._lock:
            return self.harness.advance_time(seconds)

    def export_artifacts(self, output_dir):
        with self._lock:
            directory = self.harness.export_artifacts(output_dir)
            # Harness created a NEW directory, so no existing evidence is overwritten.
            (Path(directory) / "conversation.json").write_text(
                json.dumps(self.get_state(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return directory

    def _parse_user_input(self, turn):
        visible = [o.offer.offer_id for o in self._offers or []]
        clean = normalize(turn.text).strip(" .!")
        pending = self._pending_interpretation
        if clean == "/proposal show":
            self._query_topic = "REQUEST"
            return ParsedInput()
        if clean == "/proposal keep":
            return self._keep_request(turn)
        if clean in {"ap dung", "apply"}:
            if pending is None:
                return ParsedInput(errors=["Không có diễn giải đang chờ áp dụng; đây không phải xác nhận quote."])
            if isinstance(pending, RequestProposal):
                return self._apply_request(turn, pending)
            self._pending_interpretation = None
            if pending["basis"] != self._interpretation_basis():
                return ParsedInput(errors=["Diễn giải đã cũ; hãy gửi lại yêu cầu."])
            parsed = deepcopy(pending["parsed"])
            if not parsed["requirements"] and not parsed["cancel"]:
                self.harness.resume_input(turn.turn_id)
            self._blocked_scope = None
            self.harness.record_host_event("INPUT_INTERPRETATION_APPLIED", actual_user_turn_id=turn.turn_id,
                source_turn_id=pending["source_turn_id"], intent=pending["intent"])
            return ParsedInput(**parsed)
        # Commands/profiles and exact approval/cancel controls are user ingress,
        # not interpretations. No LLM output can manufacture confirmation.
        control = clean in {"dong y", "xac nhan", "toi dong y", "toi xac nhan", "yes", "confirm",
            "khong dong y", "khong xac nhan", "no", "tiep tuc", "continue", "huy", "huy dat ve",
            "dung", "dung dat ve", "cancel"}
        explicit = (turn.text.strip().startswith("/") or clean.startswith("hanh khach:") or
                    bool(re.fullmatch(r"chon\s+(?:so\s+)?(?:\d+|off-[\w-]+)", clean)))
        if control or explicit:
            parsed = parse_user_input(turn.text, visible)
            if isinstance(pending, RequestProposal) and (parsed.selection_requested or parsed.passengers_supplied):
                return ParsedInput(errors=["Proposal yêu cầu chưa được duyệt; hãy sửa hoặc /proposal keep, rồi áp dụng trước khi chọn offer/hồ sơ."])
            if not isinstance(pending, RequestProposal) and (
                    parsed.passengers_supplied or parsed.selection_requested or parsed.cancel):
                self._pending_interpretation = None
            return parsed
        if self._input_interpreter is None:
            # Offline mode is explicitly not an NLP evaluation. Keep deterministic
            # fixtures and support read-only inspection without continuing tools.
            if clean in {"ve toi dang chon la gi", "toi dang chon ve nao", "ve dang chon", "current selection"}:
                self._query_topic = "CURRENT_SELECTION"
                return ParsedInput()
            return parse_user_input(turn.text, visible)
        if not self.harness.record_model_decision(node="conversation.input"):
            return ParsedInput()
        context = {"actual_user_turn": turn.model_dump(mode="json"), "public_state": self._public_context()}
        stage = "api_call"
        try:
            raw, decision = self._input_interpreter.interpret(context, self._messages)
            self._messages.append(raw.model_copy(deep=True))  # Preserve Gemini thought signatures.
            self.harness.record_host_event("MODEL_OUTPUT", text=decision.model_dump_json(), tool_calls=[], usage=raw.usage_metadata)
            if decision.intent == "UPDATE_SEARCH":
                stage = "proposal_validation"
                self._prepare_request(turn, decision)
                parsed = ParsedInput()
            else:
                stage = "grounding"
                parsed = grounded_input(decision, turn.text, visible)
            self.harness.record_host_event("INPUT_INTERPRETATION", user_turn_id=turn.turn_id,
                                          decision=decision.model_dump(mode="json"))
            if decision.intent == "ASK_INFO":
                self._query_topic = decision.topic
            elif decision.intent == "CLARIFY":
                self._preserve_request(turn, "CLARIFICATION_REQUIRED")
                self._input_question = QUESTIONS[decision.clarification]
                # Ambiguity must not silently authorize continuation of old work.
                if self.harness.get_state().boundary == TransactionBoundary.POST_TRANSACTION:
                    self.harness.request_handoff("AMBIGUOUS_INPUT_POST_TRANSACTION")
                else:
                    self._blocked_scope = "passengers" if decision.clarification in {"PASSENGERS", "DOCUMENT_TYPE"} else "request"
                    self.harness.block_input(turn.turn_id)
            elif decision.intent == "CONTINUE":
                # Hosted semantic guesses never resume a transaction; use an
                # explicit continue control or provide actionable user input.
                self._input_question = "Bạn muốn bổ sung thông tin hay tiếp tục? Nhắn riêng 'tiếp tục' nếu muốn tiếp tục phiên."
            elif decision.intent != "UPDATE_SEARCH":
                if isinstance(self._pending_interpretation, RequestProposal):
                    raise ValueError("Resolve request proposal before other mutations")
                stage = "proposal_validation"
                self._prepare_interpretation(turn, decision, parsed)
            return ParsedInput()
        except Exception as exc:
            self._preserve_request(turn, "INPUT_INTERPRETATION_REJECTED")
            if not isinstance(self._pending_interpretation, RequestProposal):
                self._pending_interpretation = None
            diagnostic = self._record_model_error(exc, node="conversation.input", stage=stage)
            self.harness.record_host_event("INPUT_INTERPRETATION_REJECTED", user_turn_id=turn.turn_id,
                                          exception_type=type(exc).__name__, diagnostic=diagnostic)
            if self.harness.get_state().terminal_status is None:
                if self.harness.get_state().boundary == TransactionBoundary.POST_TRANSACTION:
                    self.harness.request_handoff("INPUT_INTERPRETATION_FAILED_POST_TRANSACTION")
                else:
                    self._blocked_scope = "request"
                    self.harness.block_input(turn.turn_id)
            return ParsedInput(errors=["Chưa tiếp nhận diễn giải input; dữ liệu đã chấp nhận vẫn được giữ."])

    def _interpretation_basis(self):
        state = self.harness.get_state()
        return {"context": state.context.model_dump(mode="json"), "constraints": state.constraints.model_dump(mode="json") if state.constraints else None,
                "selected_offer_id": state.selected_offer_id, "quote_id": state.quote.quote_id if state.quote else None,
                "passengers": [p.model_dump(mode="json") for p in state.passengers],
                "draft": deepcopy(self._draft), "request_revision": self._request_revision,
                "offer_ids": [o.offer.offer_id for o in self._offers or []]}

    def _prepare_interpretation(self, turn, decision, parsed):
        state = self.harness.get_state()
        if state.boundary == TransactionBoundary.POST_TRANSACTION:
            self.harness.request_handoff("INTERPRETED_CHANGE_POST_TRANSACTION")
            return
        if parsed.requirements:
            preview = {**self._draft, **parsed.requirements}
            errors = request_errors(preview)
            if errors:
                raise ValueError("Invalid proposed request")
            if not missing_fields(preview):
                constraints = SearchConstraints.model_validate(preview)
                if constraints.departure_date < self.harness.current_time().astimezone(VIETNAM_TZ).date():
                    raise ValueError("Past departure date")
        elif parsed.passengers_supplied:
            if not state.constraints or not state.selected_offer_id:
                self._input_question = "Hãy tìm và chọn offer trước khi cung cấp hồ sơ hành khách."
                return
            try:
                passengers = [Passenger.model_validate(p) for p in parsed.passengers]
                if self.harness.constraint_policy.validate_passengers(passengers, state.constraints.passenger_count,
                                                                     self.harness.current_time().astimezone(VIETNAM_TZ).date()):
                    raise ValueError("Invalid proposed passengers")
            except ValueError:
                category = "DOCUMENT_TYPE" if any(not p.get("document_type") for p in parsed.passengers) else "PASSENGERS"
                self._input_question = QUESTIONS[category]
                self._blocked_scope = "passengers"
                self.harness.block_input(turn.turn_id)
                return
            preview = {"passengers": [p.model_dump(mode="json") for p in passengers]}
        elif parsed.selection_requested:
            preview = {"selected_offer_id": parsed.offer_id}
        else:
            preview = {"cancel": True}
        self.harness.block_input(turn.turn_id)
        self._blocked_scope = "request" if parsed.requirements else "passengers" if parsed.passengers_supplied else "selection"
        self._pending_interpretation = {"intent": decision.intent, "preview": preview, "parsed": asdict(parsed),
            "source_turn_id": turn.turn_id, "basis": self._interpretation_basis()}

    def _input_paused(self):
        return bool(self._pending_interpretation or self._input_question or self._query_topic or
                    self._input_errors or self._request_error or self._model_error or self.harness.get_state().input_blocked)

    def _ingest(self, turn):
        self._input_question = self._query_topic = None
        parsed = self._parse_user_input(turn)
        self._input_errors = parsed.errors
        if parsed.errors and isinstance(self._pending_interpretation, RequestProposal) and self._pending_interpretation.status == "READY":
            self._preserve_request(turn, "NATIVE_INPUT_REJECTED")
        self._notice = None
        state_before = self.harness.get_state()
        if parsed.cancel:
            self._pending_interpretation = None
            self.harness.cancel(turn.turn_id)
            return
        if parsed.requirements or parsed.replace_requirements:
            draft = parsed.requirements if parsed.replace_requirements else {**self._draft, **parsed.requirements}
            try:
                normalized = normalize_snapshot(draft, self._review_today()) if state_before.boundary == TransactionBoundary.PRE_TRANSACTION else draft
                sources = merge_sources(normalized, {} if parsed.replace_requirements else self._accepted_sources,
                    {k: source_record("USER_COMMAND", turn.turn_id) for k in parsed.requirements if k in normalized})
                if self._commit_request(turn, normalized, sources):
                    self._pending_interpretation = None
            except (ValueError, TypeError):
                self._preserve_request(turn, "NATIVE_REQUEST_REJECTED")
                if not self.harness.get_state().terminal_status:
                    self.harness.block_input(turn.turn_id)
                self._blocked_scope = "request"
                self._request_error = "Yêu cầu không hợp lệ; hãy sửa sân bay, ngày, số người, ngân sách và giờ bay."
            if self.harness.get_state().constraints != state_before.constraints:
                self._offers, self._inspected_offers = None, set()
        if self.harness.get_state().terminal_status:
            return
        if parsed.selection_requested:
            try:
                # Even an invalid index clears old selection/approval, never
                # silently continue charging the previously selected offer.
                self.harness.select_offer(turn.turn_id, parsed.offer_id or "INVALID-SELECTION")
                self._inspected_offers.discard(parsed.offer_id)
            except (ValueError, TypeError):
                self._input_errors.append("Hãy chọn một offer trong kết quả search của phiên này.")
        if self.harness.get_state().terminal_status:
            return
        if parsed.passengers_supplied:
            try:
                response = self.harness.set_passengers(turn.turn_id, parsed.passengers)
                self._passengers_blocked = not response.ok
                if response.ok and self._blocked_scope == "passengers" and not self._request_error:
                    self.harness.resume_input(turn.turn_id)
                    self._blocked_scope = None
            except (ValueError, TypeError):
                self._passengers_blocked = True
            if self._passengers_blocked:
                state = self.harness.get_state()
                if state.terminal_status is None and not state.input_blocked:
                    # Do not let a rogue business model reprice old valid
                    # profiles after the user submitted a rejected replacement.
                    self._blocked_scope = "passengers"
                    self.harness.block_input(turn.turn_id)
                self._input_errors.append("Cần đủ N hồ sơ hợp lệ, không trùng giấy tờ; không dùng dữ liệu thật trong lab.")
        state = self.harness.get_state()
        if not self._query_topic and not self._input_question and not state.input_blocked and state.phase == SessionPhase.WAITING_CONFIRMATION and state.displayed_quote_id and not (
            parsed.requirements or parsed.replace_requirements or parsed.selection_requested or parsed.passengers_supplied
        ):
            try:
                self.harness.confirm_quote(turn.turn_id, state.displayed_quote_id)
            except ValueError:
                self._notice = "Chưa xác nhận hợp lệ (hoặc quote hết hạn). Hãy xem quote hiện hành rồi nhắn riêng 'đồng ý'."
        if self._input_errors or self._request_error:
            self.harness.record_host_event("INPUT_REJECTED", user_turn_id=turn.turn_id,
                                          reasons=self._input_errors + ([self._request_error] if self._request_error else []))

    def send(self, message, session_id=None):
        with self._lock:
            actual_session = self.harness.get_state().context.session_id
            if session_id is not None and session_id != actual_session:
                raise ValueError("This adapter belongs to a different session")
            if not isinstance(message, str) or not message.strip():
                raise ValueError("User message must be nonempty text")
            turn = self.harness.receive_user_turn(message)
            self._messages.append(HumanMessage(content=message, id=turn.turn_id))
            self._model_error = False
            self._last_model_error = None
            # Terminal episodes need explicit reset. Requirement changes after
            # a transaction still produce a handoff, including after ticketing.
            state = self.harness.get_state()
            if state.terminal_status is None or state.boundary == TransactionBoundary.POST_TRANSACTION:
                self._ingest(turn)
            if self.harness.get_state().terminal_status is None and not self._input_paused():
                self._run_agent()
            answer = self._render_response()
            self._messages.append(AIMessage(content=answer))
            state = self.harness.get_state()
            waiting = state.terminal_status is None
            self.harness.record_host_event("AGENT_RESPONSE", text=answer, phase=state.phase.value)
            if waiting:
                self.harness.record_host_event("SESSION_PAUSED", phase=state.phase.value)
            return AgentTurnResult(assistant_message=answer, session_id=actual_session,
                phase=state.phase, terminal_status=state.terminal_status, waiting_for_user=waiting,
                completion=self.harness.check_completion())

    def _run_agent(self):
        raise NotImplementedError

    def _observe(self, name, response):
        self._last_tool_result = deepcopy(response)
        if response["ok"]:
            if name == "search_flights":
                self._offers = [OfferDetails.model_validate(o) for o in response["data"]["offers"]]
            elif name == "get_offer":
                self._inspected_offers.add(response["data"]["offer"]["offer_id"])
        elif response["code"] in {"SOLD_OUT", "OFFER_EXPIRED"}:
            self._offers, self._inspected_offers = None, set()

    def _pause_after_observation(self):
        state = self.harness.get_state()
        return bool(state.terminal_status or
            (state.quote and state.displayed_quote_id != state.quote.quote_id and not state.confirmation) or
            (not state.selected_offer_id and self._offers is not None) or
            (state.selected_offer_id and not self._passengers_valid()))

    def _render_request_proposal(self, *, bind=False):
        p = self._pending_interpretation
        if not isinstance(p, RequestProposal):
            return ""
        if bind and p.status == "READY":
            # Locate the latest genuine user turn; model output may follow it.
            user = next(m for m in reversed(self._messages) if isinstance(m, HumanMessage))
            turn_number = int(user.id.rsplit("user-", 1)[1])
            if p.displayed_revision != p.revision or p.displayed_hash != p.snapshot_hash:
                p = p.model_copy(update={"displayed_revision": p.revision,
                    "displayed_hash": p.snapshot_hash, "displayed_at_turn": turn_number})
                self._pending_interpretation = p
                self.harness.record_host_event("REQUEST_PROPOSAL_DISPLAYED", proposal_id=p.proposal_id,
                    revision=p.revision, snapshot_hash=p.snapshot_hash, after_user_turn_id=user.id)
        request = p.snapshot
        soft = ", ".join(request.get("preferred_airlines") or []) or "không có ưu tiên"
        hard = ", ".join(request.get("allowed_airlines") or []) or "không giới hạn hãng"
        text = (f"Mình hiểu yêu cầu là (proposal #{p.revision}, {p.status}): "
                f"{json.dumps(request, ensure_ascii=False)}. Ưu tiên (không bắt buộc): {soft}. "
                f"Hãng được phép (bắt buộc): {hard}. Ngân sách là tổng cho cả nhóm.")
        missing = missing_fields(request)
        if missing:
            text += " Cần bổ sung: " + ", ".join(FIELD_LABELS[f] for f in missing) + "."
        if p.warnings:
            text += " CẢNH BÁO evidence chưa được kiểm chứng: " + "; ".join(
                f"{w['field']} ({w['reason_code']})" for w in p.warnings) + ". Hãy kiểm tra toàn bộ thông tin."
        if p.last_rejected_turn_json:
            text += " Input bị từ chối trước đó không được áp dụng."
        if p.status == "READY":
            text += " Nhắn riêng 'áp dụng' để chấp nhận toàn bộ snapshot, hoặc gửi thông tin sửa tiếp."
        else:
            text += " Proposal được giữ nhưng KHÓA áp dụng. Hãy sửa yêu cầu hoặc /proposal keep để xem lại rồi xác nhận ở lượt mới."
        return text + " Bước này KHÔNG xác nhận quote và chưa cho phép giữ chỗ/thanh toán."

    def _render_response(self):
        """Business claims always come from code, never unverified assistant prose."""
        state = self.harness.get_state()
        if state.terminal_status == TerminalStatus.HANDOFF_REQUIRED:
            packet = self.harness.get_handoff()
            return (f"HANDOFF_REQUIRED: {packet.reason}. Booking: {packet.booking_id or '-'}; "
                    f"payment operation: {packet.payment_operation_id or '-'}. {packet.recommended_action}")
        if state.terminal_status == TerminalStatus.BOOKED:
            report = self.harness.check_completion()
            if not report.booking_completed:
                self.harness.request_handoff("COMPLETION_CHECK_FAILED")
                return self._render_response()
            return f"BOOKED: {state.booking_id}; payment {state.payment_id}; vé: {', '.join(state.ticket_numbers)}. Đã kiểm chứng backend."
        if state.terminal_status == TerminalStatus.FAILED:
            status = state.payment_status.value if state.payment_status else "-"
            return f"FAILED: thanh toán bị từ chối hoặc yêu cầu không hoàn tất; không báo đã xuất vé. Booking: {state.booking_id or '-'}; payment: {status}."
        if state.terminal_status == TerminalStatus.CANCELLED:
            return "CANCELLED: đã dừng trước giao dịch; không tạo booking/payment."
        if self._query_topic:
            return self._render_information(self._query_topic)
        prefix = "\n".join(self._input_errors + ([self._request_error] if self._request_error else []) +
                           ([self._notice] if self._notice else []))
        if self._model_error:
            error = self._last_model_error or {}
            if error.get("category") == "provider":
                status = error.get("http_status")
                body = f"Google từ chối request (HTTP {status})." if status and 400 <= status < 500 else "Gọi model thất bại."
                if status == 400:
                    body += " Cần kiểm tra request/schema/cấu hình API, không phải yêu cầu bạn bổ sung tuyến/ngày."
            elif error.get("category") == "parse":
                body = "Model đã trả phản hồi nhưng không parse được JSON diễn giải."
            elif error.get("category") == "validation":
                body = "Diễn giải bị từ chối vì schema, bằng chứng hoặc dữ liệu không hợp lệ."
            elif error.get("category") == "response":
                body = "Phản hồi model bị chặn, thiếu dữ liệu hoặc không đúng loại kết quả yêu cầu."
            else:
                body = "Model chưa trả lời được hoặc gặp lỗi xử lý nội bộ."
            return ((prefix + "\n" if prefix else "") + body + " State và keys được giữ nguyên; xem /trace hoặc /state → last_model_error." +
                    ("\n" + self._render_request_proposal() if isinstance(self._pending_interpretation, RequestProposal) else ""))
        if self._input_question:
            return self._input_question + ("\n" + self._render_request_proposal() if isinstance(self._pending_interpretation, RequestProposal) else "")
        if isinstance(self._pending_interpretation, RequestProposal):
            return (prefix + "\n" if prefix else "") + self._render_request_proposal(bind=not bool(prefix))
        if self._pending_interpretation:
            preview = json.dumps(self._pending_interpretation["preview"], ensure_ascii=False)
            request = self._pending_interpretation["preview"]
            meaning = ""
            if self._pending_interpretation["intent"] == "UPDATE_SEARCH":
                soft = ", ".join(request.get("preferred_airlines") or []) or "không có ưu tiên"
                hard = ", ".join(request.get("allowed_airlines") or []) or "không giới hạn hãng"
                meaning = f" Ưu tiên (không bắt buộc): {soft}. Hãng được phép (bắt buộc): {hard}. Ngân sách là tổng cho cả nhóm."
            return (f"Mình hiểu yêu cầu là: {preview}.{meaning} Nhắn riêng 'áp dụng' để chấp nhận, hoặc gửi lại thông tin. "
                    "Bước này KHÔNG xác nhận quote và chưa cho phép giữ chỗ/thanh toán.")
        if state.input_blocked:
            fields = missing_fields(self._draft)
            missing = " Cần bổ sung: " + ", ".join(FIELD_LABELS[f] for f in fields) + "." if fields else ""
            return ((prefix + "\n" if prefix else "") + "Phiên đang tạm dừng vì input chưa hợp lệ/chưa áp dụng. "
                    "Hãy sửa yêu cầu; thông tin đã chấp nhận vẫn được giữ." + missing)
        if state.constraints is None:
            fields = missing_fields(self._draft)
            body = ("Cần bổ sung: " + ", ".join(FIELD_LABELS[f] for f in fields) + ".") if fields else "Hãy sửa yêu cầu bằng /request {...}."
        elif not state.selected_offer_id:
            if self._offers is None:
                body = "Chưa có kết quả search hợp lệ. Nhắn 'tiếp tục' hoặc sửa yêu cầu."
            elif not self._offers:
                body = "Không có chuyến phù hợp. Bạn có thể đổi ngày/tuyến/ngân sách; tôi không tự nới điều kiện."
            else:
                lines = ["Các offer (tổng giá cho cả nhóm):"]
                for i, details in enumerate(self._offers, 1):
                    lines.append(f"{i}. {details.offer.offer_id} | {details.flight.airline.value} | "
                        f"{details.flight.departure_time.isoformat()} | {details.total_price_vnd:,} VND")
                body = "\n".join(lines) + "\nNhắn 'chọn 1' hoặc 'chọn OFF-...'."
        elif not self._passengers_valid():
            body = (f"Đã chọn {state.selected_offer_id}. Cần {state.constraints.passenger_count} hồ sơ giả. "
                    "Nhắn 'hành khách: Họ|Tên|YYYY-MM-DD|IDENTITY_CARD|Số giấy tờ; ...' hoặc /passengers [JSON].")
        elif state.quote and not state.confirmation and not state.booking_id:
            quote = state.quote
            if state.displayed_quote_id != quote.quote_id:
                quote = self.harness.present_quote()
            body = (f"Quote {quote.quote_id} v{quote.quote_version}: {quote.origin} → {quote.destination}, "
                f"{quote.departure_date}, {quote.passenger_count} người; offer {quote.offer_id}.\n"
                f"Đơn giá {quote.unit_price_vnd:,} VND; tổng {quote.total_price_vnd:,} {quote.currency}. "
                f"Hành khách: {', '.join(p.last_name + ' ' + p.first_name for p in state.passengers)}.\n"
                f"Hết hạn {quote.expires_at.isoformat()}. Nhắn riêng 'đồng ý' để giữ chỗ, thanh toán mock và xuất vé; "
                "hoặc sửa thông tin/hủy.")
        else:
            body = (f"Phiên chưa hoàn tất: {state.phase.value}; booking {state.booking_id or '-'}, "
                    f"payment {state.payment_status.value if state.payment_status else '-'}. Nhắn 'tiếp tục' để tiếp tục cùng giao dịch.")
        return (prefix + "\n" if prefix else "") + body

    def _render_information(self, topic):
        """LLM chooses conversational intent; factual claims are state-bound."""
        state = self.harness.get_state()
        suspended = " Phiên đang tạm dừng vì có input chưa được chấp nhận." if state.input_blocked else ""
        if topic == "CURRENT_SELECTION":
            if not state.selected_offer_id:
                return "Bạn chưa chọn offer nào trong phiên hiện tại." + suspended
            details = next((o for o in self._offers or [] if o.offer.offer_id == state.selected_offer_id), None)
            text = f"Bạn đang chọn {state.selected_offer_id}."
            if details:
                text += (f" {details.flight.airline.value}, {details.flight.origin} → {details.flight.destination}, "
                         f"{details.flight.departure_time.isoformat()}; tổng giá khi tìm {details.total_price_vnd:,} VND.")
            return text + suspended
        if topic == "REQUEST":
            if isinstance(self._pending_interpretation, RequestProposal):
                return "Yêu cầu đã chấp nhận: " + json.dumps(self._draft, ensure_ascii=False) + "\n" + self._render_request_proposal()
            pending = (" Diễn giải CHƯA áp dụng: " + json.dumps(self._pending_interpretation["preview"], ensure_ascii=False)
                       if self._pending_interpretation else "")
            if not state.constraints:
                return "Yêu cầu chưa đầy đủ. Thông tin đang thu thập: " + json.dumps(self._draft, ensure_ascii=False) + pending
            return "Yêu cầu đã chấp nhận: " + state.constraints.model_dump_json() + pending + suspended
        if topic == "STATUS":
            return (f"Phiên ở bước {state.phase.value}; booking: {state.booking_id or '-'}, "
                    f"payment: {state.payment_status.value if state.payment_status else '-'}, "
                    f"tickets: {', '.join(state.ticket_numbers) or '-'}." + suspended)
        if topic == "OFFERS":
            return "Offer đã tìm: " + (", ".join(o.offer.offer_id for o in self._offers or []) or "chưa có") + suspended
        if topic == "GREETING":
            return "Chào bạn! Bạn muốn bay từ đâu đến đâu, ngày nào và cho bao nhiêu người?"
        return ("Bạn có thể nói yêu cầu tự nhiên; diễn giải của LLM cần 'áp dụng' trước khi sử dụng. "
                "Dùng /request {JSON}, /passengers [JSON], 'chọn 1' hoặc /help. "
                "Hồ sơ giả: hành khách: Họ|Tên|YYYY-MM-DD|IDENTITY_CARD|Số giấy tờ.")
