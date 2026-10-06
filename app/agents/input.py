"""Conservative, local user ingress. No model-generated confirmation or PII.

Supported Vietnamese request phrases and structured commands are deliberately
bounded; unknown/ambiguous input asks for clarification instead of inventing
route, date, count, budget or passenger documents.
"""

import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from app.models import SearchConstraints


REQUIRED_FIELDS = ("origin", "destination", "departure_date", "passenger_count")
FIELD_LABELS = {"origin": "điểm đi", "destination": "điểm đến",
                "departure_date": "ngày bay (YYYY-MM-DD)", "passenger_count": "số hành khách"}
AIRPORT_ALIASES = {
    "sgn": "SGN", "sai gon": "SGN", "tp hcm": "SGN", "tphcm": "SGN",
    "ho chi minh": "SGN", "tan son nhat": "SGN",
    "han": "HAN", "ha noi": "HAN", "noi bai": "HAN",
    "dad": "DAD", "da nang": "DAD",
}


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    return " ".join("".join(c for c in value if unicodedata.category(c) != "Mn").split())


@dataclass
class ParsedInput:
    requirements: dict[str, Any] = field(default_factory=dict)
    replace_requirements: bool = False
    offer_id: str | None = None
    selection_requested: bool = False
    passengers: Any = None
    passengers_supplied: bool = False
    cancel: bool = False
    errors: list[str] = field(default_factory=list)


def _airport(value):
    return AIRPORT_ALIASES.get(normalize(value), value.strip().upper())


def _date(value):
    if "/" in value:
        day, month, year = map(int, value.split("/"))
        return date(year, month, day).isoformat()
    return date.fromisoformat(value).isoformat()


def parse_user_input(text: str, visible_offer_ids: list[str]) -> ParsedInput:
    """Only read the actual user's text. Structured fields are still untrusted."""
    result = ParsedInput()
    clean = normalize(text)
    if clean in {"huy", "huy dat ve", "dung", "dung dat ve", "cancel"}:
        result.cancel = True
        return result
    if clean.strip(" .!") in {"dong y", "xac nhan", "toi dong y", "toi xac nhan", "yes", "confirm",
                              "khong dong y", "khong xac nhan", "no", "tiep tuc", "continue"}:
        # The harness, not this parser, decides whether this genuine turn
        # grants approval. 'tôi'/'tới' and 'tục'/'từ' must not become airports.
        return result
    command, _, payload = text.strip().partition(" ")
    if command.lower() in {"/request", "/passengers"}:
        try:
            value = json.loads(payload)
            if command.lower() == "/request":
                if not isinstance(value, dict):
                    raise ValueError("Expected a JSON object")
                # Extra fields reach harness validation; never silently drop them.
                result.requirements = value
                result.replace_requirements = True
            else:
                result.passengers_supplied, result.passengers = True, value
        except (ValueError, TypeError):
            result.errors.append("JSON không hợp lệ; dùng /request {...} hoặc /passengers [...].")
            if command.lower() == "/request":
                result.requirements = {"_invalid_request": payload}
            else:
                result.passengers_supplied, result.passengers = True, payload
        return result
    if clean.startswith("hanh khach:"):
        result.passengers_supplied = True
        result.passengers = []
        for index, entry in enumerate(text.split(":", 1)[1].split(";"), 1):
            fields = [part.strip() for part in entry.split("|")]
            if len(fields) != 5:
                result.passengers = text  # Invalid payload reaches the harness counter.
                result.errors.append("Hồ sơ: Họ|Tên|YYYY-MM-DD|IDENTITY_CARD hoặc PASSPORT|Số giấy tờ; ...")
                break
            last_name, first_name, dob, document_type, document_number = fields
            result.passengers.append({"passenger_id": f"P{index}", "first_name": first_name,
                "last_name": last_name, "date_of_birth": dob, "document_type": document_type.upper(),
                "document_number": document_number})
        return result
    if "|" in text:
        # An unprefixed profile must never fall through to departure-date
        # extraction, even when the DOB or document type is invalid.
        if all(len(entry.split("|")) == 5 for entry in text.split(";")):
            return parse_user_input("hành khách: " + text, visible_offer_ids)
        result.passengers_supplied, result.passengers = True, text
        result.errors.append("Hồ sơ cần 5 trường: Họ|Tên|YYYY-MM-DD|IDENTITY_CARD hoặc PASSPORT|Số giấy tờ.")
        return result
    choice = re.fullmatch(r"(?:chon|/select)\s+(?:so\s+)?(\d+|off-[\w-]+)", clean)
    if choice:
        result.selection_requested = True
        value = choice[1]
        if value.isdigit():
            index = int(value) - 1
            if 0 <= index < len(visible_offer_ids):
                result.offer_id = visible_offer_ids[index]
            else:
                result.errors.append("Số lựa chọn không có trong danh sách đã hiển thị.")
        else:
            result.offer_id = value.upper()
        return result
    if re.match(r"^(?:chon|/select)\b", clean):
        result.selection_requested = True
        result.errors.append("Lựa chọn chưa rõ; dùng 'chọn 1' hoặc 'chọn OFF-...'.")
        return result
    if re.search(r"\b(?:doi|sua)\b.*\b(?:hanh khach|ho ten|giay to)\b", clean):
        result.passengers_supplied, result.passengers = True, text
        result.errors.append("Hãy gửi lại đủ hồ sơ bằng 'hành khách:' hoặc /passengers [...].")
        return result
    # Explicit route delimiters; never infer a missing endpoint from a mention.
    endpoint = r"([a-z ]+?)"
    suffix = r"(?=\s+(?:ngay|cho|\d+\s+(?:ve|nguoi)|ngan sach|truoc|sau|uu tien|chi)\b|[,.;]|$)"
    route = re.search(r"(?:\btu\s+)?" + endpoint + r"\s*(?:->|→|\bden\b|\btoi\b|\bra\b)\s*" + endpoint + suffix, clean)
    if route:
        origin = re.sub(r"^.*\b(?:tu|ve)\s+", "", route[1]).strip()
        result.requirements.update(origin=_airport(origin), destination=_airport(route[2]))
    else:
        for label, target in ((r"diem di|tu", "origin"), (r"diem den|den", "destination")):
            # Diacritic folding must not turn 'tự thanh toán' into a route
            # change ('từ ...'), or 'đen' into 'đến'. Preserve the original cue.
            original_cue = r"\b(?:điểm đi|diem di|từ|tu)(?:\s*:\s*|\s+)" if target == "origin" else (
                r"\b(?:điểm đến|diem den|đến|den)(?:\s*:\s*|\s+)")
            if not re.search(original_cue, unicodedata.normalize("NFC", text), flags=re.IGNORECASE):
                continue
            part = re.search(r"(?:\b(?:" + label + r")(?:\s*:\s*|\s+))" + endpoint + suffix, clean)
            if part:
                result.requirements[target] = _airport(part[1])
        if clean in AIRPORT_ALIASES:
            result.errors.append("Hãy ghi rõ 'từ SGN' hoặc 'đến HAN'.")
    dates = re.findall(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})\b", clean)
    if dates:
        try:
            if len(set(dates)) != 1:
                raise ValueError("Ambiguous dates")
            result.requirements["departure_date"] = _date(dates[0])
        except ValueError:
            result.requirements["departure_date"] = dates[0] if len(dates) == 1 else None
            result.errors.append("Ngày không hợp lệ hoặc có nhiều ngày; hãy ghi một ngày YYYY-MM-DD.")
    elif re.search(r"\b(?:ngay|ngay mai|hom nay)\b", clean):
        result.requirements["departure_date"] = None
        result.errors.append("Cần ngày bay cụ thể có năm; ví dụ 2026-10-10.")
    counts = re.findall(r"\b(\d+)\s*(?:ve|nguoi|hanh khach)\b", clean)
    if counts:
        result.requirements["passenger_count"] = int(counts[0]) if len(set(counts)) == 1 else None
        if len(set(counts)) != 1:
            result.errors.append("Có nhiều số hành khách; hãy ghi một số duy nhất.")
    budget = re.search(r"(?:ngan sach|budget)\s*:?\s*(\d+(?:[.,]\d+)?)\s*(trieu|tr|vnd|dong)?\b", clean)
    if budget:
        number, unit = budget[1].replace(",", "."), budget[2]
        if unit in {"trieu", "tr"}:
            from decimal import Decimal
            result.requirements["max_budget_vnd"] = int(Decimal(number) * 1_000_000)
        elif number.isdigit():
            result.requirements["max_budget_vnd"] = int(number)
        else:
            result.requirements["max_budget_vnd"] = number  # No silent float/coercion.
    elif re.search(r"\b(?:ngan sach|gia duoi|toi da)\b", clean):
        result.requirements["max_budget_vnd"] = "UNPARSED"
        result.errors.append("Ngân sách chưa rõ; dùng 'ngân sách 5 triệu' hoặc max_budget_vnd trong /request.")
    if len(re.findall(r"\bngan sach\b", clean)) > 1:
        result.requirements["max_budget_vnd"] = "AMBIGUOUS"
        result.errors.append("Có nhiều ngân sách; hãy ghi một tổng ngân sách duy nhất.")
    for word, field_name, inclusive in (("truoc", "latest_departure", "latest_inclusive"),
                                         ("sau", "earliest_departure", "earliest_inclusive")):
        match = re.search(r"\b" + word + r"\s+(\d{1,2}):([0-9]{2})\b", clean)
        if match:
            result.requirements[field_name] = f"{int(match[1]):02d}:{match[2]}:00"
            result.requirements[inclusive] = False
        elif re.search(r"\b" + word + r"\s+\d", clean):
            result.requirements[field_name] = "UNPARSED"
            result.errors.append("Giờ bay cần dạng HH:MM, ví dụ 'trước 16:00'.")
        if len(re.findall(r"\b" + word + r"\s+\d", clean)) > 1:
            result.requirements[field_name] = "AMBIGUOUS"
            result.errors.append("Có nhiều giới hạn giờ bay cùng loại; dùng một giá trị hoặc /request.")
    airline = {"vietnam airlines": "Vietnam Airlines", "vietjet air": "VietJet Air", "vietjet": "VietJet Air",
               "bamboo airways": "Bamboo Airways", "bamboo": "Bamboo Airways"}
    for phrase, field_name in (("uu tien", "preferred_airlines"), ("chi", "allowed_airlines")):
        match = re.search(r"\b" + phrase + r"\s+(.+?)(?=,?\s+(?:chi|uu tien)\b|[.;]|$)", clean)
        if match:
            remaining, names = match[1], []
            for alias, value in airline.items():
                pattern = r"\b" + alias + r"\b"
                if re.search(pattern, remaining):
                    if value not in names:
                        names.append(value)
                    remaining = re.sub(pattern, "", remaining)
            remaining = re.sub(r"\b(?:va|hoac)\b|[,/&]", "", remaining).strip()
            result.requirements[field_name] = names if names and not remaining else ["UNPARSED_AIRLINE"]
            if remaining or not names:
                result.errors.append("Hãng chưa rõ/không hỗ trợ; dùng Vietnam Airlines, VietJet, Bamboo hoặc /request.")
    if "khong" in clean and any(alias in clean for alias in airline):
        result.requirements["allowed_airlines"] = ["UNPARSED_NEGATION"]
        result.errors.append("Điều kiện phủ định hãng cần ghi rõ allowed_airlines bằng /request; không tự nới điều kiện.")
    return result


def missing_fields(draft):
    return [name for name in REQUIRED_FIELDS if draft.get(name) is None]


def request_errors(draft):
    if missing_fields(draft):
        return []
    try:
        SearchConstraints.model_validate(draft)
        return []
    except ValueError:
        return ["Yêu cầu không hợp lệ; kiểm tra sân bay SGN/HAN/DAD, ngày, số người, ngân sách và giờ bay."]
