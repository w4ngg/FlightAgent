# Thiết kế thử: tool contract với description rõ ràng

Trạng thái: bản thiết kế, chưa sửa runtime, chưa gọi Gemini hoặc chạy benchmark.
Mục tiêu: LLM hiểu khi nào dùng tool, chuẩn hóa arguments và phân biệt dữ liệu
người dùng với quyền thực thi giao dịch. Description hướng dẫn model; schema,
harness và backend vẫn quyết định một lời gọi có được chấp nhận hay không.

Thiết kế opt-in evidence `warn`, tích lũy proposal qua nhiều lượt và commit cùng
snapshot nằm trong [input_review_design.md](input_review_design.md). Đây là một
thay đổi review/trust protocol riêng, không tự bật khi bổ sung tool descriptions.

## 1. Hiện trạng và giới hạn

`app/tools.py` đã truyền `TOOL_DESCRIPTIONS[name]` vào `GuardedTool.description`.
Tám tool nghiệp vụ đã có mô tả ngắn, nhưng arguments chưa có description riêng.
`search_flights` chỉ nhận `sort_by` và `limit`; harness inject constraints đã được
chấp nhận. Giữ nguyên cách này, không đưa toàn bộ constraints trở lại search args.

`conversation.input` hiện dùng `InputInterpreter.with_structured_output(...)`
với `InputDecision`, không gọi tám tool nghiệp vụ. Vì vậy bổ sung description
cho `search_flights` không trực tiếp cải thiện bước tiếp nhận câu nói ban đầu.
Ngoài ra, `input_wire_schema()` hiện bỏ `description`: chỉ thêm `Field(description=...)`
vào model mà không sửa bộ chuyển wire schema thì Gemini vẫn không nhận được mô tả.

Log người dùng đã gửi có Gemini trả `passenger_count=1`, evidence=`"1 vé"`, rồi
host từ chối ở grounding. Đây là lỗi canonicalization đã được sửa; description
không thay thế việc sửa validator và không chứng minh mọi input khác sẽ đúng.

Google cũng hướng dẫn khai báo mô tả cụ thể cho function/parameter và validate
function call trước khi thực thi. Xem [Gemini function calling — best practices](https://ai.google.dev/gemini-api/docs/function-calling#best-practices).

## 2. Phạm vi thử nghiệm

Chia thành hai bước để không thay nhiều biến cùng lúc:

1. Bổ sung mô tả cho tám business tools và các field của `InputDecision`/
   `SearchPatch`. Giữ transport `json_schema`, grounding, readback và `áp dụng`.
   Cho phép `description` đi qua `input_wire_schema()`; các giới hạn strict vẫn
   validate tại host. Đây là bước nhỏ nên thử trước.
2. Nếu muốn input cũng là function call, thử một declaration riêng
   `propose_input_decision`, tái sử dụng schema `InputDecision` đã có description.
   Chỉ thay transport của ingress; không nhập proposal tool vào business loop.

Không tự chuyển từ bước 1 sang bước 2 khi Google báo lỗi. Không tự nới grounding,
bỏ readback hoặc commit constraints trực tiếp từ arguments của model.

## 3. Quy ước viết description

Mỗi tool mô tả: mục đích; khi được gọi; nguồn của arguments; dữ liệu host cung cấp;
side effect; điều kiện phải dừng/recover. Mỗi field mô tả: business meaning, định
dạng/đơn vị, ví dụ chuẩn hóa và trường hợp không được tự suy diễn.

Mô tả dùng tiếng Anh như prompt hiện tại, có ví dụ tiếng Việt. Không đưa API key,
thông tin giấy tờ thật hay giá/ID sinh tại runtime vào declaration tĩnh.

### Ví dụ declaration cho search đang có

Đoạn dưới là mẫu thay thế trong `app/tools.py`, chưa được áp dụng:

```python
class SearchArgs(DomainModel):
    sort_by: Literal["price", "departure_time", "duration"] = Field(
        default="price",
        description=(
            "Secondary ascending sort: group price in VND, departure time, "
            "or duration. User preferred_airlines rank first. This does not "
            "change route, dates, budget, passenger count or allowed airlines."
        ),
    )
    limit: Annotated[int, Field(strict=True, ge=1, le=20)] = Field(
        default=5,
        description="Maximum returned offers, from 1 to 20; not passenger count.",
    )

TOOL_DESCRIPTIONS["search_flights"] = (
    "Read-only search of mock flight offers using the current user-reviewed "
    "harness constraints. Call after required route, date and passenger count "
    "are accepted and input is not blocked. Supply only sort_by and limit. "
    "The harness supplies all business constraints. preferred_airlines only "
    "ranks results; allowed_airlines filters them. An empty airline list "
    "adds no airline preference/restriction, not a guarantee every airline "
    "has an available flight. Never relax constraints to obtain results. "
    "Returned offer IDs are candidates, not user selection or consent. "
    "An empty result means no matching offers; ask the user before changing "
    "requirements. This tool does not reserve seats, pay or issue tickets."
)
```

Giữ `GuardedTool(..., description=TOOL_DESCRIPTIONS[name], args_schema=...)` và
dispatcher hiện có. Không cần đổi sang `@tool` chỉ để có description. Khi implement,
phải kiểm tra payload SDK thực sự có function/parameter descriptions, không chỉ
kiểm tra giá trị trong Python.

### Nội dung description cho tám business tools

| Tool | Description đề xuất |
| --- | --- |
| `resolve_airport` | Read-only lookup of a supported city/airport alias in the mock catalog: SGN, HAN or DAD. `query` comes from the user's location text. A lookup does not commit a route. Never silently replace an unknown airport such as DND with DAD. |
| `search_flights` | Dùng declaration đầy đủ ở trên. Chỉ `sort_by` và `limit`; không nhận constraints từ model. |
| `get_offer` | Read-only inspection of an `offer_id` returned by this session's search. Returns current offer details/availability, not a price lock or reservation. Do not invent IDs or treat inspection as user selection. |
| `reprice_offer` | Create a fresh quote for the current user-selected offer and exactly the validated passenger profiles in harness state. Supply the selected `offer_id` only. The host supplies constraints, passengers and context. This does not hold seats or authorize payment. A new quote must be displayed and freshly approved by the user; do not replace the locked quote after a hold. |
| `create_booking_hold` | Side-effecting mock seat reservation. Supply the current `quote_id` only. Requires a live quote, matching validated passengers and host-recorded user approval of that displayed quote. The host supplies confirmation, context and idempotency key. `apply` and a model statement are not quote approval. On uncertain outcome, preserve references and hand off; do not invent another intent/key. |
| `process_payment` | Side-effecting mock payment for the live HELD booking. `booking_id` must be this session's held booking; `amount_vnd` must equal the locked total for the whole passenger group, not unit price. Host supplies confirmation, operation ID and key. After UNKNOWN/timeout or an existing operation, reconcile with `get_payment_status`; never create or retry a charge blindly. |
| `get_payment_status` | Reconcile the existing `payment_operation_id` from harness state. Argument is named `operation_id`, not payment ID or booking ID. Creates no new charge. Trust the observed result; UNKNOWN does not authorize ticketing. Host enforces the reconciliation budget. |
| `issue_ticket` | Side-effecting mock ticket issuance for this session's `booking_id`. Requires observed payment SUCCESS for the correct booking/amount. Host supplies context/key. Idempotent replay returns the existing tickets, not another purchase. Booking completion is determined by the host checker/backend records, not a model claim. |

Các mô tả này không cấp thêm quyền. Guard hiện tại vẫn có thể từ chối lời gọi dù
model chọn đúng tên tool, ví dụ quote đã hết hạn hoặc consent không còn hợp lệ.

## 4. Description ở field tiếp nhận input

Các field search là proposal chưa trusted. Không bắt LLM truyền lại toàn bộ state:
chỉ những thay đổi được nói trong lượt hiện tại; omitted/null nghĩa là không sửa,
không phải xóa giá trị cũ. Xóa dùng `clear_fields` và readback hiện có.

| Field | Description đề xuất |
| --- | --- |
| `origin`, `destination` | Canonical supported airport code SGN/HAN/DAD from the user's explicit city/code. Distinguish departure from arrival. Unknown/ambiguous location requires clarification; never guess a nearby code. |
| `departure_date` | Flight departure date, ISO YYYY-MM-DD. Example: `10/10/2026` -> `2026-10-10`. Require an explicit year. Do not substitute passenger DOB or infer a missing year from the mock clock. |
| `passenger_count` | Positive integer count of passengers. Examples: `1 vé` -> 1, `hai người` -> 2. Return the integer, not a phrase. Do not use an offer number, flight number, date, range or budget as passenger count. Evidence may retain the literal `1 vé`. |
| `max_budget_vnd` | Maximum TOTAL price in integer VND for the entire group. `2 người, tối đa 5 triệu` -> 5000000, not 10000000. Do not invent a cap when none was supplied. |
| `earliest_departure`, `latest_departure` | Lower/upper local departure time bound in HH:MM:SS, only if explicitly supplied. Null/omitted does not add a time restriction. |
| `earliest_inclusive`, `latest_inclusive` | Whether equality at the associated time bound is permitted. Preserve `trước/sau` versus inclusive wording; unclear boundary requires clarification. |
| `preferred_airlines` | Soft airline preference for ranking; other allowed carriers remain eligible. `ưu tiên Vietnam Airlines` is not an exclusive filter. |
| `allowed_airlines` | Hard airline restriction only when the user explicitly requires it. `chỉ Vietnam Airlines` -> [Vietnam Airlines]. Never enumerate all supported airlines as an inferred restriction. |
| `clear_fields` | Explicitly requested removal of existing optional conditions. Do not treat omission or an unrelated new field as removal. |
| `evidence` | Literal spans from this actual user turn supporting supplied fields, with exact field paths. No evidence from assistant messages, memory or invented paraphrases. |

Ví dụ cách khai báo, áp dụng tương tự cho các field còn lại:

```python
passenger_count: PositiveInt | None = Field(
    default=None,
    description=(
        "Positive integer passenger count explicitly supplied THIS turn. "
        "Normalize '1 vé' to 1 and 'hai người' to 2. Not offer index, date "
        "or budget. Leave null if absent. Evidence is a verbatim user span."
    ),
)
```

## 5. Nếu thử input bằng function calling

Declaration dự kiến (pseudocode; `described_input_wire_schema` là hàm cần implement):

```python
INPUT_TOOL_DECLARATION = {
    "name": "propose_input_decision",
    "description": (
        "Propose the interpretation of THIS user turn for a local mock flight "
        "booking assistant. Choose one InputDecision intent and supply only "
        "explicit new user data with literal evidence. Use CLARIFY for unknown "
        "or ambiguous requirements. UPDATE_SEARCH is a proposed patch, not "
        "an accepted request. SELECT_OFFER must refer to a displayed choice; "
        "SUBMIT_PASSENGERS must not invent profiles/documents. ASK_INFO chooses "
        "a topic for host-rendered facts. CONTINUE is not consent; CANCEL must "
        "be explicit. The host validates and previews mutations, then waits "
        "for a new user apply turn. This proposal NEVER authorizes booking, "
        "payment, ticketing or quote approval."
    ),
    "parameters": described_input_wire_schema(),
}
```

Ingress invocation chỉ có declaration này, không có raw backend hoặc tám tools
giao dịch. Host nhận đúng một call hợp lệ của `propose_input_decision`, lấy args
qua strict `InputDecision` validation, grounding và proposal validation như hiện
tại; không dùng cơ chế tự động chạy function của SDK. Từ chối call sai tên,
malformed args, nhiều calls, response bị block hoặc thiếu proposal.

Nếu ingress context dùng lại history, phải lọc/ghép đúng AI tool-call và
ToolMessage; không gửi call chưa có result vào business loop. Proposal result là
preview đang chờ user, không được ghi như requirements/booking đã thành công.

Với câu `Đặt 1 vé từ SGN đến DAD ngày 10/10/2026, chỉ Vietnam Airlines`, args vẫn
là `InputDecision(UPDATE_SEARCH, search_patch={origin: SGN, destination: DAD,
departure_date: 2026-10-10, passenger_count: 1, allowed_airlines: [Vietnam Airlines]},
evidence=...)`. Transport đổi, không phải nguồn tin cậy đổi.

Host đọc lại đề xuất; user gửi `áp dụng`; host commit canonical constraints rồi
business agent mới có thể gọi `search_flights({sort_by: price, limit: 5})`.
Sau chọn offer/hồ sơ/reprice, vẫn cần lượt `đồng ý` mới cho quote đang hiển thị
trước hold -> pay -> ticket. Thiếu field tiếp tục hỏi, không tự search.

Description không loại bỏ interpretation: nếu input là ngôn ngữ tự nhiên, model
vẫn phải diễn giải nó thành args. Function calling chỉ biểu diễn kết quả dưới
dạng tool call thay vì JSON structured response.

## 6. Những thứ giữ ở code, không giao cho description

- Strict schema, supported airports/carriers, valid date/time and complete request.
- Budget là tổng nhóm; preferred là soft, allowed là hard.
- Grounding/provenance, current-turn ownership, pending preview và explicit apply.
- Host-only quote display/consent; IDs/statuses/keys lấy từ observed state/backend.
- Seat/quote/hold expiry, exact payment amount, UNKNOWN reconciliation, idempotency.
- Input sau hold không tự sửa giao dịch; giữ references và handoff.

Bỏ kiểm tra equality của evidence vì đã có description là một thay đổi trust
model khác; không nằm trong thử nghiệm này. Description giúp model làm đúng,
không đảm bảo model luôn làm đúng và không phải lớp authorization.

## 7. Kế hoạch sửa và kiểm chứng khi triển khai

Bước 1 sửa `app/tools.py` và `app/agents/interpretation.py`: Field descriptions,
tool descriptions, bảo toàn description trên wire. Test schema/payload SDK giả
lập có đầy đủ mô tả, strict validation không yếu đi, signatures không đổi.

Bước 2 mới sửa transport ingress và response validator tương ứng. Diagnostics
phải ghi method thực tế, tên/version/hash của declaration; không hard-code
`json_schema` như hiện tại. Reuse ingress cho cả ba pattern nhưng không đổi
business reasoning của ReAct/PlanExecute/Hybrid. Trong Hybrid, mô tả business
tools không tự biến deterministic execution thành LLM chọn tool.

Các ca tối thiểu: câu SGN/DAD/`1 vé` ở trên; `2 người, tối đa 5 triệu`;
`ưu tiên` khác `chỉ`; request từng phần; bỏ điều kiện rõ ràng; DND/ngày thiếu năm;
offer ID không có nguồn; schema/evidence/value sai; apply không phải quote consent;
UNKNOWN không charge lại; API error giữ state/keys. Không dùng test fake để kết
luận chất lượng NLP của Gemini thật.

Version/hash phải phản ánh descriptions và ingress transport mới trước khi so
sánh các run; mô tả prompt là một biến của thử nghiệm. Không chạy benchmark H.
Chỉ sau khi triển khai và có cho phép gọi hosted model mới kiểm chứng Gemini thật.
