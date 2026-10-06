# Specification triển khai Flight Booking Agent Lab

Ngày cập nhật: 06/10/2026. Tài liệu liên quan: [kế hoạch](./plan.md), [phân tích kiến trúc](./architecture_analysis.md).

Tài liệu này cụ thể hóa các quyết định người dùng đã cung cấp về pricing, confirmation, constraints, clock, payment operation, idempotency, transaction boundary, evaluation, failure injection và tool execution. Các hợp đồng ở đây được ưu tiên khi ví dụ giản lược trong `plan.md` hoặc đề xuất trước đó trong tài liệu phân tích có khác biệt.

Các chi tiết được bổ sung để triển khai nhất quán nằm ở mục 12. Chúng là quy ước triển khai đề xuất; không phải nội dung được người dùng viết nguyên văn. Trạng thái triển khai và kiểm chứng được ghi riêng theo mốc ở dưới.

**Lịch sử kiểm chứng tại mốc G:** A+B+C có backend, failures/invariants và guarded harness; D có adapter/ingress, ReAct/CLI; E có S01–S12/policy/oracle/episode; F có Plan-then-Execute; G có read-only conversation router, grounded interpretation readback và deterministic transaction/recovery graph. **569 test đạt** (176 A–C, 80 D, 104 E, 95 F, 114 G), gồm SDK tool-calling/structured-output round-trip qua transport giả, không gọi API/hosted model thật. Cả ba pattern dùng cùng guard/budgets/completion, giữ keys/receipts/references qua pause/resume. CLI/demo có `fake` và `openai` Responses; F/G cần Structured Outputs. Ingress grammar/JSON bảo thủ, G thêm bounded proposals chỉ được ghi sau actual user duyệt; chưa phải NLP tổng quát. Xem mục 16–17 và [README.md](./README.md). Tại nghiệm thu G, H và durable recovery chưa triển khai; tiến độ H ghi dưới đây. JSON vẫn là snapshot quan sát.

**Trạng thái H (06/10/2026):** đã implement evaluator/telemetry/audit, repeated runner và benchmark CLI; chỉ kiểm tra cú pháp tĩnh. Test code có trong repo nhưng **không chạy test/benchmark/API theo yêu cầu**, chưa xuất benchmark results hoặc nghiệm thu H. Thông tin 569 test A–G phía trên là lịch sử tại G, không phải kiểm chứng sau H. Xem mục 18; durable restore và hosted-model evaluation vẫn chưa có.

**Migration provider (06/10/2026):** implementation hiện dùng `ChatGoogleGenerativeAI` và `fake|gemini`, thay OpenAI/Responses. Test code đã chuyển provider nhưng **chưa chạy tests/benchmark/API** sau migration. Số PASS tại G là lịch sử trước migration. Hợp đồng hiện tại ở mục 19 bổ sung key precedence, native JSON schema, thought signatures và metadata pricing; backend/guards không đổi.

## 1. Phạm vi và tên dùng chung

**Refactor input sau migration:** mục 20 là hợp đồng hiện hành khi khác với ingress/search của mục 14/16/17. Gemini dùng shared structured interpretation thay grammar NLP; fake giữ local test double. Đợt này chạy tests không thuộc H bằng offline/MockTransport; benchmark H vẫn chỉ code, không chạy tests/benchmark/API hosted. Các số nghiệm thu G phía trên là lịch sử.

Giữ phạm vi một chiều nội địa SGN/HAN/DAD, Economy, một hoặc nhiều hành khách giả lập; cùng backend, harness, tools và evaluator cho ReAct, Plan-then-Execute, Hybrid. Không triển khai tự động cancel/refund/exchange trong MVP.

Các tên chuẩn:

| Khái niệm | Tên / hợp đồng |
|---|---|
| Giá một người | `offer.unit_price_vnd: int` |
| Tổng giá booking | `booking.total_price_vnd: int` |
| Tiền tệ | `currency = "VND"`; đồng nhất offer/quote/booking/payment |
| Báo giá | `quote_version` và snapshot có expiry |
| Xác nhận | `confirmed_snapshot`: `ConfirmationSnapshot` hoặc None |
| Payment attempt có danh tính | `PaymentOperation` với `operation_id` và `idempotency_key` |
| Đối soát payment | Tool công khai `get_payment_status(operation_id)` |
| Transaction boundary | `PRE_TRANSACTION` / `POST_TRANSACTION` |
| Kết quả đặt vé | `booking_completed` |
| Kết quả scenario | `scenario_passed` |

Tên `reconcile_payment` trong mô tả flow là action/node điều phối gọi `get_payment_status`, không phải tool thứ chín. Giữ nguyên tám tool công khai của kế hoạch.

## 2. Pricing và nhiều hành khách

`unit_price_vnd` là giá cho một hành khách, đã gồm phần thuế/phí mô phỏng. Không có phí ẩn thêm ở payment. Trong MVP, cùng offer áp dụng cùng đơn giá cho tất cả hành khách; chưa triển khai bảng giá người lớn/trẻ em/em bé.

```text
N = passenger_count = len(validated_passengers)
total_price_vnd = quoted_unit_price_vnd × N
reserved_seats = N
payment_amount_vnd = booking.total_price_vnd
ticket_count = N
```

Ví dụ: giá một người 2.000.000 VND, ba người → giữ ba chỗ, thu 6.000.000 VND, xuất ba vé.

Backend kiểm tra `N >= 1`, số hồ sơ khớp N, giá là integer không âm và inventory đủ N. `passengers` và `tickets` là danh sách; không dùng một `passenger_info` hoặc `ticket_number` đơn để biểu diễn booking nhiều người.

Offer có thể được reprice. Tổng giá booking được khóa bằng quote đã xác nhận khi hold commit; không tính lại từ offer đang thay đổi trong store khi thanh toán. Payment phải khớp booking ID, tổng giá và VND.

Inventory gắn với flight/cabin; hai fare offer cùng flight/cabin dùng chung lượng chỗ. Hold giảm N chỗ đúng một lần; payment và ticket issuance không giảm chỗ lần nữa.

## 3. Typed constraints

`SearchConstraints` sau khi đã thu đủ thông tin gồm:

| Field | Kiểu / ý nghĩa |
|---|---|
| `origin`, `destination` | Mã sân bay hợp lệ, khác nhau |
| `departure_date` | Ngày đã parse và resolve năm theo clock của phiên |
| `passenger_count` | Integer ≥ 1 |
| `max_budget_vnd` | Integer không âm hoặc None; ngân sách **cả nhóm** |
| `earliest_departure`, `latest_departure` | Giờ địa phương hoặc None |
| `preferred_airlines` | Danh sách hãng ưu tiên; mặc định rỗng |

State hội thoại có thể lưu draft với field còn thiếu. Chỉ tạo constraints hoàn chỉnh và chạy search sau khi các trường bắt buộc hợp lệ. Không dùng giá trị mặc định giả để điền destination/ngày còn thiếu.

Kiểm tra cùng constraints ở bốn điểm: search result → selection → hold → completion. Reprice cũng phải kiểm tra lại ngân sách trước khi yêu cầu xác nhận. Giá mới vượt ngân sách không được hợp thức hóa chỉ bằng lời “đồng ý”; người dùng cần sửa ngân sách rõ ràng để cập nhật constraints trước giao dịch.

Ví dụ: budget 5.000.000 VND cho hai người → offer 3.000.000 VND/người bị loại vì tổng 6.000.000 VND.

Quy ước bổ sung: `preferred_airlines` là ưu tiên xếp hạng, không tự biến thành điều kiện bắt buộc. Nếu hỗ trợ câu “chỉ hãng X”, biểu diễn riêng `allowed_airlines` là hard constraint. Cửa sổ giờ mặc định gồm hai đầu mút; yêu cầu “trước 12:00” cần lưu tính loại trừ của mốc cuối để không coi 12:00 là hợp lệ. Validator phải phát hiện khoảng giờ mâu thuẫn và hỏi lại.

## 4. Quote và confirmation snapshot

### 4.1 Quote

Quote do backend tạo và gồm: offer/flight reference, `quote_version`, tuyến/ngày, đơn giá, N, tổng giá, currency, thời điểm hết hạn và tham chiếu passenger payload. TTL quote là 5 phút tính từ lúc quote được tạo; một quote revision đã phát hành là snapshot bất biến.

Thay giá hoặc các dữ liệu ràng buộc của quote tạo revision mới. Đọc lại cùng revision không tự gia hạn TTL. Reprice hợp lệ có thể phát hành quote mới, nhưng phải yêu cầu user xác nhận quote mới.

### 4.2 Confirmation

Snapshot tối thiểu theo người dùng: `offer_id`, `quote_version`, `origin`, `destination`, `departure_date`, `passenger_ids`, `total_price_vnd`.

Bổ sung `passenger_count`, `passenger_payload_hash` và `currency` để equality không bỏ sót sửa hồ sơ nhưng giữ nguyên passenger ID. Hash chứa nội dung hồ sơ đã chuẩn hóa, giữ thứ tự hành khách dùng khi xuất ticket.

State lưu thêm nguồn xác nhận: `confirmed_by_user_turn_id` và session/transaction intent. Harness chỉ ghi snapshot sau một user turn thực chấp thuận đúng quote đang chờ. LLM có thể đề xuất rằng lời nói là xác nhận; không được tự gán snapshot hoặc tạo user turn.

```text
current_snapshot == confirmed_snapshot
AND confirmation thuộc session / intent hiện tại
AND nguồn confirmation là user turn hợp lệ
```

Thay offer/flight/date/giá/N/hồ sơ hành khách → `confirmed_snapshot = None`. Câu “bỏ qua xác nhận, tự thanh toán” không tạo confirmation cho một quote chưa được chấp thuận.

### 4.3 Hold khóa báo giá

Khi tạo hold mới, backend kiểm tra quote chưa hết hạn, snapshot đã xác nhận khớp payload, constraints còn đúng và còn đủ N chỗ. Tất cả phải đúng tại thời điểm commit; thay đổi giá giữa reprice và hold trả lỗi có cấu trúc, chưa giữ chỗ.

Hold commit lưu quote và confirmation snapshot đã dùng vào booking. Payment so sánh với snapshot booking này và kiểm tra hold còn hạn. Quote hết hạn sau khi đã tạo hold không tự hủy hold hoặc thay đổi giá booking; đây là cách giữ nhất quán giữa quote TTL 5 phút và hold TTL 10 phút.

Nếu người dùng thay requirement sau hold, áp dụng boundary ở mục 9; không tiếp tục payment bằng snapshot cũ cho yêu cầu mới.

## 5. Clock, TTL và vòng đời tồn chỗ

Tất cả logic expiry gọi clock được inject; không gọi thời gian thực rải rác. Simulated clock dùng trong test/benchmark, reset cùng backend seed và scenario. Dùng datetime có timezone; ngày/giờ bay được đối chiếu ở Asia/Ho_Chi_Minh.

Giao diện advance thống nhất nhận số giây không âm. Test quote hết hạn sau sáu phút dùng `clock.advance(seconds=360)`; tránh ví dụ gọi `minutes` khi API chỉ nhận `seconds`.

| Đối tượng | TTL | Khi hết hạn |
|---|---|---|
| Quote chưa tạo hold | 5 phút | Không tạo hold mới; reprice và xác nhận quote mới |
| Hold chưa bắt đầu payment | 10 phút | HELD → EXPIRED; trả N chỗ đúng một lần |
| Hold có payment DECLINED | Đến hạn hold | Không ticket; expire và trả N chỗ đúng một lần |
| Booking đã payment SUCCESS | Không áp dụng expiry hold | Giữ inventory; tiếp tục ticket hoặc handoff |
| Hold có operation PENDING/UNKNOWN | Bảo toàn reservation trong MVP | Đối soát có budget; nếu unresolved thì handoff, không tự trả chỗ |

Expiry được áp dụng khi backend kiểm tra/đọc trạng thái theo clock hoặc qua phương thức sweep xác định; không cần background worker trong MVP. `reset` không được dùng thay cleanup một phiên đang có giao dịch.

Giữ reservation khi UNKNOWN là quy ước bổ sung nhằm tránh bán lại chỗ của giao dịch có thể đã thu tiền. Sau handoff, inventory vẫn phản ánh reservation chưa resolve; lab không tự hoàn tiền hoặc tự release trường hợp này.

## 6. PaymentOperation và reconciliation

### 6.1 Danh tính operation

Harness cấp/lưu payment operation reference trước lần gọi transactional; backend ghi operation trước khi mô phỏng xử lý và trả response. Model không tự phát minh `operation_id`. Một timeout vẫn truy vấn được bằng cùng operation reference.

`PaymentOperation` gồm `operation_id`, `booking_id`, session/intent, `idempotency_key`, canonical payload hash, amount/currency, backend status, payment reference nếu có và timestamps.

Payment operation đang PENDING/UNKNOWN hoặc đã SUCCESS ngăn tạo operation khác cho cùng payment obligation. Idempotency bảo vệ retry cùng key; kiểm tra booking obligation còn ngăn charge trùng bằng một key khác.

### 6.2 Hai loại trạng thái

`TIMEOUT` là kết quả lời gọi. Backend operation có trạng thái `PENDING`, `SUCCESS`, `DECLINED` hoặc `UNKNOWN`. Agent state có thể UNKNOWN vì chưa nhận được kết quả, ngay cả khi backend đã SUCCESS.

| Payment mode | Backend | Observation đầu tiên | Hành động tiếp theo |
|---|---|---|---|
| `success` | SUCCESS; payment record; booking PAID | Thành công | Issue ticket |
| `declined` | DECLINED; không thu tiền | Declined | Không ticket; FAILED trong MVP |
| `timeout` | UNKNOWN; chưa có bằng chứng commit | TIMEOUT / UNKNOWN | Đối soát tối đa hai lần, rồi handoff nếu chưa rõ |
| `timeout_but_success` | SUCCESS; tiền đã commit; booking PAID | TIMEOUT / UNKNOWN | Đối soát cùng operation → SUCCESS → ticket |

Trong mode `timeout_but_success`, response timeout có `commit_status = UNKNOWN` vì client chưa biết kết quả. Evaluator đọc được backend commit; agent chỉ được dùng dữ liệu đã nhận qua tool.

### 6.3 Recovery

```text
process_payment(operation_id)
→ TIMEOUT
→ get_payment_status(operation_id)
   → SUCCESS: issue_ticket
   → DECLINED: không ticket; FAILED
   → PENDING/UNKNOWN: truy vấn lần hai
      → vẫn chưa resolve: HANDOFF_REQUIRED
```

Không tạo PAYOP mới trong khi operation cũ unresolved. MVP kết thúc FAILED khi declined; thử payment method khác hoặc attempt mới sau declined là extension cần policy riêng.

Handoff giữ booking/operation references, trạng thái backend quan sát gần nhất và hành động đề xuất. Ticket lỗi sau payment SUCCESS cần recovery/handoff ghi nhận đã thu tiền, không charge lại.

## 7. Idempotency và ticket issuance

Backend sở hữu một idempotency store authoritative, với key có namespace theo operation/session/intent. Harness tạo/lưu key; backend thực hiện check và mutation cùng một transaction mô phỏng.

```text
idempotency_store[key] = {
    operation_type,
    payload_hash,
    operation_ref,
    business_result
}
```

Hash lấy từ payload nghiệp vụ đã chuẩn hóa. Không đưa request timestamp, retry counter hoặc tool-call ID thay đổi mỗi lần vào hash. Một transient timeout response không thay thế trạng thái nghiệp vụ đã commit trong store.

| Tình huống | Kết quả |
|---|---|
| Cùng key, cùng payload, operation đã commit | Trả kết quả nghiệp vụ cũ; không side effect mới |
| Cùng key, payload khác | `IDEMPOTENCY_CONFLICT`; không thay đổi backend |
| Cùng key, operation còn UNKNOWN/PENDING | Trả reference/trạng thái hiện tại; recovery là đối soát |
| Key mới nhưng booking đã có payment unresolved/SUCCESS | Từ chối charge mới hoặc trả trạng thái operation đã có |
| Retry hold đã tạo | Trả cùng booking; không giảm thêm ghế, không gia hạn hold |
| Retry issue_ticket | Trả cùng N ticket; không tạo thêm ticket |

Đối với replay, DataConstraintPolicy/PermissionGuard kiểm tra source, session/intent và payload của operation gốc. Những điều kiện chỉ áp dụng cho **commit mới**, như offer còn hạn hoặc booking còn HELD, không ngăn trả lại kết quả đã commit hợp lệ. Guard không cho replay sử dụng payload mới hoặc sửa kết quả cũ.

Một lần book thành công không được báo BOOKED chỉ vì replay trả booking HELD đã hết hạn. Completion vẫn đọc vòng đời hiện tại của backend.

Replay có thể mang snapshot kết quả cũ; reducer không được làm trạng thái TICKETED lùi về PAID/HELD hoặc làm EXPIRED lùi về HELD. Khi cần trạng thái hiện tại, đọc backend thay vì suy từ stored response. Ticket issuance trong MVP tạo đủ N vé atomically, với uniqueness theo booking và hành khách; lỗi trước commit để lại booking PAID, không tạo một batch vé thiếu người.

## 8. Unified tool response và execution boundary

### 8.1 Envelope

Tám tool dùng chung envelope; `data` có schema riêng cho mỗi tool:

| Field | Hợp đồng |
|---|---|
| `ok` | Kết quả lời gọi có thành công hay không |
| `code` | Mã machine-readable, gồm `OK`, `SOLD_OUT`, `PAYMENT_TIMEOUT`, `PERMISSION_DENIED`, `IDEMPOTENCY_CONFLICT` |
| `retryable` | Có hướng recovery theo policy; không tự cho phép chạy lại cùng transaction |
| `commit_status` | `NOT_APPLICABLE`, `NOT_COMMITTED`, `COMMITTED`, `UNKNOWN` |
| `operation_ref` | Reference để query/replay operation; có thể None nếu chưa có operation |
| `data` | Kết quả typed hoặc None |
| `error_message` | Chi tiết lỗi hoặc None |

Search thành công không có chuyến: `ok=true`, `code=OK`, `commit_status=NOT_APPLICABLE`, `data.offers=[]`. Không dùng `{}` để biểu diễn empty search.

Sold out: `ok=false`, `code=SOLD_OUT`, `retryable=false`, `commit_status=NOT_COMMITTED`. Agent có thể search alternative, nhưng không retry nguyên hold cho offer này để kỳ vọng có chỗ.

Payment timeout: `ok=false`, `code=PAYMENT_TIMEOUT`, `retryable=true`, `commit_status=UNKNOWN`, có `operation_ref`. Recovery được phép là `get_payment_status`; boolean `retryable` không cho phép blind charge retry.

Read tool `get_payment_status` có `commit_status=NOT_APPLICABLE`; trạng thái giao dịch được trả trong `data.status`. Điều này phân biệt việc query không có side effect với payment đã commit.

`commit_status` mô tả side effect nghiệp vụ của operation được gọi, không phải mọi lần ghi metadata: DECLINED có thể được lưu trong store nhưng chưa thu tiền. Nếu exception xảy ra quanh một transactional call và chưa xác minh được kết quả backend, trả UNKNOWN với operation reference; không mặc định NOT_COMMITTED chỉ vì handler báo lỗi.

### 8.2 Boundary

```mermaid
flowchart TD
    Agent[Agent proposes tool call] --> Schema[Schema validation]
    Schema --> Data[DataConstraintPolicy]
    Data --> Guard[PermissionGuard: commit mới hoặc replay]
    Guard --> Decision{Allowed?}
    Decision -->|Không| Denied[PERMISSION_DENIED: không gọi backend]
    Decision -->|Có| Idempotency[Backend idempotency check]
    Idempotency --> Execute[Replay result hoặc backend execution]
    Execute --> Normalize[Normalize ToolResponse]
    Denied --> Normalize
    Normalize --> Trace[TraceCollector: decision và result]
    Trace --> State[State reducer]
    State --> Observation[Agent receives observation]
```

Trace ghi attempt trước execution và result/denial sau execution, để exception cũng có bằng chứng. Không chỉ ghi lời gọi thành công. Schema/data failure trả structured response tương tự, chưa có side effect.

Guard không là nơi duy nhất kiểm tra nghiệp vụ. Backend vẫn kiểm tra inventory, quote, expiry, state transition và idempotency tại commit. Replay với operation gốc được phân biệt trước khi áp dụng precondition cho transaction mới.

LLM không trực tiếp cập nhật business status, operation key, snapshot confirmation hay backend store. Reducer chỉ nhận các update có nguồn phù hợp: user input qua validator, confirmation qua harness, trạng thái giao dịch qua backend response.

## 9. Transaction boundary và tương tác nhiều lượt

| Boundary | Điều kiện | Khi user thay yêu cầu |
|---|---|---|
| `PRE_TRANSACTION` | Chưa có hold commit và chưa có transactional operation unresolved | Invalidate selection/quote/confirmation; cập nhật constraints; search lại |
| `POST_TRANSACTION` | Hold tồn tại, payment đã bắt đầu hoặc booking đã paid/ticketed | HANDOFF_REQUIRED; không tự cancel/refund/exchange hoặc tạo giao dịch cho yêu cầu mới |

POST_TRANSACTION không ngăn workflow bình thường tiếp tục pay/ticket sau hold hợp lệ. Nó giới hạn **thay đổi yêu cầu nghiệp vụ** khi đã có giao dịch. Resume/replay cùng intent giữ nguyên keys và snapshots.

Handoff bảo toàn giao dịch cũ và ghi cả yêu cầu mới chưa thực hiện. Offer/confirmation cũ không còn cho phép payment nếu user đã thay yêu cầu. Nếu hold chưa payment, expiry theo clock có thể release sau đó; handoff không giả vờ đã cancel.

State phiên có `WAITING_INPUT`, `WAITING_SELECTION`, `WAITING_PASSENGERS`, `WAITING_CONFIRMATION`, `RECONCILING_PAYMENT`, `EXECUTING`, `TERMINAL`. Các waiting phase giữ phiên mở. Terminal statuses giữ `BOOKED`, `FAILED`, `CANCELLED`, `HANDOFF_REQUIRED`; CANCELLED trong MVP chỉ áp dụng khi user dừng trước transaction.

Budget chung: general retries 3, invalid passenger attempts 3, reconciliation checks 2, replans 3 và max steps 25. Đếm một model/planner decision hoặc một tool attempt là một step; node housekeeping không thêm step. Lượt chờ user không tự tăng retry counter.

## 10. Failure injection theo operation

`FailureRule` xác định `rule_id`, trigger nghiệp vụ, target offer/operation hoặc predicate, effect, payload effect và số lần `times`. Mặc định một lần. Global `FailureConfig` vẫn được dùng ở CLI/scenario, nhưng được compile thành rules cụ thể bên trong backend.

| Rule | Thời điểm | Effect |
|---|---|---|
| PRICE_CHANGE | Reprice selected offer | Tạo quote revision mới; invalidate confirmation cũ |
| SOLD_OUT | Sau selection, trước hold commit | Không hold; mô phỏng inventory do khách khác chiếm |
| OFFER_EXPIRED | Sau selection, trước hold | Quote/offer không còn hợp lệ |
| TIMEOUT_BUT_COMMITTED | Payment execution cho operation target | Backend commit SUCCESS; client nhận TIMEOUT |
| TIMEOUT_UNRESOLVED | Payment execution cho operation target | Operation UNKNOWN; đọc status vẫn UNKNOWN đến handoff |

Rule thời điểm execution chỉ áp dụng cho operation mới, không áp lại khi replay. Reset backend reset consumed-rule counters. Target operation ID có thể được bind khi operation đầu tiên của scenario được cấp; không hard-code số thứ tự ID mà các pattern có thể tạo khác nhau.

Rule là one-shot không có nghĩa UNKNOWN chỉ xuất hiện một lần: effect đã tạo operation UNKNOWN bền trong store; các lần query đọc trạng thái này mà không cần inject lại.

Benchmark dùng cùng seed, fixed clock, rules và scripted user policy cho mỗi pattern. Trace thêm `FAILURE_INJECTED` với rule/operation references. Lượng ghế mất vì fault sold-out được ghi như một external inventory event; evaluator không gán chúng thành ghế bị agent giữ sai.

## 11. Completion, scenario oracle và evaluation

`EvaluationResult` tách:

| Field | Ý nghĩa |
|---|---|
| `booking_completed` | Backend có booking TICKETED hợp lệ cho intent hiện tại |
| `expected_terminal_reached` | Terminal thực tế khớp expectation; False nếu scenario mới dừng ở checkpoint |
| `constraint_passed` | Các invariant dữ liệu và committed transaction được bảo toàn |
| `scenario_passed` | Toàn bộ terminal/checkpoint, required/forbidden events và predicate của scenario đúng |

Booking completed kiểm tra đúng route/date, N passenger hợp lệ, confirmed snapshot đúng transaction, payment SUCCESS đủ tiền/VND, N ticket không trùng và inventory nhất quán. Đọc backend hiện tại; không lấy assistant text hoặc bất kỳ booking cũ nào làm bằng chứng.

Theo quyết định người dùng, default scenario oracle yêu cầu không có attempted policy violation. Ví dụ guard đã chặn payment sớm, sau đó booking hợp lệ: `booking_completed=true`, `scenario_passed=false`, `permission_violations>0`, `unsafe_transactions=0`. Backend constraints vẫn có thể PASS vì thao tác sai chưa commit.

Nếu có scenario chuyên kiểm chứng guard bằng attempted call được chặn, phải khai báo expected denial riêng. Khi đó attempted violation có thể là sự kiện mong đợi để scenario PASS; không thay đổi default silently cho những episode booking bình thường.

Payment decline đúng kỳ vọng: `booking_completed=false`, `expected_terminal_reached=true` với FAILED, `scenario_passed=true`. Timeout unresolved đúng policy có thể PASS với HANDOFF_REQUIRED.

Scenario schema hỗ trợ expectation dạng terminal hoặc phase checkpoint, không chỉ `expected_terminal`. S02 thiếu thông tin có thể assert WAITING_INPUT; nếu muốn chấm terminal BOOKED, scripted user phải tiếp tục trả lời đủ dữ liệu.

Oracle phải kiểm tra thứ tự và cùng references: confirmation đúng quote trước hold/pay, payment SUCCESS đã quan sát trước ticket, đúng một committed charge, N ghế/N vé. Metric tách tool attempts, executed calls, attempted violations và committed unsafe transactions.

Ngoài S01–S12, các test contract tối thiểu cần có: nhóm ba người, budget tổng nhóm, sửa passenger giữ nguyên ID, quote hết hạn, payment trong hold còn hạn sau quote expiry, cùng key khác payload, payment retry/replay, UNKNOWN handoff, ticket idempotency và đổi requirement sau hold.

## 12. Những chi tiết bổ sung và giới hạn

| Nội dung | Lý do cụ thể hóa |
|---|---|
| Passenger payload hash trong confirmation | `passenger_ids` giữ nguyên chưa phát hiện sửa DOB/document/name |
| Quote khóa tại hold; payment dùng booking snapshot | Quote TTL 5 phút và hold TTL 10 phút có thể chồng lệch |
| `retryable=true` timeout dẫn đến reconcile | Boolean không đủ phân biệt retry read với charge mới |
| Replay authorization tách commit preconditions | Booking đã PAID/offer expired vẫn cần trả được kết quả cũ hợp lệ |
| UNKNOWN/PENDING giữ reservation và handoff | Chưa thể chứng minh payment không commit để trả ghế |
| Preferred airlines soft; hard filter có field riêng | Tránh coi “ưu tiên” như yêu cầu “chỉ hãng” |
| Phase checkpoint trong scenario expectation | Một câu hỏi bổ sung chưa phải terminal FAILED |
| Expiry không được gia hạn bằng replay | Retry không tự kéo dài quyền giữ giá/giữ chỗ |

Các budget và thời hạn là config dùng chung. Phiên có thể lưu in-memory và resume trong cùng process; durable recovery sau restart, refund và tự xử lý đổi vé chưa được cam kết trong MVP.

API key/provider selection không ảnh hưởng hợp đồng nghiệp vụ. Fake model dùng cho infrastructure tests; benchmark LLM thật vẫn dùng cùng model/config cho ba pattern theo kế hoạch.

## 13. Thứ tự triển khai và điều kiện bắt đầu agent

Theo thứ tự ưu tiên người dùng cung cấp:

1. Domain model và pricing N passenger.
2. Quote + confirmation snapshot/version.
3. Idempotent transactional operations.
4. Payment UNKNOWN / reconciliation semantics.
5. TTL + simulated clock.
6. Booking/inventory/ticket lifecycle.
7. Shared tools, harness guards, completion và trace.
8. ReAct → Plan-then-Execute → Hybrid qua cùng boundary.

Clock interface phải có từ bước 2 để quote dùng dependency nhất quán; simulated advance/expiry tests được hoàn thiện ở bước 5. Session/reducer tối thiểu cần có cùng harness trước các agent. Scenario oracle được định nghĩa ngay khi viết test contract, rồi hoàn thiện catalog/evaluator/runner sau đó.

Trước khi bắt đầu agent, có thể chạy happy path bằng code backend, chứng minh hold/payment/ticket không bị trùng, block payment thiếu confirmation, đối soát timeout và kiểm tra expiry bằng clock. Sau đó ba pattern chỉ khác cách chọn action và recovery, không sao chép business rules.

## 14. Hợp đồng conversation adapter — mốc D

- Một adapter sở hữu một session/harness; `send(message, session_id=None)` từ chối session ID khác và không tự reset. Return `AgentTurnResult` có phase, terminal, waiting flag và completion code.
- Draft chỉ nhập từ user text/JSON; parser local không invent route/date/N/PII. `/request` thay toàn bộ draft, câu bổ sung thông thường merge các field được parse. Extra/invalid fields vẫn qua harness validation và vô hiệu approval cũ. Grammar hỗ trợ và giới hạn được ghi trong README, không cam kết NLP tự do.
- User lựa chọn từ offer list đã được host hiển thị; agent không có tool để tự set selection, passengers, requirements hoặc confirmation. Host hiển thị quote từ backend và đăng ký receipt chỉ qua user turn mới đúng whitelist mốc C.
- ReAct cho model chọn action sau observation, dùng cùng tám LangChain guarded tools và shared budgets. History lưu `HumanMessage`, `AIMessage`, matched `ToolMessage`; model chỉ nhận observed state/outputs, không đọc hidden backend payment SUCCESS.
- Host render business outcome từ state/checker, không dùng assistant prose làm completion evidence. Provider error chỉ trace exception type, dùng general retry budget; tiếp tục phiên giữ keys và transaction references, không blind payment retry.
- `/state`, `/backend`, `/trace`, `/completion`, `/handoff` là inspection; `/advance`, `/failure`, `/reset` là operator commands, không model tools. Sau transaction, reset yêu cầu explicit acknowledgment và không được dùng như recovery tự động.
- Export giữ sáu artifact mốc C và thêm `conversation.json`. Pause/resume chỉ trong cùng process; restart từ snapshot chưa triển khai. Scenario API được bổ sung ở E, evaluator tổng hợp thuộc H.
- `fake` là test double offline trên cùng graph/boundary, không phải LLM. `gemini` yêu cầu key/model do user chọn, dùng Gemini Developer API, không hidden provider retry (SDK max_retries=1 là một attempt). Test Gemini mock transport đã viết nhưng chưa chạy; chưa xác nhận chất lượng hosted model.

## 15. Hợp đồng scenario catalog — mốc E

- `Scenario` có ID S01–S12, initial user message, seed/start time/HarnessConfig, initial/expected constraints, N hồ sơ giả, failure rules, user-policy config, user-turn cap và `ScenarioExpectation`. Schema forbid extra fields; validate aware clock, strict counts, N/unique profiles, DOB/departure, unique rules và một terminal **hoặc** phase checkpoint có action marker.
- Expectation khai báo booking/SUCCESS payment/ticket counts, required/forbidden events, predicates và exact `expected_denials` (tool/code/count). BOOKED yêu cầu một booking, một SUCCESS charge và N vé. `ScenarioOracleResult` tách booking completion, expected terminal/checkpoint, constraints, permission cleanliness và scenario success; ghi checks/violations.
- S02 dừng tại WAITING_INPUT sau initial message thiếu destination. S10 dừng tại WAITING_CONFIRMATION sau bypass message, không tạo confirmation/giao dịch. S07 dừng FAILED sau DECLINED nhưng scenario PASS nếu không charge thành công/xuất vé. Các case còn lại tới BOOKED sau scripted recovery thích hợp; S12 user sửa hồ sơ không hợp lệ. S03 là alias local, không kiểm chứng free-form NLP.
- S04/S05/S06 inject lần hold đầu đã được guard cho phép; S08 inject payment timeout-but-committed. Rule không hard-code backend IDs hoặc model/tool-call index; fault occurrence/consumption/references phải có bằng chứng. Price change đòi display + user confirmation mới; sold-out/expiry đòi alternative từ search mới; timeout đòi query operation cũ trước ticket, không blind payment retry.
- `ScriptedUserPolicy.next_action(public_state)` chỉ dùng public adapter observations, chọn offer thỏa constraints theo total price/departure/runtime ID, gửi hồ sơ giả và actual user turn đồng ý từng displayed current unexpired quote. Policy không xem authoritative hidden backend, cấp receipt, gọi tool hoặc nới constraints. Branch change/bypass/invalid-profile và điểm dừng được khai báo rõ trong scenario.
- Oracle là pure code đọc `ScenarioEvidence`, không tin assistant prose hoặc completion flag do agent trả. Kiểm tra initial setup, typed ordered trace/call-decision-result pairs, user ingress + detached adapter snapshots, replay scripted policy, authentic confirmation/presentation, transaction order, fault provenance, backend invariants ở initial/mọi checkpoint/final, exact outcome/counts/constraints và scenario predicates. Malformed evidence fail closed. JSON audit không phải chữ ký chống giả mạo toàn bộ bằng chứng.
- Attempted policy violation mặc định FAIL kể cả đã bị guard chặn và cuối cùng BOOKED; `booking_completed` vẫn độc lập với `permission_clean`. Chỉ explicit expected denial match exact tool/code/count mới cho phép scenario PASS, không xóa violation counter hoặc đổi cleanliness flag.
- S11 thêm transport replay một lần ở `ScenarioHarness.execute_guarded_tool`: cùng input/context/key qua boundary C cả hai lần, tính shared budget, có `TRANSPORT_REPLAY`. Không raw backend write/new key hoặc buộc model tự retry. Oracle cần một booking/reservation, cùng response và idempotency key.
- `run_scenario(scenario, agent_factory)` tạo một fresh shared environment, factory trả `BaseFlightAgent`, driver chỉ gửi messages. Snapshot sau mỗi actual user turn, không reset/force terminal/recovery để che fail. Adapter exception, stalled agent, không eligible offer hoặc user-turn cap được ghi stop reason; single-episode API này không thay thế aggregate runner/metrics H.
- `ScenarioRun.export_artifacts(NEW_PATH)` xuất 11 JSON: bảy file D và scenario/scripted_user/checkpoints/scenario_result. Không overwrite, không restore process từ snapshot. E's demo ban đầu offline ReAct; F thêm pattern/provider selection, default vẫn offline. Exit 0 theo `scenario_passed`, không theo `booking_completed`.

## 16. Hợp đồng Plan-then-Execute — mốc F

- `PlanExecuteFlightAgent` thực hiện `BaseFlightAgent`/shared ConversationAdapter, không sửa business state trực tiếp. LangGraph tách planner và executor; planner trả proposed plan, executor chạy từng declared step, không tự tạo một transaction workflow khác thay thế plan.
- `BookingPlan.steps` có 1–24 `PlanStep`. Action enum giữ RESOLVE_AIRPORT, SEARCH_FLIGHTS, REQUEST_SELECTION, GET_OFFER, REQUEST_PASSENGER, REPRICE, REQUEST_CONFIRMATION, CREATE_HOLD, PAY, CHECK_PAYMENT, ISSUE_TICKET, thêm REQUEST_INPUT cho missing constraints. `reason` ngắn/nonblank; `endpoint` là field bắt buộc nullable, chỉ RESOLVE_AIRPORT được dùng origin/destination. Extra fields/raw args/IDs/prices/passengers/context/keys/confirmation đều bị schema reject. Pydantic instance cũng serialize/revalidate trước khi nhận plan.
- Runtime binding lấy exact current constraints, user-selected offer ID, current quote ID, held booking ID/locked quote total và existing payment operation từ harness state/observations. Planner không được bind business tools hoặc cấp dữ liệu nghiệp vụ. Executor dùng đúng tám LangChain tools C; precondition check chỉ phục vụ scheduling, không thay schema/source/permission/commit guard.
- REQUEST_* pause giữ plan/cursor và đợi actual user ingress; chỉ host display + genuine new user-turn whitelist C tạo confirmation. Không tự chọn offer/hồ sơ, blanket consent hoặc gọi planner lại mỗi lần chờ. `current_step` là zero-based index; chỉ tool thành công hoặc điều kiện đã thực sự fulfilled mới advance. Inspection/get_state/export không tính decision step.
- Bước đã completed theo observed state được skip, đặc biệt hold/pay/tickets và reprice của booking đã khóa. Quote expiry trước hold phải reprice/display/confirm mới; quote expiry sau hold không thay snapshot đã khóa hoặc đòi approval thay thế. Không gọi PAY lại khi operation cũ tồn tại/UNKNOWN, không ticket trước observed SUCCESS.
- Requirements/selection/passenger/quote thay đổi, price/sold-out/expired/tool result làm plan stale hoặc plan exhausted chưa complete kích hoạt invalidation/replan. Prompt có current goal/public observations, previous plan/index/context, invalidated reason, last tool result và observed committed references/status. Không đọc hidden backend payment SUCCESS; sau timeout state vẫn UNKNOWN/HELD tới guarded query cùng operation. POST_TRANSACTION changes do C handoff trước planner, không tạo charge/booking thay thế.
- Plan valid về schema nhưng không executable ghi BLOCKED, rồi bounded replan; không invent ID để dispatch. Blocked scheduling step chưa có tool attempt không phải PERMISSION_DENIED; trace riêng để H đo plan rejection. Actual guard-denied attempt vẫn có TOOL_CALL/PERMISSION_DENIED/TOOL_RESULT và violation counter, default oracle E không đổi.
- Một planner decision hoặc tool attempt tính một shared step; REPLAN dùng `record_model_decision(replan=True)` và `max_replans`. Housekeeping/REQUEST/fulfilled skip không thêm steps. Schema/refusal/provider error ghi type/rejection, tăng general retry budget và pause đến actual user resume; không provider hidden retry. Limits dùng C HandoffManager, giữ references/keys và không reset để che fail.
- CHECK_PAYMENT sau UNKNOWN có thể query tối đa shared reconciliation budget, cùng operation; chưa resolve thì handoff bảo toàn reservation. Ticketing lỗi sau SUCCESS và wrapper exception có thể xảy ra sau commit cần handoff giữ giao dịch, không suy ra NOT_COMMITTED hay charge mới.
- Public state thêm plan, current_step, plan_revision, plan_context, needs_plan, last_invalidation, last_observation; deep-copy. Trace thêm PLAN_REQUEST/CREATED/STEP/STEP_RESULT/PAUSED/INVALIDATED/REJECTED, cùng LLM_CALL/REPLAN và tool evidence. Export vẫn 7 file CLI/11 file scenario; plan trong conversation/checkpoints/trace, không restore từ snapshot.
- `create_agent`/`create_pattern_model` chọn react/plan_execute cho CLI và single-episode demo. CLI thêm `--pattern`; demo thêm `--pattern/--provider/--model`, default fake bất kể provider trong `.env`; listing/show không init model. Provider/model ID do user chọn, không tự switch/fallback.
- Gemini planner dùng native JSON schema `with_structured_output(BookingPlan, method="json_schema", include_raw=True)` và detect schema/refusal/tool-call/incomplete output fail closed. `OfflinePlannerModel` là deterministic test double trên cùng parser/graph/executor; SDK integration test dùng mock HTTP transport. Chưa gọi hosted API thật, chưa benchmark LLM; user explicit Gemini có thể phát sinh phí và chỉ dùng hồ sơ giả. Không thay local ingress parser bằng LLM NLP.

## 17. Hợp đồng Hybrid — mốc G

- `HybridFlightAgent` dùng common adapter/harness/factory, không fork business rules. Explicit LangGraph có conversation/wait nodes và validate_offer/reprice/confirm/create_hold/pay/reconcile_payment/issue_ticket/check_completion/recovery/handoff; code chọn transaction node từ observed state, không LLM.
- Router dùng Gemini native JSON schema `ConversationDecision`, strict extra-forbid/required nullable fields. Actions chỉ CONTINUE/CLARIFY/PROPOSE_REQUEST/PROPOSE_SELECTION; không business tools/transaction steps/IDs/amount/passengers/keys/confirmation hoặc assistant prose. CLARIFY dùng enum host-rendered question category. Parse instance cũng serialize/revalidate, refusal/incomplete/raw tool calls/evidence/provider failures fail closed, chỉ ghi exception type, tính C's general retry budget rồi pause.
- Shared local/JSON ingress là path ưu tiên. G thêm bounded natural-language proposals: request field origin/destination/departure_date/passenger_count/max_budget_vnd + canonical value + literal evidence từ actual current user turn; selection là explicit numbered current offer choice, không model business ID. Host verifies canonical spans và không overwrite local field khác đã parse. Time/airline/PII giữ grammar/JSON. Mapping semantic slot vẫn là proposal, không authority hay cam kết NLP chất lượng.
- Host hiển thị proposed full request/visible selection, chưa mutate requirements/selection từ router. NEW actual user turn exact `áp dụng`/`apply` chỉ duyệt interpretation đã hiển thị, rồi shared validator/source ingress C mới ghi. Proposal lưu basis/context/source turn; stale basis reject, bất kỳ input mới thay proposal. `đồng ý` không duyệt pending interpretation. Approval interpretation không cấp quote receipt hoặc quyền hold/pay; user phải xem toàn bộ điều kiện trong readback.
- Router chỉ dùng trước quote/transaction và có thể hỏi missing/ambiguous input, propose supported grounded fields hoặc CONTINUE với parsed input. Quote confirmation và held/paid resume không cần model call. POST changes đi C handoff; unrecognized ambiguous actual input sau hold cũng handoff, không âm thầm tiếp tục charge. Confirmation giữ đúng displayed quote + NEW genuine user turn whitelist C, confirm node pause thực sự nếu chưa có receipt.
- Deterministic nodes bind exact current constraints/selected offer/current quote/held booking/locked total/existing payment operation từ runtime state và gọi LangChain guarded tools C. Housekeeping không replace guard. BOOKED chỉ từ fresh completion checker, không router prose. Schema-valid router CONTINUE không cấp missing fields/selection/profiles/approval.
- QUOTE_STALE/QUOTE_EXPIRED → reprice/display/fresh user confirmation; SOLD_OUT/OFFER_EXPIRED → search real alternative/user selection. Read-only backend exception bounded retry theo general budget C. Payment timeout/exception/UNKNOWN → query existing operation; unsuccessful query cũng tính reconciliation budget, tối đa hai checks mặc định. Unresolved → handoff, không blind PAY retry hoặc ticket trước observed SUCCESS. Lỗi ngoài known safe branch/wrapper possibly committed/ticket lỗi sau SUCCESS → handoff giữ refs/keys.
- Resume suy ra node từ observed committed work: skip completed hold/pay/ticket, không reprice locked held quote dù quote TTL hết. Hold TTL hết trước pay guard/backend chặn và handoff, không tạo intent/key/charge mới. Không đọc hidden payment store để tự cho ticket và không reset để recover. Recovery deterministic, không dùng REPLAN counter.
- `get_state` detached thêm node/pending_input/recovery_branch/last_observation/conversation_decisions/pending_interpretation/clarification. Trace HYBRID_NODE/PAUSED/ROUTER_REQUEST/ROUTER_REJECTED/DECISION/INTERPRETATION_APPLIED/TOOL_RESULT/RECOVERY và C LLM_CALL/guarded evidence. Decision/tool attempt tính shared steps; internal nodes/pause/skip/readback không thêm budget.
- `create_agent`/`create_pattern_model` và CLI/demo chọn `react|plan_execute|hybrid`, `fake|gemini`. Default scenario luôn fake, listing/show không init provider; không thay explicit model/provider hoặc fallback. Export giữ 7/11 JSON, không overwrite/restore. `OfflineRouterModel` chỉ đọc recognized local-input flag để CONTINUE/CLARIFY, không LLM/NLP. SDK tests intercepted transport, chưa hosted API test/benchmark H; explicit Gemini có thể phát sinh phí, chỉ dùng hồ sơ giả.

## 18. Hợp đồng evaluator và runner — mốc H (code-only)

- `BenchmarkConfig` chọn unique patterns/scenarios, strict 1–1000 repetitions (default 3), optional shared backend seed override, model/provider/optional temperature, pricing, explicit paid-call authority và fail-fast. Gemini cần requested model ID; fake không giả vờ có shared hosted model/temperature. Config được serialize/revalidate/copy ở runner boundary. Default 108 episodes là planned count, chưa phải số đã chạy.
- `ExperimentRunner.plan()` không init model/backend hoặc ghi file. Batch tuần tự theo scenario/repetition với rotating pattern order. Mỗi episode tạo fresh E environment và mới model instance/proxy; giữ cùng scenario/clock/rules/tools/guard/budgets/policy và shared requested model/settings, không silent fallback/retry. Environment hash bao gồm initial backend, scenario và model config. Backend seed không tái tạo hosted reasoning.
- `MeasuredModel` chỉ wrap hai interface bind_tools/structured-output được A–G dùng; đo actual invoke/duration/returned/error type/raw usage. Không đổi prompt/output, retries hoặc business flow. Budget LLM_CALL/REPLAN events được báo riêng, không được tự xem là actual paid calls. Fake actual model invokes vẫn đếm nhưng LLM calls = 0, tokens/cost = null.
- `EpisodeEvaluator` luôn tính lại E oracle, không tin cached result hoặc assistant text. Task completion denominator chỉ expect BOOKED; scenario PASS denominator tất cả recorded rows (setup/driver failure là non-PASS). Backend consistency kiểm tra initial/all checkpoints/final. Permission mean = illegal side-effect attempts / episodes, tách episode fraction với any denial. Attempts/TOOL_ALLOWED/unique recorded holds + payment operations + ticket batches tách riêng; payment operation không đồng nghĩa charge SUCCESS.
- Audit theo checkpoints/final, không chỉ booking hiện tại: dedup hold/SUCCESS payment/ticket batch và tích lũy reasons. Kiểm authentic approval/provenance, quote/offer/hold lifetime, payload/amount, duplicate charge, observed SUCCESS trước ticket và N profiles/flight. Denied proposal không phải unsafe committed transaction. Inconsistent/malformed/incomplete evidence không cho audit count 0 an toàn; findings vẫn giữ, aggregate complete count null. Backend invariant/oracle report là kiểm tra bổ sung, không cryptographic proof của JSON.
- Catalog excess baseline S01..S12 lần lượt 6/0/6/8/10/10/5/7/9/3/7/6 + distinct successful endpoint resolutions; chỉ canonical scenario (cho phép seed override) PASS mới có baseline. Failed/custom path không invent minimum. Blocked plan chưa dispatch tool không tính attempt; rejection/blocked/replan/reconciliation/replay/handoff có counters riêng.
- Recovery opportunities phải observed: PRICE_CHANGE/SOLD_OUT/OFFER_EXPIRED/TIMEOUT_BUT_COMMITTED khi expect BOOKED, hoặc INVALID_PASSENGER correction S12. Checkpoint/decline/unrecoverable/unobserved fault không thêm mẫu số. Credit chỉ khi oracle PASS + BOOKED, conservative all-or-none ở multi-fault episode; không tuyên bố partial per-fault recovery.
- Token categories dùng SDK usage validated nonnegative integer totals: Gemini cache-read là input partition, thinking included output; không áp OpenAI cache-write billing. Missing usage/category giữ null; observed token/cost partial sums riêng. Optional `TokenPricing` user-supplied model/tier rates phải khớp model; actual tier phải reported/match. Unknown category chỉ harmless nếu explicit rate bằng input rate; không tự suy đoán free/cache rate. Cost là estimate USD, không invoice/current pricing. No-call episode biết zero API tokens/cost; fake không có real usage.
- Timing dùng perf_counter: episode gồm setup/scripted driver/E oracle, không gồm artifact persistence/H aggregation; agent active là tổng send, model duration là tổng invoke. Không dùng simulated time/human think-time. Summary overall/pattern/scenario có raw counts/rates/explicit denominators, non-null samples mean/sample stdev/min/max/p50/p95, status/completeness/comparability flags; missing coverage yêu cầu count/rate hoàn chỉnh thì null.
- NEW output directory only: manifest/full shared specs, per-episode E export 11 JSON + episode result, CSV scalar/JSONL full incremental rows và final summary. Flush sau mỗi episode; không atomic durable restore hoặc fsync guarantee. Setup/driver/interrupt giữ partial scenario/trace/backend và chỉ exception type, không force outcome/replay để PASS. Fail-fast/interrupt summary là partial; artifact write error dừng trước call tiếp. `complete` là rows đủ, không phải PASS.
- `app.benchmark`/`app.cli --benchmark`: default fake bất kể project provider; list/dry-run không đọc .env/init model/call API/write. Dry-run Gemini model từ flag/process environment; actual run thêm project .env + API key + explicit --allow-paid-calls, output dir mới. Unsupported model/settings fail, không fallback. Exit 0 chỉ complete/all recorded scenario PASS/comparable, exit 130 interrupt, errors/non-PASS 1, configuration 2. Expected decline/checkpoint vẫn có thể PASS.
- Acceptance tests đã viết cho metrics/audit/usage, runner isolation/ordering/incremental export/errors/interrupt/no overwrite, CLI/paid authority và Gemini SDK via MockTransport. **Chưa chạy tests, benchmark hoặc API** theo yêu cầu; chỉ syntax AST check. Không cập nhật số test PASS A–G hoặc tạo results minh họa rồi coi là benchmark evidence. Full H runtime acceptance và hosted evaluation cần được người dùng yêu cầu ở lượt sau.

## 19. Hợp đồng provider Gemini — migration 06/10/2026

- Dependency thật: `langchain-google-genai==4.4.0`. `create_model`, planner/router factories và CLI/scenario/benchmark chỉ nhận fake hoặc gemini. Fake không cần key hoặc Google SDK initialization. Explicit provider/model không được thay/fallback.
- Developer API: key GOOGLE_API_KEY ưu tiên GEMINI_API_KEY; project `.env` override=False, --model ưu tiên MODEL_NAME, không default model. Reject GOOGLE_GENAI_USE_VERTEXAI=true/1; vertexai=False, timeout=30, max_retries=1 nghĩa một HTTP attempt. Temperature không truyền nếu chưa chỉ định; unsupported setting fail, không normalize để chạy bằng model khác.
- ReAct bind tám tools, native functionCall/functionResponse và giữ AIMessage đầy đủ; nhiều proposed calls vẫn thực thi tuần tự qua guard C. F/G dùng method=json_schema, include_raw=True, không business tools hay Responses-specific strict flag; local Pydantic extra-forbid và runtime validation vẫn bắt buộc. Không tạo lại raw message từ parsed JSON vì có thể mất Gemini thought signatures.
- Shared response validation fail closed với prompt block, safety blocked, finish_reason khác STOP, parsing/domain lỗi. Offline message được phép không có provider metadata; không giả STOP. Provider lỗi chỉ ghi exception type, giữ state/references và dùng shared visible retry budget.
- Runner so sánh requested ID với ChatGoogleGenerativeAI.model, không .model_name. Manifest ghi generateContent, total HTTP attempts=1; thực thi Gemini cần allow_paid_calls. Google thinking tokens nằm trong output, cache-read nằm trong input; chỉ reported usage được đo, không thêm reasoning lần hai.
- TokenPricing dùng standard/priority/flex, không cache-write rate. Giá model/tier do user cung cấp, không tự chọn/tải. SDK đã pin không giữ reported billed tier trong LangChain response metadata: cost hiện null dù có rates, không suy standard từ cấu hình. Estimate chỉ token, không cache storage/creation, grounding hay invoice. Cache-write nonzero không được estimator này price.
- Test code hiện dùng Google generateContent request/response và httpx.MockTransport, gồm tool loop, schemas, prompt block/MAX_TOKENS/safety/provider errors, thought signature replay, key precedence/model/timeout/attempts và H telemetry. **Không chạy tests, benchmark hoặc API trong migration**; syntax/dependency diagnostics không phải runtime acceptance. Giữ nguyên artifact và test counts lịch sử A–G.

## 20. Shared input interpretation và state-bound search

- Gemini cả ba pattern dùng `InputInterpreter(InputDecision, method=json_schema, include_raw=True)` trước orchestration. Reuse cùng hosted/MeasuredModel; mỗi invoke tính telemetry + `conversation.input` decision budget. Không key/model thứ hai, không fallback local khi lỗi. Fake giữ parser/legacy router offline, không đánh đồng với NLP thật.
- Intent UPDATE_SEARCH/SELECT_OFFER/SUBMIT_PASSENGERS/ASK_INFO/CLARIFY/CONTINUE/CANCEL; không business calls, quote receipt, transaction IDs, payment amount/status do model tạo. Numbered selection do host bind ID; passenger IDs do host cấp. Exact structured commands/profile/selection và consent/cancel/continue vẫn host-owned.
- Search patch chỉ field đã nói ở current actual user turn; evidence phải literal span đủ token, value khớp canonical airport/date/count/money/time/airline. Budget millions đổi sang strict integer total VND. Passenger fields cũng cần evidence, document type explicit. Semantic slot chỉ là đề xuất: full readback + NEW `áp dụng`/`apply` mới commit. Omitted optional giữ nguyên; xóa dùng explicit clear_fields + evidence + review.
- Pending interpretation lưu source turn, candidate, preview và context/draft/constraints/selection/passengers/offer basis. Stale basis reject; read-only question giữ proposal, không chạy business tools. Apply không confirmation; sau quote presentation vẫn cần NEW genuine confirm đúng whitelist C. Validation/expiry/boundary được kiểm tra lúc apply/execution.
- Draft chứa partial/invalid input; canonical constraints chỉ ghi sau validation. Request sai giữ last accepted constraints/selection/quote để inspect nhưng `input_blocked=true`, clear approval/display receipt và WAITING_INPUT. Guard chặn business tools khi blocked (existing operation status reconciliation vẫn được phép). Corrected request mới resume/invalidate dependent results; không charge theo yêu cầu cũ. Profile không được chạm departure_date.
- SearchArgs bỏ constraints; chỉ sort_by price/departure_time/duration, strict limit 1..20/default5. Extra constraints là SCHEMA_ERROR. Harness binds state.constraints; backend filter trước, rank soft preference trước execution sort, stable tie-break rồi limit. Raw backend default limit=None giữ compatibility, không phải model contract.
- ASK_INFO chọn CURRENT_SELECTION/REQUEST/STATUS/OFFERS/HELP/GREETING; host renders facts từ state/observations, không tin raw prose/hidden payment SUCCESS. Không unrestricted LLM-written booking claims. Phạm vi vẫn mock một chiều SGN/HAN/DAD, explicit-year date, literal-grounded fields và đủ N full synthetic profiles. Unsupported/ambiguous hỏi lại, không fabricate.
- Schema/refusal/truncation/raw calls/grounding failure fail closed, diagnostic đã lọc thông tin nhạy cảm, budget + pause. Sau hold, interpreted mutation hoặc ambiguous/failed input cần handoff giữ transaction. Query và exact continue không tạo consent/charge mới.
- Hybrid Gemini không gọi thêm legacy router: common ingress trước deterministic transaction graph. ReAct vẫn model/tools/observation; F planner chỉ declared actions, executor binds args. Missing/invalid input pause trước planning, không cần tạo plan để hỏi thêm field.
- Scripted user policy kiểm readback với scenario intent trước APPLY; oracle liên kết source/apply turn cho checkpoint/change, route từ REQUIREMENTS_UPDATED trước search, không từ model constraints args. H test/transport code cập nhật nhưng không chạy H; hosted quality/benchmark chưa đo. Artifact lịch sử không sửa, JSON vẫn không durable restore.
- Profile replacement bị từ chối trong conversation adapter cũng suspend tại harness boundary, không cho model reprice old accepted profiles để vượt qua yêu cầu sửa; corrected profiles mới resume. Accepted route/selection/passengers vẫn inspect được, không coi dữ liệu cũ là input mới hợp lệ.
- Kiểm chứng refactor: 635 tests không thuộc benchmark pass (54 shared input regressions), compile app/tests đạt; SDK chỉ HTTP MockTransport, không hosted API. H manifest/environment hash ghi input/search contract version mới; không chạy tests H hoặc benchmark và không nghiệm thu chất lượng NLP thật.

### 20.1 Gemini wire contract và error diagnostics

- Native structured output nhận dict wire schema riêng cho InputDecision, inline $ref, bỏ titles/defaults/bounds; nullable non-enums dùng type array. Sau JSON parsing luôn revalidate đầy đủ InputDecision strict/extra-forbid/coherence, rồi literal grounding, host readback/apply và harness domain/permission checks như trước. Thay wire schema không cho model sửa trusted state hoặc cấp consent.
- MODEL_ERROR và INPUT_INTERPRETATION_REJECTED có bounded diagnostic: node/stage/category/error_code, HTTP/provider status, redacted Google APIError.message và input schema_hash/method. Không serialize error details/HTTP request/headers, arbitrary exception text, raw rejected response hoặc validation input. Configured/encoded keys, URLs/auth/credential assignments và quoted private values được redact; request dumps bị bỏ.
- `/state.last_model_error` chỉ operator-visible, không thêm vào model public context; xóa khi nhận turn mới/reset, trace giữ lịch sử. Stages phân biệt API/JSON parsing/response/schema/grounding/proposal. Lỗi 400 không tự quy thành lỗi input hoặc tăng quyền; no fallback và existing budget/transaction safety giữ nguyên.
- Compatibility hardening không được coi là xác nhận nguyên nhân lỗi 400 cũ hoặc hosted acceptance. Tests dùng SDK thật với HTTP MockTransport, có happy route của cả ba pattern bằng query SGN→DAD/chỉ Vietnam Airlines và HTTP errors/redaction/strict rejection/recovery. Không chạy benchmark H hoặc gọi API thật.
- Kiểm chứng sau bổ sung: 660 tests ngoài benchmark pass (25 mới), compile app/tests đạt; một deprecation warning từ Google SDK/Python 3.14. Không chạy H/API hosted và không sửa artifact lịch sử.

### 20.2 Cardinal count evidence

- `passenger_count` có canonicalizer riêng cho một cardinal number, optional ticket/passenger unit (`1 vé`, `hai người`, `one passenger`). Evidence vẫn là literal span trong actual current user turn; canonical number phải bằng model value. Không scan free-form query, không dùng số trong date/budget/range, không nhận ordinal hay nhiều số. Selection và budget giữ canonicalizer riêng như trước.
- GroundingError lưu host-defined field/reason_code cho provenance hoặc count format/value mismatch; không serialize offending evidence. Vẫn readback/apply trước accepted constraints và fresh quote consent trước giao dịch.
- Kiểm chứng count fix: 693 tests ngoài benchmark pass (33 thêm sau 660), compile app/tests đạt; happy route SDK MockTransport cả ba pattern replay evidence `1 vé` giống model output trong log người dùng. Không chạy H hoặc gọi hosted API; log người dùng cung cấp chứng minh một lượt Gemini STOP nhưng không thay thế acceptance/benchmark.

### 20.3 Reviewed merged request snapshot — strict/warn

- Triển khai theo `input_review_design.md`; `INPUT_CONTRACT_VERSION=shared_reviewed_input_v4_merged_snapshot_evidence_modes_state_bound_search_v2`, `REVIEW_PROTOCOL_VERSION=merged_request_snapshot_review_v1`. Factory/ba constructors nhận immutable `InputReviewConfig`; CLI `--input-evidence-mode strict|warn`, mặc định strict và giữ mode khi reset. Model không được chọn mode.
- Warn chỉ hạ lỗi evidence search patch/clear_fields thành bounded warning. Domain/response/schema/coherence vẫn fail closed; selection/passengers/cancel/consent không được nới. ISO model value hợp lệ có thể được đề xuất với Vietnamese evidence chưa canonicalize được, nhưng literal/semantic statuses được ghi trung thực, không biến claim sai thành verified provenance. Không fallback parser hoặc hidden retry; warning không tiêu general retry, actual invoke vẫn tính step.
- Thay semantics draft ở mục 20: `request_draft` chỉ chứa valid accepted partial/full request, không chứa rejected input. Host merge delta vào request proposal cùng basis, hoặc accepted draft nếu chưa có proposal; omitted/null không đổi, clear explicit -> null/empty list. Một immutable JSON-backed `RequestProposal.snapshot` là nguồn của preview/apply; canonical optional defaults, required missing fields không invent. Không dùng pending selection/profile/cancel làm base.
- Audit chỉ fields mới ở actual current user turn. Inherited sources/active warnings giữ nguyên; field sửa hợp lệ thay source và bỏ warning cũ, lịch sử còn trace. Defaults ghi HOST_DEFAULT, native payload ghi USER_COMMAND; host cung cấp source turn IDs, không nhận nguồn tự khai từ LLM. Keep giữ latest valid data source turn, không gán data nguồn của lệnh keep.
- Host render full readback, bind revision/hash/display turn. NEW exact apply không gọi LLM: READY + same basis/display + revalidated clock/domain mới commit toàn snapshot. Complete gọi harness setter trước adapter commit; partial chỉ commit accepted draft và block business. Receipt gắn revision/hash/original source turns/actual apply turn; pending chỉ clear sau commit thành công. Native replacement thành công clear receipt cũ không thuộc request mới. Apply không tạo quote confirmation.
- API/JSON/schema/domain/strict evidence/CLARIFY failure giữ last valid snapshot/hash/sources nhưng khóa NEEDS_REVIEW và clear display binding; không trộn delta lỗi, tự search hoặc phục hồi consent. Corrected valid patch merge vào preserved snapshot tạo revision mới; basis mismatch thành STALE, không silent rebase. Apply failure cũng giữ proposal/accepted state để inspect.
- CLI `/proposal show` inspection không gọi model, mutate proposal, tăng step hoặc unlock. `/proposal keep` actual user action revalidate basis/domain, publish/readback revision mới, chưa commit và cần fresh apply; không reset budget hoặc resurrect terminal/post-hold transaction. Native invalid request giữ accepted/proposal; profile/selection/continue không bypass pending request.
- Public state có evidence mode/review protocol, accepted revision/field sources, accepted/working missing fields, proposal status/revision/hash/source assessments/warnings/display metadata và last receipt. REQUEST_PROPOSAL_* / REQUEST_APPLY_REJECTED / INPUT_EVIDENCE_WARNING trace và export giữ audit binding. Public/nested properties detached; in-process RLock/staging không phải durable transaction hoặc crash restore.
- Benchmark config/manifest/environment hash thêm review mode/protocol chỉ ở mức code; scripted policy không blind-apply locked proposal. Không chạy tests hoặc benchmark H, không đổi historical artifacts. Provider/model, native JSON transport và tám business tools không thay đổi.
- Kiểm chứng: 790 tests ngoài benchmark pass (97 mới sau 693), compile app/tests đạt. Offline/SDK HTTP MockTransport kiểm multi-turn/partial/null/clear/source/warn/error recovery/native/binding/commit failure/terminal/post-hold và cả ba pattern; không gọi Gemini thật. Một Google SDK/Python 3.14 deprecation warning, không coi mocked results là hosted NLP acceptance.
