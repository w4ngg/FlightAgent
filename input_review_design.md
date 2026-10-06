# Thiết kế: evidence warn, proposal đa lượt và apply cùng snapshot

Trạng thái: **đã triển khai runtime/config** ngày 06/10/2026 cho cả ba pattern.
Kiểm chứng: 790 tests ngoài benchmark pass, compile app/tests đạt; SDK chỉ dùng
HTTP MockTransport. Không gọi Gemini thật hoặc chạy tests/benchmark H. Nội dung
dưới đây là contract triển khai; wire transport/tool-description experiment
không được thay đổi trong lượt này.

## 1. Mục tiêu và phạm vi

LLM đọc history/state và trả delta của lượt hiện tại. Host tích lũy delta thành
một proposal đầy đủ, giữ nguồn riêng cho từng field. Preview và apply phải dùng
cùng snapshot do host tạo, không để model viết lại snapshot ở lượt apply.

Tách hai vấn đề của runtime trước bản triển khai này:

- `grounded_input()` có thể từ chối giá trị ISO đúng vì không hiểu evidence ngày
  tiếng Việt. `warn` cho phép đề xuất tiếp tục sau lỗi evidence, không bỏ domain checks.
- `_prepare_interpretation()` merge vào `_draft` mà bỏ proposal chưa apply; apply
  lại lấy `pending.parsed` chỉ chứa delta. Cần sửa cả base merge và commit payload.

Áp dụng proposal accumulator cho `UPDATE_SEARCH`, dùng chung ReAct, PlanExecute
và Hybrid. Không đổi pattern reasoning, transport `json_schema`, provider/model,
mock backend, transaction guards hoặc cơ chế xác nhận quote. Tool-description
experiment trong `tool_description_design.md` là thay đổi độc lập.

`warn` chỉ áp dụng evidence của **search patch và clear_fields**. Selection,
passenger profiles, cancellation, native commands và quote consent vẫn giữ
validation/authority riêng như hiện tại. Việc nới evidence cho hồ sơ hoặc cancel
là phạm vi khác, không tự đưa vào bản triển khai này.

## 2. Hai chế độ evidence

Config: `InputReviewConfig` với `evidence_mode: Literal["strict", "warn"]`.
Default vẫn `strict`; mock demo chọn `warn` một cách tường minh. CLI:

```sh
.venv/bin/python -m app.cli --pattern react --provider gemini \
  --model gemini-3.1-flash-lite --input-evidence-mode warn
```

Flag này đã có trong CLI. Mode cố định trong một episode,
không do LLM/tool args chọn và không tự chuyển mode khi gặp lỗi. Factory và cả ba
agent phải nhận cùng config; ghi mode/version trong state, trace và artifact.

### Strict

Evidence thiếu, không thuộc lượt hiện tại, không canonicalize được hoặc không
khớp model value sẽ chặn proposal. Không làm yếu strict mode hiện có.

### Warn

Vẫn kiểm tra evidence theo field, nhưng lỗi evidence không chặn search proposal
nếu schema, normalized values và domain constraints hợp lệ. Host dùng **giá trị
chuẩn từ LLM**, lưu assessment/warning và hiển thị cảnh báo trong full readback.

Ví dụ `departure_date="2026-10-10"`, evidence=`"10 tháng 10 năm 2026"`:
ngày ISO được domain validator chấp nhận; semantic evidence checker có thể trả
`EVIDENCE_FORMAT_UNSUPPORTED`; proposal có warning, chưa được apply hay search.
Không cần thêm parser hiểu lại toàn bộ câu ngày để chạy được demo ở mode này.

Warning không tăng `general_retry_count`, không đổi thành `MODEL_ERROR` và không
tự gọi LLM lại. Actual LLM invocation vẫn tính step/usage như trước. API/JSON/
schema/domain errors vẫn chặn input và dùng budget hiện có.

### Những kiểm tra luôn chặn, kể cả warn

- Response bị block/truncated, malformed JSON, raw tool calls trong ingress.
- `InputDecision` strict types, extra-forbid, intent coherence, duplicate/conflicting fields.
- Model date/time có canonical format và giá trị hợp lệ; ngày không ở quá khứ
  theo mock clock; airports/carriers được hỗ trợ; route/time window hợp lệ.
- Count là integer dương, money là integer VND không âm; budget là tổng nhóm.
- Offer/index có nguồn từ danh sách hiện tại; passenger profiles đầy đủ/hợp lệ.
- Proposal stale, chưa được display hoặc đang `NEEDS_REVIEW`; thiếu fresh apply.
- Quote consent, exact payment amount, expiry, idempotency và UNKNOWN reconciliation.

Tách extraction và evidence audit: không bọc toàn bộ `grounded_input()` bằng
`try/except ValueError: pass`. Cách đó có thể bỏ luôn selection range/domain
checks hoặc trả một `ParsedInput` đã bị cắt dở.

Warn là **opt-in giảm độ tin cậy của evidence**: model có thể bịa/misinterpret
field dù output đúng schema. Literal warning không được ghi thành provenance đã
xác thực. Prompt vẫn cấm đoán năm/đổi preference thành restriction, nhưng host
không thể bảo đảm semantic correctness trong warn; user phải duyệt full readback.
Không dùng mode này cho giao dịch thật; không gọi proposal là trusted constraints.

## 3. Ba lớp state

1. `messages`: conversational history, tiếp tục gửi cho LLM như hiện tại.
2. `accepted_request` / alias `request_draft`: snapshot đã được user apply hoặc
   nhập bằng native command hợp lệ. Có thể partial; không chứa input bị từ chối.
3. `pending_interpretation`: working proposal tích lũy, chưa authoritative.
   `harness.constraints` chỉ nhận full validated request sau explicit apply.

Không lưu thêm một mutable `working_draft` rồi sync thủ công với `preview`.
`pending_interpretation.snapshot` là nguồn duy nhất của working request;
`preview` là alias/serialization từ snapshot đó để giữ compatibility.

### RequestProposal

```text
proposal_id, revision                 host-generated identity/version
intent                               UPDATE_SEARCH
snapshot                             complete canonical request or valid partial draft
snapshot_hash                        host hash of normalized snapshot
base_request_revision, basis_hash    committed state on which the proposal is based
field_sources                        provenance/assessment per field
status                               READY | NEEDS_REVIEW | STALE
displayed_revision, displayed_hash   most recent full host-rendered readback
displayed_at_turn                     genuine user turn after which it was shown
latest_source_turn_id                 actual turn contributing the latest valid delta
warnings                             active warnings on current field values
last_rejected_turn                    safe turn ID + reason codes, no rejected values
```

Snapshot được deep-copy/đóng băng sau normalize; không sửa dict in-place. Frozen
Pydantic model không tự làm nested dict/list immutable: implementation phải đảm
bảo detached copies hoặc immutable payload. Public state cũng trả detached copy.

Canonical snapshot điền optional defaults đã công bố (null/true/empty lists),
không điền origin/destination/date/count còn thiếu. Default là host default,
không gán nguồn giả là user. Nếu đủ field, normalize bằng `SearchConstraints`
rồi serialize trước khi hash/display, để backend không áp default khác với preview.

`basis_hash` dùng accepted request revision, session/intent, selection, quote,
passenger hash và current offer basis. Không dùng message count, thời gian hay
`input_blocked` vì chính preview/turn apply làm các giá trị đó thay đổi. Clock và
domain validity được kiểm tra lại tại apply; không suy ra snapshot còn hợp lệ
chỉ vì hash không đổi. Nếu business basis đổi, proposal thành STALE, không rebase
ngầm rồi apply dữ liệu khác với thứ đã hiển thị.

## 4. Nguồn và evidence theo từng field

Metadata do host tạo, không nhận `source_turn_id` từ LLM:

```text
field_sources[field] = {
    source_turn_id,
    source_kind: USER_LLM_PATCH | USER_COMMAND | HOST_DEFAULT,
    claimed_evidence: bounded model evidence or null,
    literal_status: VERIFIED | MISSING | NOT_IN_CURRENT_TURN | NOT_APPLICABLE,
    literal_span: actual [start, end] in that turn or null,
    semantic_status: VERIFIED | UNSUPPORTED_FORMAT | VALUE_MISMATCH | NOT_APPLICABLE,
    reason_codes: bounded host-defined codes,
    introduced_in_revision
}
```

- Field mới/thay đổi: audit evidence **chỉ với actual current user turn**.
- Field giữ lại: copy value/source/assessment từ base, không audit với follow-up.
- `claimed_evidence` không nằm trong turn: giữ như claim chưa verified hoặc omit
  khi export/redact; tuyệt đối không biến thành literal span của người dùng.
- Optional defaults không cần evidence. Native command có host-owned source
  từ actual command payload; không bịa evidence do LLM tạo.
- Cùng value nhưng user nhắc lại rõ ràng: có thể cập nhật source mới nếu audit
  hợp lệ; nếu warn, phải ghi status mới trung thực, không dùng verified cũ để che warning.
- Khi apply, giữ original field sources và thêm receipt `applied_by_turn_id`;
  không đổi mọi source thành turn `áp dụng`.
- Evidence thừa/không thuộc field path hợp lệ: audit warning có bounded reason,
  không tạo field/source mới. Duplicate/coherence vẫn bị strict schema chặn.

Warnings active gắn với current value/source. Field được sửa với evidence hợp
lệ thì bỏ warning cũ khỏi current snapshot; warning lịch sử vẫn nằm trong trace.
Không copy toàn bộ exception, user input, raw response hoặc giấy tờ thật vào diagnostics.

## 5. Nhận patch và tạo proposal mới

Quy trình host, dưới session lock:

1. Kiểm tra response/schema, lấy canonical delta từ model values; không lấy giá
   trị từ evidence trong warn. Null/omitted trong SearchPatch là **không đổi**.
2. Audit supplied fields/clear_fields với turn hiện tại. Strict chặn lỗi evidence;
   warn ghi assessment nhưng không chặn chỉ vì evidence.
3. Nếu có pending UPDATE_SEARCH cùng basis, còn hiệu lực: dùng snapshot và sources
   đó, kể cả nó đang NEEDS_REVIEW vì follow-up trước bị từ chối. Nếu không có
   pending request: dùng accepted_request. Không dùng selection/profile/cancel
   preview làm request base. Pending STALE không được kế thừa tự động.
4. Apply delta và explicit clear operations lên copy của base. Omitted fields
   giữ nguyên; `budget y nguyên` không cần model trả lại budget. Clear budget ->
   null; clear airline restriction/preference -> empty list. Clearing cũng có
   source/evidence của chính turn yêu cầu xóa.
5. Validate toàn candidate: từng field, route/window coherence và domain checks.
   Request partial hợp lệ khác với request invalid; thiếu required field được
   thu thập tiếp, không tự invent và không chạy search.
6. Chỉ khi candidate hợp lệ mới publish revision mới, snapshot/hash/sources/
   warnings đồng bộ. Không overwrite last valid proposal trước khi validate xong.
7. Host full readback, ghi displayed revision/hash. Có proposal thì giữ
   `input_blocked=true`; không chạy business model/planner/transaction graph.

Pseudocode mô tả accumulator (implementation nằm ở shared adapter/review module):

```python
delta = normalize_search_patch(decision)       # errors here always block
assessment = audit_evidence(decision, turn)   # checks only the new delta
enforce_evidence_mode(assessment, config)
base = current_valid_request_proposal_or_accepted_request()
candidate = merge_snapshot(base.snapshot, delta, decision.clear_fields)
snapshot = validate_and_normalize_request(candidate)
sources = merge_field_sources(base.sources, assessment, turn)
proposal = make_new_revision(snapshot, sources, base.basis)
publish_and_render_full_preview(proposal)
```

Ngay cả strict cũng chỉ audit new fields. Không bắt model truyền route/date từ
history để rồi yêu cầu evidence trong câu follow-up chứa mỗi `1 vé`.

## 6. Preview và apply dùng cùng snapshot

Preview hiển thị full snapshot: route, date, passenger count, **group total cap**,
time bounds/inclusivity, soft preferences và hard restrictions. Kèm revision,
missing fields, warnings và trạng thái có được apply hay không. Không chỉ hiển
thị changed fields; đặc biệt không diễn đạt thiếu budget như budget đã bị xóa.

Trong warn: ghi rõ field nào chưa kiểm chứng semantic/literal, nhắc người dùng
kiểm tra rồi mới apply. Ví dụ không báo "đã xác thực ngày" khi source chỉ literal
verified nhưng format chưa đối chiếu được.

### Apply protocol

1. Nhận một NEW actual user turn exact `áp dụng`/`apply`; không gọi LLM cho apply.
2. Kiểm tra pending READY, displayed revision/hash khớp current revision/hash,
   turn mới hơn display, basis còn hiệu lực, session chưa terminal/post-transaction.
   `NEEDS_REVIEW` không apply được dù payload/schema vẫn đúng.
3. Deep-copy và revalidate **toàn snapshot** đã display, không lấy current delta
   từ `pending.parsed.requirements` rồi merge lại với một base khác.
4. Snapshot complete: stage toàn bộ accepted snapshot/sources; gọi
   `harness.set_requirements(apply_turn_id, full_snapshot)` trước khi update
   accepted_request. Chỉ sau khi setter thành công mới commit adapter state,
   record receipt/archive proposal và clear pending. Harness tự invalidate
   offer/selection/quote/consent khi requirements thay đổi như hiện tại.
5. Snapshot partial hợp lệ: commit toàn partial snapshot vào accepted_request
   và sources; giữ business actions blocked, hỏi missing fields. Không gọi
   SearchConstraints setter bằng dữ liệu thiếu rồi coi exception là apply thành công.
6. Nếu apply validation/guard fail, giữ snapshot để inspect, đánh dấu NEEDS_REVIEW
   hoặc STALE; không clear pending trước khi biết commit thành công.

Receipt ghi proposal ID/revision, snapshot hash, source turn IDs và actual apply
turn ID. Kiểm chứng normalized accepted request/harness constraints có cùng
payload/hash với preview đối với complete request. Hash là audit binding do
host quản lý, không phải authorization token model được phép truyền.

In-process lock/staged update là atomicity trong runtime mock, không phải durable
transaction hay crash recovery từ JSON. Không làm side effect hold/pay/ticket
ở apply; `áp dụng` tuyệt đối không tạo quote confirmation receipt.

## 7. Follow-up lỗi: giữ proposal, khóa apply

Nếu API/parse/schema/domain/strict evidence lỗi hoặc model trả CLARIFY:

- Giữ snapshot, hash và field_sources của last valid proposal nguyên vẹn.
- Đổi envelope sang NEEDS_REVIEW; lưu rejected turn/reason, khóa apply/display
  authorization. Không trộn field từ candidate lỗi vào proposal hay accepted state.
- Giữ `input_blocked=true`, revoke quote approval/display consent như hiện tại;
  không tự apply, search, hold, pay hoặc ticket. Không dùng output lỗi làm memory
  authoritative dù raw message vẫn có thể nằm trong conversational history.
- Response cho biết proposal nào được giữ, field lỗi/missing, và lượt vừa gửi
  chưa được áp dụng. `last_model_error`/trace giữ bounded diagnostics.

Trong warn, **chỉ evidence warning không phải invalid follow-up**: candidate
domain-valid có thể thành READY mới và được readback, nhưng vẫn chờ fresh apply.
Mọi lỗi thuộc nhóm luôn-chặn ở mục 2 vẫn đi NEEDS_REVIEW.

### Phục hồi tường minh

- User gửi patch sửa hợp lệ: merge vào last valid snapshot, publish revision mới
  và full readback. Nhắc lỗi ở lượt trước chưa được áp dụng; sau đó mới nhận apply.
- `/proposal show`: chỉ đọc proposal/status/sources; không gỡ
  khóa apply và không chạy model/business tools.
- `/proposal keep`: actual user chủ động bỏ follow-up bị từ
  chối và giữ proposal trước. Host kiểm basis/domain, tạo revision/readback mới
  từ snapshot giữ lại. **Không commit**; vẫn cần lượt `áp dụng` sau readback này.
- Nếu không có last valid proposal, yêu cầu nhập request mới; không có gì để keep.
- Basis stale không được keep/rebase ngầm. Session terminal vẫn cần explicit
  reset; không dùng keep để xóa retry counters hay hồi sinh transaction.

Hai command mới phải được khai báo ở CLI/host controls, không expose thành tool
cho model. Natural `giữ nguyên` không tự biến thành apply/keep authority.

Sau hold/payment: giữ behavior handoff và transaction references hiện tại. Không
cho accumulator tạo request thay thế, keep/unblock hoặc operation/key mới.
Existing payment status reconciliation tuân guard/budget riêng, không tự chạy
chỉ vì follow-up lỗi.

## 8. Native input và câu hỏi chỉ đọc

`/request`, `/passengers`, `/select`, exact `chọn ...` vẫn host-owned. `/request`
là explicit replacement đã validate, không phải patch LLM; chỉ bỏ pending request
sau khi command commit thành công. Command lỗi không được phá last valid proposal.
Field sources đánh dấu actual USER_COMMAND; empty optional defaults khác user claims.

Nếu pending request/NEEDS_REVIEW còn tồn tại, selection/profile/`tiếp tục` không
được âm thầm bỏ qua request chưa duyệt. Host giải thích cần sửa/keep và apply
proposal trước, thay vì báo sai rằng offer không tồn tại. Sau request được duyệt,
native choice tiếp tục dùng danh sách hiển thị/IDs có nguồn như hiện tại.

ASK_INFO/`/state`/`/trace` giữ proposal và không chạy business actions. Khi hỏi về
request, host phân biệt accepted request với working proposal và trạng thái khóa;
không dùng việc hỏi thông tin để tự bật lại apply. Mọi mode vẫn gửi chat history
và labeled accepted/pending state cho LLM; model trả delta, không được ghi trusted state.

## 9. Ví dụ đúng theo log người dùng

Turn 1: `đặt 3 vé từ SGN đến HAN ngày 10/10/2026, budget 6 triệu`.
Host tạo revision 1 đầy đủ; mọi explicit field có source turn 1. Accepted request
chưa đổi, không gọi search.

Turn 2: `tôi đặt lộn chỉ đặt 1 vé budget y nguyên`.
Model trả delta count=1, evidence=`1 vé`; host tạo revision 2:

```json
{
  "origin": "SGN",
  "destination": "HAN",
  "departure_date": "2026-10-10",
  "passenger_count": 1,
  "max_budget_vnd": 6000000
}
```

Snapshot minh họa trên rút gọn optional defaults. Count có source turn 2; route,
date và budget vẫn source turn 1. User được xem full revision 2.

Turn 3: `áp dụng`.
Host commit full revision 2, receipt có source turns 1/2 và applied_by turn 3;
search mới được phép dùng harness constraints. Không phát sinh quote consent.

Nếu turn 2 thay bằng một candidate schema/domain-invalid: giữ revision 1 nhưng
NEEDS_REVIEW. Turn 3 `áp dụng` bị từ chối. User phải sửa, hoặc `/proposal keep`
để host đọc lại revision mới rồi gửi một lượt apply khác.

## 10. Observability và acceptance tests

State giữ compatibility `request_draft` (accepted) và
`pending_interpretation.preview` (alias snapshot), thêm evidence_mode, proposal
revision/status/hash/sources/warnings. Tách `accepted_missing_fields` và
`working_missing_fields`; UI không dùng missing fields của accepted draft rỗng
để nói full pending proposal còn thiếu route. `/state` không mutate/bump revision.

Trace: INPUT_EVIDENCE_WARNING, REQUEST_PROPOSAL_CREATED/UPDATED/DISPLAYED,
REQUEST_PROPOSAL_PRESERVED, REQUEST_APPLY_REJECTED, REQUEST_PROPOSAL_KEPT và
REQUEST_PROPOSAL_APPLIED. Add event allowlist tại harness. Giữ event ingress cũ
khi cần compatibility; APPLIED chỉ ghi sau actual commit, không trước setter.
Warning/receipt có field/reason/source/revision/hash, không arbitrary exception
strings, credentials, private provider response hay real PII.

Acceptance tests dùng offline injected interpreter và SDK HTTP MockTransport,
không cần Gemini thật:

1. Full request -> count-only follow-up -> apply: route/date/budget không mất;
   preview/accepted/harness snapshot hashes khớp, business tools chỉ sau apply.
2. Tích lũy route/date/count/budget qua >= 3 lượt trước apply; partial apply chỉ
   nhận full partial snapshot, không search cho tới complete reviewed request.
3. Omission/null giữ nguyên; clear budget/airline tạo đúng null/[] và nguồn mới;
   preferred không tự thành allowed, group cap không nhân theo count.
4. Sources cũ giữ turn cũ, chỉ fields mới audit current turn; warn field có
   literal/semantic status đúng, sửa evidence hợp lệ bỏ current warning cũ.
5. ISO value + Vietnamese date evidence: strict hiện tại reject; warn có bounded
   warning/readback rồi user apply mới commit. Warning không tiêu retry budget.
6. Warn vẫn reject invalid date, past date, unsupported airport, wrong schema,
   negative/bool count/money, reversed window và incoherent decision.
7. Failed API/JSON/schema/domain/CLARIFY giữ proposal/sources/hash; apply ngay sau
   lỗi bị chặn, không tool call/keys/booking/payment/confirmation mới.
8. Show chỉ đọc; keep revalidate và chỉ readback; phải fresh apply. Valid corrected
   patch tạo revision mới. Không keep khi stale/terminal/post-transaction.
9. Stale basis, mismatched display/hash, modified public copies, apply failure
   và clock đổi không commit dữ liệu khác preview, không xóa proposal trước thất bại.
10. Native invalid command không phá proposal; native selection không bypass
    pending request lock; generic error phải phân biệt blocked và unknown offer.
11. Same behavior cho react/plan_execute/hybrid; selection/profile/cancel còn
    strict; fresh quote approval/UNKNOWN/idempotency/completion không yếu đi.

Các file triển khai: `app/agents/input_review.py` (shared models/config),
`app/agents/interpretation.py` (normalize vs audit), `app/agents/adapter.py`
(accumulator/render/apply/recovery), factory/constructors/CLI (config/controls),
harness event allowlist và tests/docs. Không đổi raw backend search contract.

Input contract version và manifest/environment hash phải ghi evidence mode và
review protocol mới trước khi so sánh experiment. H chỉ cập nhật code khi triển
khai, không chạy benchmark hay tests H. Không coi thiết kế hoặc mocked tests là
nghiệm thu hosted NLP; không sửa artifact lịch sử thành kết quả của protocol mới.
