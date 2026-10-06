"""Multi-turn CLI. Inspection/operator commands are not model tools."""

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from dotenv import load_dotenv

from app.agents.factory import AGENT_TYPES, create_agent
from app.agents.models import create_model, create_planner_model, create_router_model
from app.agents.input_review import InputReviewConfig
from app.backend import MockFlightBackend
from app.failures import FailureConfig
from app.harness.runtime import FlightBookingHarness


HELP = """Ví dụ (clock mặc định 05/10/2026; ngày mock 10–12/10):
  Đặt 2 vé SGN đến HAN ngày 10/10/2026, ngân sách 5 triệu
  chọn 1
  hành khách: Nguyen|An|1990-01-01|IDENTITY_CARD|MOCK-0001; Tran|Binh|1992-02-02|PASSPORT|MOCK-0002
  đồng ý
Yêu cầu từng phần: 'từ SGN', 'đến HAN', 'ngày 2026-10-10', '2 người'.
Structured ingress: /request {JSON constraints}; /passengers [JSON profiles]; /select 1.
Gemini (cả 3 mode): 'áp dụng' duyệt diễn giải đã hiển thị, KHÔNG phải xác nhận quote.
Follow-up sửa proposal chưa áp dụng; preview/apply cùng toàn snapshot đã merge.
--input-evidence-mode warn chỉ cảnh báo evidence search; schema/domain/guards vẫn bắt buộc.
  /proposal show  xem proposal, không mở khóa apply
  /proposal keep  giữ proposal trước sau lỗi, đọc lại; cần lượt 'áp dụng' mới
Fake dùng grammar offline, KHÔNG đánh giá khả năng hiểu ngôn ngữ của LLM.
  /state       state hội thoại + harness (không đọc hidden payment status)
  /backend     dữ liệu backend authoritative
  /trace       trace có thứ tự, kể cả tool bị chặn
  /completion  kiểm chứng completion bằng code
  /handoff     packet hoặc null
  /export PATH xuất 7 JSON vào thư mục MỚI; không phải durable recovery
  /advance N   tiến simulated clock N giây (operator lab)
  /failure MODE  success|declined|timeout|timeout_but_success (trước giao dịch)
  /reset       episode mới; nếu đã có giao dịch, dùng /reset confirm
  /quit        thoát; RAM không được phục hồi tự động
  /help        hướng dẫn này
Dùng hồ sơ giả. Các lệnh inspect không gọi model và không tăng step budget.
"""


def _print_json(value, output):
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    print(json.dumps(value, indent=2, ensure_ascii=False), file=output)


def run_cli(agent, input_stream=None, output_stream=None):
    source, output = input_stream or sys.stdin, output_stream or sys.stdout
    print(f"FlightAgent — {agent.name} / mock backend. /help để xem cú pháp.", file=output)
    interactive = getattr(source, "isatty", lambda: False)()
    while True:
        if interactive:
            print("Bạn> ", end="", file=output, flush=True)
        line = source.readline()
        if not line:
            break
        text = line.strip()
        if not text:
            continue
        try:
            command = text.partition(" ")[0].lower()
            if command in {"/quit", "/exit"}:
                break
            if command == "/help":
                print(HELP, file=output)
            elif command == "/state":
                _print_json(agent.get_state(), output)
            elif command == "/backend":
                _print_json(agent.get_backend_state(), output)
            elif command == "/trace":
                _print_json(agent.get_trace(), output)
            elif text == "/proposal show":
                _print_json(agent.get_state()["pending_interpretation"], output)
            elif command == "/completion":
                _print_json(agent.check_completion(), output)
            elif command == "/handoff":
                _print_json(agent.get_handoff(), output)
            elif command == "/export":
                parts = shlex.split(text)
                if len(parts) != 2:
                    raise ValueError("Dùng /export PATH (đặt path có khoảng trắng trong dấu nháy)")
                print("Đã xuất: " + agent.export_artifacts(parts[1]), file=output)
            elif command == "/advance":
                parts = text.split()
                if len(parts) != 2:
                    raise ValueError("Dùng /advance N với N là số giây nguyên không âm")
                print("Clock: " + agent.advance_time(int(parts[1])).isoformat(), file=output)
            elif command == "/failure":
                parts = text.split()
                if len(parts) != 2:
                    raise ValueError("Dùng /failure success|declined|timeout|timeout_but_success")
                agent.configure_failure(FailureConfig(payment_mode=parts[1]))
                print("Đã cấu hình failure cho episode hiện hành.", file=output)
            elif command == "/reset":
                if text not in {"/reset", "/reset confirm"}:
                    raise ValueError("Dùng /reset hoặc /reset confirm")
                agent.reset(force=text == "/reset confirm")
                print("Episode mới: backend, clock, messages, trace và keys đã reset; không có phục hồi phiên cũ.", file=output)
            elif text.startswith("/") and command not in {"/request", "/passengers", "/select"} and text != "/proposal keep":
                print("Lệnh không hợp lệ. Dùng /help.", file=output)
            else:
                result = agent.send(text)
                print("Agent> " + result.assistant_message, file=output)
        except (ValueError, TypeError, OSError) as exc:
            # Operator/input errors only. Provider exceptions are redacted and
            # handled in the agent; never print their secret-bearing messages.
            print("Lỗi: " + str(exc), file=output)


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--benchmark" in arguments:
        from app.benchmark import main as benchmark_main
        return benchmark_main([arg for arg in arguments if arg != "--benchmark"])
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    parser = argparse.ArgumentParser(description="Flight booking conversation with shared mock data/harness",
        epilog="Batch mode: python -m app.cli --benchmark --dry-run (see app.benchmark --help)")
    parser.add_argument("--pattern", choices=tuple(AGENT_TYPES), default="react", help="Agent orchestration pattern")
    parser.add_argument("--provider", choices=("fake", "gemini"), default=os.environ.get("FLIGHT_AGENT_PROVIDER", "fake"))
    parser.add_argument("--model", help="Gemini model supporting this pattern; otherwise MODEL_NAME")
    parser.add_argument("--input-evidence-mode", choices=("strict", "warn"), default="strict",
                        help="Search evidence only: warn requires full readback/apply; domain and guards stay strict")
    parser.add_argument("--session-id", default="cli-session")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--payment-mode", choices=("success", "declined", "timeout", "timeout_but_success"), default="success")
    parser.add_argument("--output-dir", help="Export snapshots at exit to a NEW directory")
    args = parser.parse_args(arguments)
    try:
        factory = {"react": create_model, "plan_execute": create_planner_model, "hybrid": create_router_model}[args.pattern]
        model = factory(args.provider, args.model)
        backend = MockFlightBackend(seed=args.seed, failure_config=FailureConfig(payment_mode=args.payment_mode))
        agent = create_agent(args.pattern, FlightBookingHarness(backend, session_id=args.session_id), model=model,
                             input_review_config=InputReviewConfig(evidence_mode=args.input_evidence_mode))
    except ValueError:
        # Configuration errors from a provider SDK can echo credential values.
        parser.error("Không khởi tạo được model. Gemini cần GEMINI_API_KEY/GOOGLE_API_KEY và MODEL_NAME/--model; dùng Developer API, không Vertex AI. fake không cần key.")
    print("Provider: " + args.provider + (" (deterministic test double, KHÔNG phải LLM)" if args.provider == "fake" else
          " (LLM thật; hội thoại/hồ sơ giả được gửi tới Google Gemini, có thể phát sinh phí)"))
    print("Input evidence: " + args.input_evidence_mode + " (chỉ search interpretation; schema/domain/transaction guards luôn bắt buộc)")
    exit_code = 0
    try:
        run_cli(agent)
    except KeyboardInterrupt:
        print("\nĐã dừng CLI; không tự hủy/reset giao dịch.")
    finally:
        if args.output_dir:
            try:
                print("Đã xuất: " + agent.export_artifacts(args.output_dir))
            except OSError as exc:
                print("Không xuất được JSON: " + str(exc), file=sys.stderr)
                exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
