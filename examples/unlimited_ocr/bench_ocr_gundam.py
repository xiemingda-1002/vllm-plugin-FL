"""Gundam OCR SSE benchmark; sends requests only when explicitly executed.

One image per request, each selected image exactly once, largest files first.
No automatic retries. Uses only Python's standard library.
"""

import argparse
import base64
import json
import math
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


def sse_events(response):
    """Read SSE data events, including events with multiple data lines."""
    lines = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if lines:
                yield "\n".join(lines)
                lines = []
        elif line.startswith("data:"):
            lines.append(line[5:].lstrip(" "))
    if lines:
        yield "\n".join(lines)


def token_count(usage, key):
    value = (usage or {}).get(key)
    return value if type(value) is int and value >= 0 else None


def progress(label, done, total, succeeded):
    width = 24
    filled = width * done // total if total else width
    print(
        f"\r{label} [{'#' * filled}{'-' * (width - filled)}] "
        f"{done}/{total} success={succeeded} failed={done - succeeded}",
        end="\n" if done == total else "",
        file=sys.stderr,
        flush=True,
    )


def request(args, index, image, queued_at):
    started = time.perf_counter()
    row = dict(
        index=index + 1,
        image=str(image),
        image_bytes=None,
        success=False,
        status=None,
        truncated=False,
        empty=True,
        usage=None,
        prompt_tokens=None,
        completion_tokens=None,
        nonempty_delta_chunks=0,
        finish_reason=None,
        stream_done=False,
        client_queue_seconds=started - queued_at,
        ttft_seconds=None,
        response_headers_seconds=None,
        http_stream_seconds=None,
    )
    parts = []
    http_start = None
    try:
        row["image_bytes"] = image.stat().st_size
        mime = (
            "image/jpeg"
            if image.suffix.lower() in (".jpg", ".jpeg")
            else "image/" + image.suffix.lower()[1:]
        )
        uri = (
            "data:"
            + mime
            + ";base64,"
            + base64.b64encode(image.read_bytes()).decode("ascii")
        )
        payload = dict(
            model=args.model,
            messages=[
                dict(
                    role="user",
                    content=[
                        dict(type="image_url", image_url=dict(url=uri)),
                        dict(type="text", text="document parsing."),
                    ],
                )
            ],
            temperature=0,
            seed=1024,
            max_tokens=args.max_tokens,
            stream=True,
            stream_options=dict(include_usage=True),
            skip_special_tokens=False,
            vllm_xargs=dict(ngram_size=35, window_size=128),
            mm_processor_kwargs=dict(image_size=640, base_size=1024, crop_mode=True),
        )
        req = urllib.request.Request(
            args.base.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        http_start = time.perf_counter()
        row["client_prepare_seconds"] = http_start - started
        with opener.open(req, timeout=args.timeout) as response:
            row["status"] = response.status
            row["response_headers_seconds"] = time.perf_counter() - http_start
            if (
                "text/event-stream"
                not in response.headers.get("Content-Type", "").lower()
            ):
                raise ValueError("Expected text/event-stream response")
            for event in sse_events(response):
                if event.strip() == "[DONE]":
                    row["stream_done"] = True
                    break
                data = json.loads(event)
                if data.get("error"):
                    raise ValueError(
                        "Server stream error: " + str(data["error"])[:1000]
                    )
                if isinstance(data.get("usage"), dict):
                    row["usage"] = data["usage"]
                for choice in data.get("choices", []):
                    if choice.get("index", 0) != 0:
                        continue
                    content = (choice.get("delta") or {}).get("content")
                    if isinstance(content, str) and content:
                        if row["ttft_seconds"] is None:
                            row["ttft_seconds"] = time.perf_counter() - http_start
                        row["nonempty_delta_chunks"] += 1
                        parts.append(content)
                    if choice.get("finish_reason") is not None:
                        row["finish_reason"] = choice["finish_reason"]
        row["truncated"] = row["finish_reason"] == "length"
        row["empty"] = not "".join(parts).strip()
        if not row["stream_done"] or row["finish_reason"] is None:
            row["error"] = "Incomplete stream: missing [DONE] or finish_reason"
        elif row["truncated"]:
            row["error"] = "Generation truncated at max_tokens"
        elif row["empty"]:
            row["error"] = "Empty OCR output"
        elif row["finish_reason"] != "stop":
            row["error"] = "Unexpected finish_reason: " + str(row["finish_reason"])
        else:
            row["success"] = True
    except urllib.error.HTTPError as exc:
        row.update(
            status=exc.code,
            error=exc.read(8192).decode("utf-8", errors="replace")[:1000],
        )
    except Exception as exc:
        row["error"] = str(exc)[:1000]
    finished = time.perf_counter()
    row["truncated"] = row["finish_reason"] == "length"
    row["empty"] = not "".join(parts).strip()
    row["prompt_tokens"] = token_count(row["usage"], "prompt_tokens")
    row["completion_tokens"] = token_count(row["usage"], "completion_tokens")
    row["latency_seconds"] = finished - started
    row["queued_to_finish_seconds"] = finished - queued_at
    if http_start is not None:
        row["http_stream_seconds"] = finished - http_start
    if args.save_text:
        row["_text"] = "".join(parts)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent
        / "test_images"
        / "omnidocbench_v1_5"
        / "images",
    )
    parser.add_argument("--base", default="http://127.0.0.1:10002")
    parser.add_argument("--model", default="Unlimited-OCR")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Concurrent requests; 512 must be selected explicitly",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Local output cap (8192); not an asserted official setting",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1200,
        help="Socket operation timeout in seconds",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Extra sequential warmup requests; default 0 follows official no-client-warmup procedure",
    )
    parser.add_argument(
        "--limit", type=int, help="Benchmark only the first N images after size sorting"
    )
    parser.add_argument(
        "--save-text", action="store_true", help="Save OCR texts after timing finishes"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON path; default results/bench-gundam-TIMESTAMP.json",
    )
    args = parser.parse_args()
    if (
        args.concurrency < 1
        or args.max_tokens < 1
        or args.timeout <= 0
        or args.warmup < 0
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error(
            "concurrency, max-tokens, timeout and limit must be positive; warmup must be >= 0"
        )
    source = args.image.resolve()
    extensions = {".png", ".jpg", ".jpeg", ".webp"}
    images = (
        [p for p in source.iterdir() if p.is_file() and p.suffix.lower() in extensions]
        if source.is_dir()
        else [source]
    )
    if not images or any(
        not p.is_file() or p.suffix.lower() not in extensions for p in images
    ):
        parser.error(
            "Provide an existing PNG/JPG/JPEG/WEBP image or directory containing images"
        )
    images.sort(key=lambda p: (-p.stat().st_size, str(p)))
    available_images = len(images)
    images = images[: args.limit]
    output = args.output or Path(
        __file__
    ).resolve().parent / "results" / datetime.now().strftime(
        "bench-gundam-%Y%m%d-%H%M%S-%f.json"
    )
    warmups = []
    if args.warmup:
        progress("Warmup", 0, args.warmup, 0)
    for index in range(args.warmup):
        row = request(args, index, images[index % len(images)], time.perf_counter())
        warmups.append(row)
        progress("Warmup", index + 1, args.warmup, sum(r["success"] for r in warmups))
        if not row["success"]:
            if index + 1 < args.warmup:
                print(file=sys.stderr)
            break
    rows = []
    aborted = any(not row["success"] for row in warmups)
    start = time.perf_counter()
    if not aborted:
        progress("Benchmark", 0, len(images), 0)
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(request, args, index, image, time.perf_counter())
                for index, image in enumerate(images)
            ]
            succeeded = 0
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                succeeded += int(row["success"])
                if not row["success"]:
                    detail = {
                        key: row.get(key)
                        for key in (
                            "index",
                            "image",
                            "status",
                            "error",
                            "finish_reason",
                            "completion_tokens",
                            "stream_done",
                            "latency_seconds",
                        )
                    }
                    print(
                        "\nFAILED " + json.dumps(detail, ensure_ascii=False),
                        file=sys.stderr,
                        flush=True,
                    )
                progress("Benchmark", len(rows), len(images), succeeded)
    wall = time.perf_counter() - start
    rows.sort(key=lambda row: row["index"])
    successful = [row for row in rows if row["success"]]
    latencies = sorted(row["latency_seconds"] for row in successful)
    known = [
        row["completion_tokens"]
        for row in successful
        if row["completion_tokens"] is not None
    ]
    complete_usage = bool(successful) and len(known) == len(successful)
    valid_performance = (
        not aborted and len(successful) == len(images) and complete_usage
    )
    summary = dict(
        aborted=aborted,
        performance_valid=valid_performance,
        dataset_source=str(source),
        dataset_directory=str(source if source.is_dir() else source.parent),
        selected_manifest=[str(image) for image in images],
        paper_exact_reproduction=False,
        available_images=available_images,
        selected_images=len(images),
        requests=len(rows),
        successful=len(successful),
        failed=len(rows) - len(successful),
        truncated=sum(row["truncated"] for row in rows),
        empty=sum(row["empty"] for row in rows),
        wall_seconds=wall,
        successful_requests_per_second=len(successful) / wall,
        known_successful_completion_tokens=sum(known),
        missing_successful_token_usage=len(successful) - len(known),
        completion_tokens_per_second=sum(known) / wall if complete_usage else None,
        successful_nonempty_delta_chunks=sum(
            row["nonempty_delta_chunks"] for row in successful
        ),
        successful_nonempty_delta_chunks_per_second=sum(
            row["nonempty_delta_chunks"] for row in successful
        )
        / wall,
        latency_mean_seconds=sum(latencies) / len(latencies) if latencies else None,
        latency_p50_seconds=latencies[math.ceil(len(latencies) * 0.50) - 1]
        if latencies
        else None,
        latency_p95_seconds=latencies[math.ceil(len(latencies) * 0.95) - 1]
        if latencies
        else None,
        base=args.base,
        model=args.model,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        warmup=args.warmup,
        order="file_size_descending",
        mm_processor_kwargs=dict(image_size=640, base_size=1024, crop_mode=True),
        notes=[
            "Gundam mode must also be configured and supported by server startup flags; no mode-probing endpoint is called.",
            f"max_tokens={args.max_tokens} is a local configuration; the paper does not disclose this output cap.",
            "This is not an exact paper reproduction: token counting, hardware, and client-observed timing boundaries may differ.",
            "SSE nonempty content chunks are not tokens. TPS uses server usage.completion_tokens only; missing usage gives null.",
            "TPS counts successful untruncated requests over full benchmark wall time, including failed request time and client preprocessing.",
            "TTFT is client-observed time from HTTP send boundary to first nonempty content delta; buffering/network affect it; no exact token TPOT is inferred.",
            "client_queue_seconds measures executor wait; http_stream_seconds includes network, server queue, prefill and decode, which cannot be separated here.",
            "latency_seconds starts at worker entry; queued_to_finish_seconds includes client queue; response_headers_seconds is observable header wait, not isolated server wait.",
            "Warmups and result/text file writes are excluded from benchmark timing. No automatic retries.",
        ],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.save_text:
        text_dir = output.parent / (output.stem + "-texts")
        text_dir.mkdir(parents=True, exist_ok=True)
        for label, group in [("warmup", warmups), ("request", rows)]:
            for row in group:
                path = text_dir / f"{label}-{row['index']:06d}.txt"
                path.write_text(row.pop("_text", ""), encoding="utf-8")
                row["text_path"] = str(path.resolve())
    output.write_text(
        json.dumps(
            dict(summary=summary, warmups=warmups, requests=rows),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Results:", output.resolve())
    return 0 if valid_performance else 2


if __name__ == "__main__":
    raise SystemExit(main())
