"""User-run single-image or directory OCR evaluation. Standard library only."""

import argparse
import base64
import concurrent.futures
import datetime
import json
import mimetypes
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_IMAGES = (
    Path(__file__).resolve().parent / "test_images" / "omnidocbench_v1_5" / "images"
)
# Official Unlimited-OCR README post-processing for OmniDocBench.
DET_RE = re.compile(r"<\|det\|>([^<\s]+)(?:\s*\[[^\]]*\])?\s*<\|/det\|>(.*)", re.DOTALL)


def remove_det(raw: str) -> str:
    """Preserve block content and Markdown using the official post-processing."""
    blocks = []
    cur = None
    for line in raw.splitlines():
        line = line.rstrip()
        if not line:
            continue
        m = DET_RE.match(line)
        if m:
            category, content = m.group(1).strip(), m.group(2).strip()
            if category == "image":
                continue
            if cur is not None:
                blocks.append(cur)
            cur = [content] if content else []
            continue
        if cur is None:
            cur = []
        cur.append(line)
    if cur is not None:
        blocks.append(cur)
    text = "\n\n".join("\n".join(b) for b in blocks).strip()
    return text


def normalize(text):
    text = re.sub(r"<\|det\|>.*?<\|/det\|>", "", text, flags=re.S)
    text = re.sub(r"<\|[^>]*\|>", "", text)
    return "".join(unicodedata.normalize("NFKC", text).split())


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y))
            )
        previous = current
    return previous[-1]


def load_image_selection(path, image_dir):
    """Read an explicit JSON list or a benchmark's selected_manifest, in order."""
    selection = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(selection, dict):
        selection = selection.get("summary", {}).get("selected_manifest")
    if not isinstance(selection, list) or not selection:
        raise ValueError(
            "Image list must be a nonempty JSON path list or benchmark JSON"
        )
    root = image_dir.resolve()
    images = []
    for name in selection:
        if not isinstance(name, str) or not name:
            raise ValueError("Each image-list entry must be a nonempty path string")
        image = Path(name)
        image = (image if image.is_absolute() else root / image).resolve()
        if (
            image.parent != root
            or not image.is_file()
            or image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}
        ):
            raise ValueError(
                "Image-list entry must be a supported image in the image directory: "
                + str(image)
            )
        images.append(image)
    if len({str(image).casefold() for image in images}) != len(images):
        raise ValueError("Image list has duplicate paths")
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=DEFAULT_IMAGES,
        help="Image file or directory (default: script-local test_images/omnidocbench_v1_5/images)",
    )
    parser.add_argument("--base", default="http://127.0.0.1:10002")
    parser.add_argument("--model", default="Unlimited-OCR")
    parser.add_argument("--prompt", default="document parsing.")
    parser.add_argument(
        "--expected",
        type=Path,
        help="Reference .txt file, or matching-stem .txt directory for image directories",
    )
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--requests",
        type=int,
        help="Request count; defaults to once per selected image",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--samples", type=int, help="Select the first N filename-sorted images"
    )
    selection.add_argument(
        "--image-list",
        type=Path,
        help="JSON path list or benchmark JSON; preserves the exact selected image order",
    )
    parser.add_argument(
        "--image-mode",
        choices=("server", "gundam"),
        default="server",
        help="Use server defaults, or explicitly request Gundam 640/1024/crop=True",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        help="New directory for basename.md exports (default: OUTPUT/predictions); repeated-image runs disable export",
    )
    args = parser.parse_args()
    if any(
        value is not None and value < 1
        for value in (
            args.requests,
            args.samples,
            args.concurrency,
            args.max_tokens,
            args.timeout,
        )
    ):
        parser.error(
            "requests, samples, concurrency, max-tokens and timeout must be positive"
        )
    directory = args.image.is_dir()
    if args.image_list:
        if not directory:
            parser.error("--image-list requires an image directory")
        try:
            images = load_image_selection(args.image_list, args.image)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            parser.error(str(exc))
        if args.requests is not None and args.requests != len(images):
            parser.error(
                "--requests must equal the image-list length when --image-list is used"
            )
    else:
        images = (
            sorted(
                path
                for path in args.image.iterdir()
                if path.is_file()
                and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            )
            if directory
            else [args.image]
        )
        images = images[: args.samples]
    if not images:
        parser.error("No supported images found")
    references = {}
    try:
        if args.expected and args.expected.is_dir() != directory:
            parser.error(
                "--expected must be a text file for one image, a directory for image directories"
            )
        for image in images:
            if not image.is_file():
                parser.error("Missing image: " + str(image))
            if args.expected:
                reference = (
                    args.expected / (image.stem + ".txt")
                    if directory
                    else args.expected
                )
                expected = normalize(reference.read_text(encoding="utf-8"))
                if not expected:
                    parser.error("Empty normalized reference: " + str(reference))
                references[image] = expected
    except (OSError, UnicodeError) as exc:
        parser.error(str(exc))
    args.requests = args.requests or len(images)
    export_enabled = args.requests <= len(images)
    if export_enabled and len({image.stem.casefold() for image in images}) != len(
        images
    ):
        parser.error(
            "Images have duplicate basenames; official Markdown export would overwrite predictions"
        )
    output = args.output or Path(
        __file__
    ).parent / "results" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    prediction_dir = args.predictions_dir or output / "predictions"
    if export_enabled and (
        prediction_dir.resolve() == output.resolve()
        or prediction_dir.resolve() in output.resolve().parents
    ):
        parser.error("--predictions-dir cannot be the run directory or its parent")
    if export_enabled and prediction_dir.exists():
        parser.error("--predictions-dir must not already exist: " + str(prediction_dir))
    output.mkdir(parents=True, exist_ok=False)
    if export_enabled:
        prediction_dir.mkdir(parents=True, exist_ok=False)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(args.base.rstrip("/") + "/v1/models", timeout=10) as response:
        models = json.load(response)
    if args.model not in [x["id"] for x in models["data"]]:
        raise SystemExit("Wrong endpoint model: " + str(models))

    def request(index):
        start = time.monotonic()
        image = images[(index - 1) % len(images)]
        row = dict(index=index, image=str(image.resolve()))
        if export_enabled:
            row["prediction"] = str((prediction_dir / (image.stem + ".md")).resolve())
        expected = references.get(image)
        try:
            mime = mimetypes.guess_type(image.name)[0] or "image/png"
            uri = (
                "data:"
                + mime
                + ";base64,"
                + base64.b64encode(image.read_bytes()).decode()
            )
            payload = dict(
                model=args.model,
                messages=[
                    dict(
                        role="user",
                        content=[
                            dict(type="image_url", image_url=dict(url=uri)),
                            dict(type="text", text=args.prompt),
                        ],
                    )
                ],
                temperature=0,
                seed=1024,
                max_tokens=args.max_tokens,
                stream=False,
                skip_special_tokens=False,
                vllm_xargs=dict(ngram_size=35, window_size=128),
            )
            if args.image_mode == "gundam":
                payload["mm_processor_kwargs"] = dict(
                    image_size=640, base_size=1024, crop_mode=True
                )
            req = urllib.request.Request(
                args.base.rstrip("/") + "/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
                req, timeout=args.timeout
            ) as response:
                data = json.load(response)
            row["response"] = data
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
            (output / f"{index:04d}.txt").write_text(text, encoding="utf-8")
            if export_enabled:
                prediction = remove_det(text)
                (prediction_dir / (image.stem + ".md")).write_text(
                    prediction, encoding="utf-8"
                )
                row["prediction_empty"] = not bool(prediction.strip())
            row.update(
                success=bool(text.strip()),
                finish_reason=choice.get("finish_reason"),
                completion_tokens=data.get("usage", {}).get("completion_tokens", 0),
            )
            if expected is not None:
                predicted = normalize(text)
                distance = edit_distance(expected, predicted)
                row.update(
                    normalized_cer=distance / len(expected),
                    edit_distance=distance,
                    reference_characters=len(expected),
                    predicted_characters=len(predicted),
                )
        except Exception as exc:
            row.update(success=False, error=str(exc))
            if isinstance(exc, urllib.error.HTTPError):
                row["error_body"] = exc.read().decode(errors="replace")
            if export_enabled:
                (prediction_dir / (image.stem + ".md")).write_text("", encoding="utf-8")
                row["prediction_empty"] = True
        row["elapsed_seconds"] = time.monotonic() - start
        (output / f"{index:04d}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return row

    start = time.monotonic()
    rows = []
    successful = 0
    progress_width = 0

    def show_progress():
        nonlocal progress_width
        done = len(rows)
        filled = 24 * done // args.requests
        line = (
            f"[{'#' * filled}{'-' * (24 - filled)}] {done}/{args.requests} "
            f"api_success={successful} failed={done - successful}"
        )
        progress_width = len(line)
        print("\r" + line, end="", file=sys.stderr, flush=True)

    show_progress()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(request, index) for index in range(1, args.requests + 1)]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            successful += bool(row["success"])
            print(
                "\r" + " " * progress_width + "\r", end="", file=sys.stderr, flush=True
            )
            print(
                json.dumps(
                    {k: v for k, v in row.items() if k != "response"},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            show_progress()
    print(file=sys.stderr, flush=True)
    rows.sort(key=lambda row: row["index"])
    elapsed = time.monotonic() - start
    successes = [row for row in rows if row["success"]]
    scored = [
        row
        for row in successes
        if row.get("finish_reason") != "length" and "normalized_cer" in row
    ]
    official_v15 = (
        directory
        and args.image.name == "images"
        and args.image.parent.name == "omnidocbench_v1_5"
    )
    by_image = {row["image"]: row for row in rows}
    manifest = []
    for image in images:
        row = by_image.get(str(image.resolve()))
        status = (
            "not_requested"
            if row is None
            else "failed"
            if not row["success"]
            else "length"
            if row.get("finish_reason") == "length"
            else "empty_prediction"
            if row.get("prediction_empty")
            else "complete"
        )
        manifest.append(
            dict(
                image=str(image.resolve()),
                status=status,
                prediction=row.get("prediction") if row else None,
            )
        )
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = dict(
        requests=args.requests,
        successful=len(successes),
        failed=len(rows) - len(successes),
        successful_includes_truncated=True,
        complete_nonempty_requests=sum(
            row["success"]
            and row.get("finish_reason") == "stop"
            and not row.get("prediction_empty", False)
            for row in rows
        ),
        truncated=sum(row.get("finish_reason") == "length" for row in rows),
        wall_seconds=elapsed,
        output_tokens_per_second=sum(
            row.get("completion_tokens", 0) for row in successes
        )
        / elapsed,
        image=str(args.image.resolve()),
        base=args.base,
        model=args.model,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        image_count=len(images),
        accuracy_scored_requests=len(scored),
        image_list=str(args.image_list.resolve()) if args.image_list else None,
        image_mode=args.image_mode,
        mm_processor_kwargs=dict(image_size=640, base_size=1024, crop_mode=True)
        if args.image_mode == "gundam"
        else None,
        dataset="OmniDocBench v1.5" if official_v15 else None,
        ground_truth=str((args.image.parent / "OmniDocBench.json").resolve())
        if official_v15
        else None,
        prediction_dir=str(prediction_dir.resolve()) if export_enabled else None,
        export_disabled_repeated_images=not export_enabled,
        empty_predictions=sum(bool(row.get("prediction_empty")) for row in rows),
        prediction_manifest=str((output / "manifest.json").resolve()),
        export_complete_selected_images=export_enabled
        and all(page["status"] == "complete" for page in manifest),
        selected_images_only=True,
        accuracy_unique_images=len({row["image"] for row in scored}),
        accuracy_coverage=len(scored) / len(rows) if references else None,
        micro_cer=(
            sum(row["edit_distance"] for row in scored)
            / sum(row["reference_characters"] for row in scored)
        )
        if scored
        else None,
        macro_cer=sum(row["normalized_cer"] for row in scored) / len(scored)
        if scored
        else None,
        accuracy_excluded_failed=len(rows) - len(successes) if references else None,
        accuracy_excluded_truncated=sum(
            row.get("finish_reason") == "length" for row in successes
        )
        if references
        else None,
        note="successful counts nonempty API replies, including truncated replies; complete_nonempty_requests requires stop. "
        "CER uses NFKC, removes whitespace and special tags. Failed/truncated requests excluded from text-reference CER; "
        "check coverage. Text-reference CER is a regression metric, not OmniDocBench scoring. "
        "Markdown uses official Unlimited-OCR remove_det. The manifest includes failed, truncated and unrequested pages; "
        "complete selected images does not certify full-dataset coverage or official scores. "
        "API throughput includes image encoding; no warmup. HTTP success is not accuracy validation.",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Results:", output)
    raise SystemExit(
        2
        if summary["failed"] or summary["truncated"] or summary["empty_predictions"]
        else 0
    )


if __name__ == "__main__":
    main()
