"""Prepare and validate Unlimited-OCR EvalScope performance runs."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path


def prepare(args):
    base = os.environ.get("BASE_URL", "http://127.0.0.1:10000").rstrip("/")
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(base + "/v1/models", timeout=5) as response:
            models_response = json.load(response)
            available = {row["id"] for row in models_response["data"]}
        if "Unlimited-OCR" not in available:
            raise ValueError(
                f"Unlimited-OCR not served; available models: {sorted(available)}"
            )
    except (urllib.error.URLError, ValueError, KeyError, TimeoutError) as exc:
        raise SystemExit(
            f"OCR service is not ready at {base}: {exc}. Follow the launch instructions in examples/unlimited_ocr/README.md."
        ) from None
    if args.check_server:
        print(f"OCR service ready: {base}, model=Unlimited-OCR")
        return

    from tokenizers import Tokenizer
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    model = Path(os.environ["MODEL"])
    if not model.is_dir():
        raise SystemExit(
            "Performance tokenizer export requires MODEL to be a local model directory."
        )
    source = model / "tokenizer.json"
    config = json.loads((model / "tokenizer_config.json").read_text())
    native = Tokenizer.from_file(str(source))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=native,
        bos_token=config["bos_token"],
        eos_token=config["eos_token"],
    )
    # Identity client template keeps random_vl's message structure without adding tokens.
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    export_dir = Path(os.environ.get("TOKENIZER_DIR", "tokenizer_export"))
    tokenizer.save_pretrained(export_dir)
    exported = AutoTokenizer.from_pretrained(export_dir, local_files_only=True)
    samples = [
        "document parsing.",
        "OCR performance test 1234.",
        "\u4e2d\u6587\u6587\u6863\u8bc6\u522b\u6d4b\u8bd5\u3002",
    ]
    for text in samples:
        assert native.encode(text, add_special_tokens=False).ids == exported.encode(
            text, add_special_tokens=False
        )
        ids = native.encode(text, add_special_tokens=False).ids
        assert native.decode(ids, skip_special_tokens=False) == exported.decode(
            ids, skip_special_tokens=False
        )
    root = os.environ.get("FL_ROOT", str(Path(__file__).resolve().parents[3]))

    def git(*arguments):
        return subprocess.check_output(["git", "-C", root, *arguments])

    # HEAD alone cannot identify a dirty candidate. Hash tracked and untracked source
    # files without importing FL or querying a worker process.
    source_extensions = {
        ".py",
        ".sh",
        ".cpp",
        ".cc",
        ".c",
        ".h",
        ".hpp",
        ".cu",
        ".json",
        ".toml",
        ".yaml",
        ".yml",
        ".cmake",
    }
    source_hashes = {}
    for raw in git(
        "ls-files", "-z", "--cached", "--others", "--exclude-standard"
    ).split(b"\0"):
        if not raw:
            continue
        relative = os.fsdecode(raw)
        path = Path(root) / relative
        parts = Path(relative).parts
        if any(
            part in {".venv", "__pycache__", "results", "tokenizer_export"}
            for part in parts
        ):
            continue
        if path.is_file() and (
            path.suffix.lower() in source_extensions or path.name == "CMakeLists.txt"
        ):
            source_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    source_hashes = dict(sorted(source_hashes.items()))
    status = git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        ".",
        ":(exclude)**/.venv/**",
        ":(exclude)**/results/**",
        ":(exclude)**/__pycache__/**",
        ":(exclude)**/tokenizer_export/**",
    ).decode(errors="replace")
    manifest_path = os.environ.get("FL_CANDIDATE_MANIFEST")
    candidate_manifest = None
    if manifest_path:
        manifest = Path(manifest_path).resolve(strict=True)
        candidate_manifest = {
            "path": str(manifest),
            "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        }

    record = {
        "fl_commit": git("rev-parse", "HEAD").decode().strip(),
        "fl_root": root,
        "fl_dirty": bool(status),
        "fl_status_porcelain": status,
        "fl_worktree_diff_sha256": hashlib.sha256(
            git("diff", "HEAD", "--binary")
        ).hexdigest(),
        "fl_source_hashes": source_hashes,
        "fl_source_manifest_sha256": hashlib.sha256(
            json.dumps(source_hashes, sort_keys=True).encode()
        ).hexdigest(),
        "fl_candidate_manifest": candidate_manifest,
        "source_identity_scope": "Current disk files, not proof of files loaded by the running service; source extensions only, excluding .venv/results/tokenizer_export/__pycache__.",
        "base_url": base,
        "models_response": models_response,
        "evalscope": metadata.version("evalscope"),
        "tokenizer_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "text_length": "random text tokens excluding client chat template; server adds BOS/template/image tokens",
        "configured_server": os.environ.get("OCR_SERVER_CONFIG", "unknown"),
        "configured_server_evidence": "Operator-provided OCR_SERVER_CONFIG; unknown if unset. /v1/models does not establish runtime scheduling, graph mode or parallel configuration.",
        "image_processor": {"image_size": 640, "base_size": 1024, "crop_mode": True},
        "paper_exact_reproduction": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


def summarize(args):
    root = args.root
    round_numbers = [args.check_round] if args.check_round is not None else (1, 2, 3)
    rows = []
    for n in round_numbers:
        paths = list((root / f"round{n}").rglob("benchmark_summary.json"))
        if len(paths) != 1:
            raise SystemExit(
                f"Round {n}: expected one benchmark_summary.json, found {len(paths)}"
            )
        row = json.loads(paths[0].read_text())
        for key, value in {
            "Total Requests": 1024,
            "Success Requests": 1024,
            "Failed Requests": 0,
        }.items():
            if float(row[key]) != value:
                raise SystemExit(
                    f"Round {n}: invalid {key}={row[key]}; inspect {paths[0]}"
                )
        if not math.isclose(
            float(row["Avg Output Tokens"]), 1024, rel_tol=0, abs_tol=1e-9
        ):
            raise SystemExit(
                f"Round {n}: output length differs from 1024; inspect raw records"
            )
        if (
            not math.isfinite(float(row["Output Throughput (tok/s)"]))
            or float(row["Output Throughput (tok/s)"]) <= 0
        ):
            raise SystemExit(f"Round {n}: invalid throughput")
        rows.append(
            {
                "round": n,
                "output_tokens_per_second": float(row["Output Throughput (tok/s)"]),
                "source": str(paths[0]),
            }
        )
    if len(round_numbers) == 1:
        print(
            f"Round {round_numbers[0]} passed aggregate success/output-length checks."
        )
        return
    a, b = (x["output_tokens_per_second"] for x in rows[1:])
    mean = (a + b) / 2
    variation = abs(a - b) / mean if mean > 0 else float("inf")
    result = {
        "rounds": rows,
        "round_2_3_mean_output_tokens_per_second": mean,
        "relative_difference": variation,
        "stable_below_1_percent": variation < 0.01,
        "difference_formula": "abs(round2-round3)/mean(round2,round3)",
        "note": "Synthetic fixed-output load; length finish is expected. TP2/crop configuration retained; not exact H100 reproduction.",
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["stable_below_1_percent"]:
        raise SystemExit(
            "Rounds 2/3 differ by >=1%; keep results and repeat the case before accepting the mean."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser(
        "prepare", help="Check readiness and export the tokenizer"
    )
    prepare_parser.add_argument("--output", type=Path, default=Path("environment.json"))
    prepare_parser.add_argument(
        "--check-server", action="store_true", help="Only check service readiness"
    )
    summarize_parser = commands.add_parser(
        "summarize", help="Validate and summarize the benchmark results"
    )
    summarize_parser.add_argument("root", type=Path)
    summarize_parser.add_argument("--check-round", type=int, choices=(1, 2, 3))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
