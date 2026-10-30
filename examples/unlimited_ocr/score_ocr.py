"""Run official OmniDocBench v1.5 scoring and export the six paper Table 1 metrics.

Run in the scoring Python environment; requires PyYAML and official evaluator/CDM
dependencies (including Node.js, TeX, ImageMagick and Ghostscript). This never calls an OCR server.
Preflight verifies required tools and an identical-formula CDM score of 1.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_EVALUATOR = HERE.parent / "ocr_paper_eval_20261026" / "evaluator_v1_5"
if (HERE / "evaluators" / "evaluator_v1_5").is_dir():
    DEFAULT_EVALUATOR = HERE / "evaluators" / "evaluator_v1_5"
DEFAULT_DATASET = HERE / "test_images" / "omnidocbench_v1_5" / "OmniDocBench.json"
NATIVE_SHA256 = "90a4cf4628321cf53191bd20e0109b5935032feedc8c6e502a3fa46a9181a51c"
TARGETS = dict(
    overall=93.23,
    text_edit=0.038,
    formula_cdm=92.61,
    table_teds=90.93,
    table_teds_s=94.07,
    reading_order_edit=0.045,
)
LABELS = [
    ("overall", "Overall ↑"),
    ("text_edit", "Text Edit ↓"),
    ("formula_cdm", "Formula CDM ↑"),
    ("table_teds", "Table TEDS ↑"),
    ("table_teds_s", "Table TEDS-S ↑"),
    ("reading_order_edit", "Read-order Edit ↓"),
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def select_pages(all_names, available, manifest, *, subset=False, auto_subset=False):
    """Select requested pages, retaining failed, empty and truncated requests."""
    requested = None
    if manifest is not None:
        if not isinstance(manifest, list):
            raise ValueError("Run manifest must be a list")
        names = []
        requested = set()
        for row in manifest:
            if not isinstance(row, dict) or row.get("status") not in {
                "complete",
                "failed",
                "length",
                "empty_prediction",
                "not_requested",
            }:
                raise ValueError("Invalid manifest entry/status")
            name = Path(row["image"]).stem + ".md"
            names.append(name)
            if row["status"] != "not_requested":
                requested.add(name)
        if len(names) != len(set(names)) or set(names) - all_names:
            raise ValueError("Manifest has duplicate pages or pages outside GT")
    inferred = auto_subset and requested is not None and requested != all_names
    use_subset = subset or inferred
    selected = (
        (requested if requested is not None else set(available))
        if use_subset
        else all_names
    )
    if not selected:
        raise ValueError("No requested pages to score")
    missing = sorted(selected - set(available))
    if missing:
        raise ValueError(
            f"Missing {len(missing)}/{len(selected)} required Markdown files "
            f"(failed/empty/truncated requests are retained): {missing[:5]}. "
            "For a selected-page run use --subset, or pass its run directory."
        )
    return selected, use_subset, bool(inferred and not subset)


def extract_metrics(result, *, unavailable=None):
    """Use official generate_result_tables.ipynb page aggregation, without fallback."""
    fields = {
        "text_edit": ("text_block", "all", "Edit_dist", "ALL_page_avg"),
        "formula_cdm": ("display_formula", "page", "CDM", "ALL"),
        "table_teds": ("table", "page", "TEDS", "ALL"),
        "table_teds_s": ("table", "page", "TEDS_structure_only", "ALL"),
        "reading_order_edit": ("reading_order", "all", "Edit_dist", "ALL_page_avg"),
    }
    unavailable = unavailable or {}
    scores = {}
    for name, keys in fields.items():
        value = result
        try:
            for key in keys:
                value = value[key]
            if isinstance(value, bool):
                raise ValueError("boolean score")
            value = float(value)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("score must be finite and within [0, 1]")
        except KeyError as exc:
            if name in unavailable:
                scores[name] = None
                continue
            raise ValueError(
                "Invalid/missing official metric " + ".".join(keys)
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid/missing official metric " + ".".join(keys)
            ) from exc
        scores[name] = (
            value * 100
            if name in {"formula_cdm", "table_teds", "table_teds_s"}
            else value
        )
    scores["overall"] = (
        None
        if any(scores[k] is None for k in ("text_edit", "formula_cdm", "table_teds"))
        else (
            (1 - scores["text_edit"]) * 100
            + scores["formula_cdm"]
            + scores["table_teds"]
        )
        / 3
    )
    return scores


def export_metrics(output, summary, native_result, pages):
    """Absent subset categories are N/A; invalid scores and populated categories stay strict."""
    counts = Counter(
        item.get("category_type") for page in pages for item in page["layout_dets"]
    )
    unavailable = {}
    if summary["subset"]:
        if not counts["equation_isolated"]:
            unavailable["formula_cdm"] = (
                "Selected GT contains no equation_isolated annotations"
            )
        if not counts["table"]:
            unavailable.update(
                {
                    key: "Selected GT contains no table annotations"
                    for key in ("table_teds", "table_teds_s")
                }
            )
    scores = extract_metrics(read_json(native_result), unavailable=unavailable)
    missing = {
        key: unavailable[key]
        for key, value in scores.items()
        if value is None and key != "overall"
    }
    if scores["overall"] is None:
        missing["overall"] = (
            "Requires Text Edit, Formula CDM and Table TEDS; no partial average is computed"
        )
    summary.update(
        status="complete",
        metrics=scores,
        native_result=str(native_result),
        unavailable_metrics=missing,
        selected_gt_category_counts=dict(counts),
        paper_comparison_eligible=not summary["subset"]
        and summary["ground_truth_sha256"] == NATIVE_SHA256,
        overall_formula="((1 - text_edit) * 100 + formula_cdm + table_teds) / 3",
    )
    summary.pop("error", None)
    write_json(output / "summary.json", summary)
    title = "OmniDocBench v1.5 Table 1 metrics" + (
        " — SUBSET; NOT PAPER COMPARABLE" if summary["subset"] else ""
    )
    table = [title, "", "| Metric | Measured | Paper target |", "|---|---:|---:|"]
    for key, label in LABELS:
        measured = "N/A" if scores[key] is None else f"{scores[key]:.3f}"
        table.append(f"| {label} | {measured} | {TARGETS[key]:.3f} |")
    coverage = summary["coverage"]
    table += [
        "",
        f"Coverage: {coverage['selected_pages']}/{coverage['dataset_pages']}; "
        f"empty Markdown retained: {coverage['empty_markdown']}.",
        "CDM/TEDS/TEDS-S/Overall use 0–100; Edit uses 0–1. Reading order and TEDS-S do not enter Overall.",
        "Run status: " + json.dumps(coverage["run_status"], ensure_ascii=False),
        summary["comparison_note"],
    ]
    table += [f"{key}: N/A — {reason}." for key, reason in missing.items()]
    (output / "table1.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print(json.dumps(scores, indent=2))
    print("Results:", output)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input", type=Path, help="test_ocr run directory or flat predictions directory"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Native v1.5 OmniDocBench.json",
    )
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=DEFAULT_EVALUATOR,
        help="Official v1.5 source directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New directory; default: input-parent/scores-TIMESTAMP",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Input is an existing score directory; export completed official results without rerunning evaluation",
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--subset",
        action="store_true",
        help="Score requested manifest pages (or available Markdown without a manifest)",
    )
    scope.add_argument(
        "--full",
        action="store_true",
        help="Require all 1355 pages even for a selected-page run",
    )
    args = parser.parse_args()
    source, gt, evaluator = (
        args.input.resolve(),
        args.dataset.resolve(),
        args.evaluator.resolve(),
    )
    output = (
        args.output
        or source.parent
        / ("scores-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    ).resolve()
    created_output = False
    try:
        if args.summarize_only:
            if args.output or args.subset or args.full:
                raise ValueError(
                    "--summarize-only uses the saved score scope; do not pass --output/--subset/--full"
                )
            saved = read_json(source / "summary.json")
            if saved.get("status") != "complete" and not (
                saved.get("status") == "failed"
                and saved.get("error", "").startswith(
                    "Invalid/missing official metric "
                )
            ):
                raise ValueError(
                    "Only completed evaluation or failed metric export can be summarized"
                )
            if saved.get("environment_check", {}).get("status") != "complete":
                raise ValueError("Saved CDM environment check did not complete")
            saved_gt = (
                source / "subset_ground_truth.json"
                if saved["subset"]
                else Path(saved["ground_truth"])
            )
            native = (
                source
                / "result"
                / (Path(saved["predictions"]).name + "_quick_match_metric_result.json")
            )
            # Keep the prior error and metadata for audit before updating this summary.
            backup = source / "summary.before-missing-category-fix.json"
            if not backup.exists():
                shutil.copy2(source / "summary.json", backup)
            export_metrics(source, saved, native, read_json(saved_gt))
            return
        if (
            not source.is_dir()
            or not gt.is_file()
            or not (evaluator / "pdf_validation.py").is_file()
        ):
            raise ValueError(
                "Input, native dataset or official evaluator missing; specify --dataset/--evaluator if needed"
            )
        if output.exists():
            raise ValueError("--output must be a new directory")
        run = (
            source
            if (source / "manifest.json").is_file()
            and (source / "summary.json").is_file()
            else None
        )
        prediction = source
        run_summary = read_json(run / "summary.json") if run else {}
        if run:
            configured = run_summary.get("prediction_dir")
            prediction = (
                Path(configured).resolve() if configured else source / "predictions"
            )
            if not prediction.is_dir():
                prediction = (
                    source / "predictions"
                )  # Supports a relocated complete run.
        elif (
            source.name == "predictions" and (source.parent / "manifest.json").is_file()
        ):
            run = source.parent
            run_summary = (
                read_json(run / "summary.json")
                if (run / "summary.json").is_file()
                else {}
            )
        if not prediction.is_dir():
            raise ValueError(
                "Prediction directory missing; repeated-image runs cannot be scored"
            )
        pages = read_json(gt)
        by_name = {Path(p["page_info"]["image_path"]).stem + ".md": p for p in pages}
        if len(by_name) != len(pages):
            raise ValueError("Dataset has duplicate prediction basenames")
        gt_hash = hashlib.sha256(gt.read_bytes()).hexdigest()
        if not args.subset and (len(pages) != 1355 or gt_hash != NATIVE_SHA256):
            raise ValueError(
                "Full protocol requires pinned native v1.5 GT (1355 pages); --subset is required for other GT"
            )
        available = {
            p.name: p for p in prediction.iterdir() if p.is_file() and p.suffix == ".md"
        }
        unknown = sorted(set(available) - set(by_name))
        if unknown:
            raise ValueError("Prediction names absent from GT: " + repr(unknown[:5]))
        manifest = read_json(run / "manifest.json") if run else None
        stats = dict(
            manifest="unknown", failed=None, length=None, empty=None, not_requested=None
        )
        if manifest is not None:
            if not isinstance(manifest, list):
                raise ValueError("Run manifest must be a list")
            statuses = Counter(row["status"] for row in manifest)
            stats = dict(
                manifest=str(run / "manifest.json"),
                status_counts=dict(statuses),
                failed=statuses["failed"],
                length=statuses["length"],
                empty=statuses["empty_prediction"],
                not_requested=statuses["not_requested"],
            )
        selected, subset, subset_auto_detected = select_pages(
            set(by_name),
            available,
            manifest,
            subset=args.subset,
            auto_subset=not args.full
            and run_summary.get("selected_images_only") is True,
        )
        print(
            f"Scoring {len(selected)}/{len(pages)} pages; scope="
            + ("SUBSET (not directly comparable to paper)" if subset else "FULL"),
            flush=True,
        )
        # Keep CPU scoring self-contained for direct docker exec invocations.
        environment = os.environ.copy()
        private_im7 = Path("/opt/omnidoc-imagemagick7/bin")
        if private_im7.is_dir():
            environment["PATH"] = (
                str(private_im7) + os.pathsep + environment.get("PATH", "")
            )
        private_texmf = Path("/opt/ocr-eval/texmf")
        if private_texmf.is_dir():
            environment["TEXMFHOME"] = str(private_texmf)
        for variable in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "MAGICK_THREAD_LIMIT",
        ):
            environment[variable] = "1"
        environment.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
        os.environ.update(environment)
        missing_tools = [
            tool
            for tool in ("xelatex", "magick", "node", "gs")
            if shutil.which(tool, path=environment.get("PATH")) is None
        ]
        if missing_tools:
            raise ValueError(
                "Required CDM tools missing from scoring PATH: "
                + ", ".join(missing_tools)
            )
        empty = sorted(
            name
            for name in selected
            if not available[name].read_text(encoding="utf-8").strip()
        )
        coverage = dict(
            dataset_pages=len(pages),
            selected_pages=len(selected),
            markdown_pages=len(available),
            evaluated_pages=len(selected),
            selected_coverage=len(selected) / len(pages),
            empty_markdown=len(empty),
            empty_markdown_names=empty,
            run_status=stats,
            excluded_by_status=False,
        )
        import yaml

        output.mkdir(parents=True, exist_ok=False)
        created_output = True
        # A separate cwd contains ALL native result/CDM files; no evaluator result is overwritten.
        (output / "result").mkdir()
        eval_gt = gt
        if subset:
            eval_gt = output / "subset_ground_truth.json"
            write_json(eval_gt, [p for name, p in by_name.items() if name in selected])
        config = {
            "end2end_eval": {
                "metrics": {
                    "text_block": {"metric": ["Edit_dist"]},
                    "display_formula": {"metric": ["CDM"]},
                    "table": {"metric": ["TEDS"]},
                    "reading_order": {"metric": ["Edit_dist"]},
                },
                "dataset": {
                    "dataset_name": "end2end_dataset",
                    "ground_truth": {"data_path": str(eval_gt)},
                    "prediction": {"data_path": str(prediction)},
                    "match_method": "quick_match",
                },
            }
        }
        config_path = output / "official_config.yaml"
        config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        summary = dict(
            status="running",
            input=str(source),
            predictions=str(prediction),
            ground_truth=str(gt),
            ground_truth_sha256=gt_hash,
            evaluator=str(evaluator),
            subset=subset,
            subset_auto_detected=subset_auto_detected,
            coverage=coverage,
            protocol="OmniDocBench v1.5 official quick_match, page averages",
            paper="https://arxiv.org/pdf/2606.23050",
            paper_table1_v15_unlimited_ocr_targets=TARGETS,
            paper_comparison_eligible=False,
            comparison_note="Subset cannot be compared directly; full native dataset uses the same scoring protocol. Generation/checkpoint settings require separate verification.",
        )
        write_json(output / "summary.json", summary)
        # Verify the actual renderer/matcher before accepting any model CDM scores.
        check_dir = output / "environment_check"
        check_dir.mkdir()
        summary["environment_check"] = dict(
            formula="x+y=z", expected_f1=1, artifacts=str(check_dir), status="running"
        )
        write_json(output / "summary.json", summary)
        try:
            sys.path.insert(0, str(evaluator))
            from metrics.cdm_metric import CDM

            check = CDM(output_root=str(check_dir)).evaluate(
                "x+y=z", "x+y=z", "identical_formula"
            )
            write_json(check_dir / "result.json", check)
            summary["environment_check"].update(result=check, status="complete")
            write_json(output / "summary.json", summary)
            if float(check["F1_score"]) != 1.0:
                raise ValueError(
                    f"Identical-formula F1_score must be 1, got {check['F1_score']!r}"
                )
        except Exception as exc:
            summary["environment_check"].update(status="failed", error=str(exc))
            write_json(output / "summary.json", summary)
            raise RuntimeError(
                f"CDM environment self-check failed; inspect {check_dir}: {exc}"
            ) from exc
        command = [
            sys.executable,
            "-u",
            str(evaluator / "pdf_validation.py"),
            "--config",
            str(config_path),
        ]
        with (output / "scoring.log").open("w", encoding="utf-8") as log:
            log.write(json.dumps(command) + "\n")
            log.flush()
            with subprocess.Popen(
                command,
                cwd=output,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=environment,
            ) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                returncode = process.wait()
        if returncode:
            raise RuntimeError(
                f"Official evaluator exited {returncode}; inspect {output / 'scoring.log'}"
            )
        native_result = (
            output / "result" / (prediction.name + "_quick_match_metric_result.json")
        )
        export_metrics(
            output, summary, native_result, [by_name[name] for name in selected]
        )
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
        summary_path = output / "summary.json"
        if created_output and summary_path.is_file():
            failed = read_json(summary_path)
            failed.update(
                status="failed", error=str(exc), paper_comparison_eligible=False
            )
            write_json(summary_path, failed)
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
