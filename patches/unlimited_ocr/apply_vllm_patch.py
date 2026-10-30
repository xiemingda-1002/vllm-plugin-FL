#!/usr/bin/env python3
"""Apply/check/reverse the pinned Unlimited-OCR core backport with hash guards."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Editable vLLM source checkout")
    parser.add_argument(
        "--check", action="store_true", help="Validate without changing files"
    )
    parser.add_argument(
        "--reverse", action="store_true", help="Restore the pinned pre-patch files"
    )
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / "manifest.json").read_text(encoding="utf-8"))
    patch = here / "vllm-0.24-unlimited-ocr.patch"
    try:
        source = args.source.resolve(strict=True)
        revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != manifest["vllm_base_commit"]:
            raise ValueError(
                "Expected vLLM commit "
                + manifest["vllm_base_commit"]
                + ", got "
                + revision
            )
        if sha256(patch) != manifest["patch_sha256"]:
            raise ValueError("Backport patch checksum mismatch")
        before_key, after_key = (
            ("after_sha256", "before_sha256")
            if args.reverse
            else ("before_sha256", "after_sha256")
        )
        states = []
        for record in manifest["core_files"]:
            path = (source / record["path"]).resolve()
            if not path.is_relative_to(source):
                raise ValueError(
                    "Manifest path escapes source checkout: " + record["path"]
                )
            states.append((record, path, sha256(path)))
        if all(value == record[after_key] for record, _, value in states):
            print(
                "Already "
                + ("reversed" if args.reverse else "applied")
                + "; all file hashes match"
            )
            return
        changed = [
            record["path"] for record, _, value in states if value != record[before_key]
        ]
        if changed:
            raise ValueError(
                "Source differs from the expected patch state; refusing overwrite: "
                + ", ".join(changed)
            )
        command = ["git", "-C", str(source), "apply"] + (
            ["--reverse"] if args.reverse else []
        )
        subprocess.run(command + ["--check", str(patch)], check=True)
        if args.check:
            print(
                "PASS: pinned revision, patch checksum, source hashes and git apply --check"
            )
            return
        subprocess.run(command + [str(patch)], check=True)
        if not all(sha256(path) == record[after_key] for record, path, _ in states):
            raise ValueError(
                "Post-apply hashes differ; inspect this checkout before use"
            )
        print(
            ("Reversed" if args.reverse else "Applied")
            + " 18-file Unlimited-OCR core backport"
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
