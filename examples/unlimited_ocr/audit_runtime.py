"""Inventory the OCR installation without loading model weights or running inference.

This preflight cannot certify the device kernels used by a future service process.
The model-load implementation audit and user-run traces provide that evidence.
"""

import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.find_spec("vllm_fl")
    if spec is None or spec.origin is None:
        parser.error("FL cannot be located by this Python interpreter")
    loaded_root = Path(spec.origin).resolve().parent.parent
    if loaded_root != root:
        parser.error(f"FL resolves to {loaded_root}, expected {root}")
    packages = {}
    for name in (
        "vllm-plugin-fl",
        "vllm",
        "torch",
        "torch-npu",
        "transformers",
        "vllm-ascend",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    import torch_npu

    required = ("npu_fused_infer_attention_score", "_npu_reshape_and_cache")
    apis = {name: hasattr(torch_npu, name) for name in required}
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    result = {
        "python": sys.executable,
        "fl_source": str(loaded_root),
        "fl_commit": revision,
        "packages": packages,
        "required_attention_apis": apis,
        "strict_ocr_env": os.environ.get("VLLM_FL_OCR_TORCH_NPU"),
        "editable": json.loads(
            metadata.distribution("vllm-plugin-fl").read_text("direct_url.json")
            or "null"
        ),
        "startup_sha256": hashlib.sha256(
            (Path(__file__).parent / "run_server.sh").read_bytes()
        ).hexdigest(),
        "vllm_ascend_imported": any(
            k == "vllm_ascend" or k.startswith("vllm_ascend.") for k in sys.modules
        ),
        "scope": "Installation/API inventory only; no model, tensor execution, NPU profile or service request.",
        "clean_environment_proven": False,
        "note": "An installed vllm-ascend package is reported, not imported. A clean dependency test requires a separate environment without it and its private extensions.",
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if not all(apis.values()) or result["vllm_ascend_imported"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
