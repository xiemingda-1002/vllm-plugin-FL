"""Check service readiness and export the native tokenizer; no inference calls."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("environment.json"))
parser.add_argument(
    "--check-server",
    action="store_true",
    help="Check readiness only; no inference request",
)
args = parser.parse_args()
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
        f"OCR service is not ready at {base}: {exc}. Start ../run_server.sh and wait for readiness. No benchmark requests sent."
    ) from None
if args.check_server:
    print(f"OCR service ready: {base}, model=Unlimited-OCR")
    raise SystemExit(0)
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
for raw in git("ls-files", "-z", "--cached", "--others", "--exclude-standard").split(
    b"\0"
):
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
