"""Optional eager dispatch evidence; observed schemas do not prove full closure.

Opaque custom operators and compiled graph replay may hide internal operators.
Use this diagnostic with eager short requests and retain the device trace too.
"""

import functools
import json
import logging
import os
import threading
from collections import Counter
from pathlib import Path

from torch.utils._python_dispatch import TorchDispatchMode

logger = logging.getLogger(__name__)
FORBIDDEN = frozenset({"_C", "_moe_C", "_C_cache_ops", "_C_ascend", "vllm_ascend"})


class OCRExecutionAudit(TorchDispatchMode):
    """Count actually intercepted schemas, rejecting forbidden namespaces."""

    def __init__(self):
        super().__init__()
        self.schemas = Counter()
        self.method_calls = Counter()
        self.method_failures = Counter()
        self.depth = 0
        self.lock = threading.RLock()

    def persist(self):
        directory = os.environ.get("VLLM_FL_OCR_AUDIT_DIR")
        if not directory:
            return
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"execution-{os.getpid()}.json"
        temporary = target.with_suffix(".json.tmp")
        payload = {
            "pid": os.getpid(),
            "method_calls": dict(self.method_calls),
            "method_failures": dict(self.method_failures),
            "schemas": dict(self.schemas),
            "scope": (
                "Observed eager dispatch only; opaque operators and graph replay "
                "internals may be hidden. This is not full compute closure proof."
            ),
        }
        try:
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        schema = str(func._schema)
        namespace = func._schema.name.split("::", 1)[0]
        self.schemas[schema] += 1
        if self.schemas[schema] == 1:
            logger.info("FL OCR observed dispatch schema: %s", schema)
        if namespace in FORBIDDEN:
            raise RuntimeError(f"Forbidden OCR device operator namespace: {schema}")
        # vllm namespace wrappers remain visible in the evidence. Their nested
        # execution is not necessarily exposed to TorchDispatchMode.
        return func(*args, **(kwargs or {}))


def install_execution_audit(model):
    """Wrap compute entry points with exception-safe mode scope; return recorder.

    Disabled by default. No hooks or global dispatch modes remain after a
    forward returns or raises. The wrapper is for diagnostic eager runs only.
    """
    if os.environ.get("VLLM_FL_OCR_AUDIT_OPS", "0") != "1":
        return None
    existing = getattr(model, "_ocr_execution_audit", None)
    if existing is not None:
        return existing
    recorder = OCRExecutionAudit()

    def wrap(original, method_name):
        @functools.wraps(original)
        def audited(*args, **kwargs):
            # Serialize diagnostic calls, and enter the mode once when entry
            # points call each other. Re-entering the same mode duplicates ops.
            with recorder.lock:
                outer = recorder.depth == 0
                recorder.depth += 1
                recorder.method_calls[method_name] += 1
                try:
                    if outer:
                        with recorder:
                            return original(*args, **kwargs)
                    return original(*args, **kwargs)
                except BaseException:
                    recorder.method_failures[method_name] += 1
                    raise
                finally:
                    recorder.depth -= 1
                    if outer:
                        # Persistence failures must not replace a model error.
                        try:
                            recorder.persist()
                        except Exception:
                            logger.exception("Could not persist OCR dispatch evidence")

        return audited

    for method_name in (
        "forward",
        "embed_multimodal",
        "get_multimodal_embeddings",
        "encoder_eager_forward",
        "get_input_embeddings",
        "embed_input_ids",
    ):
        original = getattr(model, method_name, None)
        if callable(original):
            setattr(model, method_name, wrap(original, method_name))
    model._ocr_execution_audit = recorder
    logger.warning(
        "FL OCR dispatch audit enabled for eager diagnostics; observed schemas "
        "alone do not prove full compute closure or graph replay internals"
    )
    return recorder
