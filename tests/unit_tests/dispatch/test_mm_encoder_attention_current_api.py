from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATH = (
    ROOT
    / "vllm_fl/dispatch/backends/vendor/ascend/impl/mm_encoder_attention.py"
)


def _load_class_with_current_vllm_base():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "AscendMMEncoderAttention"
    )

    class CurrentMMEncoderAttention:
        def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float | None = None,
            num_kv_heads: int | None = None,
            prefix: str = "",
        ) -> None:
            self.init_kwargs = {
                "num_heads": num_heads,
                "head_size": head_size,
                "scale": scale,
                "num_kv_heads": num_kv_heads,
                "prefix": prefix,
            }

    class Tensor:
        pass

    namespace = {
        "__name__": "test_mm_encoder_attention_current_api",
        "MMEncoderAttention": CurrentMMEncoderAttention,
        "torch": SimpleNamespace(Tensor=Tensor),
        "F": SimpleNamespace(),
        "torch_npu": SimpleNamespace(),
        "einops": SimpleNamespace(),
        "MIN_PAD_SIZE": 64,
        "MAX_PAD_SIZE": 128,
    }
    module = ast.Module(body=[class_node], type_ignores=[])
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return namespace["AscendMMEncoderAttention"]


def test_constructor_matches_current_vllm_024_contract() -> None:
    cls = _load_class_with_current_vllm_base()

    layer = cls(
        num_heads=8,
        head_size=128,
        scale=0.125,
        num_kv_heads=2,
        prefix="model.encoder.attn",
    )

    assert layer.init_kwargs == {
        "num_heads": 8,
        "head_size": 128,
        "scale": 0.125,
        "num_kv_heads": 2,
        "prefix": "model.encoder.attn",
    }
    assert "multimodal_config" not in inspect.signature(cls).parameters


def test_forward_oot_accepts_current_vllm_optional_metadata() -> None:
    cls = _load_class_with_current_vllm_base()

    signature = inspect.signature(cls.forward_oot)
    assert list(signature.parameters) == [
        "self",
        "query",
        "key",
        "value",
        "cu_seqlens",
        "max_seqlen",
        "sequence_lengths",
    ]
    signature.bind(
        object(),
        object(),
        object(),
        object(),
        sequence_lengths=object(),
    )


def test_text_qwen_fix_does_not_pull_encoder_graph_runtime() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")

    assert "encoder_acl_graph" not in source
    assert "get_encoder_forward_context" not in source
    assert "weak_ref_tensors" not in source
