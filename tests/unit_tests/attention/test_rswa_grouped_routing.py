"""CPU-only metadata routing checks; no vLLM import and no model execution."""

import ast
import importlib.util
import logging
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

repo = Path(__file__).resolve().parents[3]
directory = repo / "vllm_fl/dispatch/backends/vendor/ascend/impl"
spec = importlib.util.spec_from_file_location(
    "integration_mask", directory / "attention_mask.py"
)
mask = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mask)
tree = ast.parse((directory / "attention.py").read_text(encoding="utf-8"))
names = {
    "RswaAttentionGroup",
    "AscendAttentionState",
    "AscendMetadata",
    "_build_grouped_rswa_masks",
    "_rswa_mask_request_lengths",
}
nodes = [
    n
    for n in tree.body
    if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
]
builder_node = next(
    n
    for n in tree.body
    if isinstance(n, ast.ClassDef) and n.name == "AscendAttentionMetadataBuilder"
)
nodes += [
    next(
        n
        for n in builder_node.body
        if isinstance(n, ast.FunctionDef) and n.name == "build"
    )
]
prelude = ast.parse(
    "from __future__ import annotations\nfrom dataclasses import dataclass\nfrom enum import Enum\n"
).body
module = types.ModuleType("grouped_integration_under_test")
sys.modules[module.__name__] = module
module.__dict__.update(torch=torch, os=os, logger=logging.getLogger(__name__))
exec(
    compile(
        ast.Module(body=prelude + nodes, type_ignores=[]), "<metadata-build>", "exec"
    ),
    module.__dict__,
)
state = module.AscendAttentionState

cases = [
    ("OCR default enabled", None, state.ChunkedPrefill, None, -1, None, 10, True),
    (
        "ordinary default disabled",
        None,
        state.ChunkedPrefill,
        None,
        -1,
        None,
        10,
        False,
    ),
    ("mixed enabled", "1", state.ChunkedPrefill, None, -1, None, 10, True),
    ("explicit disabled", "0", state.ChunkedPrefill, None, -1, None, 10, False),
    ("pure decode", "1", state.DecodeOnly, None, -1, None, 4, False),
    ("unpaged prefill", "1", state.PrefillNoCache, None, -1, None, 10, False),
    ("speculative fallback", "1", state.ChunkedPrefill, object(), -1, None, 10, False),
    ("graph padding fallback", "1", state.ChunkedPrefill, None, 10, None, 10, False),
    ("request padding fallback", "1", state.ChunkedPrefill, None, -1, 3, 10, False),
    ("token padding fallback", "1", state.ChunkedPrefill, None, -1, None, 9, False),
]


@pytest.mark.parametrize(
    "name,enabled,phase,speculative,graph_pad,actual_reqs,total,grouped",
    cases,
    ids=[case[0] for case in cases],
)
def test_metadata_grouped_routing(
    name, enabled, phase, speculative, graph_pad, actual_reqs, total, grouped
):
    lengths = [1, 1, 1, 1] if phase == state.DecodeOnly else [1, 1, 3, 5]
    starts = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()))
    common = NS(
        num_reqs=4,
        num_actual_tokens=total,
        query_start_loc_cpu=starts,
        block_table_tensor=torch.arange(16).view(4, 4),
        _seq_lens_cpu=torch.tensor([13, 8, 5, 9]),
        slot_mapping=torch.arange(total),
        attn_state=phase,
        rswa_prefix_lens=torch.tensor([0, 2, 12, 0]),
        rswa_num_actual_reqs=actual_reqs,
        graph_pad_size=graph_pad,
        causal=True,
        max_query_len=max(lengths),
    )
    calls = []
    real_builder = mask.AttentionMaskBuilder(torch.device("cpu"))

    def build_mask(**kwargs):
        calls.append((kwargs["query_lens"].numel(), kwargs["max_query_len"]))
        return real_builder.get_rswa_attn_mask(**kwargs)

    fake_builder = NS(get_rswa_attn_mask=build_mask)
    decodes = 4 if phase == state.DecodeOnly else 2
    self = NS(
        device=torch.device("cpu"),
        model_config=NS(
            rswa_window=3,
            max_model_len=32,
            runner_type="generate",
            hf_config=NS(
                model_type="unlimited-ocr" if name.startswith("OCR") else "ordinary"
            ),
        ),
        vllm_config=NS(quant_config=None, speculative_config=speculative),
        _make_attention_mask=lambda phase: None,
        _get_mask_builder=lambda: fake_builder,
        _split_decodes_and_prefills=lambda metadata: (
            decodes,
            4 - decodes,
            decodes,
            total - decodes,
        ),
    )
    environment = dict(os.environ)
    environment["VLLM_FL_OCR_TORCH_NPU"] = "1"
    environment.pop("VLLM_FL_RSWA_GROUPED_MASK", None)
    if enabled is not None:
        environment["VLLM_FL_RSWA_GROUPED_MASK"] = enabled
    environment["VLLM_FL_RSWA_MASK_MAX_MIB"] = "2048"
    with (
        patch.dict(
            os.environ,
            environment,
            clear=True,
        ),
        patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor),
    ):
        result = module.build(self, 0, common)
    assert (result.rswa_groups is not None) == grouped, name
    if grouped:
        assert result.attn_mask is None and calls == [(2, 1), (2, 5)], (name, calls)
    else:
        assert len(calls) == 1 and result.attn_mask is not None, (name, calls)
