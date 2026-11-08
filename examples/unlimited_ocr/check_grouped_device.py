"""Compare actual torch-npu grouped and whole-mask paged FIA on device.

Run from the repository: python examples/unlimited_ocr/check_grouped_device.py
No vLLM or vllm-ascend imports are needed.
"""

import ast
import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch_npu

ROOT = Path(__file__).resolve().parents[2]
IMPL = ROOT / "vllm_fl/dispatch/backends/vendor/ascend/impl"
spec = importlib.util.spec_from_file_location(
    "device_rswa_mask", IMPL / "attention_mask.py"
)
mask_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mask_module)
names = {"RswaAttentionGroup", "_build_grouped_rswa_masks", "_run_grouped_rswa_fia"}
tree = ast.parse((IMPL / "attention.py").read_text(encoding="utf-8"))
helpers = types.ModuleType("device_grouped_helpers")
sys.modules[helpers.__name__] = helpers
nodes = ast.parse(
    "from __future__ import annotations\nfrom dataclasses import dataclass\nimport torch\n_SWA_INT_MAX=2147483647\n"
).body
nodes += [
    n
    for n in tree.body
    if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
]
exec(
    compile(
        ast.Module(body=nodes, type_ignores=[]), str(IMPL / "attention.py"), "exec"
    ),
    helpers.__dict__,
)


def main():
    torch.manual_seed(11)
    device = torch.device("npu:0")
    builder = mask_module.AttentionMaskBuilder(device)
    heads, kv_heads, dim, block_size = 10, 10, 128, 128
    for lengths in ([1, 1, 5], [1, 1, 3, 7]):
        count = len(lengths)
        seq = [17 + i for i in range(count)]
        prefix = torch.tensor([4] * count)
        qlens = torch.tensor(lengths)
        cumulative = qlens.cumsum(0).tolist()
        tokens = sum(lengths)
        q = torch.randn(tokens, heads, dim, device=device, dtype=torch.bfloat16)
        k = torch.randn(
            count, block_size, kv_heads * dim, device=device, dtype=torch.bfloat16
        )
        v = torch.randn_like(k)
        bt = torch.zeros((count, 32768 // block_size), dtype=torch.int32, device=device)
        bt[:, 0] = torch.arange(count, dtype=torch.int32, device=device)
        groups = helpers._build_grouped_rswa_masks(
            builder, prefix, torch.tensor(seq), qlens, 2, 2, 32768, 4
        )
        whole_mask = builder.get_rswa_attn_mask(
            prefix, torch.tensor(seq), qlens, 32768, max(lengths), 4
        )
        expected, _ = torch_npu.npu_fused_infer_attention_score(
            query=q,
            key=k,
            value=v,
            block_table=bt,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=cumulative,
            actual_seq_lengths_kv=seq,
            num_key_value_heads=kv_heads,
            num_heads=heads,
            scale=dim**-0.5,
            sparse_mode=0,
            atten_mask=whole_mask,
            pre_tokens=2147483647,
            next_tokens=2147483647,
        )
        output = torch.full(
            (tokens + 2, heads, dim), -999.0, device=device, dtype=q.dtype
        )
        helpers._run_grouped_rswa_fia(
            torch_npu.npu_fused_infer_attention_score,
            groups,
            q,
            k,
            v,
            bt,
            seq,
            block_size,
            heads,
            kv_heads,
            dim,
            dim**-0.5,
            output,
        )
        torch.npu.synchronize()
        torch.testing.assert_close(
            output[:tokens], expected.view(tokens, heads, dim), atol=2e-3, rtol=2e-3
        )
        assert bool((output[tokens:] == -999).all())
        print(
            f"PASS lengths={lengths}; max_abs={(output[:tokens] - expected.view_as(output[:tokens])).abs().max().item():.6g}; padding preserved"
        )


if __name__ == "__main__":
    main()
