"""User-run small R-SWA FIA/device check; no model weights or HTTP service.

Set ASCEND_RT_VISIBLE_DEVICES before launching. Uses logical NPU 0 only.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import torch
import torch_npu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--rtol", type=float, default=0.02)
    args = parser.parse_args()
    if args.atol <= 0 or args.rtol <= 0:
        parser.error("Tolerances must be positive")
    root = Path(__file__).resolve().parents[2]
    path = root / "vllm_fl/dispatch/backends/vendor/ascend/impl/attention_mask.py"
    spec = importlib.util.spec_from_file_location("fl_rswa_device_mask", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    torch.manual_seed(42)
    torch.npu.manual_seed_all(42)
    print("FIA schema:", torch.ops.npu.npu_fused_infer_attention_score.default._schema)
    builder = module.AttentionMaskBuilder(device)
    cases = [("prefill", 32, 32, 32), ("chunk", 16, 80, 200), ("decode", 1, 192, 32)]
    for name, q_len, kv_len, prefix in cases:
        heads, dim, window = 2, 64, 128
        mask = builder.get_rswa_attn_mask(
            torch.tensor([prefix], dtype=torch.int32),
            torch.tensor([kv_len], dtype=torch.int32),
            torch.tensor([q_len], dtype=torch.int32),
            max_kv_len=max(kv_len, prefix),
            max_query_len=q_len,
            rswa_window=window,
        )
        positions = torch.arange(kv_len - q_len, kv_len)[:, None]
        keys = torch.arange(kv_len)[None, :]
        keep = (keys <= positions) & ((keys < prefix) | (keys > positions - window))
        actual_mask = mask[0, 0, :q_len, :kv_len].cpu().bool()
        torch.testing.assert_close(actual_mask, ~keep)
        q = torch.randn(q_len, heads, dim).to(device=device, dtype=torch.bfloat16)
        k = torch.randn(kv_len, heads, dim).to(device=device, dtype=torch.bfloat16)
        v = torch.randn(kv_len, heads, dim).to(device=device, dtype=torch.bfloat16)
        output, _ = torch_npu.npu_fused_infer_attention_score(
            query=q,
            key=k,
            value=v,
            atten_mask=mask,
            input_layout="TND",
            actual_seq_lengths=[q_len],
            actual_seq_lengths_kv=[kv_len],
            num_heads=heads,
            num_key_value_heads=heads,
            scale=dim**-0.5,
            sparse_mode=0,
        )
        torch.npu.synchronize()
        # FP32 CPU reference starts from the same rounded BF16 inputs.
        q_ref, k_ref, v_ref = (x.cpu().float().transpose(0, 1) for x in (q, k, v))
        scores = (q_ref @ k_ref.transpose(-1, -2)) * dim**-0.5
        scores.masked_fill_(~keep.unsqueeze(0), float("-inf"))
        expected = (scores.softmax(-1) @ v_ref).transpose(0, 1)
        actual = output.cpu().float().reshape_as(expected)
        torch.testing.assert_close(actual, expected, atol=args.atol, rtol=args.rtol)
        print(
            json.dumps(
                {
                    "case": name,
                    "query": q_len,
                    "kv": kv_len,
                    "prefix": prefix,
                    "max_abs_error": (actual - expected).abs().max().item(),
                    "passed": True,
                }
            )
        )
    print(
        "PASS: three small mask/FIA cases. This does not validate paged KV, TP2, FULL graphs or model accuracy."
    )


if __name__ == "__main__":
    main()
