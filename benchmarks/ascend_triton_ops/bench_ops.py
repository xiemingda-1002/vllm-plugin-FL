#!/usr/bin/env python3
"""Per-operator micro-benchmark: native torch_npu.npu_* vs the Triton port."""
import argparse, importlib, json, statistics, sys, time
import torch, torch_npu

PKG = "vllm_fl.dispatch.backends.vendor.ascend"
TARGETS = [
 ("npu_swiglu","impl.swiglu","swiglu"),("npu_clipped_swiglu","impl.clipped_swiglu","clipped_swiglu"),
 ("npu_gelu_mul","impl.gelu_mul","gelu_mul"),("npu_grouped_matmul","impl.grouped_matmul","grouped_matmul"),
 ("npu_moe_token_unpermute","impl.moe_token_unpermute","moe_token_unpermute"),
 ("npu_moe_token_permute","impl.moe_token_permute","moe_token_permute"),
 ("npu_moe_init_routing_v2","impl.moe_init_routing_v2","moe_init_routing_v2"),
 ("npu_rms_norm","impl.rms_norm","npu_rms_norm"),
 ("npu_kv_rmsnorm_rope_cache","impl.kv_rmsnorm_rope_cache","npu_kv_rmsnorm_rope_cache"),
 ("npu_dynamic_quant","impl.dynamic_quant","dynamic_quant"),
 ("npu_dynamic_mx_quant","impl.dynamic_mx_quant","dynamic_mx_quant"),
 ("npu_dynamic_block_quant","impl.dynamic_block_quant","dynamic_block_quant"),
 ("npu_quant_matmul","impl.quant_matmul","quant_matmul"),("npu_quantize","impl.quantize","quantize"),
 ("npu_format_cast","impl.format_cast","format_cast"),
 ("npu_fused_infer_attention_score","impl.fused_infer_attention_score","fused_infer_attention_score"),
 ("npu_fusion_attention","impl.fusion_attention","fusion_attention"),
 ("npu_gather_pa_kv_cache","impl.gather_pa_kv_cache","gather_pa_kv_cache"),
 ("npu_kv_quant_sparse_flash_attention","impl.kv_quant_sparse_flash_attention","kv_quant_sparse_flash_attention"),
 ("npu_scatter_nd_update","impl.scatter_nd_update","scatter_nd_update"),
 ("npu_transpose_batchmatmul","impl.transpose_batchmatmul","transpose_batchmatmul"),
 ("npu_transpose_quant_batchmatmul","impl.transpose_quant_batchmatmul","transpose_quant_batchmatmul"),
]
# Approximate production shapes for Qwen3.6-35B-A3B:
#   hidden 2048, moe intermediate 1024, E=256, topk=8, T=1024
T=1024; H=2048; I=1024; E=256; K=8; DEV="npu:0"
def build(op):
    """Inputs per operator, following the native schema dumped from torch.ops.npu."""
    import torch
    bf = torch.bfloat16
    f32 = torch.float32
    dev = DEV
    if op == "npu_swiglu":
        return (torch.randn(T * K, 2 * I, dtype=bf, device=dev),), {}
    if op == "npu_clipped_swiglu":
        # native default interleaved=True; the port implements False, so pin it.
        return (torch.randn(T * K, 2 * I, dtype=bf, device=dev),), dict(interleaved=False)
    if op == "npu_gelu_mul":
        return (torch.randn(T * K, 2 * I, dtype=bf, device=dev),), dict(approximate="none")
    if op == "npu_grouped_matmul":
        # x: [M, K], weight: [K, N] per expert; group_list in counts mode.
        per = max(1, (T * K) // E)
        gl = torch.full((E,), per, dtype=torch.int64, device=dev)
        x = torch.randn(E * per, H, dtype=bf, device=dev)
        w = torch.randn(E, H, I, dtype=bf, device=dev)
        return ([x], [w]), dict(split_item=2, group_type=0, group_list_type=1, group_list=gl)
    if op == "npu_moe_token_unpermute":
        return (torch.randn(T * K, I, dtype=bf, device=dev),
                torch.randint(0, T * K, (T * K,), dtype=torch.int32, device=dev),
                torch.rand(T, K, dtype=f32, device=dev)), {}
    if op == "npu_moe_token_permute":
        return (torch.randn(T, I, dtype=bf, device=dev),
                torch.randint(0, E, (T, K), dtype=torch.int32, device=dev)), {}
    if op == "npu_moe_init_routing_v2":
        return (torch.randn(T, H, dtype=bf, device=dev),
                torch.randint(0, E, (T, K), dtype=torch.int32, device=dev)), dict(
                    active_num=T * K, expert_num=E, expert_tokens_num_type=1,
                    expert_tokens_num_flag=True, active_expert_range=[0, E], quant_mode=-1)
    if op == "npu_rms_norm":
        return (torch.randn(T, H, dtype=bf, device=dev),
                torch.randn(H, dtype=bf, device=dev), 1e-6), {}
    if op == "npu_kv_rmsnorm_rope_cache":
        # (kv, gamma, cos, sin, index, k_cache, ckv_cache) per the native schema
        head, rope, blk, nblk = 8, 64, 128, 64
        return (torch.randn(T, head * rope, dtype=bf, device=dev),
                torch.randn(head * rope, dtype=bf, device=dev),
                torch.randn(T, rope, dtype=bf, device=dev),
                torch.randn(T, rope, dtype=bf, device=dev),
                torch.arange(T, dtype=torch.int32, device=dev),
                torch.zeros(nblk, blk, head, rope, dtype=bf, device=dev),
                torch.zeros(nblk, blk, head, rope, dtype=bf, device=dev)), dict(
                    cache_mode="PA")
    if op == "npu_dynamic_quant":
        return (torch.randn(T, H, dtype=bf, device=dev),), dict(dst_type=torch.int8)
    if op == "npu_dynamic_mx_quant":
        return (torch.randn(T, H, dtype=bf, device=dev),), dict(dst_type=torch.float8_e4m3fn)
    if op == "npu_dynamic_block_quant":
        return (torch.randn(T, H, dtype=bf, device=dev),), {}
    if op == "npu_quant_matmul":
        return (torch.randint(-8, 8, (T, H), dtype=torch.int8, device=dev),
                torch.randint(-8, 8, (H, I), dtype=torch.int8, device=dev),
                torch.rand(1, dtype=f32, device=dev) + 0.5), {}
    if op == "npu_quantize":
        return (torch.randn(T, H, dtype=bf, device=dev),
                torch.rand(H, dtype=f32, device=dev) + 0.5,
                torch.zeros(H, dtype=torch.int32, device=dev),
                1), dict(axis=1)
    if op == "npu_format_cast":
        return (torch.randn(T, H, dtype=bf, device=dev), 2), {}
    if op == "npu_fused_infer_attention_score":
        return (torch.randn(T, 8, 128, dtype=bf, device=dev),
                torch.randn(T, 8, 128, dtype=bf, device=dev),
                torch.randn(T, 8, 128, dtype=bf, device=dev)), dict(num_heads=8, input_layout="BSH")
    if op == "npu_fusion_attention":
        return (torch.randn(T, 8, 128, dtype=bf, device=dev),
                torch.randn(T, 8, 128, dtype=bf, device=dev),
                torch.randn(T, 8, 128, dtype=bf, device=dev), 8, "BSH"), {}
    if op == "npu_gather_pa_kv_cache":
        nblk, blk, head, dim, nb, nseq = 64, 128, 8, 128, 8, 16
        return (torch.randn(nblk, blk, head, dim, dtype=bf, device=dev),
                torch.randn(nblk, blk, head, dim, dtype=bf, device=dev),
                torch.randint(0, nblk, (nseq, 8), dtype=torch.int32, device=dev),
                torch.randint(1, 8, (nseq,), dtype=torch.int32, device=dev),
                torch.zeros(nseq, 8, head, dim, dtype=bf, device=dev),
                torch.zeros(nseq, 8, head, dim, dtype=bf, device=dev)), {}
    if op == "npu_kv_quant_sparse_flash_attention":
        return None, None
    if op == "npu_scatter_nd_update":
        return (torch.randn(4096, 128, dtype=bf, device=dev),
                torch.randint(0, 4096, (T, 1), dtype=torch.int64, device=dev),
                torch.randn(T, 128, dtype=bf, device=dev)), {}
    if op in ("npu_transpose_batchmatmul", "npu_transpose_quant_batchmatmul"):
        return (torch.randn(8, 128, 256, dtype=bf, device=dev),
                torch.randn(8, 256, 128, dtype=bf, device=dev)), {}
    return None, None


def bench(fn, warm, iters):
    for _ in range(warm): fn()
    torch.npu.synchronize(); s=[]
    for _ in range(iters):
        torch.npu.synchronize(); t0=time.perf_counter(); fn(); torch.npu.synchronize()
        s.append((time.perf_counter()-t0)*1e6)
    return statistics.median(s)

ap=argparse.ArgumentParser(); ap.add_argument("--only",default=""); ap.add_argument("--warmup",type=int,default=5)
ap.add_argument("--iters",type=int,default=20); ap.add_argument("--out",default="")
a=ap.parse_args()
only=set(x.strip() for x in a.only.split(",") if x.strip())
rows=[]
for attr,mod,fn in TARGETS:
    if only and attr not in only: continue
    native=getattr(torch_npu,attr,None)
    if native is None: print("%-40s SKIP no-native"%attr); continue
    try: impl=getattr(importlib.import_module("."+mod,package=PKG),fn)
    except Exception as e: print("%-40s SKIP import:%r"%(attr,e)); continue
    args,kw=build(attr)
    if args is None: print("%-40s SKIP no-shape"%attr); continue
    try:
        with torch.no_grad(): o_n=native(*args,**kw)
        with torch.no_grad(): o_t=impl(*args,**kw)
    except Exception as e:
        print("%-40s ERROR %r"%(attr,e)); rows.append({"op":attr,"error":repr(e)[:160]}); continue
    md=None
    try:
        x=o_n[0] if isinstance(o_n,(list,tuple)) else o_n
        y=o_t[0] if isinstance(o_t,(list,tuple)) else o_t
        if hasattr(x,"shape") and hasattr(y,"shape") and x.shape==y.shape:
            md=float((x.float()-y.float()).abs().max().item())
    except Exception: pass
    with torch.no_grad(): nu=bench(lambda: native(*args,**kw), a.warmup, a.iters)
    with torch.no_grad(): tu=bench(lambda: impl(*args,**kw), a.warmup, a.iters)
    r=tu/nu if nu else 0.0
    v="FASTER" if r<0.8 else ("SLOWER" if r>1.2 else "even")
    print("%-40s native=%9.1fus triton=%9.1fus ratio=%.3f %s maxdiff=%s"%(attr,nu,tu,r,v,md))
    rows.append({"op":attr,"native_us":nu,"triton_us":tu,"ratio":r,"maxdiff":md})
print()
print("%-40s %12s %12s %8s"%("op","native_us","triton_us","ratio"))
print("-"*76)
for r in sorted([x for x in rows if "ratio" in x],key=lambda z:-z["ratio"]):
    print("%-40s %12.1f %12.1f %8.3f"%(r["op"],r["native_us"],r["triton_us"],r["ratio"]))
if a.out:
    json.dump(rows,open(a.out,"w"),indent=2); print("\nwrote "+a.out)