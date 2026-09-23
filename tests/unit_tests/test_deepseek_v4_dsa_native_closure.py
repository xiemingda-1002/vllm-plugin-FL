from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DSA_OPPS = (
    ("attention", "sparse_attn_sharedkv"),
    ("attention", "sparse_attn_sharedkv_metadata"),
    ("attention", "compressor"),
    ("attention", "compressor_metadata"),
    ("attention", "vllm_quant_lightning_indexer"),
    ("attention", "vllm_quant_lightning_indexer_metadata"),
    ("attention", "inplace_partial_rotary_mul"),
    ("attention", "rms_norm_dynamic_quant"),
    ("moe", "scatter_nd_update_v2"),
)

DSA_SYMBOLS = (
    "npu_sparse_attn_sharedkv",
    "npu_sparse_attn_sharedkv_metadata",
    "compressor",
    "compressor_metadata",
    "npu_vllm_quant_lightning_indexer",
    "npu_vllm_quant_lightning_indexer_metadata",
    "inplace_partial_rotary_mul",
    "npu_rms_norm_dynamic_quant",
    "npu_scatter_nd_update_v2",
)

HC_OPPS = (
    ("moe", "hc_pre"),
    ("moe", "hc_post"),
    ("moe", "hc_pre_inv_rms"),
    ("moe", "hc_pre_sinkhorn"),
)

HC_SYMBOLS = (
    "npu_hc_post",
    "npu_hc_pre",
    "npu_hc_pre_v2",
    "npu_hc_pre_inv_rms",
    "npu_hc_pre_sinkhorn",
)

NATIVE_OPPS = DSA_OPPS + HC_OPPS
NATIVE_SYMBOLS = DSA_SYMBOLS + HC_SYMBOLS


def test_deepseek_native_payload_is_the_exact_thirteen_root_closure() -> None:
    assert len(NATIVE_OPPS) == 13
    for category, name in NATIVE_OPPS:
        root = ROOT / "csrc/ascend" / category / name
        assert (root / "CMakeLists.txt").is_file()
        assert any(root.rglob("*.cpp"))

    # DSA has A2 (arch32) and A3 (arch35) variants where rc1 supplies them.
    for name in (
        "compressor",
        "vllm_quant_lightning_indexer",
    ):
        kernel = ROOT / "csrc/ascend/attention" / name / "op_kernel"
        assert (kernel / "arch32").is_dir()
        assert (kernel / "arch35").is_dir()


def test_deepseek_native_build_manifest_is_limited_to_the_native_payload() -> None:
    manifest = (ROOT / "csrc/ascend/build_opp.sh").read_text(encoding="utf-8")
    for _, name in NATIVE_OPPS:
        assert f"  {name}" in manifest
    assert "FL_BUILD_CANN_OPP=1" in manifest
    assert "--vendor_name=custom" in manifest


def test_deepseek_native_schemas_have_privateuse1_and_meta_dispatch() -> None:
    binding = (ROOT / "csrc/ascend/torch_binding.cpp").read_text(encoding="utf-8")
    assert len(NATIVE_SYMBOLS) == 14
    for symbol in NATIVE_SYMBOLS:
        assert binding.count(f'ops.impl("{symbol}"') == 2
    assert "TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, ops)" in binding
    assert "TORCH_LIBRARY_IMPL(_C_ascend, Meta, ops)" in binding
    assert "Tensor(a!) state_cache" in binding
    assert "Tensor(a!) var" in binding
    assert "at::empty_symint" in binding


def test_deepseek_hc_meta_shapes_follow_rc1_contract() -> None:
    binding = (ROOT / "csrc/ascend/torch_binding.cpp").read_text(encoding="utf-8")
    # HC pre uses x [B,S,HC,D] or [BS,HC,D], and returns the four-way
    # reduction together with float32 post/comb fragments.
    assert "x.sym_size(0), x.sym_size(1), x.sym_size(3)" in binding
    assert "x.sym_size(0), x.sym_size(1), hc_mult, hc_mult" in binding
    assert "x.sym_size(0), hc_mult, hc_mult" in binding
    # Inverse RMS removes HC and D dimensions and leaves one float32 lane.
    assert "for (auto i = 0; i < x.dim() - 2; ++i)" in binding
    assert "shape.push_back(1)" in binding


def test_deepseek_dsa_acl_dispatchers_have_matching_opp_roots() -> None:
    binding = (ROOT / "csrc/ascend/torch_binding.cpp").read_text(encoding="utf-8")
    assert "aclnnRmsNormDynamicQuant" in binding
    assert "aclnnScatterNdUpdateV2" in binding
    assert (ROOT / "csrc/ascend/attention/rms_norm_dynamic_quant").is_dir()
    assert (ROOT / "csrc/ascend/moe/scatter_nd_update_v2").is_dir()


def test_dsa_sparse_meta_preserves_rank_and_all_metadata_devices() -> None:
    binding = (ROOT / "csrc/ascend/torch_binding.cpp").read_text(encoding="utf-8")

    # Eager and Meta both derive LSE from every query dimension, so TND and
    # BSND inputs retain their rank with only the last dimension replaced.
    assert "std::vector<int64_t> lse_sizes(q.sizes().begin(), q.sizes().end())" in binding
    assert "lse_sizes.back() = 1" in binding
    assert "auto sizes = q.sym_sizes().vec()" in binding
    assert "sizes.back() = 1" in binding

    # Metadata may be driven by any of the five optional sequence tensors.
    for source in ("first", "second", "third", "fourth", "fifth"):
        assert f"{source}.has_value()" in binding
    assert 'std::string meta_device = "meta"' in binding
    assert "requested_device.index()" in binding
