from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_deepseek_moe_native_sources_and_build_manifest_close_rc1_paths() -> None:
    binding = (ROOT / "csrc/ascend/torch_binding.cpp").read_text(encoding="utf-8")
    manifest = (ROOT / "csrc/ascend/build_opp.sh").read_text(encoding="utf-8")

    for category, name, symbol in (
        ("moe", "moe_gating_top_k_hash", "moe_gating_top_k_hash"),
        ("moe", "dequant_swiglu_quant", "npu_dequant_swiglu_quant"),
    ):
        root = ROOT / "csrc/ascend" / category / name
        assert (root / "CMakeLists.txt").is_file()
        assert any(root.rglob("*.cpp"))
        assert f"  {name}" in manifest
        assert binding.count(f'ops.impl("{symbol}"') == 2

    assert "aclnnMoeGatingTopKHash" in binding
    assert "aclnnDequantSwigluQuantV2" in binding


def test_deepseek_moe_native_schemas_and_meta_shapes_match_rc1_contract() -> None:
    binding = (ROOT / "csrc/ascend/torch_binding.cpp").read_text(encoding="utf-8")

    assert "Tensor? tid2eid=None" in binding
    assert "bool out_flag=False" in binding
    assert "Tensor? group_index=None" in binding
    assert "int swiglu_mode=0" in binding
    assert "x.size(x.dim() - 1) % 2 == 0" in binding
    assert "x.sym_size(x.dim() - 1) / c10::SymInt(2)" in binding
