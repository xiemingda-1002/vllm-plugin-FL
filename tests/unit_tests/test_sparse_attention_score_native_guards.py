from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPARSE_ROOT = _REPO_ROOT / "csrc" / "ascend" / "attention" / "sparse_attention_score"


def test_torch_adapter_bounds_and_shape_contracts_are_fail_closed():
    source = (_SPARSE_ROOT / "sparse_attention_score_torch_adpt.h").read_text()

    assert "topK >= MIN_TOP_K && topK <= MAX_TOP_K" in source
    assert "selectIdx.size(DIM_D) == topK" in source
    assert "snIdx.size(DIM_T) == selectIdx.size(DIM_T)" in source
    assert "snIdx.size(DIM_N) == selectIdx.size(DIM_N)" in source
    assert "MAX_TOP_K = 16" in source
    assert "select_num_idx must be provided for float8_e4m3fn input." in source


def test_host_tiler_checks_topk_before_fixed_size_kernel_dispatch():
    source = (_SPARSE_ROOT / "op_host" / "sparse_attention_score_tiling.cpp").read_text()

    assert "*topKPtr < MIN_TOP_K || *topKPtr > MAX_TOP_K" in source
    assert "selectIdxTopK < MIN_TOP_K || selectIdxTopK > MAX_TOP_K" in source
    assert "static_cast<int64_t>(topK_) != selectIdxTopK" in source
    assert "selectNumIdxShape->GetStorageShape().GetDimNum() != SELECT_NUM_IDX_DIM_NUM" in source
    assert "SelectNumIdx dimensions must equal SelectIdx dimensions 0 and 1." in source
    assert "dataType_ == ge::DT_FLOAT8_E4M3FN && selectNumIdxShape == nullptr" in source
