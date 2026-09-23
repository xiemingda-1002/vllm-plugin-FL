"""Keep the migrated DSA/MoE call sites covered by FL-owned schemas."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
ASCEND = ROOT / "vllm_fl/dispatch/backends/vendor/ascend"
ATTENTION = ROOT / "vllm_fl/attention/ascend"

# These references are in code shared with other quantization/topology paths,
# not the selected A3 W8A8 fused-MC2 payload. Do not silently treat them as
# implemented native capabilities when extending that acceptance scope.
CONDITIONAL_NATIVE_PATHS = {
    "grouped_matmul_swiglu_quant_v2": "W4A8 fused MLP, not W8A8",
    "grouped_matmul_swiglu_quant_weight_nz_tensor_list": "dynamic EPLB fusion, disabled",
    "npu_swiglu_group_quant": "MXFP shared experts, not current W8A8 checkpoint",
}


def test_dsa_and_moe_native_calls_have_schemas_or_explicit_conditional_scope():
    sources = [
        ROOT / "vllm_fl/models/deepseek_v4_ascend.py",
        ASCEND / "ops/dsa.py",
        ATTENTION / "dsa_v1.py",
        ASCEND / "impl/device_operator.py",
        *sorted((ASCEND / "impl/moe").glob("*.py")),
        *sorted((ASCEND / "impl/quantization").glob("*.py")),
    ]
    calls = set()
    for source in sources:
        calls.update(re.findall(r"torch\.ops\._C_ascend\.(\w+)", source.read_text()))
    binding = (ROOT / "csrc/ascend/torch_binding.cpp").read_text()
    schemas = set(re.findall(r'ops\.def\(\s*"(\w+)\(', binding))
    assert calls - schemas == set(CONDITIONAL_NATIVE_PATHS)
    assert {"moe_gating_top_k_hash", "npu_dequant_swiglu_quant"} <= schemas
    for name in calls - set(CONDITIONAL_NATIVE_PATHS):
        assert binding.count(f'ops.impl("{name}"') == 2, name
