from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
PAYLOAD_ROOT = REPO_ROOT / "csrc/ascend/moe/moe_gating_top_k"
PROVIDER_SOURCE = (
    REPO_ROOT
    / "vllm_fl/dispatch/backends/vendor/ascend/impl/fused_moe.py"
)


# SHA256 after normalizing current-rc1's CRLF files to LF and removing only
# trailing newlines.  The adapter header is compared after restoring the
# authority namespace; that is the single intentional payload edit.
_CURRENT_RC1_NORMALIZED_SHA256 = {
    "CMakeLists.txt": "716dd07dd44379e61a2c5b2cf77b7d131fb7596847706b04d70c17c18c72b684",
    "moe_gating_top_k_torch_adpt.h": "b475c974a0d5bf4e8bb5e607a2c717bd2d59fbf5e248cabbb873e2529560aec0",
    "op_host/CMakeLists.txt": "13f2bb3fda8658dc330d65656421a93c1a9e6646f5549b0b719b8f9a4f2267cb",
    "op_host/math_util.h": "2f9c3a84572fc81b8ab447395d2cd9dc1b514116522176274207e68e74df7d7e",
    "op_host/moe_gating_top_k_def.cpp": "02b94a68c4c18d8cb3cb13fc38eecdbfade4aa57d0f80692b438c254a3d14e3d",
    "op_host/moe_gating_top_k_infershape.cpp": "eaa42620f1b8bf1edc35aeacee3cd1db686e30b88bd97538b3a576473be62c82",
    "op_host/moe_gating_top_k_proto.cpp": "2660a8667d1b06be0705d986b41cc372be6002895734399ea9df2e0618613a1d",
    "op_host/moe_gating_top_k_proto.h": "2d846e4b803825071d4df36fabc7acdf23a5634907ee7deab2be83bd572ad01a",
    "op_host/moe_gating_top_k_tiling.cpp": "5678df9b7dba2676a18abcd0da227b274b62b4d3c269af3c315cbe34f2462eee",
    "op_host/moe_gating_top_k_tiling.h": "c3d4cd150e348e9ed475e0580e8da40500c28fccc5d434fb1e4523e1c2add7e1",
    "op_host/moe_gating_top_k_tiling_arch35.cpp": "f952adb7bcbf310cad36015cb519aa4fd09d02710329efeb963910c05c6ffaea",
    "op_host/moe_gating_top_k_tiling_base.cpp": "8c66061a87a978f653170b565b6b3197df04cad911ff345a3e081c984436e6fa",
    "op_kernel/common.h": "8a671274e8be968f5f3107954a608b8211ee3738adae32af90a300c3fe649dc4",
    "op_kernel/error_log.h": "02369559e7bc46ee660a434242cffb9cd105f23f32b58ed74f1e1b5088575457",
    "op_kernel/moe_gating_top_k.cpp": "4a1fc50d0eb6dcc5e0facc17cdcba8665116f76598be6485c8b605ce803e7ac3",
    "op_kernel/moe_gating_top_k_apt.cpp": "b6052eaaf3907c1656aa515cd1f21e1b71be910f7c73c1c762fc7ecc3782b4b1",
    "op_kernel/moe_gating_top_k_e_k_fullload.h": "cdb9886c94e0c62cbdca9fcd8ecd0bc2af78c6244a3016b7c80e2b9cb34f6539",
    "op_kernel/moe_gating_top_k_generalized.h": "f0d91d606be0bd7efc2363531eab0fe8ab0d68a8125662667f21294482709368",
    "op_kernel/moe_gating_top_k_without_group.h": "0a3ba990a9503c48129a03b28d178f0040fb73b34eb37b7dfdda25726e5845d6",
}


def _text(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def test_complete_current_rc1_payload_is_present_with_one_namespace_edit() -> None:
    actual = {
        path.relative_to(PAYLOAD_ROOT).as_posix()
        for path in PAYLOAD_ROOT.rglob("*")
        if path.is_file()
    }
    assert actual == set(_CURRENT_RC1_NORMALIZED_SHA256)

    for relative, expected in _CURRENT_RC1_NORMALIZED_SHA256.items():
        data = (PAYLOAD_ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
        if relative == "moe_gating_top_k_torch_adpt.h":
            assert b"namespace vllm_ascend" not in data
            data = data.replace(
                b"namespace vllm_fl_native", b"namespace vllm_ascend"
            )
        digest = hashlib.sha256(data.rstrip(b"\n")).hexdigest()
        assert digest == expected, relative


def test_a3_uses_base_kernel_and_apt_override_is_a5_only() -> None:
    op_def = _text(
        "csrc/ascend/moe/moe_gating_top_k/op_host/"
        "moe_gating_top_k_def.cpp"
    )
    assert 'this->AICore().AddConfig("ascend910_93");' in op_def
    assert op_def.count('"moe_gating_top_k_apt"') == 1
    assert (
        '.ExtendCfgInfo("opFile.value", "moe_gating_top_k_apt")'
        in op_def
    )
    assert 'this->AICore().AddConfig("ascend950", regbaseCfg);' in op_def
    assert op_def.index('AddConfig("ascend910_93")') < op_def.index(
        '"moe_gating_top_k_apt"'
    )

    base_kernel = _text(
        "csrc/ascend/moe/moe_gating_top_k/op_kernel/"
        "moe_gating_top_k.cpp"
    )
    apt_kernel = _text(
        "csrc/ascend/moe/moe_gating_top_k/op_kernel/"
        "moe_gating_top_k_apt.cpp"
    )
    assert "arch35/moe_gating_top_k_regbase.h" not in base_kernel
    assert "arch35/moe_gating_top_k_regbase.h" in apt_kernel
    assert "moe_gating_top_k" in _text("csrc/ascend/build_opp.sh")


def test_schema_dispatch_and_provider_registration_are_closed() -> None:
    binding = _text("csrc/ascend/torch_binding.cpp")
    assert binding.count('ops.impl("moe_gating_top_k"') == 2
    assert "TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, ops)" in binding
    assert "TORCH_LIBRARY_IMPL(_C_ascend, Meta, ops)" in binding
    assert (
        "moe_gating_top_k(Tensor x, int k, int k_group, int group_count, "
        in binding
    )

    device_operator = _text(
        "vllm_fl/dispatch/backends/vendor/ascend/impl/device_operator.py"
    )
    assert "torch.ops._C_ascend.moe_gating_top_k" in device_operator
    imports = []
    for node in ast.walk(ast.parse(device_operator)):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any(
        name == "vllm_ascend" or name.startswith("vllm_ascend.")
        for name in imports
    )

    registration = _text(
        "vllm_fl/dispatch/backends/vendor/ascend/register_ops.py"
    )
    assert 'op_name="topk_softmax"' in registration
    assert 'impl_id="vendor.ascend"' in registration
    assert 'vendor="ascend"' in registration
    config = _text("vllm_fl/dispatch/config/ascend.yaml")
    assert "topk_softmax:\n    - vendor:ascend" in config


class _FakeTensor:
    def __init__(self, shape, dtype, *, device="npu:0", data=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.data = data
        self.storage_identity = object()
        self.copy_sources = []

    @property
    def ndim(self):
        return len(self.shape)

    def to(self, dtype):
        return _FakeTensor(
            self.shape, dtype, device=self.device, data=self.data
        )

    def copy_(self, source):
        self.copy_sources.append(source)
        self.data = source.data
        return self

    def reshape(self, rows, columns):
        assert rows * columns == len(self.data)
        values = [
            self.data[row * columns : (row + 1) * columns]
            for row in range(rows)
        ]
        return _FakeTensor(
            (rows, columns), self.dtype, device=self.device, data=values
        )

    @property
    def T(self):
        values = [list(column) for column in zip(*self.data)]
        return _FakeTensor(
            (self.shape[1], self.shape[0]),
            self.dtype,
            device=self.device,
            data=values,
        )


class _FakeTorch:
    Tensor = _FakeTensor
    float32 = "float32"
    int32 = "int32"
    bfloat16 = "bfloat16"

    @staticmethod
    def arange(size, *, device, dtype):
        return _FakeTensor(
            (size,), dtype, device=device, data=list(range(size))
        )


class _FakeDeviceOperator:
    call = None

    @classmethod
    def moe_gating_top_k(cls, gating_output, **kwargs):
        cls.call = (gating_output, kwargs)
        rows = gating_output.shape[0]
        top_k = kwargs["k"]
        weights = [
            [float(row * top_k + col) / 10 for col in range(top_k)]
            for row in range(rows)
        ]
        indices = [
            [row * top_k + col for col in range(top_k)]
            for row in range(rows)
        ]
        return (
            _FakeTensor(
                (rows, top_k),
                _FakeTorch.bfloat16,
                device=gating_output.device,
                data=weights,
            ),
            _FakeTensor(
                (rows, top_k),
                "int64",
                device=gating_output.device,
                data=indices,
            ),
            _FakeTensor(
                gating_output.shape,
                _FakeTorch.float32,
                device=gating_output.device,
            ),
        )


def _load_isolated_provider():
    tree = ast.parse(PROVIDER_SOURCE.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "topk_softmax_ascend"
    )
    namespace = {
        "torch": _FakeTorch,
        "DeviceOperator": _FakeDeviceOperator,
    }
    ast.fix_missing_locations(function)
    exec(
        compile(
            ast.Module([function], type_ignores=[]),
            str(PROVIDER_SOURCE),
            "exec",
        ),
        namespace,
    )
    return namespace["topk_softmax_ascend"]


def test_provider_mutates_all_preallocated_outputs_and_preserves_aliases() -> None:
    provider = _load_isolated_provider()
    gating = _FakeTensor((3, 256), _FakeTorch.bfloat16)
    weights = _FakeTensor((3, 2), _FakeTorch.float32)
    indices = _FakeTensor((3, 2), _FakeTorch.int32)
    token_indices = _FakeTensor((3, 2), _FakeTorch.int32)
    storage_ids = tuple(
        tensor.storage_identity for tensor in (weights, indices, token_indices)
    )

    returned_weights, returned_indices = provider(
        weights, indices, token_indices, gating, renormalize=True
    )

    assert returned_weights is weights
    assert returned_indices is indices
    assert tuple(
        tensor.storage_identity for tensor in (weights, indices, token_indices)
    ) == storage_ids
    assert weights.dtype == _FakeTorch.float32
    assert indices.dtype == _FakeTorch.int32
    assert weights.copy_sources[-1].dtype == _FakeTorch.float32
    assert indices.copy_sources[-1].dtype == _FakeTorch.int32
    assert token_indices.copy_sources[-1].dtype == _FakeTorch.int32
    assert weights.data == [[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]]
    assert indices.data == [[0, 1], [2, 3], [4, 5]]
    # CUDA's source_rows contract: [row, k_idx] = k_idx * M + row.
    assert token_indices.data == [[0, 3], [1, 4], [2, 5]]

    called_gating, kwargs = _FakeDeviceOperator.call
    assert called_gating is gating
    assert kwargs == {
        "k": 2,
        "k_group": 1,
        "group_count": 1,
        "group_select_mode": 1,
        "renorm": 1,
        "norm_type": 0,
        "out_flag": False,
        "routed_scaling_factor": 1.0,
        "eps": 1e-20,
        "bias_opt": None,
    }


@pytest.mark.parametrize(
    ("buffer", "dtype", "message"),
    [
        ("weights", _FakeTorch.bfloat16, "topk_weights"),
        ("indices", "int64", "topk_indices"),
        ("token_indices", "int64", "token_expert_indices"),
    ],
)
def test_provider_rejects_wrong_preallocated_dtype(
    buffer, dtype, message
) -> None:
    provider = _load_isolated_provider()
    tensors = {
        "weights": _FakeTensor((1, 8), _FakeTorch.float32),
        "indices": _FakeTensor((1, 8), _FakeTorch.int32),
        "token_indices": _FakeTensor((1, 8), _FakeTorch.int32),
    }
    tensors[buffer].dtype = dtype
    with pytest.raises(TypeError, match=message):
        provider(
            tensors["weights"],
            tensors["indices"],
            tensors["token_indices"],
            _FakeTensor((1, 256), _FakeTorch.bfloat16),
        )


def test_provider_rejects_topk_larger_than_expert_count() -> None:
    provider = _load_isolated_provider()
    with pytest.raises(ValueError, match="top-k width"):
        provider(
            _FakeTensor((1, 257), _FakeTorch.float32),
            _FakeTensor((1, 257), _FakeTorch.int32),
            _FakeTensor((1, 257), _FakeTorch.int32),
            _FakeTensor((1, 256), _FakeTorch.bfloat16),
        )
