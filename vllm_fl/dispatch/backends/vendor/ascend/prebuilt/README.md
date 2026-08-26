# FL-built Ascend artifacts

This directory is an output location, not a copy of vLLM-Ascend. Build with
`VLLM_VENDOR=ascend` to generate the artifacts for the SoC family selected by
the build image's `SOC_VERSION`:

```text
<soc>/lib/_C_ascend*.so
<soc>/opp/vendors/custom_transformer/
```

The reduced `_C_ascend` library is compiled from `csrc/ascend/torch_binding.cpp`
and registers only the operators used by the Qwen3.6 path. The OPP package is
built from the five selected source directories under `csrc/ascend`.

| CANN family | Output directory | Accepted values |
| --- | --- | --- |
| Atlas A2 / `ascend910b` | `ascend910b1` | `ascend910b*`, `910b` |
| Atlas A3 / `ascend910_93` | `ascend910_93` | `ascend910_93*`, `910c` |

Artifacts must be built with the same Python, PyTorch/torch-npu and CANN stack
used at runtime. The recommended A2/A3 vLLM-Ascend images already provide the
matching `SOC_VERSION`; another build image must provide it explicitly. A3
never falls back to A2 artifacts.
