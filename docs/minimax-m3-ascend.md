# MiniMax M3 BF16 on Ascend A3

This provider selects the complete upstream Ascend platform and runtime for a
dedicated MiniMax BF16 process. It does not combine the FL model runner with
Ascend-specific cache metadata or copy numerical kernels into the plugin.
The normal FL platform remains the default when the provider is not selected.

## Scope and validation status

The intended model architectures are `MiniMaxM3SparseForCausalLM` and
`MiniMaxM3SparseForConditionalGeneration`, using BF16 model weights and `auto`
KV-cache dtype on Ascend A3. A3 denotes the target device family, not a promise
that every configuration with the same device name has been validated.
Quantized checkpoints and quantized KV caches are outside this provider's scope.
Other model architectures must use the normal FL runtime in a separate process.

The integration inherits the vendor worker, runner, attention backends, compiler,
graph wrappers, collective communicator and custom cache registry together.
The vendor `NPUModelRunner` owns the MiniMax sparse index-cache allocation and
binding; merely selecting its model class while retaining `WorkerFL` is not an
equivalent configuration.

No target-machine execution, native build, checkpoint loading, numerical
comparison, real graph capture/replay or performance validation was performed
for this change. Source contracts and a fixed dependency are not a hardware
support or accuracy result. The upstream repository includes an A3 BF16
MiniMax configuration, but those upstream results are not results for this FL
integration.

## Fixed dependency and build environment

The optional provider uses these public upstream revisions:

| Component | Revision or ABI requirement | Upstream evidence |
| --- | --- | --- |
| vLLM | `ee0da84ab9e04ac7610e28580af62c365e898389` (`v0.24.0`) | [vendor Dockerfile](https://github.com/vllm-project/vllm-ascend/blob/412cda26814ff70c326f6eb6510f1b610f67bbc0/Dockerfile) |
| vLLM Ascend | `412cda26814ff70c326f6eb6510f1b610f67bbc0` (`v0.24.0rc1`) | [fixed source](https://github.com/vllm-project/vllm-ascend/tree/412cda26814ff70c326f6eb6510f1b610f67bbc0) |
| Torch / torch-npu | `2.10.0` / `2.10.0.post2` | [vendor requirements](https://github.com/vllm-project/vllm-ascend/blob/412cda26814ff70c326f6eb6510f1b610f67bbc0/requirements.txt) |
| torchvision / torchaudio | `0.25.0` / `2.10.0` | vendor requirements |
| Triton Ascend / Transformers | `3.2.1` / `5.13.0` | vendor requirements |
| CANN toolchain reference | `9.0.1`, Ubuntu 22.04, Python 3.12 | vendor Dockerfile base image |

The upstream Dockerfile's default SOC is A2; use the A3 SOC that matches the
deployment. The upstream setup script explicitly lists `ascend910_9391` as an
A3 example. `get_ascend_device_type()` reports build metadata, whereas the
vendor's `check_ascend_device_type()` additionally compares it with the physical
device. A correctly named wheel or successful import does not establish a
matching device or native ABI.

[requirements/minimax-ascend.txt](../requirements/minimax-ascend.txt) pins the
provider's source and core Python ABI. It is not a complete transitive lockfile.
Follow the fixed upstream dependency and native-build recipe for the remaining
build requirements, ATB, driver and CANN installation. Do not reuse a CUDA
wheel or a native payload built for a different SOC. The existing generic FL
Ascend CI image is not evidence that this new fixed-v0.24 provider has run.

For a future installation in an isolated matching environment, the source-build
inputs are:

```bash
# Source the matching CANN and ATB environment scripts first.
export SOC_VERSION=ascend910_9391
export VLLM_TARGET_DEVICE=empty
export COMPILE_CUSTOM_KERNELS=1
python -m pip install -r requirements/minimax-ascend.txt
# Match the vendor Dockerfile's final Triton namespace installation order.
python -m pip uninstall -y triton triton-ascend
python -m pip install triton-ascend==3.2.1 \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
python -m pip install --no-build-isolation --no-deps -e .
```

Use the Ascend package index and full prerequisite installation described by
the fixed upstream Dockerfile before these commands. The `empty` vLLM build
target follows that recipe and avoids selecting a CUDA build for the NPU
runtime. The vendor package must build/install its native extensions; disabling
custom kernels is not a supported substitute. These commands are installation
instructions and were not executed for this change.

The Triton cleanup and final reinstall are required by that upstream recipe:
vLLM can introduce generic CUDA `triton`, whose Python namespace conflicts with
`triton-ascend`. Listing both source projects and the Ascend version in one
requirements file does not enforce the correct final namespace contents.

## Select the dedicated provider

Set the provider before vLLM platform initialization and use a fresh process:

```bash
export VLLM_PLUGINS=fl
export VLLM_FL_MINIMAX_ASCEND_RUNTIME=1
export MODEL_PATH=/path/to/MiniMax-M3
vllm serve "$MODEL_PATH" \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --data-parallel-size 2 \
  --enable-expert-parallel \
  --enforce-eager
```

The command uses vLLM's existing serving entry point and a user-supplied local
checkpoint, not a different checkpoint format. Tensor-parallel size must fit the
model, installed devices and vendor runtime. The eager example does not certify
graph mode; graph and distributed settings require their own target-environment
validation. TP8 with DP2 follows the upstream BF16 A3 parallel configuration and
requires 16 visible logical NPU devices; it is not a single-device example.
The upstream BF16 A3 configuration is available
[here](https://github.com/vllm-project/vllm-ascend/blob/412cda26814ff70c326f6eb6510f1b610f67bbc0/tests/e2e/nightly/single_node/models/configs/MiniMax-M3-BF16-A3.yaml).

The opt-in flag is needed because platform selection precedes model
configuration. It selects an entire process-level provider, not a per-token
fallback. It must propagate to spawned workers through the platform's
`additional_env_vars`; it is not enough to set it only after the engine starts.
The provider invokes the vendor's four real general-plugin entry points
(`register_connector`, `register_model_loader`, `register_service_profiling`,
`register_model`). Their own initialization installs the vendor global patches,
model registration and loader/connector integration in each relevant process.
Default FL operator patches are not layered over that same provider process.

Do not switch providers in an already initialized process. Upstream Ascend
initialization intentionally installs process-wide registrations and patches;
use a fresh process after an initialization failure. Missing dependencies,
unsupported model/dtype or incompatible native capabilities must fail before
execution. An execution error is not permission to retry with another backend.
