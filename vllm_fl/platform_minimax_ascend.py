# Copyright (c) 2026 BAAI. All rights reserved.

"""MiniMax BF16 boundary around the complete, versioned Ascend platform.

Inherit the vendor worker/runner, cache registry, attention, compiler, graph,
communicator and forward-context hooks together. No FL runner or numerical
fallback is combined with the vendor's MiniMax cache ABI.
"""

import inspect

from vllm_fl.models.minimax_m3_ascend_runtime import (
    ENV_NAME,
    check_dependencies,
    register_general_plugins,
    validate_config,
)

check_dependencies()

from vllm_ascend.platform import NPUPlatform
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

# Validate the public lifecycle ABI before this platform can be selected. The
# actual vendor implementation remains responsible for binary/schema loading
# and the real device check in NPUWorker, before cache allocation/execution.
for _name, _args in (
    ("pre_register_and_update", ()),
    ("check_and_update_config", (None,)),
    ("register_custom_kv_cache_specs", (None,)),
    ("get_compile_backend", ()),
    ("get_pass_manager_cls", ()),
    ("get_device_communicator_cls", ()),
    ("get_static_graph_wrapper_cls", ()),
    ("get_attn_backend_cls", (None, None)),
    ("set_additional_forward_context", (None, None, None)),
    ("opaque_attention_op", ()),
    ("support_hybrid_kv_cache", ()),
    ("support_static_graph_mode", ()),
):
    _hook = getattr(NPUPlatform, _name, None)
    if not callable(_hook):
        raise RuntimeError(f"vllm-ascend is missing platform hook {_name}")
    inspect.signature(_hook).bind(*_args)
del _name, _args, _hook


class MiniMaxAscendPlatform(NPUPlatform):
    """Use the vendor runtime unchanged after validating the MiniMax contract."""

    additional_env_vars = list(
        dict.fromkeys(
            [*getattr(NPUPlatform, "additional_env_vars", ()), ENV_NAME, "VLLM_PLUGINS"]
        )
    )

    @classmethod
    def check_and_update_config(cls, vllm_config) -> None:
        validate_config(vllm_config)
        check_dependencies()
        # This is the installed native package's target, not a hardware probe.
        # The inherited NPUWorker also validates the actual SoC at startup.
        if get_ascend_device_type() != AscendDeviceType.A3:
            raise RuntimeError(
                "MiniMax BF16 runtime requires an Ascend A3 native build"
            )
        register_general_plugins()
        super().check_and_update_config(vllm_config)
