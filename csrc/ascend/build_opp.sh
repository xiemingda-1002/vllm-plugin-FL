#!/usr/bin/env bash
# Build the current vLLM-Ascend 0.24.0rc1 native closure (Qwen plus
# DeepSeek V4 DSA and GLM SFA). Catlass remains an external build-only dependency for
# the pre-existing Qwen operators; no Catlass source is vendored into FL.

set -euo pipefail

repo_dir=${1:?repository root is required}
soc_version=${2:?SOC_VERSION is required}
soc_family=${3:?SOC family is required}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

case "${soc_family}" in
  ascend910b|ascend910_93) ;;
  *) echo "Unsupported Qwen GDN SOC family: ${soc_family}" >&2; exit 2 ;;
esac

# CATLASS is a build-only dependency. Keep it outside the tracked FL source
# tree and pin the revision used by vLLM-Ascend 0.24.0rc1. Offline builds can
# provide an existing checkout through CATLASS_PATH.
catlass_root=${CATLASS_PATH:-${repo_dir}/.deps/catlass}
catlass_include=${catlass_root}/include
catlass_commit=41bf90da655bba3c66d0acd7e00abe33960ecfd6
if [[ ! -d "${catlass_include}/catlass" ]]; then
  mkdir -p "$(dirname "${catlass_root}")"
  if [[ ! -d "${catlass_root}/.git" ]]; then
    git clone --filter=blob:none https://gitcode.com/cann/catlass.git \
      "${catlass_root}"
  fi
  git -C "${catlass_root}" fetch origin "${catlass_commit}"
  git -C "${catlass_root}" checkout --detach "${catlass_commit}"
fi
export CPATH="${catlass_include}${CPATH:+:${CPATH}}"
export FL_BUILD_CANN_OPP=1
ops=(
  causal_conv1d
  recurrent_gated_delta_rule
  fused_gdn_gating
  chunk_gated_delta_rule_fwd_h
  chunk_fwd_o
  moe_gating_top_k
  moe_gating_top_k_hash
  dequant_swiglu_quant
  moe_init_routing_custom
  scatter_nd_update_v2
  hc_pre
  hc_post
  hc_pre_inv_rms
  hc_pre_sinkhorn
  sparse_attn_sharedkv
  sparse_attn_sharedkv_metadata
  sparse_flash_attention
  compressor
  compressor_metadata
  vllm_quant_lightning_indexer
  vllm_quant_lightning_indexer_metadata
  inplace_partial_rotary_mul
  rms_norm_dynamic_quant
)
if [[ "${soc_family}" == "ascend910_93" ]]; then
  # These are the A3 fused-MC2 payloads. Keep A2's selected OPP closure
  # unchanged: DispatchFFNCombineBF16 relies on the A3 CATLASS_ARCH=2201 ABI.
  ops+=(dispatch_ffn_combine dispatch_ffn_combine_bf16)
fi
ops_arg=$(IFS=';'; echo "${ops[*]}")

cd "${script_dir}"
rm -rf -- output build_out
bash build.sh --pkg --ops="${ops_arg}" --soc="${soc_family}" \
  --vendor_name=custom

shopt -s nullglob
installers=(build/cann-ops-transformer*.run)
shopt -u nullglob
if [[ ${#installers[@]} -ne 1 ]]; then
  echo "Expected exactly one OPP installer, got ${#installers[@]}" >&2
  exit 3
fi

install_root="${repo_dir}/vllm_fl/_cann_ops_custom"
mkdir -p "${install_root}"
find "${install_root}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
chmod +x "${installers[0]}"
"${installers[0]}" --install-path="${install_root}"
printf '%s\n' "${soc_version}" > "${install_root}/FL_SOC_VERSION"
