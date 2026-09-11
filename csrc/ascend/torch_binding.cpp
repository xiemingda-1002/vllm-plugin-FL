/*
 * FL-owned dispatcher bridge for the Qwen GDN native closure.
 * Function signatures and operator schemas track vLLM-Ascend 0.24.0rc1.
 */

#include <ATen/ATen.h>
#include <torch/library.h>

#include "aclnn_torch_adapter/op_api_common.h"
#include "attention/fused_gdn_gating/fused_gdn_gating_torch_adpt.h"
#include "attention/recurrent_gated_delta_rule/recurrent_gated_delta_rule_torch_adpt.h"
#include "moe/moe_gating_top_k/moe_gating_top_k_torch_adpt.h"
#include "moe/moe_init_routing_custom/moe_init_routing_custom_torch_adpt.h"

namespace vllm_fl_native {

at::Tensor npu_causal_conv1d_custom(
    const at::Tensor& output,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& conv_state,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& query_start_loc,
    const c10::optional<at::Tensor>& cache_indices,
    const c10::optional<at::Tensor>& initial_state_mode,
    const c10::optional<at::Tensor>& num_accepted_tokens,
    int64_t activation_mode,
    int64_t pad_slot_id,
    int64_t run_mode) {
  EXEC_NPU_CMD(aclnnCausalConv1d, x, weight, bias, conv_state,
               query_start_loc, cache_indices, initial_state_mode,
               num_accepted_tokens, activation_mode, pad_slot_id, run_mode,
               output);
  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
chunk_gated_delta_rule_fwd_h(
    const at::Tensor& k,
    const at::Tensor& w,
    const at::Tensor& u,
    const c10::optional<at::Tensor>& g,
    const c10::optional<at::Tensor>& gk,
    const c10::optional<at::Tensor>& initial_state,
    c10::optional<bool> output_final_state,
    c10::optional<int64_t> chunk_size,
    c10::optional<bool> save_new_value,
    c10::optional<at::IntArrayRef> cu_seqlens,
    c10::optional<at::IntArrayRef> chunk_indices,
    c10::optional<bool> use_exp2,
    c10::optional<bool> transpose_state_layout) {
  const bool output_final_state_value = output_final_state.value_or(false);
  const int64_t chunk_size_value = chunk_size.value_or(64);
  const at::Tensor& g_value = c10::value_or_else(g, [] { return at::Tensor(); });
  const at::Tensor& gk_value = c10::value_or_else(gk, [] { return at::Tensor(); });
  const at::Tensor& initial_state_value =
      c10::value_or_else(initial_state, [] { return at::Tensor(); });

  const int64_t B = k.size(0);
  const int64_t K = k.size(3);
  const int64_t T = k.size(2);
  const int64_t HV = u.size(1);
  const int64_t V = u.size(3);
  const int64_t NT = chunk_indices.has_value()
      ? static_cast<int64_t>(chunk_indices->size() / 2)
      : (T + chunk_size_value - 1) / chunk_size_value;

  at::Tensor h_out = at::zeros({B, HV, NT, K, V}, k.options());
  at::Tensor v_new_out = at::zeros(u.sizes(), u.options());
  at::Tensor final_state_out = output_final_state_value
      ? at::empty(
            {cu_seqlens.has_value()
                 ? static_cast<int64_t>(cu_seqlens->size() - 1)
                 : B,
             HV, K, V},
            initial_state.has_value() ? initial_state->options()
                                      : h_out.options())
      : at::empty({1}, k.options());

  bool save_new_value_value = save_new_value.value_or(true);
  bool use_exp2_value = use_exp2.value_or(false);
  bool transpose_state_layout_value = transpose_state_layout.value_or(false);
  EXEC_NPU_CMD(aclnnChunkGatedDeltaRuleFwdH, k, w, u, g_value, gk_value,
               initial_state_value, output_final_state_value,
               chunk_size_value, save_new_value_value, cu_seqlens,
               chunk_indices, use_exp2_value, transpose_state_layout_value,
               h_out, v_new_out, final_state_out);
  return {h_out, v_new_out,
          output_final_state_value ? final_state_out : at::Tensor()};
}

at::Tensor chunk_fwd_o(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& h,
    double scale,
    const c10::optional<at::Tensor>& g,
    const c10::optional<at::Tensor>& g_gamma,
    c10::optional<at::IntArrayRef> cu_seqlens,
    c10::optional<at::IntArrayRef> chunk_indices,
    c10::optional<int64_t> chunk_size,
    c10::optional<bool> transpose_state_layout) {
  at::Tensor output = at::zeros(v.sizes(), v.options());
  int64_t chunk_size_value = chunk_size.value_or(64);
  const at::Tensor& g_value = c10::value_or_else(g, [] { return at::Tensor(); });
  (void)g_gamma;
  (void)transpose_state_layout;
  EXEC_NPU_CMD(aclnnChunkFwdO, q, k, v, h, g_value, cu_seqlens,
               chunk_indices, scale, chunk_size_value, output);
  return output;
}

namespace meta {

at::Tensor npu_causal_conv1d_custom(
    const at::Tensor& output, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&,
    int64_t, int64_t, int64_t) {
  return output;
}

at::Tensor npu_recurrent_gated_delta_rule(
    const at::Tensor&, const at::Tensor&, const at::Tensor& value,
    at::Tensor&, const c10::optional<at::Tensor>&,
    const c10::optional<double>, const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&) {
  return at::empty_symint(
      value.sym_sizes(), value.options().dtype(at::ScalarType::BFloat16));
}

std::tuple<at::Tensor, at::Tensor> npu_fused_gdn_gating(
    const at::Tensor&, const at::Tensor& a, const at::Tensor& b,
    const at::Tensor&, double, double) {
  auto batch = a.sym_size(0);
  auto heads = a.sym_size(1);
  return {
      at::empty_symint({c10::SymInt(1), batch, heads},
                       a.options().dtype(c10::kFloat)),
      at::empty_symint({c10::SymInt(1), batch, heads}, b.options()),
  };
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
chunk_gated_delta_rule_fwd_h(
    const at::Tensor& k, const at::Tensor&, const at::Tensor& u,
    const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>& initial_state,
    c10::optional<bool> output_final_state,
    c10::optional<int64_t> chunk_size, c10::optional<bool>,
    c10::optional<at::IntArrayRef> cu_seqlens,
    c10::optional<at::IntArrayRef> chunk_indices, c10::optional<bool>,
    c10::optional<bool>) {
  auto B = k.sym_size(0);
  auto K = k.sym_size(3);
  auto T = k.sym_size(2);
  auto HV = u.sym_size(1);
  auto V = u.sym_size(3);
  const int64_t chunk = chunk_size.value_or(64);
  c10::SymInt NT = chunk_indices.has_value()
      ? c10::SymInt(chunk_indices->size() / 2)
      : (T + chunk - 1) / chunk;
  auto h = at::empty_symint({B, HV, NT, K, V}, k.options());
  auto v_new = at::empty_symint(u.sym_sizes(), u.options());
  if (!output_final_state.value_or(false)) {
    return {h, v_new, at::Tensor()};
  }
  c10::SymInt N = cu_seqlens.has_value()
      ? c10::SymInt(cu_seqlens->size() - 1)
      : B;
  auto options = initial_state.has_value() ? initial_state->options()
                                           : h.options();
  return {h, v_new, at::empty_symint({N, HV, K, V}, options)};
}

at::Tensor chunk_fwd_o(
    const at::Tensor&, const at::Tensor&, const at::Tensor& v,
    const at::Tensor&, double, const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&, c10::optional<at::IntArrayRef>,
    c10::optional<at::IntArrayRef>, c10::optional<int64_t>,
    c10::optional<bool>) {
  return at::empty_symint(v.sym_sizes(), v.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> moe_gating_top_k(
    const at::Tensor& x, int64_t k, int64_t, int64_t, int64_t, int64_t,
    int64_t, bool, double, double,
    const c10::optional<at::Tensor>& bias_opt) {
  TORCH_CHECK(x.dim() == 2, "The x should be 2D");
  TORCH_CHECK(
      x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat ||
          x.scalar_type() == at::kBFloat16,
      "float16, float32 or bfloat16 tensor expected but got a tensor with dtype: ",
      x.scalar_type());
  TORCH_CHECK(k > 0, "k must be greater than zero");

  if (bias_opt.has_value()) {
    const auto& bias = bias_opt.value();
    TORCH_CHECK(bias.scalar_type() == x.scalar_type(),
                "The dtype of x and bias should be same");
    TORCH_CHECK(bias.dim() == 1, "The bias should be 1D");
  }

  const auto rows = x.sym_size(0);
  const auto experts = x.sym_size(1);
  auto values = at::empty_symint({rows, k}, x.options());
  auto indices = at::empty_symint(
      {rows, k}, x.options().dtype(at::ScalarType::Int));
  auto auxiliary = at::empty_symint(
      {rows, experts}, x.options().dtype(at::ScalarType::Float));
  return {values, indices, auxiliary};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_moe_init_routing_custom(
    const at::Tensor& x, const at::Tensor& expert_idx,
    const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&,
    int64_t active_num, int64_t expert_capacity, int64_t expert_num,
    int64_t drop_pad_mode, int64_t expert_tokens_num_type, bool,
    int64_t quant_mode, at::IntArrayRef active_expert_range, int64_t) {
  constexpr int64_t kExpertRangeSize = 2;
  constexpr int64_t kUnquantized = -1;
  constexpr int64_t kCumsum = 0;
  constexpr int64_t kCount = 1;
  constexpr int64_t kKeyValue = 2;

  TORCH_CHECK(x.dim() == 2, "The x should be 2D");
  TORCH_CHECK(expert_idx.dim() == 2, "The expert_idx should be 2D");
  if (active_expert_range.empty()) {
    active_expert_range = at::IntArrayRef({0, expert_num});
  }
  TORCH_CHECK(active_expert_range.size() == kExpertRangeSize,
              "active_expert_range should contain [start, end]");

  const auto batch = x.sym_size(0);
  const auto hidden = x.sym_size(1);
  const auto top_k = expert_idx.sym_size(1);
  const auto active_experts =
      active_expert_range[1] - active_expert_range[0];
  c10::SymInt expanded_scale_length(0);
  at::Tensor expanded_x;

  if (drop_pad_mode == 1) {
    c10::SymDimVector shape = {c10::SymInt(expert_num),
                               c10::SymInt(expert_capacity), hidden};
    expanded_x = at::empty_symint(
        shape, quant_mode == kUnquantized
                   ? x.options()
                   : x.options().dtype(at::ScalarType::Char));
    expanded_scale_length = c10::SymInt(expert_num * expert_capacity);
  } else if (active_num > 0) {
    const c10::SymInt output_tokens =
        (batch * top_k).min(c10::SymInt(active_num));
    expanded_x = at::empty_symint(
        {output_tokens, hidden}, quant_mode == kUnquantized
                                     ? x.options()
                                     : x.options().dtype(at::ScalarType::Char));
    expanded_scale_length = output_tokens;
  } else {
    expanded_x = at::empty_symint(
        {batch * top_k, hidden}, quant_mode == kUnquantized
                                     ? x.options()
                                     : x.options().dtype(at::ScalarType::Char));
    expanded_scale_length = batch * top_k;
  }

  auto expanded_row_idx = at::empty_symint(
      {batch * top_k}, expert_idx.options());
  at::Tensor expert_tokens;
  if (expert_tokens_num_type >= kCumsum &&
      expert_tokens_num_type <= kCount) {
    expert_tokens = at::empty_symint(
        {c10::SymInt(active_experts)}, x.options().dtype(at::ScalarType::Long));
  } else if (expert_tokens_num_type == kKeyValue) {
    expert_tokens = at::empty_symint(
        {c10::SymInt(expert_num), c10::SymInt(2)},
        x.options().dtype(at::ScalarType::Long));
  }
  auto expanded_scale = at::empty_symint(
      {expanded_scale_length}, x.options().dtype(at::ScalarType::Float));
  return {expanded_x, expanded_row_idx, expert_tokens, expanded_scale};
}

}  // namespace meta
}  // namespace vllm_fl_native

TORCH_LIBRARY(_C_ascend, ops) {
  ops.def("npu_causal_conv1d_custom(Tensor output, Tensor x, Tensor weight, "
          "Tensor conv_state, Tensor? bias_opt, Tensor? query_start_loc_opt, "
          "Tensor? cache_indices_opt, Tensor? initial_state_mode_opt, "
          "Tensor? num_accepted_tokens_opt, int activation_mode, "
          "int pad_slot_id, int run_mode) -> Tensor");
  ops.def("npu_recurrent_gated_delta_rule(Tensor query, Tensor key, "
          "Tensor value, Tensor(a!) state, *, Tensor? beta=None, "
          "float? scale=None, Tensor? actual_seq_lengths=None, "
          "Tensor? ssm_state_indices=None, Tensor? num_accepted_tokens=None, "
          "Tensor? g=None, Tensor? gk=None) -> Tensor");
  ops.def("npu_fused_gdn_gating(Tensor A_log, Tensor a, Tensor b, "
          "Tensor dt_bias, float beta=1.0, float threshold=20.0) "
          "-> (Tensor g, Tensor beta_output)");
  ops.def("chunk_gated_delta_rule_fwd_h(Tensor k, Tensor w, Tensor u, "
          "Tensor? g=None, *, Tensor? gk=None, Tensor? initial_state=None, "
          "bool? output_final_state=False, int? chunk_size=None, "
          "bool? save_new_value=True, int[]? cu_seqlens=None, "
          "int[]? chunk_indices=None, bool? use_exp2=False, "
          "bool? transpose_state_layout=False) "
          "-> (Tensor h_out, Tensor v_new_out, Tensor final_state_out)");
  ops.def("chunk_fwd_o(Tensor q, Tensor k, Tensor v, Tensor h, float scale, "
          "*, Tensor? g=None, Tensor? g_gamma=None, int[]? cu_seqlens=None, "
          "int[]? chunk_indices=None, int? chunk_size=None, "
          "bool? transpose_state_layout=False) -> Tensor");
  ops.def("moe_gating_top_k(Tensor x, int k, int k_group, int group_count, "
          "int group_select_mode, int renorm, int norm_type, bool out_flag, "
          "float routed_scaling_factor, float eps, Tensor? bias_opt=None) "
          "-> (Tensor y, Tensor expert_idx, Tensor out)");
  ops.def(
      "npu_moe_init_routing_custom(Tensor x, Tensor expert_idx, *, "
      "Tensor? scale=None, Tensor? offset=None, int active_num=-1, "
      "int expert_capacity=-1, int expert_num=-1, int drop_pad_mode=0, "
      "int expert_tokens_num_type=0, bool expert_tokens_num_flag=False, "
      "int quant_mode=0, int[2] active_expert_range=[], int row_idx_type=0) "
      "-> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, ops) {
  ops.impl("npu_causal_conv1d_custom",
           &vllm_fl_native::npu_causal_conv1d_custom);
  ops.impl("npu_recurrent_gated_delta_rule",
           &vllm_ascend::npu_recurrent_gated_delta_rule);
  ops.impl("npu_fused_gdn_gating", &vllm_ascend::npu_fused_gdn_gating);
  ops.impl("chunk_gated_delta_rule_fwd_h",
           &vllm_fl_native::chunk_gated_delta_rule_fwd_h);
  ops.impl("chunk_fwd_o", &vllm_fl_native::chunk_fwd_o);
  ops.impl("moe_gating_top_k", &vllm_fl_native::moe_gating_top_k);
  ops.impl("npu_moe_init_routing_custom",
           &vllm_ascend::npu_moe_init_routing_custom);
}

TORCH_LIBRARY_IMPL(_C_ascend, Meta, ops) {
  ops.impl("npu_causal_conv1d_custom",
           &vllm_fl_native::meta::npu_causal_conv1d_custom);
  ops.impl("npu_recurrent_gated_delta_rule",
           &vllm_fl_native::meta::npu_recurrent_gated_delta_rule);
  ops.impl("npu_fused_gdn_gating",
           &vllm_fl_native::meta::npu_fused_gdn_gating);
  ops.impl("chunk_gated_delta_rule_fwd_h",
           &vllm_fl_native::meta::chunk_gated_delta_rule_fwd_h);
  ops.impl("chunk_fwd_o", &vllm_fl_native::meta::chunk_fwd_o);
  ops.impl("moe_gating_top_k", &vllm_fl_native::meta::moe_gating_top_k);
  ops.impl("npu_moe_init_routing_custom",
           &vllm_fl_native::meta::npu_moe_init_routing_custom);
}
