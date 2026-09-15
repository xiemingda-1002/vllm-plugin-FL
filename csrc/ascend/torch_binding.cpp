/*
 * FL-owned dispatcher bridge for the Qwen GDN native closure.
 * Function signatures and operator schemas track vLLM-Ascend 0.24.0rc1.
 */

#include <ATen/ATen.h>
#include <torch/library.h>

#include <array>
#include <cmath>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "aclnn_torch_adapter/op_api_common.h"
#include "attention/fused_gdn_gating/fused_gdn_gating_torch_adpt.h"
#include "attention/recurrent_gated_delta_rule/recurrent_gated_delta_rule_torch_adpt.h"
#include "mc2/dispatch_ffn_combine/dispatch_ffn_combine_torch_adpt.h"
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

std::tuple<at::Tensor, at::Tensor> dsa_sparse_attn(
    const at::Tensor& q, const c10::optional<at::Tensor>& ori_kv,
    const c10::optional<at::Tensor>& cmp_kv,
    const c10::optional<at::Tensor>& ori_sparse_indices,
    const c10::optional<at::Tensor>& cmp_sparse_indices,
    const c10::optional<at::Tensor>& ori_block_table,
    const c10::optional<at::Tensor>& cmp_block_table,
    const c10::optional<at::Tensor>& cu_seqlens_q,
    const c10::optional<at::Tensor>& cu_seqlens_ori_kv,
    const c10::optional<at::Tensor>& cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor>& seqused_q,
    const c10::optional<at::Tensor>& seqused_kv,
    const c10::optional<at::Tensor>& sinks, const c10::optional<at::Tensor>& metadata,
    double softmax_scale, int64_t cmp_ratio, int64_t ori_mask_mode,
    int64_t cmp_mask_mode, int64_t ori_win_left, int64_t ori_win_right,
    c10::string_view layout_q, c10::string_view layout_kv, bool return_softmax_lse) {
  auto out = at::empty(q.sizes(), q.options());
  at::Tensor lse;
  if (return_softmax_lse) {
    std::vector<int64_t> lse_sizes(q.sizes().begin(), q.sizes().end());
    lse_sizes.back() = 1;
    lse = at::empty(lse_sizes, q.options().dtype(at::kFloat));
  } else {
    lse = at::empty({0}, q.options().dtype(at::kFloat));
  }
  const auto ori_stride = ori_kv.has_value() ? ori_kv->stride(0) : 0;
  const auto cmp_stride = cmp_kv.has_value() ? cmp_kv->stride(0) : 0;
  auto q_layout = std::string(layout_q);
  auto kv_layout = std::string(layout_kv);
  char* q_layout_ptr = const_cast<char*>(q_layout.c_str());
  char* kv_layout_ptr = const_cast<char*>(kv_layout.c_str());
  EXEC_NPU_CMD(aclnnSparseAttnSharedkv, q, ori_kv, cmp_kv, ori_sparse_indices,
      cmp_sparse_indices, ori_block_table, cmp_block_table, cu_seqlens_q,
      cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q, seqused_kv, sinks, metadata,
      softmax_scale, cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_stride, cmp_stride,
      ori_win_left, ori_win_right, q_layout_ptr, kv_layout_ptr, return_softmax_lse,
      out, lse);
  return {out, lse};
}

at::Tensor dsa_metadata_output(
    const c10::optional<at::Tensor>& first,
    const c10::optional<at::Tensor>& second,
    const c10::optional<at::Tensor>& third,
    const c10::optional<at::Tensor>& fourth,
    const c10::optional<at::Tensor>& fifth, c10::string_view device) {
  const auto options = first.has_value()  ? first->options()
                       : second.has_value() ? second->options()
                       : third.has_value()  ? third->options()
                       : fourth.has_value() ? fourth->options()
                       : fifth.has_value()  ? fifth->options()
                       : at::TensorOptions().device(at::Device(std::string(device)));
  return at::empty({1024}, options.dtype(at::kInt));
}

at::Tensor sparse_attn_metadata(int64_t hq, int64_t hkv, int64_t d,
    const c10::optional<at::Tensor>& q, const c10::optional<at::Tensor>& okv,
    const c10::optional<at::Tensor>& ckv, const c10::optional<at::Tensor>& sq,
    const c10::optional<at::Tensor>& skv, int64_t batch, int64_t maxq, int64_t maxkv,
    int64_t otopk, int64_t ctopk, int64_t ratio, int64_t omask, int64_t cmask,
    int64_t left, int64_t right, c10::string_view lq, c10::string_view lkv,
    bool has_ori, bool has_cmp, c10::string_view device) {
  auto output = dsa_metadata_output(q, okv, ckv, sq, skv, device);
  auto q_layout = std::string(lq); auto kv_layout = std::string(lkv);
  const auto empty = at::empty({0}, output.options());
  auto q_value = q.value_or(empty);
  auto okv_value = okv.value_or(empty);
  auto ckv_value = ckv.value_or(empty);
  auto sq_value = sq.value_or(empty);
  auto skv_value = skv.value_or(empty);
  char* q_layout_ptr = const_cast<char*>(q_layout.c_str());
  char* kv_layout_ptr = const_cast<char*>(kv_layout.c_str());
  EXEC_NPU_CMD(aclnnSparseAttnSharedkvMetadata, q_value, okv_value,
      ckv_value, sq_value, skv_value, hq, hkv, d, batch,
      maxq, maxkv, otopk, ctopk, ratio, omask, cmask, left, right, q_layout_ptr,
      kv_layout_ptr, has_ori, has_cmp, output);
  return output;
}

at::Tensor compressor(const at::Tensor& x, const at::Tensor& wkv, const at::Tensor& wgate,
    at::Tensor& state, const at::Tensor& ape, const at::Tensor& weight,
    const at::Tensor& sin, const at::Tensor& cos, const c10::optional<at::Tensor>& table,
    const c10::optional<at::Tensor>& cu, const c10::optional<at::Tensor>& used,
    const c10::optional<at::Tensor>& pos, int64_t head_dim, int64_t ratio, int64_t coff,
    double eps, int64_t mode, int64_t cache_mode) {
  auto out = x.dim() == 3 ? at::empty({x.size(0), (x.size(1) + ratio - 1) / ratio, weight.size(0)}, x.options())
                          : at::empty({sin.size(0), weight.size(0)}, x.options());
  int64_t state_stride = state.stride(0);
  EXEC_NPU_CMD(aclnnCompressor, x, wkv, wgate, state, ape, weight, sin, cos, table, cu,
      used, pos, head_dim, ratio, coff, eps, mode, cache_mode, state_stride, out);
  return out;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> compressor_metadata(
    const at::Tensor& cos, const at::Tensor& sin, const at::Tensor& cu,
    const at::Tensor& pos, const at::Tensor& table, int64_t block, int64_t format,
    int64_t ratio, int64_t compressed, int64_t requests) {
  auto out_cos = at::empty({compressed, 1, 1, cos.size(1)}, cos.options());
  auto out_sin = at::empty({compressed, 1, 1, sin.size(1)}, sin.options());
  auto slots = format == 2 ? at::empty({compressed, 2}, table.options().dtype(at::kInt))
                           : at::empty({compressed}, table.options().dtype(at::kInt));
  EXEC_NPU_CMD(aclnnCompressorMetadata, cos, sin, cu, pos, table, block, format, ratio,
      requests, out_cos, out_sin, slots);
  return {out_cos, out_sin, slots};
}

std::tuple<at::Tensor, at::Tensor> lightning_indexer(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& weights,
    const at::Tensor& q_scale, const at::Tensor& k_scale, int64_t q_mode, int64_t k_mode,
    const c10::optional<at::Tensor>& q_lens, const c10::optional<at::Tensor>& k_lens,
    const c10::optional<at::Tensor>& table, const c10::optional<at::Tensor>& metadata,
    c10::string_view q_layout, c10::string_view k_layout, int64_t count, int64_t mode,
    int64_t pre, int64_t next, int64_t ratio, bool return_value) {
  const auto heads = k_layout == "TND" ? key.size(1) : key.size(2);
  auto shape = q_layout == "BSND" ? std::vector<int64_t>{query.size(0), query.size(1), heads, count}
                                   : std::vector<int64_t>{query.size(0), heads, count};
  auto indices = at::empty(shape, query.options().dtype(at::kInt));
  auto values = return_value ? at::empty(shape, query.options().dtype(at::kFloat))
                             : at::empty({0}, query.options().dtype(at::kFloat));
  auto ql = std::string(q_layout); auto kl = std::string(k_layout);
  char* ql_ptr = const_cast<char*>(ql.c_str());
  char* kl_ptr = const_cast<char*>(kl.c_str());
  int64_t key_stride = key.stride(0);
  int64_t scale_stride = k_scale.stride(0);
  EXEC_NPU_CMD(aclnnVllmQuantLightningIndexer, query, key, weights, q_scale, k_scale,
      q_lens, k_lens, table, metadata, q_mode, k_mode, ql_ptr, kl_ptr, count, mode,
      pre, next, ratio, return_value, key_stride, scale_stride, indices, values);
  return {indices, values};
}

at::Tensor lightning_indexer_metadata(int64_t hq, int64_t hk, int64_t dim,
    int64_t qm, int64_t km, const c10::optional<at::Tensor>& qlens,
    const c10::optional<at::Tensor>& klens, int64_t batch, int64_t maxq, int64_t maxk,
    c10::string_view qlayout, c10::string_view klayout, int64_t count, int64_t mode,
    int64_t pre, int64_t next, int64_t ratio, c10::string_view device) {
  const c10::optional<at::Tensor> empty_opt;
  auto output = dsa_metadata_output(qlens, klens, empty_opt, empty_opt, empty_opt, device);
  auto ql = std::string(qlayout); auto kl = std::string(klayout);
  const auto empty = at::empty({0}, output.options());
  auto qlens_value = qlens.value_or(empty);
  auto klens_value = klens.value_or(empty);
  char* ql_ptr = const_cast<char*>(ql.c_str());
  char* kl_ptr = const_cast<char*>(kl.c_str());
  EXEC_NPU_CMD(aclnnVllmQuantLightningIndexerMetadata, qlens_value, klens_value,
      hq, hk, dim, qm, km, batch, maxq, maxk, ql_ptr, kl_ptr, count, mode, pre, next,
      ratio, output);
  return output;
}

void inplace_partial_rotary_mul(at::Tensor& x, const at::Tensor& r1, const at::Tensor& r2,
                                c10::string_view rotary_mode, at::IntArrayRef partial) {
  static const std::unordered_map<std::string, int> modes{{"half", 0}, {"interleave", 1}, {"quarter", 2}, {"interleave-half", 3}};
  const auto found = modes.find(std::string(rotary_mode));
  if (found != modes.end()) {
    int64_t rotary_mode_value = found->second;
    EXEC_NPU_CMD(aclnnInplacePartialRotaryMul, x, r1, r2, rotary_mode_value, partial);
  }
}

std::tuple<at::Tensor, at::Tensor> rms_norm_dynamic_quant(const at::Tensor& x,
    const at::Tensor& gamma, const c10::optional<at::Tensor>& smooth,
    const c10::optional<at::Tensor>& beta, double eps) {
  auto y = at::empty_like(x, x.options().dtype(at::kChar));
  std::vector<int64_t> scale_shape(x.sizes().begin(), x.sizes().end() - 1);
  auto scale = at::empty(scale_shape, x.options().dtype(at::kFloat));
  auto y2 = at::empty({1}, y.options()); auto scale2 = at::empty_like(scale);
  std::array<bool, 2>* mask = nullptr; int64_t* dtype = nullptr; at::Tensor smooth2{nullptr};
  EXEC_NPU_CMD(aclnnRmsNormDynamicQuant, x, gamma, smooth, smooth2, beta, eps, mask, dtype, y, y2, scale, scale2);
  return {y, scale};
}

void scatter_nd_update_v2(at::Tensor& var, const at::Tensor& indices, const at::Tensor& update) {
  at::IntArrayRef var_stride = var.strides();
  EXEC_NPU_CMD(aclnnScatterNdUpdateV2, var, indices, update, var_stride);
}

// DeepSeek V4 HC native closure. These dispatcher implementations preserve
// the matching vLLM-Ascend 0.24.0rc1 allocation and ACLNN call contracts.
at::Tensor construct_hc_post_output_tensor(const at::Tensor& residual) {
  constexpr int64_t SIZE = 8;
  constexpr int64_t DIM_0 = 0;
  constexpr int64_t DIM_1 = 1;
  constexpr int64_t DIM_2 = 2;
  constexpr int64_t DIM_3 = 3;
  at::SmallVector<int64_t, SIZE> output_size = {
      residual.size(DIM_0), residual.size(DIM_1), residual.size(DIM_2),
      residual.size(DIM_3)};
  return at::empty(output_size, residual.options().dtype(residual.dtype()));
}

void check_hc_post_shape_and_dtype(const at::Tensor& x,
                                   const at::Tensor& residual,
                                   const at::Tensor& post,
                                   const at::Tensor& com) {
  TORCH_CHECK(x.dim() == 3, "Input tensor x's dim num should be 3, actual ",
              x.dim(), ".");
  for (size_t i = 0; i < 3; i++) {
    TORCH_CHECK(x.size(i) > 0,
                "Input tensor x's shape should be positive, but x.shape[", i,
                "] is :", x.size(i), ".");
  }
  auto batch = x.size(0);
  auto sequence = x.size(1);
  auto d = x.size(2);
  TORCH_CHECK(residual.dim() == 4,
              "Input tensor residual's dim num should be 4, actual ",
              residual.dim(), ".");
  auto hc = residual.size(2);
  TORCH_CHECK(hc > 0, "The hc of residual should be positive, actual ", hc,
              ".");
  TORCH_CHECK(residual.size(0) == batch,
              "The residual.shape[0] should be batch, actual residual.shape[0] is ",
              residual.size(0), ", batch is ", batch, ".");
  TORCH_CHECK(residual.size(1) == sequence,
              "The residual.shape[1] should be sequence, actual residual.shape[1] is ",
              residual.size(1), ", sequence is ", sequence, ".");
  TORCH_CHECK(residual.size(3) == d,
              "The residual.shape[3] should be d, actual residual.shape[3] is ",
              residual.size(3), ", d is ", d, ".");
  TORCH_CHECK(post.dim() == 3, "Input tensor post's dim num should be 3, actual ",
              post.dim(), ".");
  TORCH_CHECK(post.size(0) == batch && post.size(1) == sequence &&
                  post.size(2) == hc,
              "post shape should be [batch, sequence, hc].");
  TORCH_CHECK(com.dim() == 4 && com.size(0) == batch &&
                  com.size(1) == sequence && com.size(2) == hc &&
                  com.size(3) == hc,
              "comb shape should be [batch, sequence, hc, hc].");
  TORCH_CHECK(x.dtype() == at::kFloat || x.dtype() == at::kHalf ||
                  x.dtype() == at::kBFloat16,
              "x should be FLOAT16, BFLOAT16, or FLOAT32.");
  TORCH_CHECK(residual.dtype() == x.dtype(),
              "x's dtype should be equal to residual's dtype.");
  TORCH_CHECK(post.dtype() == at::kFloat || post.dtype() == at::kHalf ||
                  post.dtype() == at::kBFloat16,
              "post should be FLOAT16, BFLOAT16, or FLOAT32.");
  TORCH_CHECK(com.dtype() == post.dtype(),
              "com's dtype should be equal to post's dtype.");
}

at::Tensor npu_hc_post_npu(const at::Tensor& x, const at::Tensor& residual,
                           const at::Tensor& post, const at::Tensor& comb) {
  check_hc_post_shape_and_dtype(x, residual, post, comb);
  auto out = construct_hc_post_output_tensor(residual);
  EXEC_NPU_CMD(aclnnHcPost, x, residual, post, comb, out);
  return out;
}

constexpr int64_t HC_PRE_HC_LIMIT = 4;
constexpr int64_t HC_PRE_D_LIMIT = 4096;
constexpr int64_t HC_PRE_D_LIMIT_EXTEND = 7168;
constexpr int64_t HC_PRE_MIX_HC_LIMIT = 24;

std::tuple<at::Tensor, at::Tensor, at::Tensor> construct_hc_pre_output_tensor(
    const at::Tensor& x, int64_t hc_mult) {
  auto xDims = x.dim();
  at::SmallVector<int64_t, 8> y_size;
  at::SmallVector<int64_t, 8> post_size;
  at::SmallVector<int64_t, 8> comb_frag_size;
  if (xDims == 4) {
    auto batch = x.size(0);
    auto size = x.size(1);
    auto d = x.size(3);
    y_size = {batch, size, d};
    post_size = {batch, size, hc_mult};
    comb_frag_size = {batch, size, hc_mult, hc_mult};
  } else if (xDims == 3) {
    auto bs = x.size(0);
    auto d = x.size(2);
    y_size = {bs, d};
    post_size = {bs, hc_mult};
    comb_frag_size = {bs, hc_mult, hc_mult};
  }
  auto y = at::empty(y_size, x.options().dtype(at::kBFloat16));
  auto post = at::empty(post_size, x.options().dtype(at::kFloat));
  auto comb_frag = at::empty(comb_frag_size, x.options().dtype(at::kFloat));
  return {y, post, comb_frag};
}

void check_hc_pre_shape_and_dtype(const at::Tensor& x, const at::Tensor& hc_fn,
                                  const at::Tensor& hc_scale,
                                  const at::Tensor& hc_base, int64_t hc_mult) {
  constexpr int64_t HC_SCALE_SIZE = 3;
  auto x_dims = x.dim();
  TORCH_CHECK(x_dims == 3 || x_dims == 4,
              "Input tensor x's dim num should be 3 or 4, actual ", x_dims,
              ".");
  for (auto i = 0; i < x_dims; i++) {
    TORCH_CHECK(x.size(i) > 0,
                "Input tensor x's shape should be positive, but x.shape[", i,
                "] is ", x.size(i), ".");
  }
  auto hc = x_dims == 4 ? x.size(2) : x.size(1);
  auto d = x_dims == 4 ? x.size(3) : x.size(2);
  TORCH_CHECK(hc_mult == HC_PRE_HC_LIMIT, "hc_mult only supports ",
              HC_PRE_HC_LIMIT, ", actual ", hc_mult, ".");
  TORCH_CHECK(hc == HC_PRE_HC_LIMIT, "The hc of x only supports ",
              HC_PRE_HC_LIMIT, ", actual ", hc, ".");
  TORCH_CHECK(d == HC_PRE_D_LIMIT || d == HC_PRE_D_LIMIT_EXTEND,
              "The d of x only supports ", HC_PRE_D_LIMIT, " or ",
              HC_PRE_D_LIMIT_EXTEND, ", actual ", d, ".");
  TORCH_CHECK(hc_fn.dim() == 2 && hc_fn.size(0) == HC_PRE_MIX_HC_LIMIT &&
                  hc_fn.size(1) == hc * d,
              "hc_fn shape should be [24, hc * d].");
  TORCH_CHECK(hc_scale.dim() == 1 && hc_scale.size(0) == HC_SCALE_SIZE,
              "hc_scale shape should be [3].");
  TORCH_CHECK(hc_base.dim() == 1 && hc_base.size(0) == HC_PRE_MIX_HC_LIMIT,
              "hc_base shape should be [24].");
  TORCH_CHECK(x.dtype() == at::kBFloat16, "x's dtype should be BFLOAT16.");
  TORCH_CHECK(hc_fn.dtype() == at::kFloat, "hc_fn's dtype should be FLOAT32.");
  TORCH_CHECK(hc_scale.dtype() == at::kFloat,
              "hc_scale's dtype should be FLOAT32.");
  TORCH_CHECK(hc_base.dtype() == at::kFloat,
              "hc_base's dtype should be FLOAT32.");
}

at::Tensor construct_hc_pre_rsqrt_output_tensor(const at::Tensor& x,
                                                 float epsilon = 1e-6) {
  constexpr int64_t SIZE = 8;
  TORCH_CHECK(epsilon >= 0, "epsilon should be greater than 0.");
  c10::SmallVector<int64_t, SIZE> yOut_shape;
  for (size_t i = 0; i < x.dim() - 2; i++) yOut_shape.push_back(x.sizes()[i]);
  yOut_shape.push_back(1);
  return at::empty(yOut_shape, x.options().dtype(at::kFloat));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_hc_pre_npu(
    const at::Tensor& x, const at::Tensor& hc_fn, const at::Tensor& hc_scale,
    const at::Tensor& hc_base, int64_t hc_mult, int64_t hc_sinkhorn_iters,
    double norm_eps, double hc_eps) {
  check_hc_pre_shape_and_dtype(x, hc_fn, hc_scale, hc_base, hc_mult);
  auto rsqrt = construct_hc_pre_rsqrt_output_tensor(x, norm_eps);
  EXEC_NPU_CMD(aclnnHcPreInvRms, x, norm_eps, rsqrt);
  auto original_type = x.dtype();
  auto x_float = x.to(at::kFloat);
  auto x_flattened = x_float.flatten(2, -1);
  if (x.dim() == 3) x_flattened = x_float.flatten(1, -1);
  auto mixes = at::linear(x_flattened, hc_fn);
  auto output_tensors = construct_hc_pre_output_tensor(x, hc_mult);
  auto y = std::get<0>(output_tensors);
  auto post = std::get<1>(output_tensors);
  auto comb_frag = std::get<2>(output_tensors);
  EXEC_NPU_CMD(aclnnHcPreSinkhorn, mixes, rsqrt, hc_scale, hc_base, x, hc_mult,
               hc_sinkhorn_iters, hc_eps, y, post, comb_frag);
  y = y.to(original_type);
  return {y, post, comb_frag};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_hc_pre_v2_npu(
    const at::Tensor& x, const at::Tensor& hc_fn, const at::Tensor& hc_scale,
    const at::Tensor& hc_base, int64_t hc_mult, int64_t hc_sinkhorn_iters,
    double norm_eps, double hc_eps) {
  check_hc_pre_shape_and_dtype(x, hc_fn, hc_scale, hc_base, hc_mult);
  auto output_tensors = construct_hc_pre_output_tensor(x, hc_mult);
  auto y = std::get<0>(output_tensors);
  auto post = std::get<1>(output_tensors);
  auto comb_frag = std::get<2>(output_tensors);
  EXEC_NPU_CMD(aclnnHcPre, x, hc_fn, hc_scale, hc_base, hc_mult,
               hc_sinkhorn_iters, hc_eps, norm_eps, y, post, comb_frag);
  return {y, post, comb_frag};
}

at::Tensor construct_hc_pre_inv_rms_output_tensor(const at::Tensor& x,
                                                   float epsilon = 1e-20) {
  constexpr int64_t SIZE = 8;
  TORCH_CHECK(epsilon >= 0, "epsilon should be greater than 0.");
  c10::SmallVector<int64_t, SIZE> yOut_shape;
  for (auto i = 0; i < x.dim() - 2; i++) yOut_shape.push_back(x.sizes()[i]);
  yOut_shape.push_back(1);
  return at::empty(yOut_shape, x.options().dtype(at::kFloat));
}

at::Tensor npu_hc_pre_inv_rms_npu(const at::Tensor& x, double epsilon = 1e-20) {
  TORCH_CHECK(x.numel() > 0, "Input tensor x should not be empty.");
  TORCH_CHECK(epsilon >= 0, "epsilon should be greater than 0.");
  TORCH_CHECK(x.dtype() == at::kFloat || x.dtype() == at::kHalf ||
                  x.dtype() == at::kBFloat16,
              "x should be FLOAT16, BFLOAT16, or FLOAT32.");
  auto yOut = construct_hc_pre_inv_rms_output_tensor(x, epsilon);
  EXEC_NPU_CMD(aclnnHcPreInvRms, x, epsilon, yOut);
  return yOut;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
construct_hc_pre_sinkhorn_output_tensor(const at::Tensor&, const at::Tensor& x,
                                        int64_t hc_mult) {
  return construct_hc_pre_output_tensor(x, hc_mult);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_hc_pre_sinkhorn_npu(
    const at::Tensor& mixes, const at::Tensor& rsqrt, const at::Tensor& hc_scale,
    const at::Tensor& hc_base, const at::Tensor& x, int64_t hc_mult,
    int64_t hc_sinkhorn_iters, double hc_eps) {
  auto output_tensors = construct_hc_pre_sinkhorn_output_tensor(mixes, x, hc_mult);
  auto y = std::get<0>(output_tensors);
  auto post = std::get<1>(output_tensors);
  auto comb_frag = std::get<2>(output_tensors);
  EXEC_NPU_CMD(aclnnHcPreSinkhorn, mixes, rsqrt, hc_scale, hc_base, x, hc_mult,
               hc_sinkhorn_iters, hc_eps, y, post, comb_frag);
  return {y, post, comb_frag};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> moe_gating_top_k_hash(
    const at::Tensor& x, int64_t k,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& input_ids_opt,
    const c10::optional<at::Tensor>& tid2eid_opt, int64_t k_group,
    int64_t group_count, double routed_scaling_factor, double eps,
    int64_t group_select_mode, int64_t renorm, int64_t norm_type,
    bool out_flag) {
  TORCH_CHECK(x.dim() == 2, "x must be 2D, but got dim=", x.dim());
  TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat ||
                  x.scalar_type() == at::kBFloat16,
              "x dtype must be float16/float32/bfloat16, but got ",
              x.scalar_type());
  TORCH_CHECK(k > 0, "k must be > 0, but got k=", k);
  TORCH_CHECK(k_group >= 1, "k_group must be >= 1, but got k_group=", k_group);
  TORCH_CHECK(group_count >= 1, "group_count must be >= 1, but got group_count=", group_count);
  TORCH_CHECK(group_select_mode == 0 || group_select_mode == 1,
              "group_select_mode must be 0 or 1, but got ", group_select_mode);
  TORCH_CHECK(renorm == 0, "renorm can only be 0 currently, but got ", renorm);
  TORCH_CHECK(norm_type == 0 || norm_type == 1 || norm_type == 2,
              "norm_type must be 0 (softmax) or 1 (sigmoid) or 2 (softplus), but got ", norm_type);
  TORCH_CHECK(eps > 0.0, "eps must be > 0, but got ", eps);
  TORCH_CHECK(routed_scaling_factor > 0.0,
              "routed_scaling_factor must be > 0, but got ", routed_scaling_factor);

  const int64_t rows = x.size(0);
  const int64_t expert_num = x.size(1);
  TORCH_CHECK(expert_num > 0, "expert_num must be > 0");
  TORCH_CHECK(expert_num <= 2048,
              "expert_num (E) must be <= 2048, but got ", expert_num);
  if (bias_opt.has_value() && bias_opt->defined()) {
    const auto& bias = *bias_opt;
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D, but got dim=", bias.dim());
    TORCH_CHECK(bias.size(0) == expert_num,
                "bias.size(0) must equal expert_num. bias.size(0)=", bias.size(0),
                ", expert_num=", expert_num);
    TORCH_CHECK(bias.scalar_type() == x.scalar_type(),
                "bias dtype must equal x dtype. x=", x.scalar_type(),
                ", bias=", bias.scalar_type());
  }
  if (input_ids_opt.has_value() && input_ids_opt->defined()) {
    const auto& input_ids = *input_ids_opt;
    TORCH_CHECK(input_ids.scalar_type() == at::kInt || input_ids.scalar_type() == at::kLong,
                "input_ids dtype must be int32 or int64, but got ", input_ids.scalar_type());
    TORCH_CHECK(input_ids.numel() == rows,
                "input_ids.numel() must equal x.size(0). input_ids.numel()=", input_ids.numel(),
                ", rows=", rows);
  }
  if (tid2eid_opt.has_value() && tid2eid_opt->defined()) {
    const auto& tid2eid = *tid2eid_opt;
    TORCH_CHECK(tid2eid.scalar_type() == at::kInt || tid2eid.scalar_type() == at::kLong,
                "tid2eid dtype must be int32 or int64, but got ", tid2eid.scalar_type());
    TORCH_CHECK(tid2eid.dim() >= 1, "tid2eid must have dim>=1, but got dim=", tid2eid.dim());
  }

  const at::Tensor& bias = c10::value_or_else(bias_opt, [] { return at::Tensor(); });
  const at::Tensor& input_ids = c10::value_or_else(input_ids_opt, [] { return at::Tensor(); });
  const at::Tensor& tid2eid = c10::value_or_else(tid2eid_opt, [] { return at::Tensor(); });
  auto y = at::empty({rows, k}, x.options());
  auto expert_idx = at::empty({rows, k}, x.options().dtype(at::kInt));
  auto out = at::empty({rows, expert_num}, x.options().dtype(at::kFloat));
  EXEC_NPU_CMD(aclnnMoeGatingTopKHash, x, bias, input_ids, tid2eid, k,
               k_group, group_count, routed_scaling_factor, eps,
               group_select_mode, renorm, norm_type, out_flag, y, expert_idx,
               out);
  return {y, expert_idx, out};
}

std::tuple<at::Tensor, at::Tensor> npu_dequant_swiglu_quant(
    const at::Tensor& x, const c10::optional<at::Tensor>& weight_scale,
    const c10::optional<at::Tensor>& activation_scale,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& quant_scale,
    const c10::optional<at::Tensor>& quant_offset,
    const c10::optional<at::Tensor>& group_index, bool activate_left,
    int64_t quant_mode, int64_t swiglu_mode, double clamp_limit,
    double glu_alpha, double glu_bias) {
  TORCH_CHECK(x.dim() > 1, "x dim should larger than 1");
  TORCH_CHECK(quant_mode == 0 || quant_mode == 1,
              "quant_mode only support 0 or 1, but got ", quant_mode);
  TORCH_CHECK(swiglu_mode == 0 || swiglu_mode == 1,
              "swiglu_mode only support 0 or 1, but got ", swiglu_mode);
  TORCH_CHECK(std::isfinite(clamp_limit) && clamp_limit >= 0.0,
              "clamp_limit should be positive finite");
  TORCH_CHECK(std::isfinite(glu_alpha), "glu_alpha should be finite");
  TORCH_CHECK(std::isfinite(glu_bias), "glu_bias should be finite");
  TORCH_CHECK(x.size(x.dim() - 1) % 2 == 0, "x last dim should be even");
  c10::SmallVector<int64_t, 8> y_size;
  c10::SmallVector<int64_t, 8> scale_size;
  for (int64_t i = 0; i < x.dim() - 1; ++i) {
    y_size.push_back(x.size(i));
    scale_size.push_back(x.size(i));
  }
  y_size.push_back(x.size(x.dim() - 1) / 2);
  auto y = at::empty(y_size, x.options().dtype(c10::ScalarType::Char));
  auto scale = at::empty(scale_size, x.options().dtype(c10::ScalarType::Float));
  std::string quant_mode_str = quant_mode == 1 ? "dynamic" : "static";
  char* quant_mode_ptr = const_cast<char*>(quant_mode_str.c_str());
  const at::Tensor& weight_scale_value = c10::value_or_else(weight_scale, [] { return at::Tensor(); });
  const at::Tensor& activation_scale_opt = c10::value_or_else(activation_scale, [] { return at::Tensor(); });
  const at::Tensor& bias_opt = c10::value_or_else(bias, [] { return at::Tensor(); });
  const at::Tensor& quant_scale_opt = c10::value_or_else(quant_scale, [] { return at::Tensor(); });
  const at::Tensor& quant_offset_opt = c10::value_or_else(quant_offset, [] { return at::Tensor(); });
  const at::Tensor& group_index_opt = c10::value_or_else(group_index, [] { return at::Tensor(); });
  static const bool is_v2_available =
      GetOpApiFuncAddr("aclnnDequantSwigluQuantV2") != nullptr &&
      GetOpApiFuncAddr("aclnnDequantSwigluQuantV2GetWorkspaceSize") != nullptr;
  if (swiglu_mode == 0 && !is_v2_available) {
    EXEC_NPU_CMD(aclnnDequantSwigluQuant, x, weight_scale_value,
                 activation_scale_opt, bias_opt, quant_scale_opt,
                 quant_offset_opt, group_index_opt, activate_left,
                 quant_mode_ptr, y, scale);
  } else {
    int64_t dst_type = 2;
    char* round_mode = const_cast<char*>("rint");
    int64_t activate_dim = -1;
    EXEC_NPU_CMD(aclnnDequantSwigluQuantV2, x, weight_scale_value,
                 activation_scale_opt, bias_opt, quant_scale_opt,
                 quant_offset_opt, group_index_opt, activate_left,
                 quant_mode_ptr, dst_type, round_mode, activate_dim,
                 swiglu_mode, clamp_limit, glu_alpha, glu_bias, y, scale);
  }
  return {y, scale};
}

namespace meta {

std::tuple<at::Tensor&, at::Tensor&> dispatch_ffn_combine(
    const at::Tensor&, const at::TensorList&, const at::TensorList&,
    const at::Tensor&, const at::TensorList&, const at::TensorList&,
    const at::TensorList&, const at::TensorList&, const at::Tensor&,
    c10::string_view, int64_t, at::Tensor& out, at::Tensor& expert_token_nums,
    const c10::optional<at::Tensor>&, double) {
  return {out, expert_token_nums};
}

at::Tensor npu_hc_post(const at::Tensor&, const at::Tensor& residual,
                       const at::Tensor&, const at::Tensor&) {
  return at::empty_symint(residual.sym_sizes(), residual.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> hc_pre_outputs(
    const at::Tensor& x, int64_t hc_mult) {
  if (x.dim() == 4) {
    return {
        at::empty_symint({x.sym_size(0), x.sym_size(1), x.sym_size(3)},
                         x.options().dtype(at::kBFloat16)),
        at::empty_symint({x.sym_size(0), x.sym_size(1), hc_mult},
                         x.options().dtype(at::kFloat)),
        at::empty_symint(
            {x.sym_size(0), x.sym_size(1), hc_mult, hc_mult},
            x.options().dtype(at::kFloat))};
  }
  return {at::empty_symint({x.sym_size(0), x.sym_size(2)},
                           x.options().dtype(at::kBFloat16)),
          at::empty_symint({x.sym_size(0), hc_mult},
                           x.options().dtype(at::kFloat)),
          at::empty_symint({x.sym_size(0), hc_mult, hc_mult},
                           x.options().dtype(at::kFloat))};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_hc_pre(
    const at::Tensor& x, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    int64_t hc_mult, int64_t, double, double) {
  return hc_pre_outputs(x, hc_mult);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_hc_pre_v2(
    const at::Tensor& x, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    int64_t hc_mult, int64_t, double, double) {
  return hc_pre_outputs(x, hc_mult);
}

at::Tensor npu_hc_pre_inv_rms(const at::Tensor& x, double) {
  c10::SymDimVector shape;
  for (auto i = 0; i < x.dim() - 2; ++i) shape.push_back(x.sym_size(i));
  shape.push_back(1);
  return at::empty_symint(shape, x.options().dtype(at::kFloat));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_hc_pre_sinkhorn(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor& x, int64_t hc_mult, int64_t, double) {
  return hc_pre_outputs(x, hc_mult);
}

// DeepSeek V4 DSA native closure.  These schemas and allocations track the
// matching vLLM-Ascend 0.24.0rc1 implementation; the OPP payload is packaged
// separately by build_opp.sh into this same extension's vendor directory.
std::tuple<at::Tensor, at::Tensor> dsa_sparse_attn(
    const at::Tensor& q, const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&,
    double, int64_t, int64_t, int64_t, int64_t, int64_t, c10::string_view,
    c10::string_view, bool return_softmax_lse) {
  auto output = at::empty_symint(q.sym_sizes(), q.options());
  if (!return_softmax_lse) {
    return {output, at::empty({0}, q.options().dtype(at::kFloat))};
  }
  auto sizes = q.sym_sizes().vec();
  sizes.back() = 1;
  return {output, at::empty_symint(sizes, q.options().dtype(at::kFloat))};
}

at::Tensor dsa_metadata(const c10::optional<at::Tensor>& first,
                        const c10::optional<at::Tensor>& second,
                        const c10::optional<at::Tensor>& third,
                        const c10::optional<at::Tensor>& fourth,
                        const c10::optional<at::Tensor>& fifth,
                        c10::string_view device) {
  const auto& source = first.has_value()    ? first
                       : second.has_value() ? second
                       : third.has_value()  ? third
                       : fourth.has_value() ? fourth
                                            : fifth;
  if (source.has_value()) {
    return at::empty_symint({c10::SymInt(1024)},
                            source->options().dtype(at::kInt));
  }
  auto requested_device = at::Device(std::string(device));
  std::string meta_device = "meta";
  if (requested_device.has_index()) {
    meta_device += ":" + std::to_string(requested_device.index());
  }
  return at::empty_symint(
      {c10::SymInt(1024)},
      at::TensorOptions().dtype(at::kInt).device(at::Device(meta_device)));
}

at::Tensor sparse_attn_metadata(
    int64_t, int64_t, int64_t, const c10::optional<at::Tensor>& q,
    const c10::optional<at::Tensor>& ori,
    const c10::optional<at::Tensor>& cmp,
    const c10::optional<at::Tensor>& seqused_q,
    const c10::optional<at::Tensor>& seqused_kv,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, c10::string_view, c10::string_view, bool, bool,
    c10::string_view device) {
  return dsa_metadata(q, ori, cmp, seqused_q, seqused_kv, device);
}

at::Tensor lightning_indexer_metadata(
    int64_t, int64_t, int64_t, int64_t, int64_t,
    const c10::optional<at::Tensor>& q_lens,
    const c10::optional<at::Tensor>& k_lens, int64_t, int64_t, int64_t,
    c10::string_view, c10::string_view, int64_t, int64_t, int64_t, int64_t,
    int64_t, c10::string_view device) {
  return dsa_metadata(q_lens, k_lens, c10::nullopt, c10::nullopt,
                      c10::nullopt, device);
}

at::Tensor compressor(const at::Tensor& x, const at::Tensor&, const at::Tensor&,
                      at::Tensor&, const at::Tensor&, const at::Tensor& norm_weight,
                      const at::Tensor& rope_sin, const at::Tensor&,
                      const c10::optional<at::Tensor>&,
                      const c10::optional<at::Tensor>&,
                      const c10::optional<at::Tensor>&,
                      const c10::optional<at::Tensor>&, int64_t, int64_t cmp_ratio,
                      int64_t, double, int64_t, int64_t) {
  const auto cmp_s = x.dim() == 3 ? (x.sym_size(1) + cmp_ratio - 1) / cmp_ratio
                                  : rope_sin.sym_size(0);
  return x.dim() == 3
      ? at::empty_symint({x.sym_size(0), cmp_s, norm_weight.sym_size(0)}, x.options())
      : at::empty_symint({cmp_s, norm_weight.sym_size(0)}, x.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> compressor_metadata(
    const at::Tensor& rope_cos, const at::Tensor& rope_sin, const at::Tensor&,
    const at::Tensor&, const at::Tensor& table, int64_t, int64_t format,
    int64_t, int64_t compressed, int64_t) {
  auto cos = at::empty_symint({compressed, 1, 1, rope_cos.sym_size(1)}, rope_cos.options());
  auto sin = at::empty_symint({compressed, 1, 1, rope_sin.sym_size(1)}, rope_sin.options());
  auto slots = format == 2
      ? at::empty_symint({compressed, 2}, table.options().dtype(at::kInt))
      : at::empty_symint({compressed}, table.options().dtype(at::kInt));
  return {cos, sin, slots};
}

std::tuple<at::Tensor, at::Tensor> lightning_indexer(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, int64_t, int64_t,
    const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&,
    c10::string_view layout_query, c10::string_view layout_key, int64_t sparse_count,
    int64_t, int64_t, int64_t, int64_t, bool return_value) {
  const auto key_heads = layout_key == "TND" ? key.sym_size(1) : key.sym_size(2);
  c10::SymDimVector shape = layout_query == "BSND"
      ? c10::SymDimVector{query.sym_size(0), query.sym_size(1), key_heads, sparse_count}
      : c10::SymDimVector{query.sym_size(0), key_heads, sparse_count};
  auto indices = at::empty_symint(shape, query.options().dtype(at::kInt));
  auto values = return_value
      ? at::empty_symint(shape, query.options().dtype(at::kFloat))
      : at::empty({0}, query.options().dtype(at::kFloat));
  return {indices, values};
}

void inplace_partial_rotary_mul(at::Tensor&, const at::Tensor&, const at::Tensor&,
                                c10::string_view, at::IntArrayRef) {}

std::tuple<at::Tensor, at::Tensor> rms_norm_dynamic_quant(
    const at::Tensor& x, const at::Tensor&, const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&, double) {
  c10::SymDimVector scale;
  for (auto i = 0; i + 1 < x.dim(); ++i) scale.push_back(x.sym_size(i));
  return {at::empty_symint(x.sym_sizes(), x.options().dtype(at::kChar)),
          at::empty_symint(scale, x.options().dtype(at::kFloat))};
}

void scatter_nd_update_v2(at::Tensor&, const at::Tensor&, const at::Tensor&) {}

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

std::tuple<at::Tensor, at::Tensor, at::Tensor> moe_gating_top_k_hash(
    const at::Tensor& x, int64_t k,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& input_ids_opt,
    const c10::optional<at::Tensor>& tid2eid_opt, int64_t k_group,
    int64_t group_count, double routed_scaling_factor, double eps,
    int64_t group_select_mode, int64_t renorm, int64_t norm_type, bool) {
  TORCH_CHECK(x.dim() == 2, "x must be 2D, but got dim=", x.dim());
  TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat ||
                  x.scalar_type() == at::kBFloat16,
              "x dtype must be float16/float32/bfloat16, but got ", x.scalar_type());
  TORCH_CHECK(k > 0, "k must be > 0, but got k=", k);
  TORCH_CHECK(k_group >= 1, "k_group must be >= 1, but got k_group=", k_group);
  TORCH_CHECK(group_count >= 1, "group_count must be >= 1, but got group_count=", group_count);
  TORCH_CHECK(group_select_mode == 0 || group_select_mode == 1,
              "group_select_mode must be 0 or 1, but got ", group_select_mode);
  TORCH_CHECK(renorm == 0, "renorm can only be 0 currently, but got ", renorm);
  TORCH_CHECK(norm_type == 0 || norm_type == 1 || norm_type == 2,
              "norm_type must be 0 (softmax) or 1 (sigmoid) or 2 (softplus), but got ", norm_type);
  TORCH_CHECK(eps > 0.0, "eps must be > 0, but got ", eps);
  TORCH_CHECK(routed_scaling_factor > 0.0,
              "routed_scaling_factor must be > 0, but got ", routed_scaling_factor);
  if (bias_opt.has_value() && bias_opt->defined()) {
    TORCH_CHECK(bias_opt->dim() == 1, "bias must be 1D, but got dim=", bias_opt->dim());
    TORCH_CHECK(bias_opt->scalar_type() == x.scalar_type(), "bias dtype must equal x dtype. x=", x.scalar_type(), ", bias=", bias_opt->scalar_type());
  }
  if (input_ids_opt.has_value() && input_ids_opt->defined()) {
    TORCH_CHECK(input_ids_opt->scalar_type() == at::kInt || input_ids_opt->scalar_type() == at::kLong, "input_ids dtype must be int32 or int64, but got ", input_ids_opt->scalar_type());
  }
  if (tid2eid_opt.has_value() && tid2eid_opt->defined()) {
    TORCH_CHECK(tid2eid_opt->scalar_type() == at::kInt || tid2eid_opt->scalar_type() == at::kLong, "tid2eid dtype must be int32 or int64, but got ", tid2eid_opt->scalar_type());
    TORCH_CHECK(tid2eid_opt->dim() >= 1, "tid2eid must have dim>=1, but got dim=", tid2eid_opt->dim());
  }
  const auto rows = x.sym_size(0);
  const auto expert_num = x.sym_size(1);
  return {at::empty_symint({rows, k}, x.options()),
          at::empty_symint({rows, k}, x.options().dtype(at::kInt)),
          at::empty_symint({rows, expert_num}, x.options().dtype(at::kFloat))};
}

std::tuple<at::Tensor, at::Tensor> npu_dequant_swiglu_quant(
    const at::Tensor& x, const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&, bool, int64_t, int64_t, double,
    double, double) {
  c10::SymDimVector y_size;
  c10::SymDimVector scale_size;
  for (int64_t i = 0; i < x.dim() - 1; ++i) {
    y_size.push_back(x.sym_size(i));
    scale_size.push_back(x.sym_size(i));
  }
  y_size.push_back(x.sym_size(x.dim() - 1) / c10::SymInt(2));
  return {at::empty_symint(y_size, x.options().dtype(c10::ScalarType::Char)),
          at::empty_symint(scale_size, x.options().dtype(c10::ScalarType::Float))};
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
  ops.def("dispatch_ffn_combine(Tensor x, Tensor[] weight1, Tensor[] weight2, "
          "Tensor expert_idx, Tensor[] scale1, Tensor[] scale2, Tensor[] bias1, "
          "Tensor[] bias2, Tensor probs, str group, int max_output_size, "
          "Tensor! out, Tensor! expert_token_nums, Tensor? x_active_mask=None, "
          "float swiglu_limit=1000000.0) -> (Tensor out, Tensor expert_token_nums)");
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
  ops.def("moe_gating_top_k_hash(Tensor x, int k, Tensor? bias=None, "
          "Tensor? input_ids=None, Tensor? tid2eid=None, int k_group=1, "
          "int group_count=1, float routed_scaling_factor=1.0, "
          "float eps=1e-20, int group_select_mode=0, int renorm=0, "
          "int norm_type=0, bool out_flag=False) "
          "-> (Tensor y, Tensor expert_idx, Tensor out)");
  ops.def("npu_dequant_swiglu_quant(Tensor x, *, Tensor? weight_scale=None, "
          "Tensor? activation_scale=None, Tensor? bias=None, "
          "Tensor? quant_scale=None, Tensor? quant_offset=None, "
          "Tensor? group_index=None, bool activate_left=True, "
          "int quant_mode=0, int swiglu_mode=0, float clamp_limit=0.0, "
          "float glu_alpha=1.0, float glu_bias=0.0) "
          "-> (Tensor y, Tensor scale)");
  ops.def(
      "npu_moe_init_routing_custom(Tensor x, Tensor expert_idx, *, "
      "Tensor? scale=None, Tensor? offset=None, int active_num=-1, "
      "int expert_capacity=-1, int expert_num=-1, int drop_pad_mode=0, "
      "int expert_tokens_num_type=0, bool expert_tokens_num_flag=False, "
      "int quant_mode=0, int[2] active_expert_range=[], int row_idx_type=0) "
      "-> (Tensor, Tensor, Tensor, Tensor)");
  ops.def("npu_hc_post(Tensor x, Tensor residual, Tensor post, Tensor comb) -> Tensor");
  ops.def("npu_hc_pre(Tensor x, Tensor hc_fn, Tensor hc_scale, Tensor hc_base, int hc_mult, int hc_sinkhorn_iters, float norm_eps, float hc_eps) -> (Tensor out0, Tensor out1, Tensor out2)");
  ops.def("npu_hc_pre_v2(Tensor x, Tensor hc_fn, Tensor hc_scale, Tensor hc_base, int hc_mult, int hc_sinkhorn_iters, float norm_eps, float hc_eps) -> (Tensor out0, Tensor out1, Tensor out2)");
  ops.def("npu_hc_pre_inv_rms(Tensor x, float epsilon=1e-20) -> Tensor");
  ops.def("npu_hc_pre_sinkhorn(Tensor mixes, Tensor rsqrt, Tensor hc_scale, Tensor hc_base, Tensor x, int hc_mult, int hc_sinkhorn_iters, float hc_eps) -> (Tensor out0, Tensor out1, Tensor out2)");
  ops.def("compressor(Tensor x, Tensor wkv, Tensor wgate, Tensor(a!) state_cache, Tensor ape, Tensor norm_weight, Tensor rope_sin, Tensor rope_cos, Tensor? state_block_table, Tensor? cu_seqlens, Tensor? seqused, Tensor? start_pos, int rope_head_dim, int cmp_ratio, int coff, float norm_eps, int rotary_mode, int cache_mode) -> Tensor");
  ops.def("compressor_metadata(Tensor rope_cos, Tensor rope_sin, Tensor cu_seqlens, Tensor start_pos, Tensor kv_block_table, int kv_block_size, int slot_mapping_format, int compress_ratio, int num_compressed_tokens, int num_reqs_actual) -> (Tensor, Tensor, Tensor)");
  ops.def("npu_vllm_quant_lightning_indexer(Tensor query, Tensor key, Tensor weights, Tensor query_dequant_scale, Tensor key_dequant_scale, int query_quant_mode=0, int key_quant_mode=0, Tensor? actual_seq_lengths_query=None, Tensor? actual_seq_lengths_key=None, Tensor? block_table=None, Tensor? metadata=None, str layout_query=\"BSND\", str layout_key=\"BSND\", int sparse_count=2048, int sparse_mode=3, int pre_tokens=9223372036854775807, int next_tokens=9223372036854775807, int cmp_ratio=1, bool return_value=False) -> (Tensor sparse_indices, Tensor sparse_values)");
  ops.def("npu_sparse_attn_sharedkv(Tensor q, *, Tensor? ori_kv=None, Tensor? cmp_kv=None, Tensor? ori_sparse_indices=None, Tensor? cmp_sparse_indices=None, Tensor? ori_block_table=None, Tensor? cmp_block_table=None, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_ori_kv=None, Tensor? cu_seqlens_cmp_kv=None, Tensor? seqused_q=None, Tensor? seqused_kv=None, Tensor? sinks=None, Tensor? metadata=None, float softmax_scale=0, int cmp_ratio=0, int ori_mask_mode=4, int cmp_mask_mode=3, int ori_win_left=128, int ori_win_right=0, str layout_q=\"BSND\", str layout_kv=\"PA_ND\", bool return_softmax_lse=False) -> (Tensor out, Tensor softmax_lse)");
  ops.def("npu_sparse_attn_sharedkv_metadata(int num_heads_q, int num_heads_kv, int head_dim, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_ori_kv=None, Tensor? cu_seqlens_cmp_kv=None, Tensor? seqused_q=None, Tensor? seqused_kv=None, int batch_size=0, int max_seqlen_q=0, int max_seqlen_kv=0, int ori_topk=0, int cmp_topk=0, int cmp_ratio=4, int ori_mask_mode=4, int cmp_mask_mode=3, int ori_win_left=128, int ori_win_right=0, str layout_q=\"BSND\", str layout_kv=\"PA_ND\", bool has_ori_kv=True, bool has_cmp_kv=True, str device=\"npu\") -> Tensor");
  ops.def("npu_vllm_quant_lightning_indexer_metadata(int num_heads_q, int num_heads_k, int head_dim, int query_quant_mode, int key_quant_mode, Tensor? actual_seq_lengths_query=None, Tensor? actual_seq_lengths_key=None, int batch_size=0, int max_seqlen_q=0, int max_seqlen_k=0, str layout_query=\"BSND\", str layout_key=\"BSND\", int sparse_count=2048, int sparse_mode=3, int pre_tokens=9223372036854775807, int next_tokens=9223372036854775807, int cmp_ratio=1, str device=\"npu\") -> Tensor");
  ops.def("inplace_partial_rotary_mul(Tensor(a!) x, Tensor r1, Tensor r2, str rotary_mode, int[] partial_slice) -> ()");
  ops.def("npu_rms_norm_dynamic_quant(Tensor x, Tensor gamma, Tensor? smooth_scale=None, Tensor? beta=None, float epsilon=1e-6) -> (Tensor y_out, Tensor scale_out)");
  ops.def("npu_scatter_nd_update_v2(Tensor(a!) var, Tensor indices, Tensor update) -> ()");
}

TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, ops) {
  ops.impl("dispatch_ffn_combine", &vllm_fl_native::dispatch_ffn_combine);
  ops.impl("npu_causal_conv1d_custom",
           &vllm_fl_native::npu_causal_conv1d_custom);
  ops.impl("npu_recurrent_gated_delta_rule",
           &vllm_ascend::npu_recurrent_gated_delta_rule);
  ops.impl("npu_fused_gdn_gating", &vllm_ascend::npu_fused_gdn_gating);
  ops.impl("chunk_gated_delta_rule_fwd_h",
           &vllm_fl_native::chunk_gated_delta_rule_fwd_h);
  ops.impl("chunk_fwd_o", &vllm_fl_native::chunk_fwd_o);
  ops.impl("moe_gating_top_k", &vllm_fl_native::moe_gating_top_k);
  ops.impl("moe_gating_top_k_hash", &vllm_fl_native::moe_gating_top_k_hash);
  ops.impl("npu_dequant_swiglu_quant",
           &vllm_fl_native::npu_dequant_swiglu_quant);
  ops.impl("npu_moe_init_routing_custom",
           &vllm_ascend::npu_moe_init_routing_custom);
  ops.impl("npu_hc_post", &vllm_fl_native::npu_hc_post_npu);
  ops.impl("npu_hc_pre", &vllm_fl_native::npu_hc_pre_npu);
  ops.impl("npu_hc_pre_v2", &vllm_fl_native::npu_hc_pre_v2_npu);
  ops.impl("npu_hc_pre_inv_rms", &vllm_fl_native::npu_hc_pre_inv_rms_npu);
  ops.impl("npu_hc_pre_sinkhorn", &vllm_fl_native::npu_hc_pre_sinkhorn_npu);
  ops.impl("compressor", &vllm_fl_native::compressor);
  ops.impl("compressor_metadata", &vllm_fl_native::compressor_metadata);
  ops.impl("npu_vllm_quant_lightning_indexer", &vllm_fl_native::lightning_indexer);
  ops.impl("npu_sparse_attn_sharedkv", &vllm_fl_native::dsa_sparse_attn);
  ops.impl("npu_sparse_attn_sharedkv_metadata", &vllm_fl_native::sparse_attn_metadata);
  ops.impl("npu_vllm_quant_lightning_indexer_metadata", &vllm_fl_native::lightning_indexer_metadata);
  ops.impl("inplace_partial_rotary_mul", &vllm_fl_native::inplace_partial_rotary_mul);
  ops.impl("npu_rms_norm_dynamic_quant", &vllm_fl_native::rms_norm_dynamic_quant);
  ops.impl("npu_scatter_nd_update_v2", &vllm_fl_native::scatter_nd_update_v2);
}

TORCH_LIBRARY_IMPL(_C_ascend, Meta, ops) {
  ops.impl("dispatch_ffn_combine", &vllm_fl_native::meta::dispatch_ffn_combine);
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
  ops.impl("moe_gating_top_k_hash",
           &vllm_fl_native::meta::moe_gating_top_k_hash);
  ops.impl("npu_dequant_swiglu_quant",
           &vllm_fl_native::meta::npu_dequant_swiglu_quant);
  ops.impl("npu_moe_init_routing_custom",
           &vllm_fl_native::meta::npu_moe_init_routing_custom);
  ops.impl("npu_hc_post", &vllm_fl_native::meta::npu_hc_post);
  ops.impl("npu_hc_pre", &vllm_fl_native::meta::npu_hc_pre);
  ops.impl("npu_hc_pre_v2", &vllm_fl_native::meta::npu_hc_pre_v2);
  ops.impl("npu_hc_pre_inv_rms", &vllm_fl_native::meta::npu_hc_pre_inv_rms);
  ops.impl("npu_hc_pre_sinkhorn", &vllm_fl_native::meta::npu_hc_pre_sinkhorn);
  ops.impl("compressor", &vllm_fl_native::meta::compressor);
  ops.impl("compressor_metadata", &vllm_fl_native::meta::compressor_metadata);
  ops.impl("npu_vllm_quant_lightning_indexer", &vllm_fl_native::meta::lightning_indexer);
  ops.impl("npu_sparse_attn_sharedkv", &vllm_fl_native::meta::dsa_sparse_attn);
  ops.impl("npu_sparse_attn_sharedkv_metadata", &vllm_fl_native::meta::sparse_attn_metadata);
  ops.impl("npu_vllm_quant_lightning_indexer_metadata", &vllm_fl_native::meta::lightning_indexer_metadata);
  ops.impl("inplace_partial_rotary_mul", &vllm_fl_native::meta::inplace_partial_rotary_mul);
  ops.impl("npu_rms_norm_dynamic_quant", &vllm_fl_native::meta::rms_norm_dynamic_quant);
  ops.impl("npu_scatter_nd_update_v2", &vllm_fl_native::meta::scatter_nd_update_v2);
}
