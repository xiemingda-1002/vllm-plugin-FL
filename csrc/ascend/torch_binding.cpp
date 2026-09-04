/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// FL-owned binding for the Ascend operators exercised by the supported model
// paths. Keeping the registration surface scoped avoids linking unrelated
// kernels into the standalone plugin.

#include <ATen/ATen.h>
#include <torch/library.h>

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#include "aclnn_torch_adapter/op_api_common.h"
#include "attention/recurrent_gated_delta_rule/recurrent_gated_delta_rule_torch_adpt.h"
#include "moe/add_rms_norm_bias/add_rms_norm_bias_torch_adpt.h"
#include "moe/moe_gating_top_k/moe_gating_top_k_torch_adpt.h"
#include "moe/moe_init_routing_custom/moe_init_routing_custom_torch_adpt.h"
#include "moe/apply_top_k_top_p_custom/apply_top_k_top_p_custom_torch_adpt.h"

namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor> npu_gemma_rms_norm(
    const at::Tensor& x,
    const at::Tensor& gamma,
    double epsilon)
{
    const int64_t diff = x.dim() - gamma.dim();
    std::vector<int64_t> rstd_shape;
    if (diff > 0) {
        rstd_shape.reserve(x.dim());
        for (int64_t i = 0; i < diff; ++i) {
            rstd_shape.push_back(x.size(i));
        }
        for (int64_t i = 0; i < gamma.dim(); ++i) {
            rstd_shape.push_back(1);
        }
    } else {
        rstd_shape.assign(x.dim(), 1);
    }
    at::Tensor rstd = at::empty(rstd_shape, x.options().dtype(at::kFloat));
    at::Tensor y = at::empty(x.sizes(), x.options());
    EXEC_NPU_CMD(aclnnGemmaRmsNorm, x, gamma, epsilon, y, rstd);
    return {y, rstd};
}

at::Tensor npu_causal_conv1d_custom(
    const at::Tensor& output,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& conv_state,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef query_start_loc,
    at::IntArrayRef cache_indices,
    at::IntArrayRef initial_state_mode,
    at::IntArrayRef num_accepted_tokens,
    int64_t activation_mode,
    int64_t pad_slot_id,
    int64_t run_mode)
{
    EXEC_NPU_CMD(
        aclnnCausalConv1d,
        x,
        weight,
        bias,
        conv_state,
        query_start_loc,
        cache_indices,
        initial_state_mode,
        num_accepted_tokens,
        activation_mode,
        pad_slot_id,
        run_mode,
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
    c10::optional<bool> transpose_state_layout)
{
    const bool output_final_state_value = output_final_state.value_or(false);
    const int64_t chunk_size_value = chunk_size.value_or(64);
    const at::Tensor& g_value = c10::value_or_else(
        g, [] { return at::Tensor(); });
    const at::Tensor& gk_value = c10::value_or_else(
        gk, [] { return at::Tensor(); });
    const at::Tensor& initial_state_value = c10::value_or_else(
        initial_state, [] { return at::Tensor(); });

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
    at::Tensor final_state_out;
    if (output_final_state_value) {
        const int64_t N = cu_seqlens.has_value()
            ? static_cast<int64_t>(cu_seqlens->size() - 1)
            : B;
        auto options = initial_state.has_value()
            ? initial_state->options()
            : h_out.options();
        final_state_out = at::empty({N, HV, K, V}, options);
    } else {
        final_state_out = at::empty({1}, k.options());
    }

    // EXEC_NPU_CMD's current adapter converts arguments through non-const
    // lvalue references. Keep optional defaults in named locals to match the
    // Ascend operator ABI contract.
    bool save_new_value_value = save_new_value.value_or(true);
    bool use_exp2_value = use_exp2.value_or(false);
    bool transpose_state_layout_value =
        transpose_state_layout.value_or(false);

    EXEC_NPU_CMD(
        aclnnChunkGatedDeltaRuleFwdH,
        k,
        w,
        u,
        g_value,
        gk_value,
        initial_state_value,
        output_final_state_value,
        chunk_size_value,
        save_new_value_value,
        cu_seqlens,
        chunk_indices,
        use_exp2_value,
        transpose_state_layout_value,
        h_out,
        v_new_out,
        final_state_out);

    return {
        h_out,
        v_new_out,
        output_final_state_value ? final_state_out : at::Tensor(),
    };
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
    c10::optional<bool> transpose_state_layout)
{
    at::Tensor output = at::zeros(v.sizes(), v.options());
    int64_t chunk_size_value = chunk_size.value_or(64);
    const at::Tensor& g_value = c10::value_or_else(
        g, [] { return at::Tensor(); });
    (void)g_gamma;
    (void)transpose_state_layout;
    EXEC_NPU_CMD(
        aclnnChunkFwdO,
        q,
        k,
        v,
        h,
        g_value,
        cu_seqlens,
        chunk_indices,
        scale,
        chunk_size_value,
        output);
    return output;
}

std::vector<bool> is_contiguous_axes(const at::Tensor& tensor)
{
    const auto sizes = tensor.sizes();
    const auto strides = tensor.strides();
    const int64_t ndim = sizes.size();
    if (ndim == 0) {
        return {};
    }

    std::vector<int64_t> contiguous_stride(ndim, 1);
    for (int64_t i = ndim - 2; i >= 0; --i) {
        contiguous_stride[i] = contiguous_stride[i + 1] * sizes[i + 1];
    }

    std::vector<bool> result(ndim, false);
    for (int64_t i = 0; i < ndim; ++i) {
        result[i] = strides[i] == contiguous_stride[i];
    }
    return result;
}

std::tuple<at::Tensor> compressor(
    const at::Tensor& x,
    const at::Tensor& wkv,
    const at::Tensor& wgate,
    at::Tensor& state_cache,
    const at::Tensor& ape,
    const at::Tensor& norm_weight,
    const at::Tensor& rope_sin,
    const at::Tensor& rope_cos,
    const c10::optional<at::Tensor>& state_block_table,
    const c10::optional<at::Tensor>& cu_seqlens,
    const c10::optional<at::Tensor>& seqused,
    const c10::optional<at::Tensor>& start_pos,
    int64_t rope_head_dim,
    int64_t cmp_ratio,
    int64_t coff,
    double norm_eps,
    int64_t rotary_mode,
    int64_t cache_mode)
{
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "compressor x must be 2D or 3D");
    TORCH_CHECK(norm_weight.dim() == 1, "compressor norm_weight must be 1D");
    TORCH_CHECK(rope_sin.dim() == x.dim(), "compressor rope_sin rank must match x");
    TORCH_CHECK(cmp_ratio > 0, "compressor cmp_ratio must be positive");
    TORCH_CHECK(state_cache.dim() == 3, "compressor state_cache must be 3D");

    const int64_t cmp_s = x.dim() == 3
        ? (x.size(1) + cmp_ratio - 1) / cmp_ratio
        : rope_sin.size(0);
    at::Tensor cmp_kv = x.dim() == 3
        ? at::empty({x.size(0), cmp_s, norm_weight.size(0)}, x.options())
        : at::empty({cmp_s, norm_weight.size(0)}, x.options());
    const int64_t state_cache_stride_dim0 = state_cache.stride(0);

    EXEC_NPU_CMD(
        aclnnCompressor,
        x,
        wkv,
        wgate,
        state_cache,
        ape,
        norm_weight,
        rope_sin,
        rope_cos,
        state_block_table,
        cu_seqlens,
        seqused,
        start_pos,
        rope_head_dim,
        cmp_ratio,
        coff,
        norm_eps,
        rotary_mode,
        cache_mode,
        state_cache_stride_dim0,
        cmp_kv);
    return {cmp_kv};
}

std::tuple<at::Tensor, at::Tensor> npu_quant_lightning_indexer(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& weights,
    const at::Tensor& query_dequant_scale,
    const at::Tensor& key_dequant_scale,
    int64_t query_quant_mode,
    int64_t key_quant_mode,
    const c10::optional<at::Tensor>& actual_seq_lengths_query,
    const c10::optional<at::Tensor>& actual_seq_lengths_key,
    const c10::optional<at::Tensor>& block_table,
    const c10::optional<at::Tensor>& metadata,
    c10::string_view layout_query,
    c10::string_view layout_key,
    int64_t sparse_count,
    int64_t sparse_mode,
    int64_t pre_tokens,
    int64_t next_tokens,
    int64_t cmp_ratio,
    bool return_value)
{
    TORCH_CHECK(sparse_count > 0, "sparse_count must be positive");
    const std::string query_layout(layout_query);
    const std::string key_layout(layout_key);
    const int64_t key_heads = key_layout == "TND" ? key.size(1) : key.size(2);
    at::SmallVector<int64_t, 4> output_shape;
    if (query_layout == "BSND") {
        output_shape = {query.size(0), query.size(1), key_heads, sparse_count};
    } else {
        output_shape = {query.size(0), key_heads, sparse_count};
    }
    at::Tensor sparse_indices = at::empty(output_shape, query.options().dtype(at::kInt));
    at::Tensor sparse_values = return_value
        ? at::empty(output_shape, query.options().dtype(at::kFloat))
        : at::empty({0}, query.options().dtype(at::kFloat));
    const int64_t key_stride = key.stride(0);
    const int64_t scale_stride = key_dequant_scale.stride(0);
    char* query_layout_ptr = const_cast<char*>(query_layout.c_str());
    char* key_layout_ptr = const_cast<char*>(key_layout.c_str());

    if (key_layout == "PA_BSND") {
        const auto key_axes = is_contiguous_axes(key);
        const auto scale_axes = is_contiguous_axes(key_dequant_scale);
        TORCH_CHECK(key_axes[1] && key_axes[2], "key must be contiguous except axis 0");
        TORCH_CHECK(scale_axes[1] && scale_axes[2], "key scale must be contiguous except axis 0");
    }

    EXEC_NPU_CMD(
        aclnnQuantLightningIndexer,
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table,
        metadata,
        query_quant_mode,
        key_quant_mode,
        query_layout_ptr,
        key_layout_ptr,
        sparse_count,
        sparse_mode,
        pre_tokens,
        next_tokens,
        cmp_ratio,
        return_value,
        key_stride,
        scale_stride,
        sparse_indices,
        sparse_values);
    return {sparse_indices, sparse_values};
}

std::tuple<at::Tensor, at::Tensor> npu_sparse_attn_sharedkv(
    const at::Tensor& q,
    const c10::optional<at::Tensor>& ori_kv,
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
    const c10::optional<at::Tensor>& sinks,
    const c10::optional<at::Tensor>& metadata,
    double softmax_scale,
    int64_t cmp_ratio,
    int64_t ori_mask_mode,
    int64_t cmp_mask_mode,
    int64_t ori_win_left,
    int64_t ori_win_right,
    c10::string_view layout_q,
    c10::string_view layout_kv,
    bool return_softmax_lse)
{
    at::Tensor output = at::empty(q.sizes(), q.options());
    std::vector<int64_t> lse_shape(q.sizes().begin(), q.sizes().end());
    lse_shape.back() = 1;
    at::Tensor softmax_lse = return_softmax_lse
        ? at::empty(lse_shape, q.options().dtype(at::kFloat))
        : at::empty({0}, q.options().dtype(at::kFloat));
    const int64_t ori_kv_stride = ori_kv.has_value() ? ori_kv->stride(0) : 0;
    const int64_t cmp_kv_stride = cmp_kv.has_value() ? cmp_kv->stride(0) : 0;
    const std::string q_layout(layout_q);
    const std::string kv_layout(layout_kv);
    char* q_layout_ptr = const_cast<char*>(q_layout.c_str());
    char* kv_layout_ptr = const_cast<char*>(kv_layout.c_str());

    EXEC_NPU_CMD(
        aclnnSparseAttnSharedkv,
        q,
        ori_kv,
        cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        ori_block_table,
        cmp_block_table,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        seqused_q,
        seqused_kv,
        sinks,
        metadata,
        softmax_scale,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_kv_stride,
        cmp_kv_stride,
        ori_win_left,
        ori_win_right,
        q_layout_ptr,
        kv_layout_ptr,
        return_softmax_lse,
        output,
        softmax_lse);
    return {output, softmax_lse};
}

at::Tensor optional_int_tensor(
    const c10::optional<at::Tensor>& value,
    const at::Device& device)
{
    return value.has_value()
        ? *value
        : at::empty({0}, at::TensorOptions().dtype(at::kInt).device(device));
}

at::Tensor npu_sparse_attn_sharedkv_metadata(
    int64_t num_heads_q,
    int64_t num_heads_kv,
    int64_t head_dim,
    const c10::optional<at::Tensor>& cu_seqlens_q,
    const c10::optional<at::Tensor>& cu_seqlens_ori_kv,
    const c10::optional<at::Tensor>& cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor>& seqused_q,
    const c10::optional<at::Tensor>& seqused_kv,
    int64_t batch_size,
    int64_t max_seqlen_q,
    int64_t max_seqlen_kv,
    int64_t ori_topk,
    int64_t cmp_topk,
    int64_t cmp_ratio,
    int64_t ori_mask_mode,
    int64_t cmp_mask_mode,
    int64_t ori_win_left,
    int64_t ori_win_right,
    c10::string_view layout_q,
    c10::string_view layout_kv,
    bool has_ori_kv,
    bool has_cmp_kv,
    c10::string_view device)
{
    at::Device output_device{std::string(device)};
    for (const auto* value : {&cu_seqlens_q, &cu_seqlens_ori_kv, &cu_seqlens_cmp_kv, &seqused_q, &seqused_kv}) {
        if (value->has_value()) {
            output_device = value->value().device();
            break;
        }
    }
    at::Tensor output = at::empty(
        {1024}, at::TensorOptions().dtype(at::kInt).device(output_device));
    at::Tensor q_seq = optional_int_tensor(cu_seqlens_q, output_device);
    at::Tensor ori_seq = optional_int_tensor(cu_seqlens_ori_kv, output_device);
    at::Tensor cmp_seq = optional_int_tensor(cu_seqlens_cmp_kv, output_device);
    at::Tensor used_q = optional_int_tensor(seqused_q, output_device);
    at::Tensor used_kv = optional_int_tensor(seqused_kv, output_device);
    const std::string q_layout(layout_q);
    const std::string kv_layout(layout_kv);
    char* q_layout_ptr = const_cast<char*>(q_layout.c_str());
    char* kv_layout_ptr = const_cast<char*>(kv_layout.c_str());
    EXEC_NPU_CMD(
        aclnnSparseAttnSharedkvMetadata,
        q_seq,
        ori_seq,
        cmp_seq,
        used_q,
        used_kv,
        num_heads_q,
        num_heads_kv,
        head_dim,
        batch_size,
        max_seqlen_q,
        max_seqlen_kv,
        ori_topk,
        cmp_topk,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        q_layout_ptr,
        kv_layout_ptr,
        has_ori_kv,
        has_cmp_kv,
        output);
    return output;
}

at::Tensor npu_quant_lightning_indexer_metadata(
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t head_dim,
    int64_t query_quant_mode,
    int64_t key_quant_mode,
    const c10::optional<at::Tensor>& actual_seq_lengths_query,
    const c10::optional<at::Tensor>& actual_seq_lengths_key,
    int64_t batch_size,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    c10::string_view layout_query,
    c10::string_view layout_key,
    int64_t sparse_count,
    int64_t sparse_mode,
    int64_t pre_tokens,
    int64_t next_tokens,
    int64_t cmp_ratio,
    c10::string_view device)
{
    at::Device output_device{std::string(device)};
    if (actual_seq_lengths_query.has_value()) {
        output_device = actual_seq_lengths_query->device();
    } else if (actual_seq_lengths_key.has_value()) {
        output_device = actual_seq_lengths_key->device();
    }
    at::Tensor output = at::empty(
        {1024}, at::TensorOptions().dtype(at::kInt).device(output_device));
    at::Tensor query_seq = optional_int_tensor(actual_seq_lengths_query, output_device);
    at::Tensor key_seq = optional_int_tensor(actual_seq_lengths_key, output_device);
    const std::string query_layout(layout_query);
    const std::string key_layout(layout_key);
    char* query_layout_ptr = const_cast<char*>(query_layout.c_str());
    char* key_layout_ptr = const_cast<char*>(key_layout.c_str());
    EXEC_NPU_CMD(
        aclnnQuantLightningIndexerMetadata,
        query_seq,
        key_seq,
        num_heads_q,
        num_heads_k,
        head_dim,
        query_quant_mode,
        key_quant_mode,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        query_layout_ptr,
        key_layout_ptr,
        sparse_count,
        sparse_mode,
        pre_tokens,
        next_tokens,
        cmp_ratio,
        output);
    return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_hc_pre(
    const at::Tensor& x,
    const at::Tensor& hc_fn,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    int64_t hc_mult,
    int64_t hc_sinkhorn_iters,
    double norm_eps,
    double hc_eps)
{
    TORCH_CHECK(x.dim() == 3 || x.dim() == 4, "npu_hc_pre x must be 3D or 4D");
    TORCH_CHECK(x.dtype() == at::kBFloat16, "npu_hc_pre x must be bfloat16");
    TORCH_CHECK(hc_mult == 4, "npu_hc_pre only supports hc_mult=4");
    const int64_t batch = x.size(0);
    const int64_t sequence = x.dim() == 4 ? x.size(1) : 0;
    const int64_t hidden = x.size(-1);
    at::SmallVector<int64_t, 4> y_shape;
    at::SmallVector<int64_t, 4> post_shape;
    at::SmallVector<int64_t, 4> comb_shape;
    if (x.dim() == 4) {
        y_shape = {batch, sequence, hidden};
        post_shape = {batch, sequence, hc_mult};
        comb_shape = {batch, sequence, hc_mult, hc_mult};
    } else {
        y_shape = {batch, hidden};
        post_shape = {batch, hc_mult};
        comb_shape = {batch, hc_mult, hc_mult};
    }

    std::vector<int64_t> rsqrt_shape;
    for (int64_t i = 0; i < x.dim() - 2; ++i) {
        rsqrt_shape.push_back(x.size(i));
    }
    rsqrt_shape.push_back(1);
    at::Tensor rsqrt = at::empty(rsqrt_shape, x.options().dtype(at::kFloat));
    EXEC_NPU_CMD(aclnnHcPreInvRms, x, norm_eps, rsqrt);

    at::Tensor x_float = x.to(at::kFloat);
    at::Tensor x_flattened = x.dim() == 3
        ? x_float.flatten(1, -1)
        : x_float.flatten(2, -1);
    at::Tensor mixes = at::linear(x_flattened, hc_fn);
    at::Tensor y = at::empty(y_shape, x.options().dtype(at::kBFloat16));
    at::Tensor post = at::empty(post_shape, x.options().dtype(at::kFloat));
    at::Tensor comb = at::empty(comb_shape, x.options().dtype(at::kFloat));
    EXEC_NPU_CMD(
        aclnnHcPreSinkhorn,
        mixes,
        rsqrt,
        hc_scale,
        hc_base,
        x,
        hc_mult,
        hc_sinkhorn_iters,
        hc_eps,
        y,
        post,
        comb);
    return {y, post, comb};
}

at::Tensor npu_hc_post(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& post,
    const at::Tensor& comb)
{
    at::Tensor output = at::empty(residual.sizes(), residual.options());
    EXEC_NPU_CMD(aclnnHcPost, x, residual, post, comb, output);
    return output;
}

void inplace_partial_rotary_mul(
    at::Tensor& x,
    const at::Tensor& r1,
    const at::Tensor& r2,
    c10::string_view rotary_mode,
    at::IntArrayRef partial_slice)
{
    static const std::unordered_map<std::string, int64_t> mode_map = {
        {"half", 0},
        {"interleave", 1},
        {"quarter", 2},
        {"interleave-half", 3},
    };
    const auto mode = mode_map.find(std::string(rotary_mode));
    TORCH_CHECK(mode != mode_map.end(), "unsupported rotary_mode");
    TORCH_CHECK(x.dim() == 4, "inplace_partial_rotary_mul x must be 4D");
    EXEC_NPU_CMD(aclnnInplacePartialRotaryMul, x, r1, r2, mode->second, partial_slice);
}

std::tuple<at::Tensor, at::Tensor> npu_rms_norm_dynamic_quant(
    const at::Tensor& x,
    const at::Tensor& gamma,
    const c10::optional<at::Tensor>& smooth_scale,
    const c10::optional<at::Tensor>& beta,
    double epsilon)
{
    TORCH_CHECK(x.numel() > 0 && gamma.numel() > 0, "rms norm inputs must be non-empty");
    at::Tensor smooth_scale2;
    at::Tensor y = at::empty_like(x, x.options().dtype(at::kChar));
    at::Tensor y2 = at::empty({1}, x.options().dtype(at::kChar));
    std::vector<int64_t> scale_shape(x.sizes().begin(), x.sizes().end() - 1);
    at::Tensor scale = at::empty(scale_shape, x.options().dtype(at::kFloat));
    at::Tensor scale2 = at::empty_like(scale);
    std::array<bool, 2>* output_mask = nullptr;
    int64_t* dst_type = nullptr;
    EXEC_NPU_CMD(
        aclnnRmsNormDynamicQuant,
        x,
        gamma,
        smooth_scale,
        smooth_scale2,
        beta,
        epsilon,
        output_mask,
        dst_type,
        y,
        y2,
        scale,
        scale2);
    return {y, scale};
}

void npu_scatter_nd_update_v2(
    at::Tensor& var,
    const at::Tensor& indices,
    const at::Tensor& update)
{
    at::IntArrayRef var_stride = var.strides();
    EXEC_NPU_CMD(aclnnScatterNdUpdateV2, var, indices, update, var_stride);
}

namespace meta {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_moe_init_routing_custom(
    const at::Tensor& x,
    const at::Tensor& expert_idx,
    const c10::optional<at::Tensor>& scale,
    const c10::optional<at::Tensor>& offset,
    int64_t active_num,
    int64_t expert_capacity,
    int64_t expert_num,
    int64_t drop_pad_mode,
    int64_t expert_tokens_num_type,
    bool expert_tokens_num_flag,
    int64_t quant_mode,
    at::IntArrayRef active_expert_range,
    int64_t row_idx_type)
{
    (void)scale;
    (void)offset;
    (void)expert_tokens_num_flag;
    (void)row_idx_type;
    const c10::SymInt batch = x.sym_size(0);
    const c10::SymInt hidden = x.sym_size(1);
    const c10::SymInt top_k = expert_idx.sym_size(1);
    const c10::SymInt expanded_tokens = active_num > 0
        ? c10::SymInt(active_num)
        : batch * top_k;
    c10::SymDimVector expanded_shape;
    if (drop_pad_mode == 1) {
        expanded_shape = {
            c10::SymInt(expert_num), c10::SymInt(expert_capacity), hidden};
    } else {
        expanded_shape = {expanded_tokens, hidden};
    }
    auto expanded_options = quant_mode == -1
        ? x.options()
        : x.options().dtype(at::kChar);
    const int64_t first_expert = active_expert_range.empty()
        ? 0
        : active_expert_range[0];
    const int64_t last_expert = active_expert_range.empty()
        ? expert_num
        : active_expert_range[1];
    c10::SymDimVector expert_tokens_shape = expert_tokens_num_type == 2
        ? c10::SymDimVector{
              c10::SymInt(expert_num), c10::SymInt(2)}
        : c10::SymDimVector{
              c10::SymInt(last_expert - first_expert)};
    return {
        at::empty_symint(expanded_shape, expanded_options),
        at::empty_symint({batch * top_k}, expert_idx.options()),
        at::empty_symint(
            expert_tokens_shape, x.options().dtype(at::kLong)),
        at::empty_symint(
            {drop_pad_mode == 1
                 ? c10::SymInt(expert_num * expert_capacity)
                 : expanded_tokens},
            x.options().dtype(at::kFloat)),
    };
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> moe_gating_top_k(
    const at::Tensor& x,
    int64_t k,
    int64_t k_group,
    int64_t group_count,
    int64_t group_select_mode,
    int64_t renorm,
    int64_t norm_type,
    bool out_flag,
    double routed_scaling_factor,
    double eps,
    const c10::optional<at::Tensor>& bias_opt)
{
    (void)k_group;
    (void)group_count;
    (void)group_select_mode;
    (void)renorm;
    (void)norm_type;
    (void)out_flag;
    (void)routed_scaling_factor;
    (void)eps;
    (void)bias_opt;
    const c10::SymInt rows = x.sym_size(0);
    const c10::SymInt experts = x.sym_size(1);
    return {
        at::empty_symint({rows, c10::SymInt(k)}, x.options()),
        at::empty_symint(
            {rows, c10::SymInt(k)}, x.options().dtype(at::kInt)),
        at::empty_symint(
            {rows, experts}, x.options().dtype(at::kFloat)),
    };
}

std::tuple<at::Tensor, at::Tensor> npu_gemma_rms_norm(
    const at::Tensor& x,
    const at::Tensor& gamma,
    double epsilon)
{
    (void)epsilon;
    c10::SymDimVector rstd_shape;
    const int64_t diff = x.dim() - gamma.dim();
    if (diff > 0) {
        rstd_shape.reserve(x.dim());
        for (int64_t i = 0; i < diff; ++i) {
            rstd_shape.push_back(x.sym_size(i));
        }
        for (int64_t i = 0; i < gamma.dim(); ++i) {
            rstd_shape.push_back(c10::SymInt(1));
        }
    } else {
        rstd_shape.assign(x.dim(), c10::SymInt(1));
    }
    return {
        at::empty_symint(x.sym_sizes(), x.options()),
        at::empty_symint(rstd_shape, x.options().dtype(at::kFloat)),
    };
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_add_rms_norm_bias(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& gamma,
    const c10::optional<at::Tensor>& beta,
    double epsilon)
{
    (void)x2;
    (void)beta;
    (void)epsilon;
    c10::SymDimVector rstd_shape;
    const int64_t diff = x1.dim() - gamma.dim();
    if (diff > 0) {
        rstd_shape.reserve(x1.dim());
        for (int64_t i = 0; i < diff; ++i) {
            rstd_shape.push_back(x1.sym_size(i));
        }
        for (int64_t i = 0; i < gamma.dim(); ++i) {
            rstd_shape.push_back(c10::SymInt(1));
        }
    } else {
        rstd_shape.assign(x1.dim(), c10::SymInt(1));
    }
    return {
        at::empty_symint(x1.sym_sizes(), x1.options()),
        at::empty_symint(rstd_shape, x1.options().dtype(at::kFloat)),
        at::empty_symint(x1.sym_sizes(), x1.options()),
    };
}

at::Tensor npu_recurrent_gated_delta_rule(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& state,
    const c10::optional<at::Tensor>& beta,
    const c10::optional<double> scale,
    const c10::optional<at::Tensor>& actual_seq_lengths,
    const c10::optional<at::Tensor>& ssm_state_indices,
    const c10::optional<at::Tensor>& num_accepted_tokens,
    const c10::optional<at::Tensor>& g,
    const c10::optional<at::Tensor>& gk)
{
    (void)query;
    (void)key;
    (void)state;
    (void)beta;
    (void)scale;
    (void)actual_seq_lengths;
    (void)ssm_state_indices;
    (void)num_accepted_tokens;
    (void)g;
    (void)gk;
    return at::empty_symint(
        value.sym_sizes(), value.options().dtype(at::kBFloat16));
}

at::Tensor npu_causal_conv1d_custom(
    const at::Tensor& output,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& conv_state,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef query_start_loc,
    at::IntArrayRef cache_indices,
    at::IntArrayRef initial_state_mode,
    at::IntArrayRef num_accepted_tokens,
    int64_t activation_mode,
    int64_t pad_slot_id,
    int64_t run_mode)
{
    (void)x;
    (void)weight;
    (void)conv_state;
    (void)bias;
    (void)query_start_loc;
    (void)cache_indices;
    (void)initial_state_mode;
    (void)num_accepted_tokens;
    (void)activation_mode;
    (void)pad_slot_id;
    (void)run_mode;
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
    c10::optional<bool> transpose_state_layout)
{
    (void)w;
    (void)g;
    (void)gk;
    (void)save_new_value;
    (void)use_exp2;
    (void)transpose_state_layout;
    const int64_t B = k.size(0);
    const int64_t K = k.size(3);
    const int64_t T = k.size(2);
    const int64_t HV = u.size(1);
    const int64_t V = u.size(3);
    const int64_t chunk_size_value = chunk_size.value_or(64);
    const int64_t NT = chunk_indices.has_value()
        ? static_cast<int64_t>(chunk_indices->size() / 2)
        : (T + chunk_size_value - 1) / chunk_size_value;
    at::Tensor h = at::empty({B, HV, NT, K, V}, k.options());
    at::Tensor v_new = at::empty(u.sizes(), u.options());
    if (!output_final_state.value_or(false)) {
        return {h, v_new, at::Tensor()};
    }
    const int64_t N = cu_seqlens.has_value()
        ? static_cast<int64_t>(cu_seqlens->size() - 1)
        : B;
    auto options = initial_state.has_value()
        ? initial_state->options()
        : h.options();
    return {h, v_new, at::empty({N, HV, K, V}, options)};
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
    c10::optional<bool> transpose_state_layout)
{
    (void)q;
    (void)k;
    (void)h;
    (void)scale;
    (void)g;
    (void)g_gamma;
    (void)cu_seqlens;
    (void)chunk_indices;
    (void)chunk_size;
    (void)transpose_state_layout;
    return at::empty(v.sizes(), v.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_hc_pre(
    const at::Tensor& x,
    const at::Tensor& hc_fn,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    int64_t hc_mult,
    int64_t hc_sinkhorn_iters,
    double norm_eps,
    double hc_eps)
{
    (void)hc_fn;
    (void)hc_scale;
    (void)hc_base;
    (void)hc_sinkhorn_iters;
    (void)norm_eps;
    (void)hc_eps;

    c10::SymDimVector y_shape;
    c10::SymDimVector post_shape;
    c10::SymDimVector comb_shape;
    if (x.dim() == 4) {
        const c10::SymInt batch = x.sym_size(0);
        const c10::SymInt sequence = x.sym_size(1);
        const c10::SymInt hidden = x.sym_size(3);
        y_shape = {batch, sequence, hidden};
        post_shape = {batch, sequence, c10::SymInt(hc_mult)};
        comb_shape = {
            batch,
            sequence,
            c10::SymInt(hc_mult),
            c10::SymInt(hc_mult),
        };
    } else {
        const c10::SymInt batch = x.sym_size(0);
        const c10::SymInt hidden = x.sym_size(2);
        y_shape = {batch, hidden};
        post_shape = {batch, c10::SymInt(hc_mult)};
        comb_shape = {
            batch,
            c10::SymInt(hc_mult),
            c10::SymInt(hc_mult),
        };
    }
    return {
        at::empty_symint(y_shape, x.options().dtype(at::kBFloat16)),
        at::empty_symint(post_shape, x.options().dtype(at::kFloat)),
        at::empty_symint(comb_shape, x.options().dtype(at::kFloat)),
    };
}

at::Tensor npu_hc_post(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& post,
    const at::Tensor& comb)
{
    (void)x;
    (void)post;
    (void)comb;
    return at::empty_symint(residual.sym_sizes(), residual.options());
}

}  // namespace meta
}  // namespace vllm_ascend

TORCH_LIBRARY(_C_ascend, ops) {
    ops.def(
        "npu_moe_init_routing_custom(Tensor x, Tensor expert_idx, *, "
        "Tensor? scale=None, Tensor? offset=None, int active_num=-1, "
        "int expert_capacity=-1, int expert_num=-1, int drop_pad_mode=0, "
        "int expert_tokens_num_type=0, bool expert_tokens_num_flag=False, "
        "int quant_mode=0, int[2] active_expert_range=[], "
        "int row_idx_type=0) -> (Tensor, Tensor, Tensor, Tensor)");
    ops.def(
        "npu_apply_top_k_top_p(Tensor logits, Tensor? p=None, Tensor? k=None) -> Tensor");
    ops.def(
        "moe_gating_top_k(Tensor x, int k, int k_group, int group_count, "
        "int group_select_mode, int renorm, int norm_type, bool out_flag, "
        "float routed_scaling_factor, float eps, Tensor? bias_opt=None) "
        "-> (Tensor y, Tensor expert_idx, Tensor out)");
    ops.def(
        "npu_gemma_rms_norm(Tensor x, Tensor gamma, float epsilon=1e-6) "
        "-> (Tensor y, Tensor rstd)");
    ops.def(
        "npu_add_rms_norm_bias(Tensor x1, Tensor x2, Tensor gamma, "
        "Tensor? beta=None, float epsilon=1e-6) "
        "-> (Tensor y, Tensor rstd, Tensor x)");
    ops.def(
        "npu_recurrent_gated_delta_rule(Tensor query, Tensor key, "
        "Tensor value, Tensor(a!) state, *, Tensor? beta=None, "
        "float? scale=None, Tensor? actual_seq_lengths=None, "
        "Tensor? ssm_state_indices=None, Tensor? num_accepted_tokens=None, "
        "Tensor? g=None, Tensor? gk=None) -> Tensor");
    ops.def(
        "npu_causal_conv1d_custom(Tensor output, Tensor x, Tensor weight, "
        "Tensor conv_state, Tensor? bias_opt, int[] query_start_loc_opt, "
        "int[] cache_indices_opt, int[] initial_state_mode_opt, "
        "int[] num_accepted_tokens_opt, int activation_mode, "
        "int pad_slot_id, int run_mode) -> Tensor");
    ops.def(
        "chunk_gated_delta_rule_fwd_h(Tensor k, Tensor w, Tensor u, "
        "Tensor? g=None, *, Tensor? gk=None, Tensor? initial_state=None, "
        "bool? output_final_state=False, int? chunk_size=None, "
        "bool? save_new_value=True, int[]? cu_seqlens=None, "
        "int[]? chunk_indices=None, bool? use_exp2=False, "
        "bool? transpose_state_layout=False) "
        "-> (Tensor h_out, Tensor v_new_out, Tensor final_state_out)");
    ops.def(
        "chunk_fwd_o(Tensor q, Tensor k, Tensor v, Tensor h, float scale, "
        "*, Tensor? g=None, Tensor? g_gamma=None, int[]? cu_seqlens=None, "
        "int[]? chunk_indices=None, int? chunk_size=None, "
        "bool? transpose_state_layout=False) -> Tensor");
    ops.def(
        "compressor(Tensor x, Tensor wkv, Tensor wgate, Tensor(a!) state_cache, "
        "Tensor ape, Tensor norm_weight, Tensor rope_sin, Tensor rope_cos, "
        "Tensor? state_block_table, Tensor? cu_seqlens, Tensor? seqused, "
        "Tensor? start_pos, int rope_head_dim, int cmp_ratio, int coff, "
        "float norm_eps, int rotary_mode, int cache_mode) -> Tensor");
    ops.def(
        "npu_quant_lightning_indexer(Tensor query, Tensor key, Tensor weights, "
        "Tensor query_dequant_scale, Tensor key_dequant_scale, "
        "int query_quant_mode=0, int key_quant_mode=0, "
        "Tensor? actual_seq_lengths_query=None, Tensor? actual_seq_lengths_key=None, "
        "Tensor? block_table=None, Tensor? metadata=None, "
        "str layout_query=\"BSND\", str layout_key=\"BSND\", "
        "int sparse_count=2048, int sparse_mode=3, "
        "int pre_tokens=9223372036854775807, int next_tokens=9223372036854775807, "
        "int cmp_ratio=1, bool return_value=False) -> (Tensor, Tensor)");
    ops.def(
        "npu_sparse_attn_sharedkv(Tensor q, *, Tensor? ori_kv=None, "
        "Tensor? cmp_kv=None, Tensor? ori_sparse_indices=None, "
        "Tensor? cmp_sparse_indices=None, Tensor? ori_block_table=None, "
        "Tensor? cmp_block_table=None, Tensor? cu_seqlens_q=None, "
        "Tensor? cu_seqlens_ori_kv=None, Tensor? cu_seqlens_cmp_kv=None, "
        "Tensor? seqused_q=None, Tensor? seqused_kv=None, Tensor? sinks=None, "
        "Tensor? metadata=None, float softmax_scale=0, int cmp_ratio=0, "
        "int ori_mask_mode=4, int cmp_mask_mode=3, int ori_win_left=128, "
        "int ori_win_right=0, str layout_q=\"BSND\", str layout_kv=\"PA_ND\", "
        "bool return_softmax_lse=False) -> (Tensor, Tensor)");
    ops.def(
        "npu_sparse_attn_sharedkv_metadata(int num_heads_q, int num_heads_kv, "
        "int head_dim, Tensor? cu_seqlens_q=None, "
        "Tensor? cu_seqlens_ori_kv=None, Tensor? cu_seqlens_cmp_kv=None, "
        "Tensor? seqused_q=None, Tensor? seqused_kv=None, int batch_size=0, "
        "int max_seqlen_q=0, int max_seqlen_kv=0, int ori_topk=0, "
        "int cmp_topk=0, int cmp_ratio=4, int ori_mask_mode=4, "
        "int cmp_mask_mode=3, int ori_win_left=128, int ori_win_right=0, "
        "str layout_q=\"BSND\", str layout_kv=\"PA_ND\", bool has_ori_kv=True, "
        "bool has_cmp_kv=True, str device=\"npu\") -> Tensor");
    ops.def(
        "npu_quant_lightning_indexer_metadata(int num_heads_q, int num_heads_k, "
        "int head_dim, int query_quant_mode, int key_quant_mode, "
        "Tensor? actual_seq_lengths_query=None, Tensor? actual_seq_lengths_key=None, "
        "int batch_size=0, int max_seqlen_q=0, int max_seqlen_k=0, "
        "str layout_query=\"BSND\", str layout_key=\"BSND\", "
        "int sparse_count=2048, int sparse_mode=3, "
        "int pre_tokens=9223372036854775807, int next_tokens=9223372036854775807, "
        "int cmp_ratio=1, str device=\"npu\") -> Tensor");
    ops.def(
        "npu_hc_pre(Tensor x, Tensor hc_fn, Tensor hc_scale, Tensor hc_base, "
        "int hc_mult, int hc_sinkhorn_iters, float norm_eps, float hc_eps) "
        "-> (Tensor, Tensor, Tensor)");
    ops.def("npu_hc_post(Tensor x, Tensor residual, Tensor post, Tensor comb) -> Tensor");
    ops.def(
        "inplace_partial_rotary_mul(Tensor(a!) x, Tensor r1, Tensor r2, "
        "str rotary_mode, int[] partial_slice) -> ()");
    ops.def(
        "npu_rms_norm_dynamic_quant(Tensor x, Tensor gamma, "
        "Tensor? smooth_scale=None, Tensor? beta=None, float epsilon=1e-6) "
        "-> (Tensor, Tensor)");
    ops.def(
        "npu_scatter_nd_update_v2(Tensor(a!) var, Tensor indices, Tensor update) -> ()");
}

TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, ops) {
    ops.impl(
        "npu_moe_init_routing_custom",
        &vllm_ascend::npu_moe_init_routing_custom);
    ops.impl("npu_apply_top_k_top_p", &vllm_ascend::npu_apply_top_k_top_p);
    ops.impl("moe_gating_top_k", &vllm_ascend::moe_gating_top_k);
    ops.impl("npu_gemma_rms_norm", &vllm_ascend::npu_gemma_rms_norm);
    ops.impl("npu_add_rms_norm_bias", &vllm_ascend::npu_add_rms_norm_bias);
    ops.impl(
        "npu_recurrent_gated_delta_rule",
        &vllm_ascend::npu_recurrent_gated_delta_rule);
    ops.impl(
        "npu_causal_conv1d_custom",
        &vllm_ascend::npu_causal_conv1d_custom);
    ops.impl(
        "chunk_gated_delta_rule_fwd_h",
        &vllm_ascend::chunk_gated_delta_rule_fwd_h);
    ops.impl("chunk_fwd_o", &vllm_ascend::chunk_fwd_o);
    ops.impl("compressor", &vllm_ascend::compressor);
    ops.impl(
        "npu_quant_lightning_indexer",
        &vllm_ascend::npu_quant_lightning_indexer);
    ops.impl(
        "npu_sparse_attn_sharedkv",
        &vllm_ascend::npu_sparse_attn_sharedkv);
    ops.impl(
        "npu_sparse_attn_sharedkv_metadata",
        &vllm_ascend::npu_sparse_attn_sharedkv_metadata);
    ops.impl(
        "npu_quant_lightning_indexer_metadata",
        &vllm_ascend::npu_quant_lightning_indexer_metadata);
    ops.impl("npu_hc_pre", &vllm_ascend::npu_hc_pre);
    ops.impl("npu_hc_post", &vllm_ascend::npu_hc_post);
    ops.impl(
        "inplace_partial_rotary_mul",
        &vllm_ascend::inplace_partial_rotary_mul);
    ops.impl(
        "npu_rms_norm_dynamic_quant",
        &vllm_ascend::npu_rms_norm_dynamic_quant);
    ops.impl(
        "npu_scatter_nd_update_v2",
        &vllm_ascend::npu_scatter_nd_update_v2);
}

TORCH_LIBRARY_IMPL(_C_ascend, Meta, ops) {
    ops.impl(
        "npu_moe_init_routing_custom",
        &vllm_ascend::meta::npu_moe_init_routing_custom);
    ops.impl("moe_gating_top_k", &vllm_ascend::meta::moe_gating_top_k);
    ops.impl("npu_gemma_rms_norm", &vllm_ascend::meta::npu_gemma_rms_norm);
    ops.impl(
        "npu_add_rms_norm_bias",
        &vllm_ascend::meta::npu_add_rms_norm_bias);
    ops.impl(
        "npu_recurrent_gated_delta_rule",
        &vllm_ascend::meta::npu_recurrent_gated_delta_rule);
    ops.impl(
        "npu_causal_conv1d_custom",
        &vllm_ascend::meta::npu_causal_conv1d_custom);
    ops.impl(
        "chunk_gated_delta_rule_fwd_h",
        &vllm_ascend::meta::chunk_gated_delta_rule_fwd_h);
    ops.impl("chunk_fwd_o", &vllm_ascend::meta::chunk_fwd_o);
    ops.impl("npu_hc_pre", &vllm_ascend::meta::npu_hc_pre);
    ops.impl("npu_hc_post", &vllm_ascend::meta::npu_hc_post);
}
