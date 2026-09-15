/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Adaptations Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OF CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef DISPATCH_FFN_COMBINE_TORCH_ADPT_H
#define DISPATCH_FFN_COMBINE_TORCH_ADPT_H

namespace vllm_fl_native {
std::tuple<at::Tensor&, at::Tensor&> dispatch_ffn_combine(
    const at::Tensor& x,
    const at::TensorList& weight1,
    const at::TensorList& weight2,
    const at::Tensor& expert_idx,
    const at::TensorList& scale1,
    const at::TensorList& scale2,
    const at::TensorList& bias1,
    const at::TensorList& bias2,
    const at::Tensor& probs,
    c10::string_view group,
    int64_t max_output_size,
    at::Tensor& out,
    at::Tensor& expert_token_nums,
    const c10::optional<at::Tensor>& x_active_mask,
    double swiglu_limit
) {
    TORCH_CHECK(!weight1.empty() && !weight2.empty(),
                "dispatch_ffn_combine requires non-empty W8A8 weight lists");
    for (const auto& weight : weight1) {
        TORCH_CHECK(weight.scalar_type() == at::kChar,
                    "FL dispatch_ffn_combine supports W8A8 int8 weight1 only");
    }
    for (const auto& weight : weight2) {
        TORCH_CHECK(weight.scalar_type() == at::kChar,
                    "FL dispatch_ffn_combine supports W8A8 int8 weight2 only");
    }
    (void)bias1;
    (void)bias2;
    char *group_ep_ptr = const_cast<char *>(group.data());
    // Deliberately do not retain rc1's W4A8/BF16 branches: their op-api
    // symbols are outside this A3 W8A8 selected-native payload.
    EXEC_NPU_CMD(aclnnDispatchFFNCombine,
                 x,
                 weight1,
                 weight2,
                 expert_idx,
                 scale1,
                 scale2,
                 probs,
                 x_active_mask.has_value() ? x_active_mask.value() : at::Tensor(),
                 group_ep_ptr,
                 max_output_size,
                 swiglu_limit,
                 out,
                 expert_token_nums);
    return {out, expert_token_nums};
}
}
#endif
