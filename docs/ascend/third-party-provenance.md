# Ascend 第三方来源与协议事实（非合规结论）

本文仅记录本 checkpoint 已知的第三方来源和未证实项；不修改原有 notice、license 文本、文件头或
打包 metadata，也不构成许可证合规认证、法律意见或分发批准。

## vLLM 与 vLLM-Ascend

迁移以 vLLM 0.24.0 和 vLLM-Ascend 0.24.0rc1 为源码基线。适配的 Python/C++ 来源保留其原 attribution。
上游仓根为 Apache-2.0，但 Ascend operator/build closure 中存在明确的 CANN Open Software License
Agreement 1.0/2.0 文件头；不得以项目根 Apache 文本覆盖逐文件 notice。

本轮 `dispatch_ffn_combine` 50-file inventory 中有 31 个 CANN 1.0、2 个 CANN 2.0、2 个 Apache
notice，另有 15 个没有可识别的 leading notice。没有 leading notice 不等于可以给文件赋予新协议。

两个明确 CANN 2.0 文件是：

- `csrc/ascend/mc2/dispatch_ffn_combine/CMakeLists.txt`，SHA-256
  `2c6fc054e718ba6b0ac2e8ac199a7f3459688d488cf50cc6ddb1fdc555572b1e`；
- `csrc/ascend/mc2/dispatch_ffn_combine/op_kernel/utils/moe_distribute_base.h`，SHA-256
  `47d20797282c301336b5d1d10fcb14837d0786f73a157525bcddcef1d9513485`。

这些文件头指向 upstream root `LICENSE`，而匹配 vLLM-Ascend rc1 根 `LICENSE` 是 Apache-2.0。因此该两份
CANN 2.0 文件的精确外部 agreement/source chain 为 **UNKNOWN**。不得将同名但来源不同的 SHMEM Version
2 协议文本代入，也不得据此推定任何 Version 2 协议覆盖全部 native 文件。

## CATLASS

CATLASS 是外部固定构建依赖，非 vendored repository。`csrc/ascend/build_opp.sh` 使用的固定 revision 是
Ascend CATLASS `41bf90da655bba3c66d0acd7e00abe33960ecfd6`。该固定依赖 `LICENSE` 的 SHA-256 为
`cb282ac2b66d7d6734c3fb354db792b28cf23ab29fbc16d5cfd8bd542bedeb1e`，标题为 CANN Open Software
License Agreement Version 2.0。

该固定 CATLASS 文本不能由 Ascend/SHMEM 的同名 Version 2 文本替换：后者 SHA-256 为
`0d7593660e3c03cc589e00540a1beb0c6960303a7ff5ac007b63f802c1de4387`，且条款不相同。若最终分发实际
包含 CATLASS 或其文件，应按最终 wheel/file list 再核对适用 notice 与 exact agreement copy。

## 交付边界与待核对项

本次仅披露事实，不修改原 notice/license/metadata。两个 CANN 2.0 来源协议指向的歧义保持为 UNKNOWN，
直至维护方提供可验证的外部仓、commit 和根 LICENSE。成功构建、运行通过或项目根协议标签都不证明
许可证兼容或合规。

可恢复证据：Task `fl-vllm024-ascend-migration-20260909` 的
`integrated-qwen-a2-post-fused-v2/runs/qwen_graph/regression-002/artifacts/license-review/`
中的 `NOTICE-draft.md`、`license-facts.md` 和固定 CATLASS LICENSE 副本。
