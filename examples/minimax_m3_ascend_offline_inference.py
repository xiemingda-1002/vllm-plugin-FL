# Copyright (c) 2026 BAAI. All rights reserved.
"""MiniMax-M3 BF16 text generation with the complete Ascend A3 runtime.

See docs/minimax-m3-ascend.md for the fixed native runtime and resource needs.
"""

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Local MiniMax-M3 BF16 checkpoint"
    )
    args = parser.parse_args()
    if (
        os.environ.get("VLLM_FL_MINIMAX_ASCEND_RUNTIME") != "1"
        or os.environ.get("VLLM_PLUGINS") != "fl"
    ):
        parser.error("set VLLM_FL_MINIMAX_ASCEND_RUNTIME=1 and VLLM_PLUGINS=fl")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        kv_cache_dtype="auto",
        trust_remote_code=True,
        tensor_parallel_size=8,
        data_parallel_size=2,
        enable_expert_parallel=True,
        max_model_len=2048,
        max_num_seqs=1,
    )
    outputs = llm.chat(
        [{"role": "user", "content": "What is 2 + 2? Answer briefly."}],
        SamplingParams(max_tokens=64, temperature=0.0),
    )
    for output in outputs:
        print(output.outputs[0].text)


if __name__ == "__main__":
    main()
