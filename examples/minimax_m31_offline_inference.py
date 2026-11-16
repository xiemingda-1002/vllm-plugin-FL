# Copyright (c) 2026 BAAI. All rights reserved.

"""MiniMax-M3.1-Flash offline inference using existing local weights.

See docs/minimax-m31.md for checkpoint configuration and environment requirements.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the local MiniMax-M3.1-Flash checkpoint.",
    )
    args = parser.parse_args()

    # Parse help and validate the command line before loading the chip runtime.
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=8,
        trust_remote_code=True,
        block_size=128,
        dtype="bfloat16",
        disable_hybrid_kv_cache_manager=True,
        max_model_len=2048,
        max_num_seqs=1,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.8,
        compilation_config={
            "mode": "NONE",
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1],
            "max_cudagraph_capture_size": 1,
        },
    )

    messages = [{"role": "user", "content": "What is 2 + 2? Answer briefly."}]
    outputs = llm.chat(messages, SamplingParams(max_tokens=64, temperature=0.0))
    for output in outputs:
        print(f"Generated: {output.outputs[0].text!r}")


if __name__ == "__main__":
    main()
