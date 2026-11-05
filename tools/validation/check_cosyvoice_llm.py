# Copyright (c) 2026 BAAI. All rights reserved.
"""Fixed prompt embeddings, greedy speech-token outputs and actual graph replay."""
import argparse
import json
from pathlib import Path

import torch

from vllm import LLM, SamplingParams


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--execution", choices=["eager", "full"], required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    torch.manual_seed(1105)
    prompts = [{"prompt_embeds": torch.randn(length, 896) * 0.02} for length in [4, 9]]
    graph = (
        {}
        if a.execution == "eager"
        else {
            "compilation_config": {
                "mode": 3,
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1, 2],
            }
        }
    )
    llm = LLM(
        model=a.model,
        dtype="float32",
        tensor_parallel_size=1,
        skip_tokenizer_init=True,
        enable_prompt_embeds=True,
        enforce_eager=a.execution == "eager",
        enable_prefix_caching=False,
        block_size=128,
        max_model_len=4096,
        max_num_seqs=2,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.18,
        worker_extension_cls="vllm_fl.dispatch.backends.vendor.ascend.impl.fp32_diagnostics.FP32Diagnostics",
        additional_config={"graph_replay_log": True},
        **graph
    )
    params = SamplingParams(
        temperature=0,
        min_tokens=16,
        max_tokens=16,
        ignore_eos=True,
        detokenize=False,
        logprobs=5,
    )
    records = []
    for repetition in range(2):
        outputs = llm.generate(prompts, params, use_tqdm=False)
        records.append(
            [
                {
                    "token_ids": list(o.outputs[0].token_ids),
                    "logprobs": [
                        {str(k): v.logprob for k, v in step.items()}
                        for step in o.outputs[0].logprobs
                    ],
                }
                for o in outputs
            ]
        )
    report = {
        "execution": a.execution,
        "repetitions": records,
        "diagnostics": llm.llm_engine.collective_rpc("fp32_dtype_report"),
    }
    Path(a.output).write_text(json.dumps(report, indent=2))
    print("LLM_EQUIVALENCE_WRITTEN", a.output, flush=True)


if __name__ == "__main__":
    main()
