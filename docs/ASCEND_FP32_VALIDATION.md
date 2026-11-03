# Ascend FP32 validation — 2026-11-03

Environment: Python 3.12.13, torch 2.10.0+cpu, torch-npu 2.10.0.post2, vLLM 0.24.0, CANN 9.0.1, Ascend 910C. USE_FLAGGEMS=0. FP32 reference path: TP=1, CP=1, no quantization or sliding window. Decode graph covers model forward; prefill, scheduling, sampling and acoustic stages remain outside it.

BGE-M3 dense embeddings passed all 13 CPU FP32 comparison cases in eager and graph modes (maximum absolute error <=1e-3, cosine >=0.9999). Graph/eager maximum difference: 1.788139343e-7. Full NFCorpus: 3633 documents, 323 queries; nDCG@10 0.3148933882, MRR@10 0.5261720478, recall@10 0.1475021314.

CosyVoice3 Chinese SEED-TTS first 12 cases: 12/12 generated and scored, macro CER 1.1544011544%, WavLM cosine similarity x100 81.7328368624. Decode-graph/eager performance used the same first four inputs, three rounds, concurrency 1/2/4. Requests/s: graph 0.258620/0.335937/0.405361; eager 0.124806/0.188383/0.269274. Graph diagnostics confirmed FP32 and graph replay for all 54 graph requests.

Coexistence smoke regression retained the FL Unlimited-OCR core patch: OCR 9/9 pages completed, Qwen3-VL text checks 10/10 plus image-text smoke passed, and Qwen3.6 35B/27B completed two performance rounds of 256 requests each without failures. Qwen3-VL greedy probe looping remains an observation. DSv4F was skipped because required devices were occupied.

These are sampled regression results with shared CPU, memory and I/O, not exclusive-machine performance or full-suite acceptance. English/hardcase/full SEED-TTS were not rerun. FlagGems-enabled paths were not validated. Raw per-request artifacts were removed after reporting; this document summarizes the accepted validation report rather than providing retained raw evidence.
