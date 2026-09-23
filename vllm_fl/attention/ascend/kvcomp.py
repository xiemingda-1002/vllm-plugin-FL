from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class KVCompMetaData:
    kvcomp_config: Any
    chunk_sizes_for_hamming_full: torch.Tensor
    topk_for_hamming_full: torch.Tensor
    topk_for_hamming_full_cpu: torch.Tensor
    seq_lens_for_hamming: torch.Tensor
    hamming_output: torch.Tensor
    seq_lens_from_hamming: torch.Tensor
    seq_lens_for_reshape: torch.Tensor
    valid_query_mask: torch.Tensor
    sink: int
    recent: int
    hash_encoder: Any
    hashk_caches: list[torch.Tensor]
    num_actual_tokens: int = 0
    max_seq_len_for_hamming: int = 0
    slot_mapping: torch.Tensor | None = None
    seq_lens_gpu: torch.Tensor | None = None
    actual_query_lens: torch.Tensor | None = None
    block_tables_for_hamming: torch.Tensor | None = None
