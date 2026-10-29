# SPDX-License-Identifier: Apache-2.0
"""Compressor state and composed paged-attention lifecycle contracts on NPU."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(current_platform.device_type != "npu", reason="Requires Ascend"),
]


@pytest.fixture(autouse=True)
def strict_dispatch():
    from vllm_fl.dispatch import get_default_manager, with_strict_mode

    with with_strict_mode():
        for name in ("save_state", "compress"):
            assert (
                get_default_manager().get_selected_impl_id("deepseek_v4_" + name)
                == "reference.torch"
            )
        for name in (
            "deepseek_v4_compressor",
            "deepseek_v4_sparse_attention",
            "deepseek_v4_indexer",
            "paged_cache_write",
        ):
            assert get_default_manager().get_selected_impl_id(name) == "vendor.ascend"
        yield


def _call(name, *args, **kwargs):
    from vllm_fl.dispatch import call_op

    return call_op("deepseek_v4_" + name, *args, **kwargs)


PAGES = [[6, 0, 4, 2], [7, 1, 5, 3]]


def _state_case(ratio):
    block = 4 if ratio == 4 else 64
    n, d, coff = 2 * ratio, 4, 2 if ratio == 4 else 1
    pos_list = list(range(n)) * 2
    slots = [
        PAGES[req][p // block] * block + p % block for req in range(2) for p in range(n)
    ]
    values = torch.arange(2 * n * coff * d, device="npu").reshape(2 * n, coff * d)
    # Projection outputs are FP32; rounding them before state storage changes math.
    kv = ((values % 37 - 18) * 0.03).float()
    gate = ((values % 13 - 6) * 0.07).float()
    ape = (
        torch.arange(ratio * coff * d, device="npu").reshape(ratio, coff * d)
        % 17
        * 0.01
    ).float()
    cache = torch.zeros(8, block, 2 * coff * d, device="npu", dtype=torch.float32)
    positions = torch.tensor(pos_list, device="npu", dtype=torch.int64)
    slot_mapping = torch.tensor(slots, device="npu", dtype=torch.int64)
    table = torch.tensor(PAGES, device="npu", dtype=torch.int32)
    weight = torch.tensor([0.7, 1.1, 0.8, 1.3], device="npu")
    angle = torch.arange(n, device="npu").float() * 0.013
    rope = torch.stack([angle.cos(), angle.sin()], dim=-1)
    return kv, gate, ape, positions, slot_mapping, cache, table, weight, rope


def _compress_oracle(
    cache, reqs, ends, weight, rope, ratio, output_dtype=torch.bfloat16
):
    d, block = weight.numel(), cache.shape[1]
    coff = 2 if ratio == 4 else 1
    outputs = []
    for req, end in zip(reqs, ends):
        if req < 0 or end < 0:
            outputs.append(torch.zeros_like(weight))
            continue
        start = end - ratio * coff + 1
        rows, scores = [], []
        for time in range(max(0, start), end + 1):
            raw = cache[PAGES[req][time // block], time % block]
            half = int(coff == 2 and time >= end - ratio + 1)
            rows.append(raw[half * d : (half + 1) * d])
            scores.append(raw[coff * d + half * d : coff * d + (half + 1) * d])
        # Independently compute each channel's weighted history reduction.
        result = torch.stack(
            [
                sum(
                    row[c] * score
                    for row, score in zip(
                        rows, torch.softmax(torch.stack([s[c] for s in scores]), dim=0)
                    )
                )
                for c in range(d)
            ]
        )
        result = result * torch.rsqrt(result.square().mean() + 1e-6) * weight
        cosine, sine = rope[(end // ratio) * ratio]
        a, b = result[-2], result[-1]
        outputs.append(
            torch.stack(
                [result[0], result[1], a * cosine - b * sine, b * cosine + a * sine]
            )
        )
    return torch.stack(outputs).to(output_dtype)


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("chunked", [False, True])
def test_save_state_chunking_and_nan_padding(ratio, chunked):
    kv, gate, ape, pos, slots, cache, *_ = _state_case(ratio)
    expected = cache.clone()
    for i in range(kv.shape[0]):
        slot = int(slots[i].item())
        expected[slot // cache.shape[1], slot % cache.shape[1]] = torch.cat(
            [
                kv[i].float(),
                gate[i].float() + ape[i % (2 * ratio) % ratio],
            ]
        )
    if chunked:
        cuts = [0, 3, kv.shape[0] - 1, kv.shape[0]]
        for start, end in zip(cuts, cuts[1:]):
            _call(
                "save_state",
                kv[start:end],
                gate[start:end],
                ape,
                pos[start:end],
                slots[start:end],
                cache,
                ratio,
            )
    else:
        _call("save_state", kv, gate, ape, pos, slots, cache, ratio)
    before = cache.clone()
    _call(
        "save_state",
        torch.full_like(kv[:1], float("nan")),
        torch.full_like(gate[:1], float("nan")),
        ape,
        torch.tensor([-1], device="npu"),
        torch.tensor([-1], device="npu"),
        cache,
        ratio,
    )
    torch.testing.assert_close(cache, before, rtol=0, atol=0)
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)
    assert torch.isfinite(cache).all()


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("boundary", [1, 2])
def test_compression_boundaries_and_pages(ratio, boundary):
    kv, gate, ape, pos, slots, cache, table, weight, rope = _state_case(ratio)
    _call("save_state", kv, gate, ape, pos, slots, cache, ratio)
    reqs, ends = [0, 1, -1], [ratio * boundary - 1, ratio - 1, -1]
    dtype = torch.bfloat16 if boundary == 1 else torch.float16
    output = _call(
        "compress",
        cache,
        table,
        torch.tensor(reqs, device="npu"),
        torch.tensor(ends, device="npu"),
        weight,
        rope,
        compress_ratio=ratio,
        output_dtype=dtype,
    )
    expected = _compress_oracle(cache, reqs, ends, weight, rope, ratio, dtype)
    assert output.dtype == dtype
    torch.testing.assert_close(output, expected, rtol=0.01, atol=0.01)
    assert torch.count_nonzero(output[-1]) == 0


@pytest.mark.parametrize("ratio", [4, 128])
def test_integrated_prefill_decode_chain(ratio):
    kv, gate, ape, pos, slots, cache, table, weight, rope = _state_case(ratio)
    cuts = [0, ratio - 1, 2 * ratio - 1, kv.shape[0]]
    for start, end in zip(cuts, cuts[1:]):
        _call(
            "save_state",
            kv[start:end],
            gate[start:end],
            ape,
            pos[start:end],
            slots[start:end],
            cache,
            ratio,
        )
    reqs = torch.tensor([0, 1], device="npu")
    end = 2 * ratio - 1
    ends = torch.tensor([end, end], device="npu")
    compressed = _call(
        "compress",
        cache,
        table,
        torch.tensor([0, 0, 1, 1], device="npu"),
        torch.tensor([ratio - 1, end, ratio - 1, end], device="npu"),
        weight,
        rope,
        compress_ratio=ratio,
    )
    torch.testing.assert_close(
        compressed,
        _compress_oracle(
            cache, [0, 0, 1, 1], [ratio - 1, end, ratio - 1, end], weight, rope, ratio
        ),
        rtol=0.01,
        atol=0.01,
    )
    compact = torch.zeros(4, 4, 4, device="npu", dtype=torch.bfloat16)
    compact[2, :2], compact[3, :2] = compressed[:2], compressed[2:]
    q = torch.full((2, 2, 4), 0.25, device="npu", dtype=torch.bfloat16)
    compact_table = torch.tensor([[2], [3]], device="npu", dtype=torch.int32)
    counts = torch.full((2,), 2, device="npu", dtype=torch.int32)
    if ratio == 4:
        indices = _call(
            "indexer",
            q,
            compact,
            torch.ones(2, 2, device="npu"),
            compact_table,
            reqs,
            ends,
            counts,
            2,
        )
        torch.testing.assert_close(
            indices.sort(dim=-1).values,
            torch.tensor([[0, 1], [0, 1]], dtype=torch.int32, device="npu"),
        )
    else:
        # C128 uses every completed compact row and has no learned indexer.
        indices = torch.tensor([[0, 1], [0, 1]], device="npu", dtype=torch.int32)
    swa = cache[..., :4].to(torch.bfloat16) * 0.03
    swa_table = table
    result = _call(
        "sparse_attention",
        q,
        swa,
        compact,
        swa_table,
        compact_table,
        reqs,
        ends,
        torch.full((2,), 2 * ratio, device="npu"),
        counts,
        indices,
        3,
        compress_ratio=ratio,
    )
    expected = torch.zeros_like(q)
    for row in range(2):
        # Retain both streams even if their underlying token coverage overlaps.
        original_rows = [
            swa[PAGES[row][i // swa.shape[1]], i % swa.shape[1]].float()
            for i in range(max(0, end - 2), end + 1)
        ]
        values = torch.stack(
            original_rows + list(compressed[row * 2 : row * 2 + 2].float())
        )
        logits = (q[row].float() @ values.t()) / 2
        expected[row] = (logits.softmax(-1) @ values).to(q.dtype)
    torch.testing.assert_close(result, expected, rtol=0.01, atol=0.01)


def test_compress_fixed_shape_changed_state_graph():
    import torch_npu  # noqa: F401

    kv, gate, ape, pos, slots, cache, table, weight, rope = _state_case(4)
    _call("save_state", kv, gate, ape, pos, slots, cache, 4)
    reqs = torch.tensor([0, 1], device="npu")
    ends = torch.tensor([7, 3], device="npu")

    def run():
        return _call("compress", cache, table, reqs, ends, weight, rope)

    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = run()
    for _ in range(2):
        cache[..., :8].add_(0.07)
        expected = _compress_oracle(cache, [0, 1], [7, 3], weight, rope, 4)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected, rtol=0.01, atol=0.01)


@pytest.mark.parametrize("ratio", [4, 128])
def test_compression_query_tiles_match_independent_requests(ratio):
    kv, gate, ape, pos, slots, cache, table, weight, rope = _state_case(ratio)
    _call("save_state", kv, gate, ape, pos, slots, cache, ratio)
    reqs = torch.tensor([0, 1, -1] * 13, device="npu")
    ends = torch.tensor([ratio - 1, 2 * ratio - 1, -1] * 13, device="npu")
    actual = _call("compress", cache, table, reqs, ends, weight, rope, ratio)
    expected = torch.cat(
        [
            _call(
                "compress",
                cache,
                table,
                reqs[row : row + 1],
                ends[row : row + 1],
                weight,
                rope,
                ratio,
            )
            for row in range(39)
        ]
    )
    torch.testing.assert_close(actual, expected)


def test_empty_chain_inputs():
    kv, gate, ape, pos, slots, cache, table, weight, rope = _state_case(4)
    _call("save_state", kv[:0], gate[:0], ape, pos[:0], slots[:0], cache, 4)
    output = _call("compress", cache, table, pos[:0], pos[:0], weight, rope)
    assert output.shape == (0, 4)
    keys, key_table = cache[..., :4].to(torch.bfloat16), table
    q = torch.empty(0, 2, 4, device="npu", dtype=torch.bfloat16)
    lengths = torch.tensor([2, 2], device="npu")
    ids = _call(
        "indexer",
        q,
        keys,
        torch.empty(0, 2, device="npu"),
        key_table,
        pos[:0],
        pos[:0],
        lengths,
        2,
    )
    assert ids.shape == (0, 2)
    attention = _call(
        "sparse_attention",
        q,
        keys,
        None,
        key_table,
        None,
        pos[:0],
        pos[:0],
        lengths,
        None,
        None,
        3,
        compress_ratio=0,
    )
    assert attention.shape == (0, 2, 4)


def test_save_real_slot_zero_with_nan_padding():
    cache = torch.full((2, 4, 16), -17.0, device="npu")
    kv = torch.stack(
        [
            torch.arange(8, device="npu").float() + 0.25,
            torch.full((8,), float("nan"), device="npu"),
        ]
    )
    gate = kv * 0.2
    ape = torch.full((4, 8), 0.125, device="npu")
    expected = cache.clone()
    expected[0, 0] = torch.cat([kv[0], gate[0] + ape[1]])
    _call(
        "save_state",
        kv,
        gate,
        ape,
        torch.tensor([1, -1], device="npu"),
        torch.tensor([0, -1], device="npu"),
        cache,
        4,
    )
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)
    assert torch.isfinite(cache).all()


def test_save_noncontiguous_page_backing_preserves_padding():
    backing = torch.full((2, 4, 20), 123.0, device="npu")
    cache = backing[..., :16]
    assert not cache.is_contiguous()
    kv = torch.arange(16, device="npu").reshape(2, 8).float() * 0.03
    gate = kv * -0.5
    ape = torch.full((4, 8), 0.0625, device="npu")
    expected = backing.clone()
    for row, slot in enumerate((0, 5)):
        expected[slot // 4, slot % 4, :16] = torch.cat([kv[row], gate[row] + ape[row]])
    _call(
        "save_state",
        kv,
        gate,
        ape,
        torch.tensor([0, 1], device="npu"),
        torch.tensor([0, 5], device="npu"),
        cache,
        4,
    )
    torch.testing.assert_close(backing, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        backing[..., 16:], torch.full_like(backing[..., 16:], 123)
    )


def test_full_reference_pipeline_changed_inputs_graph():
    import torch_npu  # noqa: F401

    kv, gate, ape, pos, slots, cache, table, weight, rope = _state_case(4)
    compressed_reqs = torch.tensor([0, 0, 1, 1], device="npu")
    compressed_ends = torch.tensor([3, 7, 3, 7], device="npu")
    reqs = torch.tensor([0, 1], device="npu")
    ends = torch.tensor([7, 7], device="npu")
    compact = torch.zeros(4, 4, 4, device="npu", dtype=torch.bfloat16)
    compact_table = torch.tensor([[2], [3]], device="npu", dtype=torch.int32)
    compact_lengths = torch.full((2,), 2, device="npu", dtype=torch.int32)
    swa_lengths = torch.full((2,), 8, device="npu", dtype=torch.int32)
    q = torch.full((2, 2, 4), 0.25, device="npu", dtype=torch.bfloat16)
    head_weights = torch.ones(2, 2, device="npu")

    def run():
        _call("save_state", kv, gate, ape, pos, slots, cache, 4)
        compressed = _call(
            "compress", cache, table, compressed_reqs, compressed_ends, weight, rope
        )
        compact[2, :2].copy_(compressed[:2])
        compact[3, :2].copy_(compressed[2:])
        ids = _call(
            "indexer",
            q,
            compact,
            head_weights,
            compact_table,
            reqs,
            ends,
            compact_lengths,
            2,
        )
        swa = cache[..., :4].to(torch.bfloat16) * 0.03
        result = _call(
            "sparse_attention",
            q,
            swa,
            compact,
            table,
            compact_table,
            reqs,
            ends,
            swa_lengths,
            compact_lengths,
            ids,
            3,
        )
        return result, compressed, ids

    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output, compressed_output, selected = run()
    torch.npu.synchronize()
    previous = output.clone()
    for _ in range(2):
        kv.add_(0.03125)
        gate.mul_(1.03)
        q.add_(torch.tensor([0.1, -0.05, 0.08, -0.04], device="npu", dtype=q.dtype))
        expected = tuple(value.clone() for value in run())
        oracle_compressed = _compress_oracle(
            cache, [0, 0, 1, 1], [3, 7, 3, 7], weight, rope, 4
        )
        oracle_output = torch.empty_like(q)
        for req in range(2):
            swa_values = [
                (cache[PAGES[req][i // 4], i % 4, :4].to(q.dtype) * 0.03).float()
                for i in (5, 6, 7)
            ]
            matrix = torch.stack(
                swa_values + list(oracle_compressed[req * 2 : req * 2 + 2].float())
            )
            logits = q[req].float() @ matrix.t() / 2
            oracle_output[req] = (logits.softmax(-1) @ matrix).to(q.dtype)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected[0], rtol=0.01, atol=0.01)
        torch.testing.assert_close(compressed_output, expected[1], rtol=0.01, atol=0.01)
        torch.testing.assert_close(selected, expected[2])
        torch.testing.assert_close(output, oracle_output, rtol=0.01, atol=0.01)
        torch.testing.assert_close(
            compressed_output, oracle_compressed, rtol=0.01, atol=0.01
        )
        assert torch.isfinite(output).all()
        assert not torch.equal(output, previous)
        previous = output.clone()


def _full_compressor_cpu_oracle(args):
    """CPU projection/state writes followed by independent history mathematics."""
    tensors = [x.detach().cpu() for x in args[:-1]]
    x, wkv, wgate, ape, state, pos, slots, table, req, ends, weight, rope = tensors
    ratio = args[-1]
    state = state.clone()
    kv = torch.nn.functional.linear(x.float(), wkv.float())
    gate = torch.nn.functional.linear(x.float(), wgate.float())
    block, d = state.shape[1], weight.numel()
    coff = 2 if ratio == 4 else 1
    for row, slot in enumerate(slots.tolist()):
        if 0 <= slot < state.shape[0] * block:
            state[slot // block, slot % block] = torch.cat(
                [kv[row], gate[row] + ape[int(pos[row]) % ratio].float()]
            )
    output = torch.zeros((ends.numel(), d), dtype=torch.float32)
    for row, (request, end) in enumerate(zip(req.tolist(), ends.tolist())):
        rope_pos = (end // ratio) * ratio
        if not (
            0 <= request < table.shape[0]
            and end >= 0
            and (end + 1) % ratio == 0
            and rope_pos < rope.shape[0]
        ):
            continue
        values, scores = [], []
        for index in range(coff * ratio):
            source = end - coff * ratio + 1 + index
            if not (source >= 0 and source // block < table.shape[1]):
                continue
            page = int(table[request, source // block])
            if not 0 <= page < state.shape[0]:
                continue
            half = index // ratio
            raw = state[page, source % block]
            values.append(raw[half * d : (half + 1) * d])
            scores.append(raw[(coff + half) * d : (coff + half + 1) * d])
        if not values:
            continue
        compressed = (torch.stack(values) * torch.softmax(torch.stack(scores), 0)).sum(
            0
        )
        normed = (
            compressed * torch.rsqrt(compressed.square().mean() + 1e-6) * weight.float()
        )
        rd = rope.shape[1]
        if rd:
            tail = normed[-rd:].clone()
            cos, sin = rope[rope_pos].float().chunk(2)
            normed[-rd::2] = tail[::2] * cos - tail[1::2] * sin
            normed[-rd + 1 :: 2] = tail[1::2] * cos + tail[::2] * sin
        output[row] = normed
    return output.to(x.dtype), state


def _full_compressor_case(
    ratio=4, dim=512, dtype=torch.bfloat16, rope_dim=64, seed=931
):
    gen = torch.Generator().manual_seed(seed)
    block = 4 if ratio == 4 else 8
    nc = (2 * ratio + block - 1) // block + 1
    pages = torch.randperm(2 * nc, generator=gen)
    zero_index = int((pages == 0).nonzero()[0])
    pages[zero_index], pages[0] = pages[0].clone(), pages[zero_index].clone()
    table = pages.reshape(2, nc).long()
    logical = [(request, pos) for request in range(2) for pos in range(2 * ratio)]
    logical = [logical[i] for i in torch.randperm(len(logical), generator=gen).tolist()]
    slots = [
        int(table[request, pos // block]) * block + pos % block
        for request, pos in logical
    ]
    n, hidden = len(logical) + 2, 33
    coff = 2 if ratio == 4 else 1
    x = (torch.randn((n, hidden), generator=gen) * 0.03).to(dtype)
    x[-2:].fill_(torch.nan)
    wkv = (torch.randn((coff * dim, hidden), generator=gen) * 0.04).to(dtype)
    wgate = (torch.randn((coff * dim, hidden), generator=gen) * 0.04).to(dtype)
    ape = torch.randn((ratio, coff * dim), generator=gen) * 0.02
    state = torch.randn((2 * nc + 1, block, 2 * coff * dim), generator=gen) * 0.02
    state[-1].fill_(torch.nan)
    pos = torch.tensor([pos for _, pos in logical] + [-1, 2**63 - 1])
    slots = torch.tensor(slots + [-1, 2**40])
    req = torch.tensor([1, 0, -1, 0, 1, 0, 2**40])
    ends = torch.tensor(
        [
            ratio - 1,
            2 * ratio - 1,
            2 * ratio - 1,
            ratio - 2,
            2 * ratio - 1,
            ratio - 1,
            2**63 - 1,
        ]
    )
    weight = (torch.rand(dim, generator=gen) + 0.5).to(dtype)
    angle = (
        torch.arange(2 * ratio + 8).float()[:, None]
        * torch.linspace(0.01, 0.07, max(1, rope_dim // 2))[None, :]
    )
    rope = (
        torch.cat([angle.cos(), angle.sin()], -1)
        if rope_dim
        else torch.empty(2 * ratio + 8, 0)
    )
    return [
        x,
        wkv,
        wgate,
        ape,
        state,
        pos,
        slots,
        table,
        req,
        ends,
        weight,
        rope,
        ratio,
    ]


def _full_to_device(args):
    return [x.npu() if isinstance(x, torch.Tensor) else x for x in args]


def _check_full_compressor(args):
    expected, state = _full_compressor_cpu_oracle(args)
    output = _call("compressor", *args)
    torch.testing.assert_close(
        output.cpu(), expected, rtol=0.01, atol=0.01, equal_nan=True
    )
    torch.testing.assert_close(
        args[4].cpu(), state, rtol=1.3e-6, atol=1e-5, equal_nan=True
    )
    return output


@pytest.mark.parametrize("ratio,dim", [(4, 512), (4, 128), (128, 512)])
def test_full_compressor_cpu_output_and_state(ratio, dim):
    _check_full_compressor(_full_to_device(_full_compressor_case(ratio, dim)))


@pytest.mark.parametrize("rope_dim", [0, 4])
def test_full_compressor_fp32_strided_and_odd_rope(rope_dim):
    args = _full_to_device(_full_compressor_case(4, 5, torch.float32, rope_dim))
    for index in (0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11):
        x = args[index]
        backing = torch.empty((*x.shape, 2), dtype=x.dtype, device=x.device)
        backing.fill_(torch.nan if x.is_floating_point() else -37)
        backing[..., 0].copy_(x)
        args[index] = backing[..., 0]
    _check_full_compressor(args)


def test_full_compressor_empty_history_and_valid_nan():
    args = _full_to_device(_full_compressor_case(4, 5, torch.float32, 4))
    args[0], args[5], args[6] = args[0][:0], args[5][:0], args[6][:0]
    _check_full_compressor(args)
    page = int(args[7][0, 0].cpu())
    args[4][page, 0, 0] = torch.nan
    _check_full_compressor(args)
    args[8], args[9] = args[8][:0], args[9][:0]
    assert _check_full_compressor(args).shape == (0, 5)


def test_full_compressor_mutating_metadata_graph():
    cpu = _full_compressor_case(4, 5, torch.bfloat16, 4)
    table, block = cpu[7], cpu[4].shape[1]
    cpu[0] = cpu[0][:6]
    cpu[0][-2:].fill_(torch.nan)
    cpu[5] = torch.tensor([4, 5, 6, 7, -1, -1])
    cpu[6] = torch.tensor([int(table[0, 1]) * block + i for i in range(4)] + [-1, -1])
    cpu[8], cpu[9] = torch.tensor([0, 1, -1]), torch.tensor([7, 7, -1])
    args = _full_to_device(cpu)
    _call("compressor", *args)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = _call("compressor", *args)
    for version in (1, 2):
        changed = [x.clone() if isinstance(x, torch.Tensor) else x for x in cpu]
        changed[0][:-2] *= 0.7 + version * 0.1
        changed[4][:-1] += version * 0.01
        if version == 2:
            changed[7] = changed[7].flip(1)
            changed[6][:4] = changed[7][0, 1] * block + torch.arange(4)
            changed[8], changed[9] = torch.tensor([1, -1, 0]), torch.tensor([3, 7, 7])
        expected, state = _full_compressor_cpu_oracle(changed)
        for actual, value in zip(args, changed):
            if isinstance(actual, torch.Tensor):
                actual.copy_(value)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(
            output.cpu(), expected, rtol=0.01, atol=0.01, equal_nan=True
        )
        torch.testing.assert_close(
            args[4].cpu(), state, rtol=1.3e-6, atol=1e-5, equal_nan=True
        )


def test_full_compressor_actual_shared_parent_high_pages():
    """Exercise all three real layouts over one 39 GB allocation, without copying it."""
    parent_bytes = 39_022_701_952
    free_bytes, _ = torch.npu.mem_get_info()
    if free_bytes < parent_bytes + 3 * 1024**3:
        pytest.skip("39 GB shared parent plus 3 GiB headroom is unavailable")
    parent = torch.empty(parent_bytes // 4, dtype=torch.float32, device="npu")
    for ratio, dim, block, width, offset in (
        (4, 512, 4, 2048, 0),
        (4, 128, 4, 512, 8192),
        (128, 512, 8, 1024, 0),
    ):
        cpu = _full_compressor_case(ratio, dim)
        page_count = cpu[4].shape[0]
        physical = torch.tensor([0] + list(range(26079 - page_count, 26078)))
        assert physical.numel() == page_count and int(physical[-1]) == 26077
        # Put a known valid request-zero block on the highest physical page.
        active_compact_page = int(cpu[7][0, 1])
        physical[active_compact_page], physical[-1] = (
            physical[-1].clone(),
            physical[active_compact_page].clone(),
        )
        assert physical[0] == 0 and physical[active_compact_page] == 26077
        # Independent compact oracle uses identical page contents and logical ids.
        # Invalid table pages are kept invalid rather than clamped to a valid page.
        cpu[7][1, 0] = -1
        cpu[7][1, 1] = 2**40
        expected, expected_state = _full_compressor_cpu_oracle(cpu)
        args = _full_to_device([*cpu[:4], torch.empty(0), *cpu[5:]])
        wide = torch.as_strided(
            parent, (26078, block, width), (374096, width, 1), offset
        )
        for compact, page in enumerate(physical.tolist()):
            wide[page].copy_(cpu[4][compact])
        args[4] = wide
        valid_slots = (cpu[6] >= 0) & (cpu[6] < page_count * block)
        slots = cpu[6].clone()
        slots[valid_slots] = (
            physical[slots[valid_slots] // block] * block + slots[valid_slots] % block
        )
        table = cpu[7].clone()
        valid_pages = (table >= 0) & (table < page_count)
        table[valid_pages] = physical[table[valid_pages]]
        table_columns = 8192 if ratio == 4 else 4096
        actual_table = torch.full((2, table_columns), -1, dtype=torch.int64)
        actual_table[:, : table.shape[1]] = table
        assert actual_table[0, 1] == 26077
        assert (slots[valid_slots] // block == 26077).any()
        history_length = (2 if ratio == 4 else 1) * ratio
        assert any(
            request == 0
            and end >= 0
            and (end + 1) % ratio == 0
            and end - history_length + 1 <= block <= end
            for request, end in zip(cpu[8].tolist(), cpu[9].tolist())
        )
        requests = torch.full((512,), -1, dtype=torch.int64)
        positions = torch.full((512,), -1, dtype=torch.int64)
        requests[: cpu[8].numel()] = cpu[8]
        positions[: cpu[9].numel()] = cpu[9]
        args[6], args[7] = slots.npu(), actual_table.npu()
        args[8], args[9] = requests.npu(), positions.npu()
        expected = torch.cat(
            [
                expected,
                torch.zeros((512 - expected.shape[0], dim), dtype=expected.dtype),
            ]
        )
        guards = []
        for page in (0, int(physical[1]), 26077):
            begin = offset + page * 374096
            for guard_start in (begin - 16, begin + block * width):
                if guard_start >= 0:
                    guard = parent.narrow(0, guard_start, 16)
                    guard.fill_(19.125)
                    guards.append(guard)
        output = _call("compressor", *args)
        for guard in guards:
            torch.testing.assert_close(
                guard.cpu(), torch.full((16,), 19.125), rtol=0, atol=0
            )
        torch.testing.assert_close(
            output.cpu(), expected, rtol=0.01, atol=0.01, equal_nan=True
        )
        # Read only the initialized pages, including untouched rows/NaN sentinels.
        for compact, page in enumerate(physical.tolist()):
            torch.testing.assert_close(
                wide[page].cpu(),
                expected_state[compact],
                rtol=1.3e-6,
                atol=1e-5,
                equal_nan=True,
            )
        print(
            f"wide R={ratio} D={dim} parent_bytes={parent_bytes} "
            f"page=26077 element_offset={offset + 26077 * 374096}"
        )
        del args, wide, output
    del parent
