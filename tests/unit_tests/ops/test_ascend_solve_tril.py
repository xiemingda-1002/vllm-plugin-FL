import torch

from vllm_fl.dispatch.backends.vendor.ascend.impl.triton.fla.solve_tril import (
    _solve_tril_torch,
    _use_stable_torch_solve,
)


def _fill_packed_lower_triangle(
    tensor: torch.Tensor,
    ranges: list[tuple[int, int, int]],
) -> None:
    _, _, heads, block_size = tensor.shape
    generator = torch.Generator().manual_seed(7)
    for batch_index, sequence_start, sequence_end in ranges:
        for chunk_start in range(sequence_start, sequence_end, block_size):
            chunk_end = min(chunk_start + block_size, sequence_end)
            chunk_len = chunk_end - chunk_start
            values = torch.randn(
                heads, chunk_len, chunk_len, generator=generator
            ) * 0.05
            tensor[
                batch_index, chunk_start:chunk_end, :, :chunk_len
            ] = torch.tril(values, diagonal=-1).permute(1, 0, 2)


def _assert_inverse(
    source: torch.Tensor,
    inverse: torch.Tensor,
    ranges: list[tuple[int, int, int]],
) -> None:
    _, _, heads, block_size = source.shape
    for batch_index, sequence_start, sequence_end in ranges:
        for chunk_start in range(sequence_start, sequence_end, block_size):
            chunk_end = min(chunk_start + block_size, sequence_end)
            chunk_len = chunk_end - chunk_start
            packed = source[
                batch_index, chunk_start:chunk_end, :, :chunk_len
            ].permute(1, 0, 2)
            actual = inverse[
                batch_index, chunk_start:chunk_end, :, :chunk_len
            ].permute(1, 0, 2)
            identity = torch.eye(chunk_len).expand(heads, -1, -1)
            system = torch.tril(packed, diagonal=-1) + identity
            torch.testing.assert_close(system @ actual, identity)


def test_torch_solve_tril_handles_packed_variable_length_sequences():
    source = torch.zeros(1, 23, 2, 16)
    ranges = [(0, 0, 5), (0, 5, 23)]
    _fill_packed_lower_triangle(source, ranges)

    actual = _solve_tril_torch(
        source, torch.tensor([0, 5, 23]), torch.float32
    )

    _assert_inverse(source, actual, ranges)


def test_torch_solve_tril_handles_batched_fixed_length_sequences():
    source = torch.zeros(2, 19, 2, 16)
    ranges = [(0, 0, 19), (1, 0, 19)]
    _fill_packed_lower_triangle(source, ranges)

    actual = _solve_tril_torch(source, None, torch.float32)

    _assert_inverse(source, actual, ranges)


def test_specialized_solve_is_the_default(monkeypatch):
    monkeypatch.delenv("VLLM_FL_GDN_TORCH_SOLVE", raising=False)

    assert not _use_stable_torch_solve()


def test_torch_solve_can_be_enabled_as_a_fallback(monkeypatch):
    monkeypatch.setenv("VLLM_FL_GDN_TORCH_SOLVE", "1")

    assert _use_stable_torch_solve()


def test_specialized_solve_can_be_selected_explicitly(monkeypatch):
    monkeypatch.setenv("VLLM_FL_GDN_TORCH_SOLVE", "0")

    assert not _use_stable_torch_solve()
