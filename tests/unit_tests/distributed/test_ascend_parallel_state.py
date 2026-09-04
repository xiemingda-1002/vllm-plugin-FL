# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest

from vllm_fl.distributed import ascend_parallel_state


def _config(*, dp=2, pp=2, pcp=1, tp=2, quantization="ascend", is_moe=True):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            quantization=quantization,
            is_moe=is_moe,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=dp,
            pipeline_parallel_size=pp,
            prefill_context_parallel_size=pcp,
            tensor_parallel_size=tp,
            enable_expert_parallel=True,
        ),
    )


@pytest.fixture(autouse=True)
def _reset_mc2_state(monkeypatch):
    monkeypatch.setattr(ascend_parallel_state, "_MC2", None)
    monkeypatch.setattr(ascend_parallel_state, "_MC2_TOPOLOGY", None)


@pytest.mark.parametrize(
    ("world_size", "dp", "pp", "pcp", "tp", "expected"),
    (
        (16, 4, 1, 1, 4, [list(range(16))]),
        (8, 2, 2, 1, 2, [[0, 1, 4, 5], [2, 3, 6, 7]]),
        (16, 2, 2, 2, 2, [
            [0, 1, 2, 3, 8, 9, 10, 11],
            [4, 5, 6, 7, 12, 13, 14, 15],
        ]),
    ),
)
def test_build_mc2_group_ranks_matches_ep_layout(
    world_size,
    dp,
    pp,
    pcp,
    tp,
    expected,
):
    assert ascend_parallel_state._build_mc2_group_ranks(
        world_size,
        dp,
        pp,
        pcp,
        tp,
    ) == expected


def test_init_is_idempotent_and_destroy_resets_group(monkeypatch):
    config = _config()
    world_group = SimpleNamespace(
        rank=0,
        world_size=8,
        local_rank=0,
        device_group=object(),
    )
    ep_group = SimpleNamespace(
        ranks=[0, 1, 4, 5],
        world_size=4,
        rank_in_group=0,
    )
    created = []

    class FakeCoordinator:
        ranks = [0, 1, 4, 5]
        world_size = 4
        rank_in_group = 0

        def __init__(self):
            self.destroy_calls = 0

        def destroy(self):
            self.destroy_calls += 1

    coordinator = FakeCoordinator()

    def fake_init(group_ranks, local_rank, backend, **kwargs):
        created.append((group_ranks, local_rank, backend, kwargs))
        return coordinator

    monkeypatch.setattr(
        ascend_parallel_state.torch.distributed,
        "is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        ascend_parallel_state.torch.distributed,
        "get_backend",
        lambda group: "hccl",
    )
    monkeypatch.setattr(
        ascend_parallel_state,
        "get_world_group",
        lambda: world_group,
    )
    monkeypatch.setattr(
        ascend_parallel_state,
        "get_ep_group",
        lambda: ep_group,
    )
    monkeypatch.setattr(
        ascend_parallel_state,
        "init_model_parallel_group",
        fake_init,
    )

    first = ascend_parallel_state.init_ascend_mc2_group(config)
    second = ascend_parallel_state.init_ascend_mc2_group(config)

    assert first is second is coordinator
    assert created == [
        (
            [[0, 1, 4, 5], [2, 3, 6, 7]],
            0,
            "hccl",
            {
                "group_name": "ascend_mc2",
                "use_device_communicator": False,
            },
        )
    ]
    assert ascend_parallel_state.get_ascend_mc2_group() is coordinator
    assert ascend_parallel_state.is_ascend_mc2_group_initialized()

    with pytest.raises(RuntimeError, match="different parallel topology"):
        ascend_parallel_state.init_ascend_mc2_group(
            _config(dp=1, pp=4, pcp=1, tp=2)
        )

    ascend_parallel_state.destroy_ascend_mc2_group()
    ascend_parallel_state.destroy_ascend_mc2_group()
    assert coordinator.destroy_calls == 1
    assert not ascend_parallel_state.is_ascend_mc2_group_initialized()
    with pytest.raises(AssertionError, match="not initialized"):
        ascend_parallel_state.get_ascend_mc2_group()


def test_init_rejects_membership_different_from_ep(monkeypatch):
    config = _config()
    monkeypatch.setattr(
        ascend_parallel_state.torch.distributed,
        "is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        ascend_parallel_state,
        "get_world_group",
        lambda: SimpleNamespace(
            rank=0,
            world_size=8,
            local_rank=0,
            device_group=object(),
        ),
    )
    monkeypatch.setattr(
        ascend_parallel_state,
        "get_ep_group",
        lambda: SimpleNamespace(
            ranks=[0, 1, 2, 3],
            world_size=4,
            rank_in_group=0,
        ),
    )

    with pytest.raises(RuntimeError, match="must match"):
        ascend_parallel_state.init_ascend_mc2_group(config)


@pytest.mark.parametrize(
    "config",
    (
        _config(quantization=None),
        _config(is_moe=False),
        SimpleNamespace(
            model_config=SimpleNamespace(
                quantization="ascend",
                is_moe=True,
            ),
            parallel_config=SimpleNamespace(enable_expert_parallel=False),
        ),
    ),
)
def test_irrelevant_configuration_does_not_create_group(config):
    assert ascend_parallel_state.init_ascend_mc2_group(config) is None
    assert not ascend_parallel_state.is_ascend_mc2_group_initialized()
