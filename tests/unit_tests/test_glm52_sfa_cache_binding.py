from types import SimpleNamespace

import pytest

from vllm_fl.kv_cache.ascend.kv_cache_binding import (
    bind_sfa_kv_cache,
)


def test_sfa_binds_main_and_indexer_without_losing_tuple_identity():
    main = (object(), object())
    indexer = (object(),)
    previous = (object(), object())
    caches = {
        "model.layers.2.self_attn.attn": main,
        "model.layers.2.self_attn.indexer.k_cache": indexer,
        "model.layers.0.self_attn.attn": previous,
    }
    context = {name: SimpleNamespace(kv_cache=None) for name in caches}
    runner = []
    bind_sfa_kv_cache(caches, context, runner)
    assert runner == [previous, main, indexer]
    for name, cache in caches.items():
        assert context[name].kv_cache is cache
    with pytest.raises(AssertionError):
        bind_sfa_kv_cache(caches, context, runner)
