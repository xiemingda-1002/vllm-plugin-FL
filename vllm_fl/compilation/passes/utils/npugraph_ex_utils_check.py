# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations

from torch._inductor.pattern_matcher import Match
from vllm.logger import init_logger

logger = init_logger(__name__)


def extra_stream_scope_check(match: Match) -> bool:
    """Reject a fusion whose call_function nodes span stream scopes."""
    non_default_streams = set()
    has_default = False
    for node in match.nodes:
        if node.op != "call_function":
            continue
        stream = node.meta.get("stream_label")
        if stream is None:
            has_default = True
        else:
            non_default_streams.add(stream)
            if len(non_default_streams) > 1:
                logger.debug("Fusion rejected across streams: %s", non_default_streams)
                return False
    if has_default and non_default_streams:
        logger.debug("Fusion rejected across default/non-default streams: %s", non_default_streams)
        return False
    return True
