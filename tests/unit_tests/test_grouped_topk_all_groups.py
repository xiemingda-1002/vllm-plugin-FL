"""CPU numerical regression for the no-bias grouped router fast path.

Load the function AST to exercise production numerics without importing the
vLLM platform integration or initializing a device backend.
"""

import ast
from pathlib import Path

import pytest
import torch


def load_router():
    path = Path(__file__).resolve().parents[2] / "vllm_fl/ops/fused_moe/router.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_fl_grouped_topk"
    )
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[function.name]


def original(logits, topk, renormalize, groups, selected, scoring, scaling):
    scores = (
        torch.softmax(logits, dim=-1) if scoring == "softmax" else torch.sigmoid(logits)
    )
    group_size = scores.size(-1) // groups
    group_scores = scores.view(-1, groups, group_size).amax(dim=-1)
    _, selected_groups = torch.topk(group_scores, selected, dim=-1, sorted=False)
    mask = torch.zeros_like(scores)
    for i in range(selected):
        start = selected_groups[:, i] * group_size
        for j in range(group_size):
            mask.scatter_(1, (start + j).unsqueeze(1), 1.0)
    weights, ids = torch.topk(scores * mask, topk, dim=-1, sorted=False)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    if scaling != 1.0:
        weights = weights * scaling
    return weights.float(), ids.int()


@pytest.mark.parametrize("groups,selected", [(1, 1), (2, 2), (8, 8), (4, 2)])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("scaling", [1.0, 2.5])
@pytest.mark.parametrize("scoring", ["softmax", "sigmoid"])
@pytest.mark.parametrize("case", ["random", "near_tie", "tie"])
def test_matches_original(groups, selected, renormalize, scaling, scoring, case):
    logits = torch.randn(64, 64, generator=torch.Generator().manual_seed(17))
    if case == "near_tie":
        logits *= 1e-6
    elif case == "tie":
        logits.zero_()
    kwargs = dict(
        topk=6,
        renormalize=renormalize,
        num_expert_group=groups,
        topk_group=selected,
        scoring_func=scoring,
        routed_scaling_factor=scaling,
    )
    actual = load_router()(torch.empty(64, 1), logits, **kwargs)
    expected = original(logits, 6, renormalize, groups, selected, scoring, scaling)
    assert actual[0].dtype == torch.float32
    assert actual[1].dtype == torch.int32
    for result, reference in zip(actual, expected):
        assert torch.equal(result, reference)


def test_all_groups_does_not_scatter(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("all-group routing must not scatter")

    monkeypatch.setattr(torch.Tensor, "scatter_", forbidden)
    logits = torch.zeros(64, 64)
    load_router()(torch.empty(64, 1), logits, 6, True, 1, 1)
