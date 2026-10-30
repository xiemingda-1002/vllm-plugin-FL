"""CPU DP boundary regression using the runner branch and actual dummy methods.

The upstream dummy snapshot preserves its first-two-forwards coordination rule.
It is a test fixture from reference_core/vllm/v1/spec_decode/llm_base_proposer.py.
The proposing side models only the upstream collective/forward protocol; this
checks ordering and participation, not device execution or distributed liveness.
"""

import ast
import copy
import unittest
from contextlib import nullcontext
from functools import cache
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[2]
# Pinned vLLM ee0da84 baseline: vllm/v1/spec_decode/llm_base_proposer.py.
BASE_DUMMY = """
@torch.inference_mode()
def dummy_run(self, num_tokens: int, use_cudagraphs: bool=True, is_graph_capturing: bool=False, slot_mappings: dict[str, torch.Tensor] | None=None) -> None:
    only_one_forward_pass = is_graph_capturing or self.parallel_drafting
    for fwd_idx in range(1 if only_one_forward_pass else self.num_speculative_tokens):
        if fwd_idx <= 1:
            cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = self._determine_batch_execution_and_padding(num_tokens, use_cudagraphs=use_cudagraphs)
        if self._draft_attn_layer_names and slot_mappings is not None and (next(iter(self._draft_attn_layer_names)) in slot_mappings):
            slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
        else:
            slot_mapping_dict = slot_mappings or {}
        with set_forward_context(None, self.vllm_config, num_tokens=num_input_tokens, num_tokens_across_dp=num_tokens_across_dp, cudagraph_runtime_mode=cudagraph_runtime_mode, slot_mapping=slot_mapping_dict):
            if self.supports_mm_inputs:
                input_ids = None
                inputs_embeds = self.inputs_embeds[:num_input_tokens]
            else:
                input_ids = self.input_ids[:num_input_tokens]
                inputs_embeds = None
            kwargs = dict(input_ids=input_ids, positions=self._get_positions(num_input_tokens), inputs_embeds=inputs_embeds)
            if self.pass_hidden_states_to_model:
                kwargs['hidden_states'] = self.hidden_states[:num_input_tokens]
            self.model(**kwargs)
"""


@cache
def method(path, name):
    tree = ast.parse((ROOT / path).read_text(encoding="utf8"))
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return copy.deepcopy(node)


def compile_nodes(nodes, namespace):
    for node in nodes:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(
        compile(
            "from __future__ import annotations\n" + ast.unparse(module),
            "<mtp-regression>",
            "exec",
        ),
        namespace,
    )


class Buffer:
    def __getitem__(self, key):
        return self

    def expand(self, *shape):
        return self


class DPBoundaryTests(unittest.TestCase):
    def make_drafter(self, prepared, stale_k):
        events = []
        ns = {"set_forward_context": lambda *a, **kw: nullcontext()}
        compile_nodes([ast.parse(BASE_DUMMY).body[0]], ns)
        base = type("Base", (), {"dummy_run": ns["dummy_run"]})
        node = method("vllm_fl/spec_decode/dsv4_mtp.py", "dummy_run")
        cls = ast.ClassDef(
            name="Drafter",
            bases=[ast.Name(id="Base", ctx=ast.Load())],
            keywords=[],
            body=[node],
            decorator_list=[],
        )
        ns["Base"] = base
        compile_nodes([cls], ns)
        draft = ns["Drafter"]()
        draft._prepare_steps = prepared
        draft._fast_mtp_metadata = prepared
        draft.num_speculative_tokens = stale_k
        draft.parallel_drafting = False
        draft._draft_attn_layer_names = set()
        draft.supports_mm_inputs = False
        draft.pass_hidden_states_to_model = True
        draft.vllm_config = NS()
        draft.input_ids = draft.hidden_states = Buffer()
        draft._get_positions = lambda count: Buffer()
        draft.model = lambda **kw: events.append("forward")

        def coordinate(*args, **kwargs):
            events.append("collective:scalar")
            return None, 1, [1, 1]

        def packed(*args):
            events.append("collective:packed")
            return (None, 1, [1, 1]), (None, 1, [1, 1])

        draft._determine_batch_execution_and_padding = coordinate
        draft._coordinate_step_sizes = packed
        return draft, events

    def run_rank(self, prepared, k, max_seq, stale_k):
        draft, events = self.make_drafter(prepared, stale_k)
        config = NS(
            method="mtp",
            use_eagle=lambda: True,
            uses_draft_model=lambda: False,
            uses_extract_hidden_states=lambda: False,
            disable_padded_drafter_batch=False,
            use_dflash=lambda: False,
        )
        runner = NS(
            speculative_config=config,
            drafter=draft,
            use_compress=True,
            parallel_config=NS(data_parallel_size=2),
            valid_sampled_token_count_event=None,
            input_batch=NS(req_ids=["req"]),
            num_spec_tokens=k,
            effective_drafter_max_model_len=128,
            device="cpu",
            _copy_draft_token_ids_to_cpu=lambda *a, **kw: None,
        )
        ns = {
            "torch": NS(zeros=lambda *a, **kw: Buffer(), int32="int32"),
            "EagleProposer": type(draft),
            "DFlashProposer": type(draft),
            "DraftModelProposer": type(draft),
            "ExtractHiddenStatesProposer": type(draft),
            "Gemma4Proposer": type(draft),
        }
        fits = method("vllm_fl/worker/model_runner.py", "_input_fits_in_drafter")
        compile_nodes([fits], ns)
        runner._input_fits_in_drafter = lambda cad: ns["_input_fits_in_drafter"](
            runner, cad
        )
        sample = method("vllm_fl/worker/model_runner.py", "sample_tokens")
        branch = next(
            n
            for n in sample.body
            if isinstance(n, ast.If)
            and ast.unparse(n.test) == "spec_config is not None"
        )

        # Execute the complete real speculative branch including stale-draft reset.
        def propose(tokens):
            draft.num_speculative_tokens = k
            if prepared:
                events.append("collective:packed")
            for step in range(k):
                if not prepared and step <= 1:
                    events.append("collective:scalar")
                events.append("forward")

        ns.update(
            self=runner,
            spec_config=config,
            spec_decode_common_attn_metadata=NS(max_seq_len=max_seq),
            sampler_output=NS(sampled_token_ids=Buffer()),
            scheduler_output=NS(num_spec_tokens_to_schedule=k),
            slot_mappings={},
            propose_draft_token_ids=propose,
        )
        compile_nodes([branch], ns)
        return events, draft

    def test_two_rank_length_boundary_protocol(self):
        for prepared in (False, True):
            for k in (1, 2, 3):
                boundary = 128 - k
                for lengths in (
                    (1, 1),
                    (boundary, boundary + 1),
                    (boundary + 1, boundary),
                    (128, 128),
                ):
                    with self.subTest(prepared=prepared, k=k, lengths=lengths):
                        stale_k = 3 if k == 1 else 1
                        left, ld = self.run_rank(prepared, k, lengths[0], stale_k)
                        right, rd = self.run_rank(prepared, k, lengths[1], stale_k)
                        expected = (
                            ["collective:packed"] + ["forward"] * k
                            if prepared
                            else [
                                item
                                for i in range(k)
                                for item in (
                                    ["collective:scalar", "forward"]
                                    if i <= 1
                                    else ["forward"]
                                )
                            ]
                        )
                        self.assertEqual(left, expected)
                        self.assertEqual(right, expected)
                        self.assertEqual(ld.num_speculative_tokens, k)
                        self.assertEqual(rd.num_speculative_tokens, k)

    def test_graph_capture_does_not_join_runtime_protocol(self):
        for prepared in (False, True):
            draft, events = self.make_drafter(prepared, 3)
            draft.dummy_run(1, is_graph_capturing=True)
            self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
