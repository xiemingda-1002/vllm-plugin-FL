"""CPU-only migration contracts; extract pure methods without loading NPU/vLLM."""
import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load_function(path, name):
    tree = ast.parse((ROOT / path).read_text(encoding="utf8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)], type_ignores=[]))
    namespace = {}
    exec(compile(module, path, "exec"), namespace)
    return namespace[name]


class MTPPortContracts(unittest.TestCase):
    def config(self):
        return NS(speculative_config=NS(method="mtp", num_speculative_tokens=3,
                  enforce_eager=True), additional_config={"enable_dsv4_mtp": True},
                  parallel_config=NS(), scheduler_config=NS(), cache_config=NS(),
                  model_config=NS(hf_config=NS(num_nextn_predict_layers=1)))

    def test_supported_and_unsupported_configs(self):
        validate = load_function("vllm_fl/spec_decode/dsv4_mtp_utils.py", "validate_dsv4_mtp_config")
        with patch.dict(os.environ, {"MTP_PREPARE_STEPS": "1", "MTP_FUSED_INPUTS": "1", "MTP_LOCAL_ARGMAX": "1"}):
            validate(self.config())
            for field, value in [("num_speculative_tokens", 4), ("enforce_eager", False), ("method", "eagle")]:
                config = self.config()
                setattr(config.speculative_config, field, value)
                with self.assertRaises(NotImplementedError):
                    validate(config)
            config = self.config()
            config.cache_config.enable_prefix_caching = True
            with self.assertRaises(NotImplementedError):
                validate(config)

    def test_quant_aliases_preserve_target_keys_and_reject_conflicts(self):
        aliases = load_function("vllm_fl/spec_decode/dsv4_mtp_utils.py", "add_mtp_quant_aliases")
        config = NS(quant_description={"target.attn.w1.weight": "keep", "mtp.0.attn.w1.weight": "INT8"})
        aliases(config)
        self.assertEqual(config.quant_description["target.attn.w1.weight"], "keep")
        self.assertEqual(config.quant_description["mtp.0.self_attn.gate_proj.weight"], "INT8")
        config.quant_description["mtp.0.self_attn.gate_proj.weight"] = "FLOAT"
        with self.assertRaises(ValueError):
            aliases(config)

    def test_count_snapshot_waits_once_and_returns_independent_lists(self):
        snapshot = load_function("vllm_fl/worker/model_runner.py", "_get_mtp_count_list_snapshot")
        class Counts:
            def __getitem__(self, item):
                return self
            def tolist(self):
                return [2, 1]
        class Event:
            waits = 0
            def synchronize(self):
                self.waits += 1
        event = Event()
        runner = NS(input_batch=NS(prev_sampled_token_ids=NS(shape=(2, 1))),
                    valid_sampled_token_count_event=event, valid_sampled_token_count_cpu=Counts(),
                    _mtp_count_list_generation=0, _mtp_count_list_snapshot=None)
        first = snapshot(runner)
        first[0] = 99
        self.assertEqual(snapshot(runner), [2, 1])
        self.assertEqual(event.waits, 1)
        runner._mtp_count_list_generation += 1
        snapshot(runner)
        self.assertEqual(event.waits, 2)
        runner.valid_sampled_token_count_event = None
        self.assertEqual(snapshot(runner), [])
        self.assertIsNone(runner._mtp_count_list_snapshot)


if __name__ == "__main__":
    unittest.main()
