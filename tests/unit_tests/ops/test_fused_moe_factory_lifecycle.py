import ast
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[3] / 'vllm_fl/ops/custom_ops.py'


def load_patch(platform, modules, upstream, replacement):
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == '_patch_fused_moe_factory')
    ns = {'FusedMoEFL': replacement, 'logger': types.SimpleNamespace(info=lambda *a: None)}
    code = compile(ast.Module([node], type_ignores=[]), str(SOURCE), 'exec')
    exec(code, ns)
    fused_pkg = types.ModuleType('vllm.model_executor.layers.fused_moe')
    fused_layer = types.ModuleType('vllm.model_executor.layers.fused_moe.layer')
    fused_pkg.FusedMoE = upstream
    fused_layer.FusedMoE = upstream
    platform_mod = types.ModuleType('vllm.platforms')
    platform_mod.current_platform = platform
    orig_mod = types.ModuleType('vllm_fl.ops.fused_moe.layer')
    orig_mod._OrigFusedMoE = upstream
    root = types.ModuleType('vllm')
    root.__path__ = []
    model_executor = types.ModuleType('vllm.model_executor')
    model_executor.__path__ = []
    layers = types.ModuleType('vllm.model_executor.layers')
    layers.__path__ = []
    platforms = types.ModuleType('vllm.platforms')
    platforms.current_platform = platform
    modules.update({
        'vllm': root,
        'vllm.model_executor': model_executor,
        'vllm.model_executor.layers': layers,
        'vllm.model_executor.layers.fused_moe': fused_pkg,
        'vllm.model_executor.layers.fused_moe.layer': fused_layer,
        'vllm.platforms': platform_mod,
        'vllm_fl.ops.fused_moe.layer': orig_mod,
    })
    with patch.dict(sys.modules, modules):
        ns['_patch_fused_moe_factory']()


class FactoryPatchTests(unittest.TestCase):
    def setUp(self):
        self.upstream = object()
        self.replacement = object()

    def test_ast_function_exists_and_patches_ascend_stale_aliases(self):
        qwen_names = ('vllm.model_executor.models.qwen3_next',
                      'vllm.model_executor.models.qwen3_moe',
                      'vllm.model_executor.models.qwen3_vl_moe')
        modules = {n: types.SimpleNamespace(FusedMoE=self.upstream) for n in qwen_names}
        platform = types.SimpleNamespace(vendor_name='ascend', device_type='npu')
        load_patch(platform, modules, self.upstream, self.replacement)
        self.assertTrue(all(modules[n].FusedMoE is self.replacement for n in qwen_names))

    def test_custom_factory_is_not_overwritten_on_ascend(self):
        custom = object()
        modules = {'vllm.model_executor.models.qwen3_moe': types.SimpleNamespace(FusedMoE=custom)}
        platform = types.SimpleNamespace(vendor_name='ascend', device_type='npu')
        load_patch(platform, modules, self.upstream, self.replacement)
        self.assertIs(modules['vllm.model_executor.models.qwen3_moe'].FusedMoE, custom)

    def test_repeated_execution_is_idempotent(self):
        modules = {'vllm.model_executor.models.qwen3_next': types.SimpleNamespace(FusedMoE=self.upstream)}
        platform = types.SimpleNamespace(vendor_name='ascend', device_type='npu')
        load_patch(platform, modules, self.upstream, self.replacement)
        load_patch(platform, modules, self.upstream, self.replacement)
        self.assertIs(modules['vllm.model_executor.models.qwen3_next'].FusedMoE, self.replacement)

    def test_non_ascend_repairs_qwen3_next_only(self):
        modules = {'vllm.model_executor.models.qwen3_next': types.SimpleNamespace(FusedMoE=self.upstream),
                   'vllm.model_executor.models.qwen3_moe': types.SimpleNamespace(FusedMoE=self.upstream)}
        platform = types.SimpleNamespace(vendor_name='other', device_type='cuda')
        load_patch(platform, modules, self.upstream, self.replacement)
        self.assertIs(modules['vllm.model_executor.models.qwen3_next'].FusedMoE, self.replacement)
        self.assertIs(modules['vllm.model_executor.models.qwen3_moe'].FusedMoE, self.upstream)


if __name__ == '__main__':
    unittest.main(verbosity=2)
