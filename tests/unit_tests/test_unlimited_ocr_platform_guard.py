"""CPU-only regression of the platform guard shipped in the OCR patch."""

import ast
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

PATCH = (
    Path(__file__).resolve().parents[2]
    / "patches/unlimited_ocr/vllm-0.24-unlimited-ocr.patch"
)


def load_config():
    lines = PATCH.read_text(encoding="utf-8").splitlines()
    start = lines.index("+class UnlimitedOCRForCausalLMConfig(VerifyAndUpdateConfig):")
    source = []
    for line in lines[start:]:
        if not line.startswith("+"):
            break
        source.append(line[1:])
    tree = ast.parse("\n".join(source))
    messages = []
    namespace = {
        "VerifyAndUpdateConfig": object,
        "logger": SimpleNamespace(info=lambda message: messages.append(message)),
    }
    exec(compile(tree, str(PATCH), "exec"), namespace)
    return namespace["UnlimitedOCRForCausalLMConfig"], messages


class PlatformGuardTests(unittest.TestCase):
    def verify(self, platform, prefix=True):
        config_class, messages = load_config()
        config = SimpleNamespace(
            cache_config=SimpleNamespace(enable_prefix_caching=prefix)
        )
        platforms = ModuleType("vllm.platforms")
        platforms.current_platform = platform
        with patch.dict(sys.modules, {"vllm.platforms": platforms}):
            config_class.verify_and_update_config(config)
        return config, messages

    def platform(self, device="npu", oot=True, **attributes):
        return SimpleNamespace(
            device_type=device, is_out_of_tree=lambda: oot, **attributes
        )

    def test_fl_and_native_ascend_disable_prefix_caching(self):
        for platform in [
            self.platform(vendor_name="ascend"),
            self.platform(
                device_name="npu",
                device_control_env_var="ASCEND_RT_VISIBLE_DEVICES",
            ),
        ]:
            with self.subTest(platform=vars(platform)):
                config, messages = self.verify(platform)
                self.assertFalse(config.cache_config.enable_prefix_caching)
                self.assertEqual(len(messages), 1)

    def test_disabled_prefix_cache_remains_disabled_without_logging(self):
        config, messages = self.verify(self.platform(vendor_name="ascend"), False)
        self.assertFalse(config.cache_config.enable_prefix_caching)
        self.assertEqual(messages, [])

    def test_other_oot_devices_are_rejected(self):
        for device in ["cuda", "musa", "gcu"]:
            with (
                self.subTest(device=device),
                self.assertRaisesRegex(ValueError, "requires the Ascend"),
            ):
                self.verify(self.platform(device=device, vendor_name="ascend"))

    def test_unknown_and_conflicting_npu_vendors_are_rejected(self):
        for attributes in [
            {},
            {"device_name": "ascend"},
            {"vendor_name": "other"},
            {
                "vendor_name": "other",
                "device_control_env_var": "ASCEND_RT_VISIBLE_DEVICES",
            },
            {"vendor_name": "", "device_control_env_var": "ASCEND_RT_VISIBLE_DEVICES"},
        ]:
            with (
                self.subTest(attributes=attributes),
                self.assertRaisesRegex(ValueError, "requires the Ascend"),
            ):
                self.verify(self.platform(**attributes))

    def test_non_oot_ascend_is_rejected_before_any_cache_mutation(self):
        config_class, messages = load_config()

        class UnreachableCache:
            @property
            def enable_prefix_caching(self):
                self.fail_access()

            def fail_access(self):
                raise AssertionError(
                    "Rejected platforms must not access cache settings"
                )

        # Use this sentinel for every rejected category to catch even reads before
        # validation, as well as prefix-cache writes or log side effects.
        for platform in [
            self.platform(oot=False, vendor_name="ascend"),
            self.platform(device="cuda"),
            self.platform(vendor_name="other"),
        ]:
            with self.subTest(platform=vars(platform)):
                platforms = ModuleType("vllm.platforms")
                platforms.current_platform = platform
                with (
                    patch.dict(sys.modules, {"vllm.platforms": platforms}),
                    self.assertRaisesRegex(ValueError, "requires the Ascend"),
                ):
                    config_class.verify_and_update_config(
                        SimpleNamespace(cache_config=UnreachableCache())
                    )
                self.assertEqual(messages, [])


if __name__ == "__main__":
    unittest.main()
