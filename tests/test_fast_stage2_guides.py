import ast
import importlib.util
from pathlib import Path
import unittest
import sys
import types
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stage2_guides", ROOT / "iamccs_h3_stage2_guides.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
extended_spec = importlib.util.spec_from_file_location(
    "extended_av", ROOT / "iamccs_minimax_h3_extended_av.py"
)
extended_av = importlib.util.module_from_spec(extended_spec)
extended_spec.loader.exec_module(extended_av)


def frame_at_latent(k):
    return (k // 5) * 17 + sum((1, 4, 4, 4, 4)[:k % 5])


tree = ast.parse((ROOT / "vendor/mmh3tools_r38b/mmh3tools/nodes_looping_sampler.py").read_text())
namespace = {"frame_at_latent": frame_at_latent}
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_window_encoded_guides")
exec(compile(ast.Module(body=[node], type_ignores=[]), "window_guides", "exec"), namespace)
window_guides = namespace["_window_encoded_guides"]


class Stage2GuidesTest(unittest.TestCase):
    def setUp(self):
        self.video = torch.zeros(1, 2, 32, 2, 2)
        self.guide = {"resolved_frame_index": 68, "latent": torch.ones(1, 2, 1, 2, 2)}

    def test_soft_second_window_rebased(self):
        self.assertEqual(window_guides([self.guide], 34, 90)[0]["resolved_frame_index"], 34)
        self.assertEqual(window_guides([self.guide], 0, 56), [])

    def test_window_boundary_exclusive(self):
        self.assertEqual(window_guides([self.guide], 0, 68), [])
        self.assertEqual(window_guides([self.guide], 68, 100)[0]["resolved_frame_index"], 0)

    def test_overlap_receives_guide(self):
        self.assertEqual(len(window_guides([self.guide], 34, 85)), 1)
        self.assertEqual(len(window_guides([self.guide], 51, 102)), 1)

    def test_masked_video_only_and_no_mutation(self):
        original = torch.ones_like(self.video)
        video, mask, count = helpers.pin_still_guides(self.video, original, [self.guide])
        self.assertEqual(count, 1)
        self.assertTrue(torch.all(video[:, :, 20] == 1))
        self.assertTrue(torch.all(mask[:, :, 20] == 0))
        self.assertEqual(torch.count_nonzero(self.video), 0)
        self.assertTrue(torch.all(original == 1))
        self.assertTrue(torch.all(mask[:, :, :20] == 1))

    def test_non_grid_frame_containing_token(self):
        guide = dict(self.guide, resolved_frame_index=72)
        _, mask, _ = helpers.pin_still_guides(self.video, None, [guide])
        self.assertTrue(torch.all(mask[:, :, 21] == 0))

    def test_duplicate_token_rejected(self):
        with self.assertRaisesRegex(ValueError, "collide"):
            helpers.pin_still_guides(self.video, None, [self.guide, self.guide])

    def test_existing_protection_rejected(self):
        with self.assertRaisesRegex(ValueError, "protected"):
            helpers.pin_still_guides(self.video, torch.zeros_like(self.video), [self.guide])

    def test_repeated_metadata_deduplicated(self):
        cond = [[None, {"minimax_keyframes": [self.guide]}]] * 2
        self.assertEqual(len(helpers.conditioning_guides(cond)), 1)

    def test_stage2_dispatch_preserves_guides_and_audio(self):
        # Execute the real dispatcher with sampler/model boundaries stubbed on CPU.
        sys.path.insert(0, str(ROOT.parents[1]))
        from comfy.nested_tensor import NestedTensor
        src = ast.parse((ROOT / "iamccs_minimax_h3_fast_latent_2pass.py").read_text())
        fn = next(n for n in src.body if isinstance(n, ast.FunctionDef) and n.name == "_sample_stage2")
        audio = torch.randn(1, 2, 2, 40)
        common = types.SimpleNamespace(
            unpack_av=lambda x, label: x["samples"].unbind(),
            pack_av=lambda x, v, a, noise_mask: dict(x, samples=NestedTensor([v, a]), noise_mask=noise_mask),
            latents_to_frames=lambda t: frame_at_latent(t), frame_at_latent=frame_at_latent)
        for mode in ("soft", "masked"):
            for full, native, task_mode in (
                (False, False, "i2va"), (True, False, "i2va"), (False, True, "i2va"),
                (False, False, extended_av.EXTENDED_AV_MODE), (True, False, extended_av.EXTENDED_AV_MODE),
            ):
                calls = []
                context_calls = []
                guider_calls = []
                def sample(**kwargs):
                    calls.append(kwargs)
                    latent = kwargs.get("latent", kwargs.get("latent_image"))
                    video, source_audio = latent["samples"].unbind()
                    final = dict(latent, samples=NestedTensor((video, source_audio + 1)), selected="final")
                    return final, dict(latent, selected="denoised")
                class ContextWindows:
                    @classmethod
                    def execute(cls, **kwargs):
                        context_calls.append(kwargs)
                        return ("joint-model", "IAMCCS internal PSkf")
                def guide(**kwargs):
                    guider_calls.append(kwargs)
                    return (object(),)
                loop = types.SimpleNamespace(
                    per_row_mask_is_continuous=lambda: True,
                    _split_mask=lambda x: x["noise_mask"].unbind())
                windows = types.SimpleNamespace(
                    _plan=lambda *args: (17, 2, 90, 27, [range(17), range(10, 27)]),
                    MMH3ContextWindows=ContextWindows,
                    ContextFuseMethods=types.SimpleNamespace(PYRAMID="pyramid"),
                    ContextSchedules=types.SimpleNamespace(STATIC_STANDARD="standard_static"))
                providers = {
                    "common": common,
                    "nodes_looping_sampler": loop,
                    "nodes_windows": windows,
                }
                def provider(name):
                    if native and name == "nodes_windows":
                        self.fail("Native route must not install the CPU provider")
                    return providers[name]
                modules = {
                    "stage2_test.iamccs_h3_stage2_guides": helpers,
                    "stage2_test.iamccs_minimax_h3_pixel_refine_variant": types.SimpleNamespace(
                        _provider=provider),
                    "comfy_extras.nodes_context_windows": types.SimpleNamespace(
                        MiniMaxH3ContextWindowsNode=ContextWindows),
                    "comfy_extras.nodes_custom_sampler": types.SimpleNamespace(
                        BasicGuider=types.SimpleNamespace(execute=guide),
                        SamplerCustomAdvanced=types.SimpleNamespace(execute=sample)),
                }
                ns = {"__package__": "stage2_test", "torch": torch, "extended_av": extended_av,
                      "LOG": types.SimpleNamespace(info=lambda *a: None, warning=lambda *a: None),
                      "IAMCCS_MiniMaxH3LatentUpresSamplingR38": lambda: types.SimpleNamespace(prepare=lambda *a: (None, None, None, None, "test")),
                      "_result_item": lambda x, i: x[i], "_resolve_shotplan": lambda x: {"task_mode": task_mode},
                      "_upres_settings": lambda p: {"fast_latent_stage2_guides": mode, "fast_latent_stage2_native_windows": native},
                      "_native_stage2_window_policy": lambda *args: (56, 22, "native-test"),
                      "_stage2_window_policy": lambda p: (None if full else 56, 5, "test")}
                exec(compile(ast.Module(body=[fn], type_ignores=[]), "dispatcher", "exec"), ns)
                latent = {"samples": NestedTensor([self.video, audio]),
                          "noise_mask": NestedTensor([torch.ones_like(self.video), torch.zeros_like(audio)])}
                with patch.dict(sys.modules, modules):
                    result, _ = ns["_sample_stage2"](None, [[None, {"minimax_keyframes": [self.guide]}]], latent, None, 0)
                self.assertTrue(torch.equal(result["samples"].unbind()[1], audio))
                self.assertTrue(torch.all(result["noise_mask"].unbind()[1] == 0))
                self.assertEqual(result["selected"], "final" if task_mode == extended_av.EXTENDED_AV_MODE else "denoised")
                if not full:
                    self.assertEqual(len(context_calls), 1)
                    self.assertEqual(context_calls[0]["context_length"], 56 if native else 17)
                    self.assertEqual(context_calls[0]["context_overlap"], 22 if native else 2)
                    self.assertEqual(context_calls[0]["fuse_method"], "pyramid")
                    self.assertFalse(context_calls[0]["freenoise"])
                    if native:
                        self.assertNotIn("accumulator_device", context_calls[0])
                    else:
                        self.assertEqual(context_calls[0]["context_schedule"], "standard_static")
                        self.assertEqual(context_calls[0]["accumulator_device"], "cpu")
                    self.assertEqual(guider_calls[-1]["model"], "joint-model")
                    self.assertEqual(
                        guider_calls[-1]["conditioning"][0][1]["minimax_keyframes"],
                        [self.guide],
                    )
                    self.assertIn("latent_image", calls[0])
                if mode == "masked":
                    self.assertTrue(torch.all(result["samples"].unbind()[0][:, :, 20] == 1))

    def test_extended_high_pin_uses_delivered_tail_and_leaves_current_audio_locked(self):
        sys.path.insert(0, str(ROOT.parents[1]))
        from comfy.nested_tensor import NestedTensor

        source = ast.parse((ROOT / "iamccs_minimax_h3_fast_latent_2pass.py").read_text(encoding="utf-8"))
        fn = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "_locked_av_latent")
        def concat(video, audio):
            return {key: NestedTensor((video[key], audio[key])) for key in ("samples", "noise_mask")}

        ns = {
            "_concat_av": concat, "torch": torch, "extended_av": extended_av,
            "LOG": types.SimpleNamespace(info=lambda *args: None),
        }
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "high_pin", "exec"), ns)
        parent_video = torch.arange(42.0).view(1, 1, 42, 1, 1).expand(1, 1, 42, 4, 6)
        parent_audio = torch.zeros(1, 2, 2, 235)
        parent = {
            "video": parent_video, "audio": parent_audio,
            "meta": {"pinned_head_frames": "0", "delivered_frames": "107", "raw_frames": "141"},
        }
        video = torch.full((1, 1, 37, 4, 6), -1.0)
        audio = torch.full((1, 2, 2, 207), 7.0)
        contract = extended_av.resolve_contract()
        with patch.object(extended_av, "core_masks_available", return_value=True):
            result = ns["_locked_av_latent"](
                {"samples": video}, {"samples": audio},
                extended_parent=parent, extended_contract=contract,
            )
        output_video, output_audio = result["samples"].unbind()
        video_mask, audio_mask = result["noise_mask"].unbind()
        torch.testing.assert_close(output_video[:, :, :12], parent_video[:, :, 20:32])
        torch.testing.assert_close(output_video[:, :, 12:], video[:, :, 12:])
        self.assertTrue(torch.equal(output_audio, audio))
        self.assertFalse(torch.any(video_mask[:, :, :12]))
        self.assertTrue(torch.all(video_mask[:, :, 12:] == 1))
        self.assertFalse(torch.any(audio_mask))

    def test_target_rebuild_defers_the_parent_pin_only_for_extended_av(self):
        source = ast.parse((ROOT / "iamccs_minimax_h3_fast_latent_2pass.py").read_text(encoding="utf-8"))
        fn = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "_target_conditioning")
        prepare = unittest.mock.Mock(return_value=("model", "rebuilt conditioning"))
        ns = {
            "extended_av": extended_av, "_resolve_shotplan": lambda plan: plan,
            "_replace_plan": lambda original, target: target,
            "IAMCCS_MiniMaxH3AtomicConditioningBackend": lambda: types.SimpleNamespace(prepare=prepare),
        }
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "target_rebuild", "exec"), ns)
        for mode in (
            "t2va", "i2va", "fl2va_stable", "fl2va_continuous",
            "ref2va", "ref2va_vfx_edit", "ref2vid_lipsync", "ref2va_omni",
            "longvid_multishot_hardcut", "keyframe_joint_native",
            "v2va_controlnet", "v2va_object_swap", extended_av.EXTENDED_AV_MODE,
        ):
            with self.subTest(mode=mode):
                plan = {"task_mode": mode, "width": 960, "height": 544, "upscale_enabled": True}
                ns["_target_conditioning"](None, None, None, None, plan, 1, 1920, 1088, None)
                call = prepare.call_args.kwargs
                self.assertEqual(call["extended_conditioning_only"], mode == extended_av.EXTENDED_AV_MODE)
                self.assertEqual((call["cine_linx"]["width"], call["cine_linx"]["height"]), (1920, 1088))
                self.assertEqual((plan["width"], plan["height"]), (960, 544))
        prepare.reset_mock()
        plan = {"task_mode": "v2va_face_swap_lora", "width": 960, "height": 544}
        result = ns["_target_conditioning"](None, None, None, None, plan, 1, 1920, 1088, "source conditioning")
        self.assertEqual(result[1], "source conditioning")
        prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
