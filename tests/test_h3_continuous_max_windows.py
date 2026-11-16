"""Maximum-window I2V/T2V continuation geometry and timed conditioning."""
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORE = load("iamccs_minimax_h3_shotboard_core")
EXT = load("iamccs_minimax_h3_extended_av")


def plan(root="t2va", policy="max_native", duration=30, evolving=False, profile="safe_8_12gb"):
    frames = round(duration * 24)
    timeline = {"fps": 24, "rows": []}
    if root == "i2va":
        timeline["rows"] = [{"id": "opener", "type": "image", "start": 0,
                             "length": frames, "imageFile": "soldier.png", "use_guide": True}]
    if evolving:
        timeline["extended_prompt_blocks"] = [
            {"id": "walk", "start_frame": 0, "end_frame": 240, "prompt": "Walk steadily."},
            {"id": "turn", "start_frame": 240, "end_frame": 480, "prompt": "Turn toward the river."},
            {"id": "run", "start_frame": 480, "end_frame": frames, "prompt": "Run along the bank."},
        ]
    return CORE.build_shotplan(
        timeline_data=timeline, global_prompt="One continuous forest take.",
        duration_seconds=duration, task_mode="fl2va_extended_av",
        extended_av_root_task=root, extended_av_profile=profile,
        extended_av_window_policy=policy, width=960, height=544,
    )


class MaximumWindowTests(unittest.TestCase):
    def test_full_windows_for_text_image_continuous_and_evolving(self):
        for root in ("t2va", "i2va"):
            for evolving in (False, True):
                with self.subTest(root=root, evolving=evolving):
                    p = plan(root=root, evolving=evolving)
                    self.assertEqual(p["chunks"][0]["task_mode"], root)
                    self.assertEqual(p["chunk_max_frames"], 345)
                    self.assertEqual([c["visible_frame_count"] for c in p["chunks"]], [311, 272, 137])
                    self.assertEqual([c["frame_count"] for c in p["chunks"]], [345, 345, 226])
                    self.assertEqual(sum(c["unique_frames"] for c in p["chunks"]), 720)
                    cursor = 0
                    for i, c in enumerate(p["chunks"]):
                        self.assertEqual(c["timeline_start_frame"], cursor)
                        self.assertEqual(c["extended_av_context_prefix_frames"], 39 if i else 0)
                        self.assertEqual(c["extended_av_boundary_runway_frames"], 34)
                        self.assertEqual(c["frame_count"] % 17, 5)
                        self.assertLessEqual(c["frame_count"], 362)
                        cursor += c["unique_frames"]

    def test_profile_mode_restores_previous_geometry(self):
        p = plan(policy="profile")
        self.assertEqual(p["chunk_max_frames"], 192)
        self.assertEqual(p["chunks"][0]["visible_frame_count"], 158)
        self.assertEqual(len(p["chunks"]), 6)

    def test_remainders_do_not_create_gaps_or_extra_delivered_frames(self):
        for frames in (361, 362, 384, 583, 584, 720, 1440, 311 + 272 * 3):
            with self.subTest(frames=frames):
                p = plan(duration=frames / 24)
                self.assertEqual(sum(c["unique_frames"] for c in p["chunks"]), frames)
                self.assertTrue(all(c["frame_count"] == 345 for c in p["chunks"][:-1]))
                for c in p["chunks"]:
                    self.assertEqual(c["frame_count"], c["extended_av_context_prefix_frames"]
                                     + c["unique_frames"] + c["extended_av_padding_frames"])

    def test_context_and_audio_contract_are_preserved(self):
        old, new = plan(policy="profile"), plan()
        for key in ("overlap_frames", "pin_mode", "audio_mask", "soft_audio_handover_ms",
                    "boundary_polish_ms", "delivery_policy", "boundary_runway_frames"):
            self.assertEqual(old["extended_av"][key], new["extended_av"][key], key)
        for profile in EXT.PROFILE_PRESETS:
            with self.subTest(profile=profile):
                p = plan(profile=profile)
                overlap = EXT.PROFILE_PRESETS[profile]["overlap_frames"]
                self.assertEqual(p["chunks"][1]["visible_frame_count"], 345 - overlap - 34)
                self.assertEqual(p["chunks"][1]["frame_count"], 345)

    def test_evolving_beats_keep_global_timing_in_longer_windows(self):
        p = plan(root="i2va", evolving=True)
        self.assertIn("Walk steadily.", p["chunks"][0]["prompt"])
        self.assertIn("Turn toward the river.", p["chunks"][0]["prompt"])
        self.assertNotIn("Run along the bank.", p["chunks"][0]["prompt"])
        self.assertNotIn("Walk steadily.", p["chunks"][1]["prompt"])
        self.assertIn("Run along the bank.", p["chunks"][1]["prompt"])
        for c in p["chunks"]:
            self.assertTrue(c.get("conditioning_revision"))
            for binding in c["conditioning_beat_bindings"]:
                self.assertGreaterEqual(binding["sample_local_start_frame"], c["extended_av_context_prefix_frames"])

    def test_short_text_continuous_remains_one_native_run(self):
        p = CORE.build_shotplan(timeline_data={}, global_prompt="Walk.", duration_seconds=10,
                               task_mode="t2va_continuous", extended_av_window_policy="max_native")
        self.assertEqual(len(p["chunks"]), 1)
        self.assertNotIn("extended_av", p)

    def test_other_model_family_keeps_profile_window(self):
        p = plan(root="ref2va")
        self.assertEqual(p["chunk_max_frames"], 192)
        self.assertEqual(p["extended_av"]["window_policy"], "profile")

    def test_invalid_policy_is_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError, "window policy"):
            plan(policy="unknown")


if __name__ == "__main__":
    unittest.main()
