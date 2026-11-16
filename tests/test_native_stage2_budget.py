import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ast.parse((ROOT / "iamccs_minimax_h3_fast_latent_2pass.py").read_text(encoding="utf-8"))
POLICY = next(n for n in SOURCE.body if isinstance(n, ast.FunctionDef) and n.name == "_native_stage2_window_policy")


class NativeStage2BudgetTests(unittest.TestCase):
    def policy(self, free=10000, ram=10000, requested="AUTO", overlap="AUTO", device="cuda:1"):
        management = SimpleNamespace(
            get_total_memory=lambda d: 20000,
            minimum_inference_memory=lambda: 100,
            get_free_memory=lambda d: ram,
        )
        def estimate(shape):
            self.assertEqual(shape[:2], [1, 1])
            self.assertEqual(len(shape), 3, "H3 estimates packed AV, not BCHTW")
            return shape[2] * 100
        model = SimpleNamespace(load_device=torch.device(device), get_free_memory=lambda d: free,
                                memory_required=estimate)
        # Tiny fake tensors keep the budget arithmetic deterministic without GPU allocation.
        video = SimpleNamespace(shape=(1, 1, 107, 1, 1), numel=lambda: 10, element_size=lambda: 2)
        audio = SimpleNamespace(numel=lambda: 5, element_size=lambda: 4)
        common = SimpleNamespace(latents_to_frames=lambda t: 345,
                                 frames_to_latents=lambda f: ((f - 5) // 17) * 5 + 2)
        ns = {"torch": torch, "math": math, "model_management": management}
        exec(compile(ast.Module(body=[POLICY], type_ignores=[]), "native_budget", "exec"), ns)
        return ns[POLICY.name](model, video, audio, {
            "fast_latent_stage2_window": requested,
            "fast_latent_stage2_overlap": overlap,
        }, common)

    def test_auto_uses_available_memory_and_float32_buffers(self):
        frames, overlap, report = self.policy(free=4500)
        self.assertEqual((frames, overlap), (73, 22))
        self.assertIn("AUTO->73f/22f", report)
        self.assertIn("device=cuda:1", report)

    def test_manual_window_is_a_ceiling(self):
        self.assertEqual(self.policy(requested="39")[0], 39)
        self.assertEqual(self.policy(free=2000, requested="107")[0], 22)

    def test_full_remains_memory_bounded(self):
        self.assertEqual(self.policy(free=4500, requested="FULL")[0], 73)

    def test_overlap_clamps_to_native_chunk_grid(self):
        self.assertEqual(self.policy(free=2000, overlap="22")[:2], (22, 5))

    def test_short_windows_cannot_hide_full_timeline_exhaustion(self):
        with self.assertRaisesRegex(RuntimeError, "full-clip device buffers"):
            self.policy(free=500)

    def test_system_ram_exhaustion_stops_before_sampling(self):
        with self.assertRaisesRegex(RuntimeError, "system RAM"):
            self.policy(ram=150)

    def test_cpu_backend_uses_the_same_budget_interface(self):
        self.assertIn("device=cpu", self.policy(device="cpu")[2])


if __name__ == "__main__":
    unittest.main()
