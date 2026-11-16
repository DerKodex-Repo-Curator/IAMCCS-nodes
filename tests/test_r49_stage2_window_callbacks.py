import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch


SOURCE = (Path(__file__).resolve().parents[1]
          / "vendor/mmh3tools_r38b/mmh3tools/nodes_windows.py")


class Stage2WindowCallbacksTest(unittest.TestCase):
    def test_callbacks_and_cache_release_across_multiple_windows(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        handler = next(node for node in tree.body
                       if isinstance(node, ast.ClassDef) and node.name == "MMH3ContextHandler")
        execute = next(node for node in handler.body
                       if isinstance(node, ast.FunctionDef) and node.name == "execute")
        events = []
        comfy = ModuleType("comfy")
        comfy.patcher_extension = SimpleNamespace(get_all_callbacks=lambda kind, callbacks: callbacks[kind])
        comfy.model_management = SimpleNamespace(soft_empty_cache=lambda: events.append("release"))
        ns = {
            "comfy": comfy,
            "IndexListCallbacks": SimpleNamespace(EXECUTE_START="start", EXECUTE_CLEANUP="cleanup"),
            "ContextFuseMethods": SimpleNamespace(RELATIVE="relative"),
        }
        exec(compile(ast.Module(body=[execute], type_ignores=[]), str(SOURCE), "exec"), ns)
        video = torch.ones(1, 1, 3, 1, 1)
        state = SimpleNamespace(latents=[video], guide_latents=[None], is_multimodal=False)

        def evaluate(*args, **kwargs):
            window_index, _ = args[5][0]
            events.append(window_index)
            return []

        mock_handler = SimpleNamespace(
            set_step=lambda *args: None,
            _build_window_state=lambda *args: state,
            get_context_windows=lambda *args: [range(2), range(1, 3)],
            _alloc_accumulators=lambda *args: ([[video.clone()]], [[torch.ones_like(video)]], [[None]]),
            fuse_method=SimpleNamespace(name="pyramid"),
            callbacks={
                "start": [lambda *args: events.append("start")],
                "cleanup": [lambda *args: events.append("cleanup")],
            },
            evaluate_context_windows=evaluate,
        )
        with patch.dict(sys.modules, {"comfy": comfy}):
            result = ns["execute"](mock_handler, None, object(), [object()], video, 1, {})
        self.assertEqual(events, ["start", 0, "release", 1, "release", "cleanup"])
        torch.testing.assert_close(result[0], video)


if __name__ == "__main__":
    unittest.main()
