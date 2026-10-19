"""CPU tests for compiled request ownership; no NPU graph or speed claims."""

import unittest

import torch
import torch_npu  # noqa: F401 -- registers the linked native library
import _strata_native as native


class NativeStateTests(unittest.TestCase):
    def setUp(self):
        self.buffers = {
            "hidden": torch.zeros(4, 1, 4), "tokens": torch.zeros(4, 1, dtype=torch.long),
            "active": torch.zeros(4, dtype=torch.bool),
            "seen": torch.zeros(4, 16, dtype=torch.bool), "uniforms": torch.full((4, 9, 16), .5),
            "trailing": torch.zeros(4, 5, 4), "text_lengths": torch.zeros(4, dtype=torch.long),
            "frames": torch.zeros(4, dtype=torch.long), "positions": torch.zeros(4, dtype=torch.long),
            "rope": torch.zeros(4, dtype=torch.long), "history": torch.zeros(4, 8, 16, dtype=torch.long),
        }
        self.kv = [torch.zeros(4, 2, 32, 2) for _ in range(4)]
        self.codec = [torch.zeros(4, 2, 3)]
        self.core = native.Runtime(self.buffers, 4, 8, 15, 4)
        self.core.set_kv(self.kv, 32)
        self.core.set_codec_state(self.codec)

    def add(self, identifier, limit=8):
        self.core.admit(identifier, torch.full((1, 1, 4), float(identifier)),
                        torch.tensor([[identifier % 15]]), torch.full((1, 2, 4), 3.),
                        torch.full((1, 1, 4), 9.), torch.full((limit + 1, 16), .2),
                        torch.zeros(1, 16, dtype=torch.bool), 7, 0, limit)

    def test_cancel_compacts_all_state_and_reuse_resets_history(self):
        for identifier in (10, 11, 12):
            self.add(identifier)
        for value in self.kv + self.codec:
            value[2].fill_(12)
        self.buffers["history"][2].fill_(12)
        self.core.cancel([11])
        self.assertEqual(self.core.identifiers(), [10, 12])
        self.assertTrue(torch.all(self.buffers["hidden"][1] == 12))
        self.assertTrue(torch.all(self.buffers["history"][1] == 12))
        for value in self.kv + self.codec:
            self.assertTrue(torch.all(value[1] == 12))
        self.assertFalse(self.buffers["active"][2])
        self.assertEqual(self.buffers["positions"][2].item(), 0)
        self.add(13)
        self.assertEqual(self.core.identifiers(), [10, 12, 13])
        self.assertTrue(torch.all(self.buffers["history"][2] == 0))
        self.assertTrue(torch.all(self.codec[0][2] == 0))
        self.assertTrue(torch.all(self.buffers["trailing"][2, :2] == 3))
        self.assertTrue(torch.all(self.buffers["trailing"][2, 2:] == 9))

    def test_bounds_and_missing_graph_do_not_advance_live_requests(self):
        self.add(10)
        for identifier, limit in ((10, 8), (11, 9), (11, 1)):
            with self.assertRaises(RuntimeError):
                self.add(identifier, limit)
        with self.assertRaisesRegex(RuntimeError, "graph has not been prepared"):
            self.core.step()
        self.assertEqual(self.core.identifiers(), [10])
        self.assertEqual(self.buffers["frames"][0].item(), 0)

    def test_cancel_unknown_empty_and_full_capacity(self):
        self.core.cancel([100])
        for identifier in (1, 2, 3, 4):
            self.add(identifier)
        self.assertEqual(self.core.batch(), 4)
        with self.assertRaisesRegex(RuntimeError, "full"):
            self.add(5)
        self.core.cancel([1, 2, 3, 4])
        self.assertEqual(self.core.size(), 0)
        self.assertEqual(self.core.step(), [])


if __name__ == "__main__":
    unittest.main()
