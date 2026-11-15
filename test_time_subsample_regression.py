"""Regression coverage for identical sensor timestamps and time_subsample caps.
Run from repository root: python -m unittest -v test_time_subsample_regression.py
"""
import unittest
import numpy as np
import hmog_feta_training_preprocessing_v2 as pre


def make_stream(elapsed):
    elapsed = np.asarray(elapsed, dtype=np.float32)
    n = len(elapsed)
    result = np.zeros((n, 5), dtype=np.float32)
    result[:, 0] = np.arange(n, dtype=np.float32)  # recover source indices
    result[:, -2] = elapsed
    result[1:, -1] = np.diff(elapsed)
    return result


class TimeSubsampleRegressionTests(unittest.TestCase):
    def check_downsample(self, elapsed, cap):
        raw = make_stream(elapsed)
        out, capped = pre.time_subsample(raw, cap)
        self.assertEqual(capped, len(raw) > cap > 0)
        self.assertEqual(len(out), min(len(raw), cap) if cap > 0 else len(raw))
        self.assertEqual(out[0, 0], 0)
        self.assertEqual(out[-1, 0], len(raw) - 1)
        self.assertTrue(np.all(np.diff(out[:, 0]) > 0), 'Indices must be unique and ordered')
        self.assertTrue(np.all(np.diff(out[:, -2]) >= 0), 'Time cannot go backwards')
        self.assertTrue(np.allclose(out[1:, -1], np.diff(out[:, -2]), atol=1e-6))
        self.assertEqual(out[0, -1], 0)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_original_failure_duplicate_timestamps(self):
        # Old implementation: two selected unique indices, one free slot,
        # more than one unused source row -> sample_indices(n, 1) raises.
        self.check_downsample([0.0, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02], 3)

    def test_large_duplicate_timestamps(self):
        self.check_downsample([0] + [0.02] * 2047, 512)

    def test_irregular_monotonic_timestamps(self):
        rng = np.random.default_rng(27)
        for n, cap in [(10, 3), (19, 8), (513, 512), (1000, 256), (4000, 512)]:
            for j in range(12):
                # 70% tied timestamps, other intervals irregular.
                delta = rng.choice([0., .001, .01, .03, .09], size=n,
                                   p=[.7, .05, .1, .1, .05])
                self.check_downsample(np.cumsum(delta), cap)

    def test_all_unique_timestamps(self):
        self.check_downsample(np.arange(1500) / 83., 256)

    def test_invalid_requested_caps_still_rejected(self):
        a = make_stream(np.arange(10) * .1)
        for cap in (1, 2):
            with self.assertRaisesRegex(ValueError, 'Maximum allowed length'):
                pre.time_subsample(a, cap)


if __name__ == '__main__':
    unittest.main()
