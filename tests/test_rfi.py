"""Tests for RFI masking."""

import numpy as np
import pytest

from casm_calibrator.rfi import RFIMask


class TestRFIMask:
    def test_empty_default_all_good(self):
        """No bad ranges => all channels good."""
        mask = RFIMask()
        freqs = np.linspace(375, 469, 100)
        result = mask(freqs)
        assert result.shape == (100,)
        assert np.all(result)

    def test_none_arg_all_good(self):
        """Explicit None => all channels good."""
        mask = RFIMask(bad_ranges_mhz=None)
        freqs = np.linspace(375, 469, 100)
        assert np.all(mask(freqs))

    def test_single_range(self):
        """Single bad range correctly flagged."""
        mask = RFIMask(bad_ranges_mhz=[(400, 410)])
        freqs = np.array([395, 400, 405, 410, 415])
        result = mask(freqs)
        expected = np.array([True, False, False, False, True])
        np.testing.assert_array_equal(result, expected)

    def test_multiple_ranges(self):
        """Multiple bad ranges all flagged."""
        mask = RFIMask(bad_ranges_mhz=[(400, 402), (450, 455)])
        freqs = np.array([399, 400, 401, 402, 403, 449, 450, 452, 455, 456])
        result = mask(freqs)
        expected = np.array([True, False, False, False, True, True, False, False, False, True])
        np.testing.assert_array_equal(result, expected)

    def test_edge_inclusive(self):
        """Boundary values are included in the flagged range."""
        mask = RFIMask(bad_ranges_mhz=[(400.0, 410.0)])
        freqs = np.array([400.0, 410.0])
        result = mask(freqs)
        assert not result[0]
        assert not result[1]
