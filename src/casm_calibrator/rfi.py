"""Configurable RFI frequency masking."""

import numpy as np


class RFIMask:
    """Boolean frequency mask based on user-specified bad ranges.

    Parameters
    ----------
    bad_ranges_mhz : list of (float, float), optional
        Pairs of (lo_mhz, hi_mhz) to flag. If None or empty,
        all channels are marked good.
    """

    def __init__(self, bad_ranges_mhz=None):
        self._ranges = list(bad_ranges_mhz) if bad_ranges_mhz else []

    def __call__(self, freqs_mhz):
        """Return boolean mask: True = good channel.

        Parameters
        ----------
        freqs_mhz : ndarray, shape (F,)
            Frequency axis in MHz.

        Returns
        -------
        ndarray, shape (F,)
            Boolean mask. True = good.
        """
        freqs_mhz = np.asarray(freqs_mhz)
        good = np.ones(len(freqs_mhz), dtype=bool)
        for lo, hi in self._ranges:
            good &= ~((freqs_mhz >= lo) & (freqs_mhz <= hi))
        return good
