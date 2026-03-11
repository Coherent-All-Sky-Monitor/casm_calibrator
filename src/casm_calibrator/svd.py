"""SVD-based calibration engine: per-channel and per-block modes."""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class SVDMode(Enum):
    """SVD input matrix preparation mode."""

    CROSS_ONLY = "cross-only"
    PHASE_ONLY = "phase-only"
    RAW = "raw"


@dataclass
class SVDConfig:
    """Configuration for SVD calibration.

    Attributes
    ----------
    threshold : float
        Minimum sigma_1/sigma_2 ratio for a channel/block to pass.
    ref_ant_idx : int
        0-indexed reference antenna (gain phase set to zero).
    svd_mode : SVDMode
        How to prepare the input matrix for SVD.
    block_size : int
        Number of channels per block. 1 = per-channel SVD.
    fill_mode : str
        How to handle failed blocks: 'interpolate', 'zero', or 'nearest'.
    """

    threshold: float = 4.0
    ref_ant_idx: int = 0
    svd_mode: SVDMode = SVDMode.PHASE_ONLY
    block_size: int = 1
    fill_mode: str = "interpolate"


@dataclass
class SVDResult:
    """Results from SVD calibration.

    Attributes
    ----------
    gains : ndarray, shape (n_ant, n_chan)
        Per-antenna complex gains (phase-only, unit amplitude).
    weights : ndarray, shape (n_ant, n_chan)
        Beamformer weights = conj(gains).
    flags : ndarray, shape (n_chan,)
        True = good channel.
    rank1_ratios : ndarray, shape (n_chan,)
        sigma_1 / sigma_2 per channel.
    singular_values : ndarray, shape (n_chan, n_ant)
        Full singular value spectrum per channel.
    block_metadata : dict
        Additional metadata for block SVD (block_flags, block_ratios, etc.).
    """

    gains: np.ndarray
    weights: np.ndarray
    flags: np.ndarray
    rank1_ratios: np.ndarray
    singular_values: np.ndarray
    block_metadata: dict = field(default_factory=dict)


class SVDCalibrator:
    """SVD-based per-antenna gain calibrator.

    Parameters
    ----------
    config : SVDConfig
        Calibration configuration.
    """

    def __init__(self, config: SVDConfig):
        self.config = config

    def calibrate(self, vis_avg: np.ndarray) -> SVDResult:
        """Run SVD calibration on time-averaged visibilities.

        Parameters
        ----------
        vis_avg : ndarray, shape (n_chan, n_ant, n_ant)
            Time-averaged visibility matrix.

        Returns
        -------
        SVDResult
        """
        if self.config.block_size > 1:
            return self._per_block_svd(vis_avg)
        return self._per_channel_svd(vis_avg)

    def _per_channel_svd(self, vis_avg: np.ndarray) -> SVDResult:
        """Per-channel SVD calibration."""
        n_chan, n_ant, _ = vis_avg.shape
        cfg = self.config

        gains = np.zeros((n_ant, n_chan), dtype=np.complex128)
        weights = np.zeros((n_ant, n_chan), dtype=np.complex128)
        rank1_ratios = np.zeros(n_chan)
        singular_values = np.zeros((n_chan, n_ant))
        flags = np.zeros(n_chan, dtype=bool)

        for ch in range(n_chan):
            V = vis_avg[ch]
            svd_input = self._prepare_svd_input(V, cfg.svd_mode)

            U, sigma, _Vh = np.linalg.svd(svd_input)
            singular_values[ch] = sigma

            if sigma[1] > 0:
                rank1_ratios[ch] = sigma[0] / sigma[1]
            else:
                rank1_ratios[ch] = np.inf

            if rank1_ratios[ch] >= cfg.threshold:
                flags[ch] = True
                g = np.exp(1j * np.angle(U[:, 0]))
                ref_phase = np.angle(g[cfg.ref_ant_idx])
                g *= np.exp(-1j * ref_phase)
                gains[:, ch] = g
                weights[:, ch] = np.conj(g)

        return SVDResult(
            gains=gains,
            weights=weights,
            flags=flags,
            rank1_ratios=rank1_ratios,
            singular_values=singular_values,
        )

    def _per_block_svd(self, vis_avg: np.ndarray) -> SVDResult:
        """Block-averaged SVD calibration with interpolation."""
        n_chan, n_ant, _ = vis_avg.shape
        cfg = self.config
        block_size = cfg.block_size
        n_blocks = int(np.ceil(n_chan / block_size))

        block_gains = np.zeros((n_ant, n_blocks), dtype=np.complex128)
        block_flags = np.zeros(n_blocks, dtype=bool)
        block_ratios = np.zeros(n_blocks)
        block_centers = np.zeros(n_blocks)

        for b in range(n_blocks):
            ch_start = b * block_size
            ch_end = min((b + 1) * block_size, n_chan)
            block_centers[b] = 0.5 * (ch_start + ch_end - 1)

            V_block = np.mean(vis_avg[ch_start:ch_end], axis=0)
            svd_input = self._prepare_svd_input(V_block, cfg.svd_mode)

            U, sigma, _Vh = np.linalg.svd(svd_input)

            if sigma[1] > 0:
                block_ratios[b] = sigma[0] / sigma[1]
            else:
                block_ratios[b] = np.inf

            if block_ratios[b] >= cfg.threshold:
                block_flags[b] = True
                g = np.exp(1j * np.angle(U[:, 0]))
                ref_phase = np.angle(g[cfg.ref_ant_idx])
                g *= np.exp(-1j * ref_phase)
                block_gains[:, b] = g

        n_good_blocks = np.sum(block_flags)
        if n_good_blocks == 0:
            raise ValueError("All blocks flagged! No good data.")

        # Fill failed blocks
        if cfg.fill_mode in ("interpolate", "nearest"):
            good_idx = np.where(block_flags)[0]
            bad_idx = np.where(~block_flags)[0]

            if len(bad_idx) > 0 and len(good_idx) >= 2:
                for ant in range(n_ant):
                    phases_good = np.angle(block_gains[ant, good_idx])
                    phases_unwrap = np.unwrap(phases_good)
                    centers_good = block_centers[good_idx]

                    if cfg.fill_mode == "interpolate":
                        phases_filled = np.interp(
                            block_centers[bad_idx], centers_good, phases_unwrap
                        )
                        block_gains[ant, bad_idx] = np.exp(1j * phases_filled)
                    else:  # nearest
                        for bi in bad_idx:
                            nearest = good_idx[
                                np.argmin(
                                    np.abs(block_centers[good_idx] - block_centers[bi])
                                )
                            ]
                            block_gains[ant, bi] = block_gains[ant, nearest]

            elif len(bad_idx) > 0 and len(good_idx) == 1:
                for ant in range(n_ant):
                    block_gains[ant, bad_idx] = block_gains[ant, good_idx[0]]

        # Expand block gains to per-channel
        gains = np.zeros((n_ant, n_chan), dtype=np.complex128)
        weights = np.zeros((n_ant, n_chan), dtype=np.complex128)
        flags = np.zeros(n_chan, dtype=bool)
        rank1_ratios = np.zeros(n_chan)

        for b in range(n_blocks):
            ch_start = b * block_size
            ch_end = min((b + 1) * block_size, n_chan)
            rank1_ratios[ch_start:ch_end] = block_ratios[b]

            if block_flags[b] or (
                cfg.fill_mode != "zero" and np.any(block_gains[:, b] != 0)
            ):
                flags[ch_start:ch_end] = True
                gains[:, ch_start:ch_end] = block_gains[:, b : b + 1]
                weights[:, ch_start:ch_end] = np.conj(block_gains[:, b : b + 1])

        return SVDResult(
            gains=gains,
            weights=weights,
            flags=flags,
            rank1_ratios=rank1_ratios,
            singular_values=np.zeros((n_chan, n_ant)),
            block_metadata={
                "block_flags": block_flags,
                "block_ratios": block_ratios,
                "block_size": block_size,
                "block_centers": block_centers,
                "fill_mode": cfg.fill_mode,
            },
        )

    @staticmethod
    def _prepare_svd_input(V: np.ndarray, mode: SVDMode) -> np.ndarray:
        """Prepare matrix for SVD based on mode."""
        if mode == SVDMode.PHASE_ONLY:
            return np.exp(1j * np.angle(V))
        elif mode == SVDMode.CROSS_ONLY:
            out = V.copy()
            np.fill_diagonal(out, 0)
            return out
        else:  # RAW
            return V
