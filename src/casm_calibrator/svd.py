"""SVD-based calibration engine: per-channel and per-block modes."""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class SVDMode(Enum):
    """SVD input matrix preparation mode.

    ``COMPLEX`` is amplitude+phase: gains are taken as
    ``U[:, 0] * sqrt(sigma[0])`` and then phase-referenced to
    ``ref_ant_idx`` while leaving |g_i| free. The other three modes
    (CROSS_ONLY/PHASE_ONLY/RAW) force unit-amplitude phase-only gains.
    """

    CROSS_ONLY = "cross-only"
    PHASE_ONLY = "phase-only"
    RAW = "raw"
    COMPLEX = "complex"


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
    min_baseline_wavelengths : float, optional
        If set, baselines whose ENU separation is shorter than this
        many wavelengths at the highest in-band frequency are excluded
        from the SVD (zeroed in the input matrix). Standard
        radio-interferometry self-cal practice: short baselines see
        large-scale extended emission (e.g. 410 MHz solar corona,
        Galactic background) that doesn't satisfy the rank-1 source
        assumption and so contaminates lambda_2. ``None`` (default) =
        no cut, current behaviour. ``1.5`` is a common starting point.
        Requires ``svd_calibrate`` to be called with the antenna
        mapping (it pulls positions from there).
    """

    threshold: float = 4.0
    ref_ant_idx: int = 0
    svd_mode: SVDMode = SVDMode.PHASE_ONLY
    block_size: int = 1
    fill_mode: str = "interpolate"
    amp_weighting: str = "none"

    # Subband-aware SVD with smooth gain interpolation. When
    # ``subband_size`` is set, ``svd_calibrate`` routes through a path
    # that (a) skips RFI-flagged channels inside each subband mean,
    # (b) marks subbands failed if too few good channels remain or if
    # lambda_1/lambda_2 falls below ``threshold``, (c) fits a smooth
    # per-antenna polynomial across valid subbands weighted by quality,
    # (d) evaluates that fit at every channel.
    #
    # ``subband_size=None`` (default) keeps the legacy per-channel
    # / block_size behaviour for back-compat.
    subband_size: int | None = None
    subband_min_good_frac: float = 0.5
    smooth_order: int = 3
    fill_failed: str = "smooth_fit"   # "smooth_fit" | "zero"

    # How to fill per-channel cal phase from per-subband SVD results.
    #
    # ``"global_poly"`` (default, legacy):
    #     ONE polynomial of degree ``smooth_order`` is fit across ALL valid
    #     subbands' phase points per antenna. Phase is unwrapped *globally*
    #     before fitting, then the polynomial is evaluated at every channel.
    #     Vulnerable to ``np.unwrap`` failures across wide RFI gaps for
    #     high-cable-delay antennas: a single bad unwrap rotates the entire
    #     fitted line.
    #
    # ``"local_linear"`` (recommended for high-tau arrays):
    #     For each failed subband, find the K=``local_interp_n_neighbors``
    #     nearest valid subbands on each side, locally unwrap just those
    #     points, fit a linear ``phi = a + b*f``, and evaluate at the
    #     failed-subband center freq. Per-channel output = phase of the
    #     containing subband. No global unwrap — immune to wide-gap
    #     unwrap errors that broke ``"global_poly"`` for antennas with
    #     cable delay > ~88 ns under the v3 RFI mask.
    subband_fill_method: str = "global_poly"
    local_interp_n_neighbors: int = 2

    # What to do at channels that the per-observation RFI mask flagged
    # as bad. Default is to zero them — RFI dominates these channels at
    # 30-50 dB above thermal noise, and a phase-only cal can't reduce
    # power, so passing the channel through any calibration gives a
    # net SNR loss vs zero (RFI gets squared into intensity).
    #
    # ``"zero"``           : gains = 0, flags = False at masked channels.
    #                        F-engine output is exactly 0; the channel
    #                        contributes nothing. Safe default.
    # ``"geo_fallback"``   : gains = 1 (unit cal), flags = True. F-engine
    #                        applies geometric phase only — useful only
    #                        if you know a "masked" channel is actually
    #                        clean tonight (low cal SNR at build time but
    #                        no real RFI).
    # ``"extrapolate"``    : gains = polynomial smooth-fit values, flags
    #                        = True. The smooth_fit polynomial covers the
    #                        whole band; this option just doesn't zero
    #                        its values at masked channels. Best choice
    #                        for low-cal-SNR-but-RFI-clean channels.
    masked_band_strategy: str = "zero"

    min_baseline_wavelengths: float | None = None

    def __post_init__(self):
        if isinstance(self.svd_mode, str):
            try:
                self.svd_mode = SVDMode(self.svd_mode)
            except ValueError:
                raise TypeError(
                    f"svd_mode must be SVDMode enum or matching string; got {self.svd_mode!r}"
                )


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
    amp_weights: np.ndarray | None = None


class SVDCalibrator:
    """SVD-based per-antenna gain calibrator.

    Parameters
    ----------
    config : SVDConfig
        Calibration configuration.
    """

    def __init__(self, config: SVDConfig,
                 baseline_mask: np.ndarray | None = None):
        self.config = config
        self.baseline_mask = baseline_mask

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
            result = self._per_block_svd(vis_avg)
        else:
            result = self._per_channel_svd(vis_avg)

        if self.config.amp_weighting == "inverse-variance":
            result = self._apply_inverse_variance(result, vis_avg)

        return result

    def _apply_inverse_variance(
        self, result: SVDResult, vis_avg: np.ndarray
    ) -> SVDResult:
        """Apply inverse-variance amplitude weighting from auto-power.

        Downweights noisy antennas: w_i proportional to 1/P_auto_i,
        normalized so the quietest antenna has amplitude 1.0 per channel.

        Parameters
        ----------
        result : SVDResult
            Phase-only SVD result to augment with amplitude.
        vis_avg : ndarray, shape (n_chan, n_ant, n_ant)
            Time-averaged visibility matrix (auto-power on diagonal).

        Returns
        -------
        SVDResult
            Updated result with amplitude-weighted gains/weights.
        """
        n_chan, n_ant, _ = vis_avg.shape

        # Auto-power per antenna per channel: P[i, f] = Re(V[f, i, i])
        auto_power = np.array([
            np.real(vis_avg[:, i, i]) for i in range(n_ant)
        ])  # (n_ant, n_chan)

        # Inverse power, guard against zeros
        inv_power = np.where(auto_power > 0, 1.0 / auto_power, 0.0)

        # Normalize per channel: quietest antenna (max 1/P) gets amplitude 1.0
        max_inv = np.max(inv_power, axis=0, keepdims=True)
        amp_weights = np.where(max_inv > 0, inv_power / max_inv, 0.0)

        # Apply to gains and weights (only on good channels)
        gains = result.gains.copy()
        gains[:, result.flags] *= amp_weights[:, result.flags]
        weights = np.conj(gains)

        return SVDResult(
            gains=gains,
            weights=weights,
            flags=result.flags,
            rank1_ratios=result.rank1_ratios,
            singular_values=result.singular_values,
            block_metadata=result.block_metadata,
            amp_weights=amp_weights.astype(np.float32),
        )

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
            svd_input = self._prepare_svd_input(V, cfg.svd_mode,
                                                baseline_mask=self.baseline_mask)

            U, sigma, _Vh = np.linalg.svd(svd_input)
            singular_values[ch] = sigma

            if sigma[1] > 0:
                rank1_ratios[ch] = sigma[0] / sigma[1]
            else:
                rank1_ratios[ch] = np.inf

            if rank1_ratios[ch] >= cfg.threshold:
                flags[ch] = True
                if cfg.svd_mode == SVDMode.COMPLEX:
                    # Amplitude+phase: scale leading singular vector by
                    # sqrt(sigma_1) so |g_i| reflects per-antenna
                    # sensitivity. Then phase-reference to ref_ant
                    # without normalising amplitudes.
                    g = U[:, 0] * np.sqrt(sigma[0])
                    ref = g[cfg.ref_ant_idx]
                    if np.abs(ref) > 0:
                        g = g * (np.conj(ref) / np.abs(ref))
                else:
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
            # Record the acceptance threshold so saved calibrations carry
            # their provenance (CalibrationWeightsWriter reads this key).
            block_metadata={"threshold": cfg.threshold},
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
            svd_input = self._prepare_svd_input(V_block, cfg.svd_mode,
                                                baseline_mask=self.baseline_mask)

            U, sigma, _Vh = np.linalg.svd(svd_input)

            if sigma[1] > 0:
                block_ratios[b] = sigma[0] / sigma[1]
            else:
                block_ratios[b] = np.inf

            if block_ratios[b] >= cfg.threshold:
                block_flags[b] = True
                if cfg.svd_mode == SVDMode.COMPLEX:
                    g = U[:, 0] * np.sqrt(sigma[0])
                    ref = g[cfg.ref_ant_idx]
                    if np.abs(ref) > 0:
                        g = g * (np.conj(ref) / np.abs(ref))
                else:
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
                    amps_good = np.abs(block_gains[ant, good_idx])
                    centers_good = block_centers[good_idx]

                    if cfg.fill_mode == "interpolate":
                        phases_filled = np.interp(
                            block_centers[bad_idx], centers_good, phases_unwrap
                        )
                        if cfg.svd_mode == SVDMode.COMPLEX:
                            amps_filled = np.interp(
                                block_centers[bad_idx], centers_good, amps_good
                            )
                        else:
                            amps_filled = np.ones_like(phases_filled)
                        block_gains[ant, bad_idx] = (
                            amps_filled * np.exp(1j * phases_filled)
                        )
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
                "threshold": cfg.threshold,
            },
        )

    @staticmethod
    def _prepare_svd_input(V: np.ndarray, mode: SVDMode,
                           baseline_mask: np.ndarray | None = None) -> np.ndarray:
        """Prepare matrix for SVD based on mode.

        The autocorrelation diagonal is always zeroed: it carries no
        phase information (V_ii is real-positive), and leaving it as
        ``exp(i*angle(V_ii)) = 1`` in PHASE_ONLY mode added a constant
        all-ones rank-1 component independent of the source signal —
        small bias in practice but inconsistent with CROSS_ONLY's
        treatment.

        ``baseline_mask`` (n_ant, n_ant) bool, optional: True = include,
        False = exclude. Excluded baselines are zeroed in the input.
        Used to drop short baselines whose extended-source structure
        breaks the rank-1 assumption.
        """
        if mode == SVDMode.PHASE_ONLY:
            out = np.exp(1j * np.angle(V))
        elif mode == SVDMode.CROSS_ONLY:
            out = V.astype(np.complex128, copy=True)
        else:  # RAW / COMPLEX (both keep amplitude)
            out = V.astype(np.complex128, copy=True)

        np.fill_diagonal(out, 0)

        if baseline_mask is not None:
            out = out * baseline_mask
        return out
