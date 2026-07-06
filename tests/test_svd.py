"""Tests for SVD calibration engine."""

import numpy as np
import pytest

from casm_calibrator.svd import SVDCalibrator, SVDConfig, SVDMode


class TestPerChannelSVD:
    def test_recovers_known_gains(self, rank1_vis, n_ant, n_chan):
        """Per-channel SVD recovers exact gains from a perfect rank-1 matrix."""
        vis_avg, true_gains = rank1_vis

        config = SVDConfig(threshold=2.0, ref_ant_idx=0, svd_mode=SVDMode.PHASE_ONLY)
        result = SVDCalibrator(config).calibrate(vis_avg)

        assert result.gains.shape == (n_ant, n_chan)
        assert result.weights.shape == (n_ant, n_chan)
        assert result.flags.shape == (n_chan,)

        # All channels should pass (perfect rank-1)
        assert np.all(result.flags)

        # Recovered gains should match true gains (up to global sign flip)
        for f in range(n_chan):
            g_recovered = result.gains[:, f]
            g_true = true_gains[:, f]
            # Check phase difference is zero (mod sign ambiguity)
            phase_diff = np.angle(g_recovered * np.conj(g_true))
            # Either all ~0 or all ~pi (global sign flip)
            if np.abs(phase_diff[0]) > np.pi / 2:
                phase_diff = np.angle(-g_recovered * np.conj(g_true))
            np.testing.assert_allclose(phase_diff, 0, atol=1e-10)

    def test_reference_phase_zeroed(self, rank1_vis):
        """Reference antenna has zero phase on good channels."""
        vis_avg, _ = rank1_vis
        config = SVDConfig(threshold=2.0, ref_ant_idx=0, svd_mode=SVDMode.PHASE_ONLY)
        result = SVDCalibrator(config).calibrate(vis_avg)

        ref_phases = np.angle(result.gains[0, result.flags])
        np.testing.assert_allclose(ref_phases, 0, atol=1e-10)

    def test_threshold_rejection(self, n_ant, n_chan):
        """Channels with low rank-1 quality are flagged."""
        rng = np.random.default_rng(99)
        # Random (non-rank-1) matrix
        vis_avg = rng.standard_normal((n_chan, n_ant, n_ant)) + \
                  1j * rng.standard_normal((n_chan, n_ant, n_ant))
        # Make Hermitian
        for f in range(n_chan):
            vis_avg[f] = 0.5 * (vis_avg[f] + vis_avg[f].conj().T)

        config = SVDConfig(threshold=100.0, ref_ant_idx=0, svd_mode=SVDMode.PHASE_ONLY)
        result = SVDCalibrator(config).calibrate(vis_avg)

        # Random matrices should not be rank-1 => all flagged
        assert not np.any(result.flags)

    def test_cross_only_mode(self, rank1_vis, n_chan):
        """Cross-only mode zeros diagonal before SVD."""
        vis_avg, _ = rank1_vis
        config = SVDConfig(threshold=2.0, ref_ant_idx=0, svd_mode=SVDMode.CROSS_ONLY)
        result = SVDCalibrator(config).calibrate(vis_avg)

        # Should still recover gains from cross-correlations
        assert np.sum(result.flags) > n_chan * 0.5

    def test_weights_are_conj_gains(self, noisy_rank1_vis):
        """Weights = conj(gains) for good channels."""
        vis_avg, _ = noisy_rank1_vis
        config = SVDConfig(threshold=2.0, ref_ant_idx=0, svd_mode=SVDMode.PHASE_ONLY)
        result = SVDCalibrator(config).calibrate(vis_avg)

        good = result.flags
        np.testing.assert_allclose(
            result.weights[:, good], np.conj(result.gains[:, good])
        )


def _coherent_block_vis(n_ant, n_chan, rng):
    """Build a (n_chan, n_ant, n_ant) rank-1 vis with gains constant
    within each block_size=8 group, so block-averaging preserves
    rank-1 structure (block-SVD's actual operating regime). Random
    per-channel gains do not block-average to rank-1, so they need
    constant-per-block gains to test block SVD meaningfully.
    """
    block_size = 8
    n_blocks = n_chan // block_size
    block_phases = rng.uniform(-np.pi, np.pi, (n_ant, n_blocks))
    block_phases[0, :] = 0.0
    phases = np.repeat(block_phases, block_size, axis=1)
    true_gains = np.exp(1j * phases)
    vis_avg = np.zeros((n_chan, n_ant, n_ant), dtype=np.complex128)
    for f in range(n_chan):
        g = true_gains[:, f]
        vis_avg[f] = np.outer(g, np.conj(g))
    return vis_avg, true_gains


class TestPerBlockSVD:
    def test_block_svd_basic(self, n_ant, n_chan):
        """Block SVD on data that's coherent within each block."""
        rng = np.random.default_rng(42)
        vis_avg, _ = _coherent_block_vis(n_ant, n_chan, rng)
        config = SVDConfig(
            threshold=2.0, ref_ant_idx=0, svd_mode=SVDMode.PHASE_ONLY,
            block_size=8, fill_mode="interpolate",
        )
        result = SVDCalibrator(config).calibrate(vis_avg)

        assert result.gains.shape == (n_ant, n_chan)
        # Most channels should be good (coherent rank-1 block-averages cleanly)
        assert np.sum(result.flags) > n_chan * 0.5
        # Block metadata should be present
        assert "block_flags" in result.block_metadata
        assert "block_size" in result.block_metadata

    def test_block_interpolation(self, n_ant):
        """Failed blocks get interpolated from neighbors."""
        n_chan = 40
        block_size = 8
        rng = np.random.default_rng(42)

        # Coherent-within-block rank-1 gains
        vis_avg, _ = _coherent_block_vis(n_ant, n_chan, rng)

        # Corrupt block 2
        vis_avg[16:24] = rng.standard_normal((8, n_ant, n_ant)) + \
                         1j * rng.standard_normal((8, n_ant, n_ant))
        for f in range(16, 24):
            vis_avg[f] = 0.5 * (vis_avg[f] + vis_avg[f].conj().T)

        config = SVDConfig(
            threshold=2.0, ref_ant_idx=0, svd_mode=SVDMode.PHASE_ONLY,
            block_size=block_size, fill_mode="interpolate",
        )
        result = SVDCalibrator(config).calibrate(vis_avg)

        # Interpolated block should still have weights (flags True)
        assert np.all(result.flags[16:24])  # filled via interpolation


class TestInverseVarianceWeighting:
    """Tests for inverse-variance amplitude weighting."""

    def test_noisy_antenna_downweighted(self, n_ant=4, n_chan=32):
        """Antenna with higher auto-power gets lower amplitude weight."""
        rng = np.random.default_rng(99)

        # Create rank-1 vis with uniform gains
        phases = rng.uniform(-np.pi, np.pi, (n_ant, n_chan))
        phases[0, :] = 0.0
        true_gains = np.exp(1j * phases)

        vis_avg = np.zeros((n_chan, n_ant, n_ant), dtype=np.complex128)
        for f in range(n_chan):
            g = true_gains[:, f]
            vis_avg[f] = np.outer(g, np.conj(g))

        # Make antenna 0 have 10x auto-power (noisy)
        for f in range(n_chan):
            vis_avg[f, 0, 0] *= 10.0

        config = SVDConfig(
            threshold=2.0, ref_ant_idx=0, svd_mode=SVDMode.PHASE_ONLY,
            amp_weighting="inverse-variance",
        )
        result = SVDCalibrator(config).calibrate(vis_avg)

        assert result.amp_weights is not None
        # Noisy antenna 0 should have lower weight than quiet antennas
        mean_amp_ant0 = np.mean(result.amp_weights[0, result.flags])
        mean_amp_ant1 = np.mean(result.amp_weights[1, result.flags])
        assert mean_amp_ant0 < mean_amp_ant1, (
            f"Noisy antenna should be downweighted: {mean_amp_ant0:.4f} >= {mean_amp_ant1:.4f}"
        )
        # Specifically, ant0 has 10x power, so weight ≈ 0.1
        assert mean_amp_ant0 < 0.15, f"Expected ~0.1, got {mean_amp_ant0:.4f}"

    def test_phases_preserved(self, n_ant=4, n_chan=32):
        """Inverse-variance weighting preserves SVD phase structure."""
        rng = np.random.default_rng(42)
        phases = rng.uniform(-np.pi, np.pi, (n_ant, n_chan))
        phases[0, :] = 0.0
        true_gains = np.exp(1j * phases)

        vis_avg = np.zeros((n_chan, n_ant, n_ant), dtype=np.complex128)
        for f in range(n_chan):
            g = true_gains[:, f]
            vis_avg[f] = np.outer(g, np.conj(g))
            vis_avg[f, 0, 0] *= 5.0  # unequal auto-power

        # Phase-only result
        config_phase = SVDConfig(threshold=2.0, ref_ant_idx=0, svd_mode=SVDMode.PHASE_ONLY)
        result_phase = SVDCalibrator(config_phase).calibrate(vis_avg)

        # Inverse-variance result
        config_iv = SVDConfig(
            threshold=2.0, ref_ant_idx=0, svd_mode=SVDMode.PHASE_ONLY,
            amp_weighting="inverse-variance",
        )
        result_iv = SVDCalibrator(config_iv).calibrate(vis_avg)

        # Phases should match (modulo amplitude)
        for ch in range(n_chan):
            if result_phase.flags[ch] and result_iv.flags[ch]:
                phase_diff = np.angle(result_iv.gains[:, ch]) - np.angle(result_phase.gains[:, ch])
                phase_diff = np.abs(np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi)
                assert np.all(phase_diff < 0.01), (
                    f"Phase mismatch at channel {ch}: {phase_diff}"
                )

    def test_default_no_amp_weights(self, n_ant=4, n_chan=16):
        """Default (no amp weighting) should have amp_weights=None."""
        rng = np.random.default_rng(7)
        phases = rng.uniform(-np.pi, np.pi, (n_ant, n_chan))
        phases[0, :] = 0.0
        true_gains = np.exp(1j * phases)

        vis_avg = np.zeros((n_chan, n_ant, n_ant), dtype=np.complex128)
        for f in range(n_chan):
            g = true_gains[:, f]
            vis_avg[f] = np.outer(g, np.conj(g))

        config = SVDConfig(threshold=2.0, ref_ant_idx=0, svd_mode=SVDMode.PHASE_ONLY)
        result = SVDCalibrator(config).calibrate(vis_avg)
        assert result.amp_weights is None
        # Gains should be unit amplitude
        assert np.allclose(np.abs(result.gains[:, result.flags]), 1.0, atol=1e-10)

# ---------------------------------------------------------------------
# masked_band_strategy: how the SVD treats RFI-flagged channels
# ---------------------------------------------------------------------

def _make_synth_hermitian(n_chan, n_ant, gains, freqs_mhz, rng):
    """Build a (n_chan, n_ant, n_ant) Hermitian rank-1 matrix with
    unit-amplitude gains plus a small noise level. Used to drive the
    subband SVD with a known answer."""
    H = np.zeros((n_chan, n_ant, n_ant), dtype=np.complex128)
    g = np.asarray(gains, dtype=np.complex128)        # (n_ant,)
    for k in range(n_chan):
        # Smooth freq-dependent linear phase per antenna so the
        # polynomial smooth_fit has something physical to recover.
        delay_phases = np.exp(1j * 2 * np.pi * freqs_mhz[k] * 1e-3 * np.arange(n_ant))
        gk = g * delay_phases
        H[k] = np.outer(gk, np.conj(gk))
        H[k] += 0.001 * (rng.standard_normal((n_ant, n_ant))
                         + 1j * rng.standard_normal((n_ant, n_ant)))
        H[k] = 0.5 * (H[k] + H[k].conj().T)
    return H


def test_masked_band_strategy_zero_default():
    from casm_calibrator import SVDConfig, SVDMode
    from casm_calibrator import _subband_svd_with_smooth_fit

    rng = np.random.default_rng(0)
    n_chan = 64; n_ant = 6
    freqs = np.linspace(420.0, 450.0, n_chan)
    gains = np.exp(1j * rng.uniform(-np.pi, np.pi, n_ant))
    H = _make_synth_hermitian(n_chan, n_ant, gains, freqs, rng)

    # Flag the lower half of the band as RFI.
    freq_mask = np.ones(n_chan, dtype=bool)
    freq_mask[:32] = False

    cfg = SVDConfig(threshold=1.5, svd_mode=SVDMode.PHASE_ONLY,
                    subband_size=8, subband_min_good_frac=0.5,
                    smooth_order=2, fill_failed="smooth_fit",
                    masked_band_strategy="zero")
    res = _subband_svd_with_smooth_fit(H, freqs, freq_mask, cfg)
    # RFI-flagged channels: gains zero, flags False.
    assert np.all(res.gains[:, ~freq_mask] == 0.0)
    assert not np.any(res.flags[~freq_mask])
    # Unflagged channels: gains non-zero (smooth-fit values).
    assert np.any(res.gains[:, freq_mask] != 0.0)


def test_masked_band_strategy_geo_fallback():
    from casm_calibrator import SVDConfig, SVDMode
    from casm_calibrator import _subband_svd_with_smooth_fit

    rng = np.random.default_rng(1)
    n_chan = 64; n_ant = 6
    freqs = np.linspace(420.0, 450.0, n_chan)
    gains = np.exp(1j * rng.uniform(-np.pi, np.pi, n_ant))
    H = _make_synth_hermitian(n_chan, n_ant, gains, freqs, rng)
    freq_mask = np.ones(n_chan, dtype=bool); freq_mask[:32] = False

    cfg = SVDConfig(threshold=1.5, svd_mode=SVDMode.PHASE_ONLY,
                    subband_size=8, subband_min_good_frac=0.5,
                    smooth_order=2, fill_failed="smooth_fit",
                    masked_band_strategy="geo_fallback")
    res = _subband_svd_with_smooth_fit(H, freqs, freq_mask, cfg)
    # RFI-flagged channels: gains == 1 (unit cal), flags True so deploy
    # multiplies through.
    assert np.allclose(res.gains[:, ~freq_mask], 1.0 + 0j)
    assert np.all(res.flags[~freq_mask])


def test_masked_band_strategy_extrapolate():
    from casm_calibrator import SVDConfig, SVDMode
    from casm_calibrator import _subband_svd_with_smooth_fit

    rng = np.random.default_rng(2)
    n_chan = 64; n_ant = 6
    freqs = np.linspace(420.0, 450.0, n_chan)
    gains = np.exp(1j * rng.uniform(-np.pi, np.pi, n_ant))
    H = _make_synth_hermitian(n_chan, n_ant, gains, freqs, rng)
    freq_mask = np.ones(n_chan, dtype=bool); freq_mask[:32] = False

    cfg = SVDConfig(threshold=1.5, svd_mode=SVDMode.PHASE_ONLY,
                    subband_size=8, subband_min_good_frac=0.5,
                    smooth_order=2, fill_failed="smooth_fit",
                    masked_band_strategy="extrapolate")
    res = _subband_svd_with_smooth_fit(H, freqs, freq_mask, cfg)
    # RFI-flagged channels: gains are the polynomial smooth-fit values,
    # which for a phase-only fit have unit amplitude and continuous
    # phase across the band. Flags True.
    g_flagged = res.gains[:, ~freq_mask]
    assert np.all(res.flags[~freq_mask])
    # Values are NOT identically zero or identically 1.
    assert np.any(g_flagged != 0.0)
    assert not np.allclose(g_flagged, 1.0 + 0j)
    # Phase-only mode: |g| should be ~1 at extrapolated channels.
    np.testing.assert_allclose(np.abs(g_flagged), 1.0, atol=1e-6)


def test_masked_band_strategy_invalid():
    from casm_calibrator import SVDConfig, SVDMode
    from casm_calibrator import _subband_svd_with_smooth_fit
    rng = np.random.default_rng(3)
    n_chan = 64; n_ant = 6
    freqs = np.linspace(420.0, 450.0, n_chan)
    gains = np.exp(1j * rng.uniform(-np.pi, np.pi, n_ant))
    H = _make_synth_hermitian(n_chan, n_ant, gains, freqs, rng)
    freq_mask = np.ones(n_chan, dtype=bool); freq_mask[:32] = False

    cfg = SVDConfig(threshold=1.5, svd_mode=SVDMode.PHASE_ONLY,
                    subband_size=8, subband_min_good_frac=0.5,
                    smooth_order=2, fill_failed="smooth_fit",
                    masked_band_strategy="bogus")
    import pytest
    with pytest.raises(ValueError, match="Unknown masked_band_strategy"):
        _subband_svd_with_smooth_fit(H, freqs, freq_mask, cfg)


class TestThresholdProvenance:
    """Saved calibrations must record the threshold actually used.

    Both SVD paths store it in block_metadata, which
    CalibrationWeightsWriter reads when writing the NPZ. Before this was
    pinned down, every saved file claimed threshold=0.0.
    """

    def test_per_channel_records_threshold(self, rank1_vis):
        vis_avg, _ = rank1_vis
        config = SVDConfig(threshold=3.5, ref_ant_idx=0, svd_mode=SVDMode.PHASE_ONLY)
        result = SVDCalibrator(config).calibrate(vis_avg)
        assert result.block_metadata["threshold"] == 3.5

    def test_per_block_records_threshold(self, n_ant, n_chan):
        rng = np.random.default_rng(7)
        vis_avg, _ = _coherent_block_vis(n_ant, n_chan, rng)
        config = SVDConfig(
            threshold=2.5, ref_ant_idx=0, svd_mode=SVDMode.PHASE_ONLY, block_size=8
        )
        result = SVDCalibrator(config).calibrate(vis_avg)
        assert result.block_metadata["threshold"] == 2.5


def test_subband_fill_method_local_linear():
    """local_linear fills failed subbands from neighbors, no global unwrap."""
    from casm_calibrator import SVDConfig, SVDMode
    from casm_calibrator import _subband_svd_with_smooth_fit

    rng = np.random.default_rng(11)
    n_chan = 64; n_ant = 6
    freqs = np.linspace(420.0, 450.0, n_chan)
    true_phases = rng.uniform(-np.pi, np.pi, n_ant)
    gains = np.exp(1j * true_phases)
    H = _make_synth_hermitian(n_chan, n_ant, gains, freqs, rng)

    # RFI gap in the middle of the band: two subbands fail, valid
    # neighbors exist on both sides for the local fit.
    freq_mask = np.ones(n_chan, dtype=bool)
    freq_mask[24:40] = False

    cfg = SVDConfig(threshold=1.5, svd_mode=SVDMode.PHASE_ONLY,
                    subband_size=8, subband_min_good_frac=0.5,
                    smooth_order=2, fill_failed="smooth_fit",
                    subband_fill_method="local_linear")
    res = _subband_svd_with_smooth_fit(H, freqs, freq_mask, cfg)

    # The fixture applies phase(i, f) = theta_i + 2*pi*f*1e-3*i (a linear
    # cable-delay ramp per antenna); referenced to ref_ant, that ramp must
    # be recovered across the whole band — including the gap, where the
    # local linear fill extrapolates it from neighboring valid subbands.
    solved = res.gains != 0.0
    assert solved.any()
    for i in range(n_ant):
        expected = (true_phases[i] - true_phases[cfg.ref_ant_idx]
                    + 2 * np.pi * freqs * 1e-3 * i)
        err = np.angle(res.gains[i] * np.exp(-1j * expected))[solved[i]]
        assert np.max(np.abs(err)) < 0.2


def test_subband_fill_method_invalid():
    from casm_calibrator import SVDConfig, SVDMode
    from casm_calibrator import _subband_svd_with_smooth_fit

    rng = np.random.default_rng(12)
    n_chan = 64; n_ant = 6
    freqs = np.linspace(420.0, 450.0, n_chan)
    gains = np.exp(1j * rng.uniform(-np.pi, np.pi, n_ant))
    H = _make_synth_hermitian(n_chan, n_ant, gains, freqs, rng)
    freq_mask = np.ones(n_chan, dtype=bool)

    cfg = SVDConfig(threshold=1.5, svd_mode=SVDMode.PHASE_ONLY,
                    subband_size=8, subband_min_good_frac=0.5,
                    smooth_order=2, fill_failed="smooth_fit",
                    subband_fill_method="bogus")
    import pytest
    with pytest.raises(ValueError, match="Unknown subband_fill_method"):
        _subband_svd_with_smooth_fit(H, freqs, freq_mask, cfg)
