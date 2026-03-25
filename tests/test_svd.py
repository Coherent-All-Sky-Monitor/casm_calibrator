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


class TestPerBlockSVD:
    def test_block_svd_basic(self, rank1_vis, n_ant, n_chan):
        """Block SVD with known rank-1 data."""
        vis_avg, true_gains = rank1_vis
        config = SVDConfig(
            threshold=2.0, ref_ant_idx=0, svd_mode=SVDMode.PHASE_ONLY,
            block_size=8, fill_mode="interpolate",
        )
        result = SVDCalibrator(config).calibrate(vis_avg)

        assert result.gains.shape == (n_ant, n_chan)
        # Most channels should be good
        assert np.sum(result.flags) > n_chan * 0.5
        # Block metadata should be present
        assert "block_flags" in result.block_metadata
        assert "block_size" in result.block_metadata

    def test_block_interpolation(self, n_ant):
        """Failed blocks get interpolated from neighbors."""
        n_chan = 40
        block_size = 8
        rng = np.random.default_rng(42)

        # Create vis where block 2 (channels 16-23) is random noise
        phases = rng.uniform(-np.pi, np.pi, (n_ant, n_chan))
        phases[0, :] = 0.0
        true_gains = np.exp(1j * phases)

        vis_avg = np.zeros((n_chan, n_ant, n_ant), dtype=np.complex128)
        for f in range(n_chan):
            g = true_gains[:, f]
            vis_avg[f] = np.outer(g, np.conj(g))

        # Corrupt block 2
        vis_avg[16:24] = rng.standard_normal((8, n_ant, n_ant)) + \
                         1j * rng.standard_normal((8, n_ant, n_ant))
        for f in range(16, 24):
            vis_avg[f] = 0.5 * (vis_avg[f] + vis_avg[f].conj().T)

        config = SVDConfig(
            threshold=3.0, ref_ant_idx=0, svd_mode=SVDMode.PHASE_ONLY,
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
