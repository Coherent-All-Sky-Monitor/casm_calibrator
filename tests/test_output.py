"""Tests for NPZ output compatibility."""

import tempfile
import os

import numpy as np
import pytest

from casm_calibrator.output import CalibrationWeightsWriter
from casm_calibrator.svd import SVDCalibrator, SVDConfig, SVDMode, SVDResult


class TestCalibrationWeightsWriter:
    def test_write_and_load_npz(self, rank1_vis, freq_mhz, ant_ids):
        """Write NPZ then load with bf_weights_generator.load_calibration_weights."""
        vis_avg, _ = rank1_vis
        n_ant, n_chan = len(ant_ids), len(freq_mhz)

        config = SVDConfig(threshold=2.0, ref_ant_idx=0, svd_mode=SVDMode.PHASE_ONLY)
        result = SVDCalibrator(config).calibrate(vis_avg)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_weights.npz")
            writer = CalibrationWeightsWriter()
            writer.write(
                path=path,
                svd_result=result,
                freqs_mhz=freq_mhz,
                ant_ids=ant_ids,
                ref_ant_id=1,
                source="SUN",
                n_time_averaged=10,
            )

            # Verify file exists and has correct keys
            data = np.load(path, allow_pickle=True)
            assert "weights" in data
            assert "flags" in data
            assert "freqs_hz" in data
            assert "ant_ids" in data
            assert "ref_ant_id" in data
            assert "source" in data

            # Shape checks
            assert data["weights"].shape == (n_ant, n_chan)
            assert data["flags"].shape == (n_chan,)
            assert data["freqs_hz"].shape == (n_chan,)
            assert data["ant_ids"].shape == (n_ant,)
            assert int(data["ref_ant_id"]) == 1
            assert str(data["source"]) == "SUN"

            # Frequencies ascending
            assert np.all(np.diff(data["freqs_hz"]) > 0)

    def test_load_with_bf_weights_generator(self, rank1_vis, freq_mhz, ant_ids):
        """Verify NPZ is loadable by bf_weights_generator.load_calibration_weights."""
        vis_avg, _ = rank1_vis

        config = SVDConfig(threshold=2.0, ref_ant_idx=0, svd_mode=SVDMode.PHASE_ONLY)
        result = SVDCalibrator(config).calibrate(vis_avg)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "compat_test.npz")
            writer = CalibrationWeightsWriter()
            writer.write(
                path=path,
                svd_result=result,
                freqs_mhz=freq_mhz,
                ant_ids=ant_ids,
                ref_ant_id=1,
                source="SUN",
            )

            try:
                from bf_weights_generator.snap_weights import load_calibration_weights

                cal = load_calibration_weights(path)
                assert cal.weights.shape[0] == len(ant_ids)
                assert cal.weights.shape[1] == len(freq_mhz)
                assert len(cal.flags) == len(freq_mhz)
                assert np.all(np.diff(cal.frequencies_hz) > 0)
                assert cal.ref_ant_id == 1
                assert cal.source == "SUN"
            except ImportError:
                pytest.skip("bf_weights_generator not installed")

    def test_rfi_mask_combined(self, rank1_vis, freq_mhz, ant_ids):
        """RFI mask is combined with SVD flags in output."""
        vis_avg, _ = rank1_vis
        n_chan = len(freq_mhz)

        config = SVDConfig(threshold=2.0, ref_ant_idx=0, svd_mode=SVDMode.PHASE_ONLY)
        result = SVDCalibrator(config).calibrate(vis_avg)

        # Flag first 10 channels via RFI
        rfi_mask = np.ones(n_chan, dtype=bool)
        rfi_mask[:10] = False

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "rfi_test.npz")
            CalibrationWeightsWriter().write(
                path=path,
                svd_result=result,
                freqs_mhz=freq_mhz,
                ant_ids=ant_ids,
                ref_ant_id=1,
                source="SUN",
                rfi_mask=rfi_mask,
            )

            data = np.load(path, allow_pickle=True)
            # First 10 channels should be flagged
            assert not np.any(data["flags"][:10])
            # Weights should be zero for flagged channels
            assert np.all(data["weights"][:, :10] == 0)
