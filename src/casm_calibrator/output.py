"""NPZ output compatible with bf_weights_generator.load_calibration_weights."""

import numpy as np

from .svd import SVDResult


class CalibrationWeightsWriter:
    """Write SVD calibration results to NPZ format.

    The output is compatible with
    ``bf_weights_generator.load_calibration_weights()``.
    """

    def write(
        self,
        path: str,
        svd_result: SVDResult,
        freqs_mhz: np.ndarray,
        ant_ids: np.ndarray,
        ref_ant_id: int,
        source: str,
        n_time_averaged: int = 0,
        rfi_mask: np.ndarray | None = None,
    ):
        """Write calibration weights to NPZ.

        Parameters
        ----------
        path : str
            Output .npz file path.
        svd_result : SVDResult
            SVD calibration results.
        freqs_mhz : ndarray, shape (n_chan,)
            Frequencies in MHz (ascending).
        ant_ids : ndarray, shape (n_ant,)
            1-indexed antenna IDs.
        ref_ant_id : int
            Reference antenna ID.
        source : str
            Calibrator source name.
        n_time_averaged : int
            Number of time integrations averaged.
        rfi_mask : ndarray, shape (n_chan,), optional
            Boolean RFI mask (True=good). Combined with SVD flags.
        """
        flags_combined = svd_result.flags.copy()
        if rfi_mask is not None:
            flags_combined &= rfi_mask

        weights = svd_result.weights.copy()
        gains = svd_result.gains.copy()
        # Zero out flagged channels
        weights[:, ~flags_combined] = 0.0
        gains[:, ~flags_combined] = 0.0

        freqs_hz = freqs_mhz.astype(np.float64) * 1e6

        save_dict = dict(
            weights=weights.astype(np.complex64),
            flags=flags_combined,
            freqs_hz=freqs_hz,
            freqs_mhz=freqs_mhz.astype(np.float64),
            ant_ids=ant_ids.astype(int),
            ref_ant_id=int(ref_ant_id),
            source=str(source),
            gains=gains.astype(np.complex64),
            rank1_ratios=svd_result.rank1_ratios.astype(np.float64),
            threshold=float(
                svd_result.block_metadata.get("threshold", 0.0)
                if svd_result.block_metadata
                else 0.0
            ),
            n_time_averaged=int(n_time_averaged),
        )

        # Add block metadata if present
        if svd_result.block_metadata:
            for key in ("block_flags", "block_ratios", "block_centers", "block_size", "fill_mode"):
                if key in svd_result.block_metadata:
                    save_dict[key] = svd_result.block_metadata[key]

        np.savez_compressed(path, **save_dict)
