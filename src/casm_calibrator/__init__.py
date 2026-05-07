"""CASM SVD-based beamformer calibration package."""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from .output import CalibrationWeightsWriter
from .svd import SVDCalibrator, SVDConfig, SVDMode, SVDResult
from .visibility import VisibilityLoader, VisibilityMatrix


class CalibrationResult(TypedDict, total=False):
    """Compose-friendly calibration output.

    Same content as :class:`SVDResult` plus provenance fields. Returned
    by :func:`svd_calibrate`. ``layout_version`` is stamped when
    available so downstream beam-weight generation can record which
    antenna positions the cal was built against.
    """
    gains: np.ndarray            # (n_chan, n_ant) complex
    weights: np.ndarray          # post-flag complex weights
    flags: np.ndarray            # (n_chan,) bool
    rank1_ratios: np.ndarray
    singular_values: np.ndarray
    freqs_mhz: np.ndarray
    ant_ids: np.ndarray
    ref_ant_id: int
    source: str
    layout_version: dict


def svd_calibrate(fs, ant, *, ref_ant=None, config: SVDConfig | None = None,
                  source: str | None = None) -> CalibrationResult:
    """Compose-friendly SVD calibration.

    Parameters
    ----------
    fs : :class:`casm_vis_analysis.fringe_stop.FringeStoppedData`
        Output of :func:`casm_vis_analysis.fringe_stop`. Must carry
        ``vis_for_calibration`` (or ``vis_stopped``), ``freq_mhz``,
        ``ref_ant``, ``source``.
    ant : :class:`casm_io.AntennaMapping`
    ref_ant : int, optional
        Reference antenna ID. Defaults to ``fs['ref_ant']``.
    config : :class:`SVDConfig`, optional
        SVD configuration. Defaults to ``SVDConfig()``.
    source : str, optional
        Source name. Defaults to ``fs['source']``.

    Returns
    -------
    cal : :class:`CalibrationResult`
        Dict with gains, weights, flags, rank1_ratios, singular_values,
        plus provenance.
    """
    cfg = config if config is not None else SVDConfig()
    if ref_ant is None:
        ref_ant = int(fs.get("ref_ant"))
    if source is None:
        source = fs.get("source", "")

    # Derive the time-averaged Hermitian visibility matrix the SVD wants.
    vis_fs = fs.get("vis_for_calibration", fs.get("vis_stopped"))
    if vis_fs is None:
        raise ValueError(
            "fs must include 'vis_for_calibration' or 'vis_stopped'."
        )
    freq_mhz = np.asarray(fs["freq_mhz"])

    # vis_fs has shape (T, F, n_targets). Time-average; reshape to
    # (n_chan, n_ant, n_ant) Hermitian. Phase-only baseline subset (ref-target)
    # only fills the first row/column; full-matrix construction is the
    # responsibility of callers using ``casm_calibrator.SVDCalibrator``
    # directly with their own VisibilityMatrix builder.
    if vis_fs.ndim != 3:
        raise ValueError(f"vis_fs must be (T, F, n_bl); got {vis_fs.shape}")
    vis_avg_baselines = np.mean(vis_fs, axis=0)  # (F, n_bl)

    # The full SVD path needs an N x N matrix. We can build it from the
    # ref-target baselines: row i, col j gets vis_baseline[ref<->i] (or
    # similar). For now, surface a clear-failure path if the caller
    # didn't pre-build the matrix; full construction lives in the
    # existing VisibilityMatrix builder which loads from disk.
    raise NotImplementedError(
        "svd_calibrate(fs, ant): the dict-wrapper for full-matrix SVD "
        "construction lands with the VisibilityMatrix dedup. For now use "
        "casm_calibrator.SVDCalibrator(config).calibrate(vis_avg) directly "
        "with a VisibilityMatrix built via the legacy VisibilityLoader."
    )


__all__ = [
    "SVDCalibrator",
    "SVDConfig",
    "SVDMode",
    "SVDResult",
    "VisibilityLoader",
    "VisibilityMatrix",
    "CalibrationWeightsWriter",
    "CalibrationResult",
    "svd_calibrate",
]
