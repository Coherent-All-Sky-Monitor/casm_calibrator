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


def _build_hermitian_matrix(vis_full, ant) -> np.ndarray:
    """Build (F, N, N) Hermitian matrix from full-triangle vis + AntennaMapping.

    ``vis_full`` is (T, F, n_inputs*(n_inputs+1)/2) — what
    ``casm_io.read_visibilities`` returns. The matrix is built over the
    *active* antennas only (so any ``ant.with_inactive([...])`` overrides
    are honoured).
    """
    from casm_io.correlator.baselines import triu_flat_index

    active = sorted(ant.active_antennas())
    n_ant = len(active)
    n_bl = vis_full.shape[-1]
    n_inputs = int((-1 + (1 + 8 * n_bl) ** 0.5) / 2)
    if n_inputs * (n_inputs + 1) // 2 != n_bl:
        raise ValueError(
            f"vis must be full upper-triangle (n*(n+1)/2 baselines); "
            f"got {n_bl}, doesn't match any integer n."
        )

    # Time-average to (F, n_bl)
    vis_avg = np.mean(vis_full, axis=0)
    n_chan = vis_avg.shape[0]

    pidx = [ant.packet_index(a) for a in active]

    mat = np.zeros((n_chan, n_ant, n_ant), dtype=np.complex128)
    for i in range(n_ant):
        for j in range(i, n_ant):
            p_i, p_j = pidx[i], pidx[j]
            i_min, i_max = (p_i, p_j) if p_i <= p_j else (p_j, p_i)
            bl = triu_flat_index(n_inputs, i_min, i_max)
            v = vis_avg[:, bl]
            if p_i > p_j:
                v = np.conj(v)        # we have V[j,i] but want V[i,j]
            mat[:, i, j] = v
            if i != j:
                mat[:, j, i] = np.conj(v)
    return mat


def svd_calibrate(fs, ant, data=None, *, config: SVDConfig | None = None
                  ) -> CalibrationResult:
    """Compose-friendly SVD calibration.

    Parameters
    ----------
    fs : :class:`casm_vis_analysis.fringe_stop.FringeStoppedData`
        Output of :func:`casm_vis_analysis.fringe_stop`. Provides
        ``freq_mhz``, ``ref_ant``, ``source`` (provenance).
    ant : :class:`casm_io.AntennaMapping`
        Active set for the matrix (respects ``with_inactive`` overrides).
    data : VisibilityResult or dict-like
        The full-triangle visibilities from
        :func:`casm_io.read_visibilities`. Required: SVD needs the full
        N x N Hermitian matrix, which can only be built from all
        baselines (ref+target subset is not enough).
    config : :class:`SVDConfig`, optional
        Defaults to ``SVDConfig()`` (per-channel, threshold=4.0,
        ref_ant_idx=0, phase-only).

    Returns
    -------
    cal : :class:`CalibrationResult`
    """
    if data is None:
        raise ValueError(
            "svd_calibrate needs the full-triangle visibility data: "
            "pass `data=` (the dict from read_visibilities). The fringe-"
            "stopped subset only carries ref<->target baselines, which "
            "is not enough for the full Hermitian matrix the SVD wants."
        )
    cfg = config if config is not None else SVDConfig()
    ref_ant = int(fs.get("ref_ant"))
    source = fs.get("source", "")

    vis_full = data["vis"] if hasattr(data, "__getitem__") else data.vis
    freq_mhz = np.asarray(fs["freq_mhz"])

    # Build the Hermitian matrix from full-triangle baselines.
    vis_avg = _build_hermitian_matrix(vis_full, ant)

    # Map the user's ref_ant onto the matrix's row index. Mutate the
    # dataclass in place rather than reconstructing it so we don't
    # have to track every SVDConfig field.
    active = sorted(ant.active_antennas())
    if ref_ant in active:
        from dataclasses import replace
        cfg = replace(cfg, ref_ant_idx=active.index(ref_ant))

    result = SVDCalibrator(cfg).calibrate(vis_avg)

    return {
        "gains": result.gains,
        "weights": result.weights,
        "flags": result.flags,
        "rank1_ratios": result.rank1_ratios,
        "singular_values": result.singular_values,
        "freqs_mhz": freq_mhz,
        "ant_ids": np.asarray(active),
        "ref_ant_id": ref_ant,
        "source": source,
        "layout_version": {},   # populated when AntennaMapping carries provenance
    }


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
