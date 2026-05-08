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


def save_calibration(cal, path, *, n_time_averaged=0, rfi_mask=None):
    """Save a :class:`CalibrationResult` dict.

    Output format is chosen by the file extension:

    * ``.h5`` / ``.hdf5`` (preferred): HDF5 with top-level datasets
      ``weights``, ``gains``, ``flags``, ``freqs_hz``, ``freqs_mhz``,
      ``ant_ids``, ``rank1_ratios``, plus attrs ``ref_ant_id``,
      ``source``, ``n_time_averaged``. Consumed by
      ``bf_weights_generator.load_calibration_weights``.
    * ``.npz`` (legacy): same field set, NPZ-archived.

    Reusable from notebooks; mirrors what
    :class:`CalibrationWeightsWriter` does for the legacy ``SVDResult``
    dataclass.

    Parameters
    ----------
    cal : :class:`CalibrationResult`
        From :func:`svd_calibrate`.
    path : str or Path
        Output path. Extension picks the format.
    n_time_averaged : int, optional
        Integration count that went into the time average. Recorded as
        provenance.
    rfi_mask : ndarray of bool, optional
        Per-channel mask (True=good). Combined (AND) with ``cal['flags']``
        before writing. Channels where the combined mask is False have
        their weights and gains zeroed.
    """
    from pathlib import Path
    p = Path(path)

    flags = np.asarray(cal["flags"]).copy()
    if rfi_mask is not None:
        flags = flags & np.asarray(rfi_mask, dtype=bool)

    weights = cal["weights"].copy()
    gains = cal["gains"].copy()
    weights[:, ~flags] = 0.0
    gains[:, ~flags] = 0.0

    freqs_mhz = np.asarray(cal["freqs_mhz"], dtype=np.float64)
    freqs_hz = freqs_mhz * 1e6
    ant_ids = np.asarray(cal["ant_ids"], dtype=int)
    rank1 = np.asarray(cal["rank1_ratios"], dtype=np.float64)

    weights64 = weights.astype(np.complex64)
    gains64 = gains.astype(np.complex64)

    ext = p.suffix.lower()
    if ext in (".h5", ".hdf5"):
        import h5py
        with h5py.File(p, "w") as f:
            f.create_dataset("weights", data=weights64, compression="gzip")
            f.create_dataset("gains",   data=gains64,  compression="gzip")
            f.create_dataset("flags",   data=flags,    compression="gzip")
            f.create_dataset("freqs_hz",  data=freqs_hz)
            f.create_dataset("freqs_mhz", data=freqs_mhz)
            f.create_dataset("ant_ids",   data=ant_ids)
            f.create_dataset("rank1_ratios", data=rank1)
            f.attrs["ref_ant_id"]      = int(cal["ref_ant_id"])
            f.attrs["source"]          = str(cal["source"])
            f.attrs["n_time_averaged"] = int(n_time_averaged)
    elif ext == ".npz":
        np.savez_compressed(
            str(p),
            weights=weights64, gains=gains64, flags=flags,
            freqs_hz=freqs_hz, freqs_mhz=freqs_mhz,
            ant_ids=ant_ids, rank1_ratios=rank1,
            ref_ant_id=int(cal["ref_ant_id"]),
            source=str(cal["source"]),
            n_time_averaged=int(n_time_averaged),
        )
    else:
        raise ValueError(
            f"Unsupported extension {ext!r} for {p}. Use .h5 (preferred) or .npz."
        )


def plot_calibration(cal, *, threshold=None, rfi_ranges=None,
                     output_path=None, ant=None):
    """Notebook-friendly diagnostic plots: σ₁/σ₂, gain phase, gain amplitude.

    Returns a list of three matplotlib Figures (rank1, phase, amplitude).
    When ``output_path`` is provided, also saves PNGs alongside it
    (``{base}_rank1.png``, ``{base}_phase.png``, ``{base}_amp.png``).

    Parameters
    ----------
    cal : :class:`CalibrationResult`
    threshold : float, optional
        Drawn as a horizontal line on the σ₁/σ₂ plot. If None, infers
        from the smallest passing ratio.
    rfi_ranges : list of (lo_mhz, hi_mhz), optional
        Shaded as gray bands on each subplot.
    output_path : str or Path, optional
        If given, save PNGs and don't display.
    ant : :class:`AntennaMapping`, optional
        Used for richer per-antenna labels (e.g. "Ant 9 N21E1"). Falls
        back to "Ant N" if not given.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    flags = np.asarray(cal["flags"])
    ratios = np.asarray(cal["rank1_ratios"])
    gains = cal["gains"]
    freqs_mhz = np.asarray(cal["freqs_mhz"])
    ant_ids = list(cal["ant_ids"])
    n_ant = gains.shape[0]
    rfi_ranges = rfi_ranges or []

    if threshold is None:
        good = ratios[flags & np.isfinite(ratios)]
        threshold = float(good.min()) if good.size else 4.0

    # Friendly per-antenna labels.
    def _label(aid):
        base = f"Ant {aid}"
        if ant is None:
            return base
        try:
            df = ant.dataframe
            row = df.loc[df.antenna_id == aid].iloc[0]
            grid = (f" {row['row']}{row['col']}"
                    if row.get("row") and row.get("col") else "")
            return f"{base}{grid}"
        except Exception:
            return base
    labels = [_label(a) for a in ant_ids]

    src = cal.get("source", "")

    # 1. σ₁ / σ₂ vs frequency
    fig1, ax = plt.subplots(figsize=(13, 4))
    ratios_clip = np.clip(ratios, 0, 50)
    ax.scatter(freqs_mhz, ratios_clip, c=np.where(flags, "C0", "red"),
               s=4, alpha=0.6)
    ax.axhline(threshold, color="orange", ls="--", lw=1.2,
               label=f"threshold = {threshold:.1f}")
    for lo, hi in rfi_ranges:
        ax.axvspan(lo, hi, alpha=0.08, color="gray")
    n_good = int(flags.sum())
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel(r"$\lambda_1 / \lambda_2$")
    ax.set_title(f"Rank-1 quality ({src}) — {n_good}/{len(flags)} channels passed")
    ax.set_xlim(freqs_mhz[0], freqs_mhz[-1])
    ax.set_ylim(0, max(threshold * 1.5, ratios_clip.max() * 1.05))
    ax.legend(loc="upper right")
    fig1.tight_layout()

    # 2. Per-antenna gain phase
    fig2, ax = plt.subplots(figsize=(13, 4))
    for k in range(n_ant):
        phase = np.angle(gains[k], deg=True).astype(float)
        phase[~flags] = np.nan
        ax.plot(freqs_mhz, phase, lw=0.7, alpha=0.8, label=labels[k])
    for lo, hi in rfi_ranges:
        ax.axvspan(lo, hi, alpha=0.08, color="gray")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Gain phase (deg)")
    ax.set_title(f"Per-antenna gain phase ({src})")
    ax.set_xlim(freqs_mhz[0], freqs_mhz[-1])
    ax.legend(fontsize=7, ncol=max(1, n_ant // 6),
              loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig2.tight_layout()

    # 3. Per-antenna gain amplitude
    fig3, ax = plt.subplots(figsize=(13, 4))
    for k in range(n_ant):
        amp = np.abs(gains[k]).astype(float)
        amp[~flags] = np.nan
        ax.plot(freqs_mhz, amp, lw=0.7, alpha=0.8, label=labels[k])
    for lo, hi in rfi_ranges:
        ax.axvspan(lo, hi, alpha=0.08, color="gray")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Gain amplitude")
    ax.set_title(f"Per-antenna gain amplitude ({src})")
    ax.set_xlim(freqs_mhz[0], freqs_mhz[-1])
    ax.legend(fontsize=7, ncol=max(1, n_ant // 6),
              loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig3.tight_layout()

    figs = [fig1, fig2, fig3]
    if output_path is not None:
        base = Path(output_path)
        stem, suffix = base.with_suffix(""), base.suffix or ".png"
        for fig, name in zip(figs, ("rank1", "phase", "amp")):
            fig.savefig(f"{stem}_{name}{suffix}", dpi=150, bbox_inches="tight")
            plt.close(fig)
    else:
        plt.show()
    return figs


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
    "save_calibration",
    "plot_calibration",
]
