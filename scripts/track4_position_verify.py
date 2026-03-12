#!/usr/bin/env python3
"""
Track 4: Antenna Position Verification.

Load recent visibility data, fringe-stop toward Sun, and for each antenna
fit a position delta that minimizes the fringe-stop residual.

Compare fitted vs CSV positions. Deltas > 5cm indicate layout errors.
"""

import os
import sys
import numpy as np
from scipy.optimize import minimize

# ── Config ──
DATA_DIR = "/mnt/nvme3/data/casm/visibilities_64ant"
OBS_ID = "2026-03-11-12:17:47"
FORMAT = "layout_64ant"
LAYOUT = os.path.expanduser("~/software/dev/antenna_layouts/antenna_layout_current.csv")
SOURCE = "sun"
REF_ANT = 5

FREQ_LOW_MHZ = 420.0
FREQ_HIGH_MHZ = 460.0

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "position_fit")
os.makedirs(OUT_DIR, exist_ok=True)

VENV = os.path.expanduser("~/software/dev/casm_venvs/casm_offline_env")


def main():
    print("=" * 70)
    print("TRACK 4: ANTENNA POSITION VERIFICATION")
    print("=" * 70)

    # ── Activate venv imports ──
    sys.path.insert(0, os.path.join(VENV, "lib/python3.12/site-packages"))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

    from casm_io.correlator.mapping import AntennaMapping
    from casm_io.correlator.reader import VisibilityReader
    from casm_io.correlator.formats import load_format
    from casm_io.constants import C_LIGHT_M_S
    from casm_vis_analysis.sources import source_enu, find_transit_window

    from casm_calibrator.visibility import VisibilityLoader, VisibilityMatrix
    from casm_calibrator.fringe_stop import FringeStopMatrix

    # ── Load data ──
    print(f"\n1. Loading visibilities: {OBS_ID}")
    mapping = AntennaMapping.load(LAYOUT)
    ant_ids = np.array(mapping.active_antennas(), dtype=int)
    n_ant = len(ant_ids)
    print(f"   {n_ant} antennas: {ant_ids}")

    ref_ant_idx = int(np.where(ant_ids == REF_ANT)[0][0])

    fmt = load_format(FORMAT)
    loader = VisibilityLoader(mapping)
    vis_matrix = loader.load(DATA_DIR, OBS_ID, fmt=fmt, nfiles=8)
    print(f"   vis shape: {vis_matrix.vis.shape}")
    print(f"   freq range: {vis_matrix.freq_mhz[0]:.2f} - {vis_matrix.freq_mhz[-1]:.2f} MHz")

    # ── Find transit window ──
    print(f"\n2. Finding transit window...")
    try:
        i_start, i_end = find_transit_window(SOURCE, vis_matrix.time_unix, min_alt_deg=10.0)
        n_transit = i_end - i_start + 1
        print(f"   Transit: {i_start}-{i_end} ({n_transit} integrations)")
    except Exception as e:
        print(f"   Transit window failed: {e}")
        print(f"   Using all integrations")
        i_start, i_end = 0, vis_matrix.vis.shape[0] - 1
        n_transit = i_end - i_start + 1

    vis_transit = VisibilityMatrix(
        vis=vis_matrix.vis[i_start:i_end + 1],
        freq_mhz=vis_matrix.freq_mhz,
        time_unix=vis_matrix.time_unix[i_start:i_end + 1],
        ant_ids=vis_matrix.ant_ids,
        positions_enu=vis_matrix.positions_enu,
    )

    # ── Fringe-stop toward source ──
    print(f"\n3. Fringe-stopping toward {SOURCE}...")
    vis_fs = FringeStopMatrix()(vis_transit, SOURCE)

    # ── Time-average ──
    print(f"\n4. Time-averaging {vis_fs.vis.shape[0]} integrations...")
    vis_avg = np.mean(vis_fs.vis, axis=0)  # (F, n_ant, n_ant)
    print(f"   vis_avg shape: {vis_avg.shape}")

    # ── Restrict to clean band ──
    freqs_mhz = vis_fs.freq_mhz
    band_mask = (freqs_mhz >= FREQ_LOW_MHZ) & (freqs_mhz <= FREQ_HIGH_MHZ)
    n_chan_band = int(np.sum(band_mask))
    vis_band = vis_avg[band_mask]
    freqs_band = freqs_mhz[band_mask]
    freqs_hz_band = freqs_band * 1e6
    print(f"   Clean band: {FREQ_LOW_MHZ}-{FREQ_HIGH_MHZ} MHz, {n_chan_band} channels")

    # ── Get source direction at transit center ──
    t_mid = vis_fs.time_unix[len(vis_fs.time_unix) // 2]
    s_enu = source_enu(SOURCE, np.array([t_mid]))[0]  # (3,)
    print(f"   Source ENU: {s_enu}")

    positions = vis_fs.positions_enu  # (n_ant, 3)

    # ── Fit position deltas ──
    print(f"\n5. Fitting position deltas...")
    print(f"   Reference antenna: Ant {REF_ANT} (index {ref_ant_idx})")

    fitted_deltas = np.zeros((n_ant, 3))
    fit_quality = np.zeros(n_ant)

    for ant in range(n_ant):
        if ant == ref_ant_idx:
            continue

        # Extract cross-correlation with ref antenna
        # vis_band[f, ref, ant] should be ~real after fringe-stopping
        vis_cross = vis_band[:, ref_ant_idx, ant]  # (n_chan_band,)

        # Current phase residual
        phase_residual = np.angle(vis_cross)
        amp = np.abs(vis_cross)

        # Weight by amplitude (high SNR channels contribute more)
        weights = amp / np.max(amp)

        def cost(delta_pos):
            """Cost function: weighted phase residual after position correction."""
            dx, dy, dz = delta_pos
            # Extra delay from position delta
            d_tau = (s_enu[0] * dx + s_enu[1] * dy + s_enu[2] * dz) / C_LIGHT_M_S
            # Phase correction
            d_phi = -2 * np.pi * freqs_hz_band * d_tau
            # Residual after correction
            corrected_phase = phase_residual - d_phi
            # Minimize weighted phase variance (modulo 2pi)
            return np.sum(weights * (1 - np.cos(corrected_phase)))

        result = minimize(cost, x0=[0.0, 0.0, 0.0], method="Nelder-Mead",
                          options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 5000})
        fitted_deltas[ant] = result.x
        fit_quality[ant] = 1.0 - result.fun / np.sum(weights)

    # ── Print results ──
    print(f"\n{'=' * 80}")
    print("POSITION FIT RESULTS")
    print(f"{'=' * 80}")
    print(f"  {'Ant':>4s}  {'dE (cm)':>8s}  {'dN (cm)':>8s}  {'dU (cm)':>8s}  "
          f"{'|delta| (cm)':>12s}  {'Quality':>8s}  {'Status':>10s}")
    print(f"  {'-' * 70}")

    for ant in range(n_ant):
        ant_id = ant_ids[ant]
        dx, dy, dz = fitted_deltas[ant] * 100  # m -> cm
        dr = np.sqrt(dx**2 + dy**2 + dz**2)
        q = fit_quality[ant]

        if ant == ref_ant_idx:
            status = "REF"
        elif dr > 10:
            status = "LARGE!"
        elif dr > 5:
            status = "check"
        else:
            status = "OK"

        flag = " <-- ANT 4" if ant_id == 4 else ""
        print(f"  {ant_id:4d}  {dx:+8.2f}  {dy:+8.2f}  {dz:+8.2f}  "
              f"{dr:12.2f}  {q:8.4f}  {status}{flag}")

    # ── Coherence check ──
    print(f"\n  Coherence vs ref ant (after fringe-stop, band-averaged):")
    for ant in range(n_ant):
        if ant == ref_ant_idx:
            continue
        vis_cross = vis_band[:, ref_ant_idx, ant]
        coherence = np.abs(np.mean(vis_cross)) / np.mean(np.abs(vis_cross))
        phase_rms = np.degrees(np.std(np.angle(vis_cross)))
        flag = " <-- ANT 4" if ant_ids[ant] == 4 else ""
        print(f"  Ant {ant_ids[ant]:2d}: coherence={coherence:.4f}  phase_rms={phase_rms:.1f}°{flag}")

    # ── Decision ──
    max_delta = np.max(np.sqrt(np.sum(fitted_deltas**2, axis=1)))
    print(f"\n{'=' * 70}")
    print(f"DECISION:")
    print(f"  Max position delta: {max_delta*100:.1f} cm")
    if max_delta < 0.05:
        print(f"  -> Positions ACCURATE (<5 cm). Layout is fine.")
    elif max_delta < 0.10:
        print(f"  -> Minor position corrections needed (5-10 cm).")
    else:
        print(f"  -> SIGNIFICANT position errors (>{max_delta*100:.0f} cm). Update layout CSV!")

    # ── Save results ──
    out_npz = os.path.join(OUT_DIR, "position_fit_results.npz")
    np.savez_compressed(out_npz,
                        ant_ids=ant_ids,
                        fitted_deltas=fitted_deltas,
                        fit_quality=fit_quality,
                        positions_csv=positions,
                        source_enu=s_enu,
                        ref_ant=REF_ANT)
    print(f"\n  Saved: {out_npz}")

    # ── Plot ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = ["dE (cm)", "dN (cm)", "dU (cm)"]
    for k, (ax, label) in enumerate(zip(axes, labels)):
        vals = fitted_deltas[:, k] * 100
        colors = ["red" if ant_ids[i] == 4 else "C0" for i in range(n_ant)]
        ax.bar(range(n_ant), vals, color=colors)
        ax.set_xlabel("Antenna")
        ax.set_ylabel(label)
        ax.set_xticks(range(n_ant))
        ax.set_xticklabels([str(a) for a in ant_ids], fontsize=7)
        ax.axhline(0, color="gray", lw=0.5)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Fitted Position Deltas (Ant 4 in red)")
    fig.tight_layout()
    plot_path = os.path.join(OUT_DIR, "position_deltas.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {plot_path}")

    print(f"\n{'=' * 70}")
    print("TRACK 4 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
