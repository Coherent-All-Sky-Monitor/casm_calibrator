#!/usr/bin/env python3
"""
Track 5: Verify Mar10 HDF5 weight consistency with Mar11 data.

Three approaches:
  A) Re-run SVD on Mar11 with relaxed parameters to extract delays.
     Also includes Mar06 comparison for multi-day trend analysis.
  B) Direct gain-ratio cross-check: phase of g_Mar11 * conj(g_Mar10)
     should give the delay drift (validates Part A delay extraction).
  C) Visibility residual check: apply Mar10 cal to Mar11 fringe-stopped
     visibilities using BLOCK-averaged data (avoids phase wrapping on
     per-channel data).

If delays are stable (< 2 ns RMS), Mar10 weights are good to deploy.
"""

import os
import sys
import subprocess

import numpy as np

# ── Paths ──
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, SCRIPTS_DIR)
from compare_cal_delays import extract_delays

DATA_DIR = "/mnt/nvme3/data/casm/visibilities_64ant"
LAYOUT = os.path.expanduser(
    "~/software/dev/antenna_layouts/antenna_layout_current.csv"
)
FORMAT = "layout_64ant"
OUT_DIR = os.path.join(PROJECT_DIR, "results", "mar10_verification")
os.makedirs(OUT_DIR, exist_ok=True)

VENV_PYTHON = os.path.expanduser(
    "~/software/dev/casm_venvs/casm_offline_env/bin/python"
)

MAR10_NPZ = os.path.join(PROJECT_DIR, "results", "svd_weights_mar10.npz")
MAR06_NPZ = os.path.join(PROJECT_DIR, "results", "multi_day_svd",
                         "svd_weights_2026-03-06.npz")
MAR11_OBS = "2026-03-11-12:17:47"
BLOCK_SIZE = 32


# =========================================================================
# Part A: Re-run SVD on Mar11 with relaxed parameters
# =========================================================================

def attempt_mar11_svd():
    """Try SVD on Mar11 with relaxed threshold and min-alt."""
    print("=" * 70)
    print("PART A: SVD on Mar11 data (relaxed parameters)")
    print("=" * 70)

    out_npz = os.path.join(OUT_DIR, "svd_weights_mar11_relaxed.npz")
    out_plots = os.path.join(OUT_DIR, "svd_diag_mar11_relaxed.pdf")

    if os.path.exists(out_npz):
        print(f"  Already exists: {out_npz}")
        return out_npz

    cmd = [
        VENV_PYTHON, "-m", "casm_calibrator.cli",
        "--data-dir", DATA_DIR,
        "--obs", MAR11_OBS,
        "--format", FORMAT,
        "--layout", LAYOUT,
        "--output", out_npz,
        "--plots", out_plots,
        "--source", "sun",
        "--ref-ant", "5",
        "--svd-mode", "phase-only",
        "--block-size", str(BLOCK_SIZE),
        "--threshold", "1.5",       # relaxed from 2.0
        "--fill-mode", "interpolate",
        "--rfi-mask-range", "375", "390",
        "--min-alt", "5",           # relaxed from 10
        "--nfiles", "9",            # all available files
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"  SVD FAILED for Mar11 (exit code {result.returncode})")
        print(f"  STDERR: {result.stderr[-500:]}")
        print(f"  STDOUT: {result.stdout[-500:]}")
        return None

    print(result.stdout[-800:] if len(result.stdout) > 800 else result.stdout)
    return out_npz


def load_npz_gains(npz_path):
    """Load gains, flags, freqs from an NPZ, ensuring ascending freq order."""
    d = np.load(npz_path, allow_pickle=True)
    gains = d["gains"]
    flags = d["flags"]
    freqs_hz = d["freqs_hz"]
    if freqs_hz[1] < freqs_hz[0]:
        freqs_hz = freqs_hz[::-1]
        gains = gains[:, ::-1]
        flags = flags[::-1]
    return gains, flags, freqs_hz, d


def multi_day_delay_comparison(mar11_npz):
    """Extract and compare delays across Mar06, Mar10, Mar11."""
    print("\n" + "=" * 70)
    print("MULTI-DAY DELAY COMPARISON")
    print("=" * 70)

    datasets = [("Mar06", MAR06_NPZ), ("Mar10", MAR10_NPZ), ("Mar11", mar11_npz)]
    available = [(label, path) for label, path in datasets if os.path.exists(path)]

    d_ref = np.load(MAR10_NPZ, allow_pickle=True)
    ant_ids = d_ref["ant_ids"]
    ref_ant_id = int(d_ref["ref_ant_id"])
    ref_ant_idx = int(np.where(ant_ids == ref_ant_id)[0][0])
    n_ant = len(ant_ids)

    all_delays = {}
    all_r2 = {}
    for label, path in available:
        gains, flags, freqs_hz, _ = load_npz_gains(path)
        n_good = flags.sum()
        print(f"\n  {label}: {n_good}/{len(flags)} good channels, "
              f"freq range: {freqs_hz[0]/1e6:.1f}-{freqs_hz[-1]/1e6:.1f} MHz")
        delays, offsets, r2 = extract_delays(gains, flags, freqs_hz, ref_ant_idx)
        all_delays[label] = delays
        all_r2[label] = r2

    # Print comparison table
    labels = [l for l, _ in available]
    print(f"\n{'Ant':>4s}", end="")
    for label in labels:
        print(f"  {label + ' (ns)':>14s}", end="")
    if len(labels) >= 2:
        print(f"  {'Diff last':>12s}", end="")
    print(f"  {'R2 (last)':>9s}  Status")
    print("-" * (4 + 16 * len(labels) + 25))

    last = labels[-1]
    prev = labels[-2] if len(labels) >= 2 else None
    diff = all_delays[last] - all_delays[prev] if prev else np.zeros(n_ant)

    for k in range(n_ant):
        row = f"{ant_ids[k]:4d}"
        for label in labels:
            row += f"  {all_delays[label][k]:+14.2f}"
        if prev:
            row += f"  {diff[k]:+12.2f}"
        row += f"  {all_r2[last][k]:9.4f}"
        status = "OK" if abs(diff[k]) < 2.0 else "DRIFT" if abs(diff[k]) < 5.0 else "*** BAD"
        row += f"  {status}"
        print(row)

    rms = np.sqrt(np.mean(diff**2))
    max_diff = np.max(np.abs(diff))
    worst_ant = ant_ids[np.argmax(np.abs(diff))]

    print(f"\n  {last} vs {prev}:")
    print(f"    RMS delay change:  {rms:.2f} ns")
    print(f"    Max delay change:  {max_diff:.2f} ns (Ant {worst_ant})")

    # Check if Mar06->Mar10 drift matches Mar10->Mar11 direction
    if "Mar06" in all_delays and "Mar10" in all_delays:
        diff_06_10 = all_delays["Mar10"] - all_delays["Mar06"]
        rms_06_10 = np.sqrt(np.mean(diff_06_10**2))
        print(f"\n  Mar10 vs Mar06:")
        print(f"    RMS delay change:  {rms_06_10:.2f} ns")

    return diff, rms, ant_ids, all_delays


# =========================================================================
# Part B: Direct gain-ratio cross-check
# =========================================================================

def gain_ratio_check(mar11_npz):
    """Compute delay drift via phase of g_Mar11 * conj(g_Mar10)."""
    print("\n" + "=" * 70)
    print("PART B: Direct gain-ratio cross-check")
    print("=" * 70)

    g10, f10, fhz10, d10 = load_npz_gains(MAR10_NPZ)
    g11, f11, fhz11, d11 = load_npz_gains(mar11_npz)

    ant_ids = d10["ant_ids"]
    ref_ant_id = int(d10["ref_ant_id"])
    ref_ant_idx = int(np.where(ant_ids == ref_ant_id)[0][0])
    n_ant = len(ant_ids)

    # Use channels good in BOTH
    good = f10 & f11
    n_good = good.sum()
    print(f"  Channels good in both: {n_good}/{len(good)}")

    freqs_good = fhz10[good]
    print(f"  Freq range: {freqs_good[0]/1e6:.1f}-{freqs_good[-1]/1e6:.1f} MHz")

    # Gain ratio: g_ratio = g_Mar11 * conj(g_Mar10)
    # Phase slope gives delay drift directly
    g_ratio = g11[:, good] * np.conj(g10[:, good])

    print(f"\n{'Ant':>4s}  {'Ratio delay (ns)':>18s}  {'R2':>8s}  {'Status':>10s}")
    print("-" * 50)

    ratio_delays = np.zeros(n_ant)
    ratio_r2 = np.zeros(n_ant)

    for k in range(n_ant):
        phase = np.unwrap(np.angle(g_ratio[k]))
        coeffs = np.polyfit(freqs_good, phase, 1)
        slope_ns = coeffs[0] / (2 * np.pi) * 1e9

        predicted = np.polyval(coeffs, freqs_good)
        ss_res = np.sum((phase - predicted) ** 2)
        ss_tot = np.sum((phase - np.mean(phase)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        ratio_delays[k] = slope_ns
        ratio_r2[k] = r2

    # Make relative to ref
    ratio_delays -= ratio_delays[ref_ant_idx]

    for k in range(n_ant):
        status = "OK" if abs(ratio_delays[k]) < 2.0 else \
                 "DRIFT" if abs(ratio_delays[k]) < 5.0 else "*** BAD"
        ref_tag = " (ref)" if k == ref_ant_idx else ""
        print(f"{ant_ids[k]:4d}  {ratio_delays[k]:+18.2f}  {ratio_r2[k]:8.4f}  "
              f"{status:>10s}{ref_tag}")

    non_ref = [k for k in range(n_ant) if k != ref_ant_idx]
    rms = np.sqrt(np.mean(ratio_delays[non_ref]**2))
    max_dr = np.max(np.abs(ratio_delays[non_ref]))
    worst = ant_ids[non_ref[np.argmax(np.abs(ratio_delays[non_ref]))]]
    print(f"\n  RMS delay drift:  {rms:.2f} ns")
    print(f"  Max delay drift:  {max_dr:.2f} ns (Ant {worst})")

    return ratio_delays, ratio_r2


# =========================================================================
# Part C: Visibility residual check (block-averaged)
# =========================================================================

def block_residual_phase_check():
    """Load Mar11 vis, fringe-stop, apply Mar10 weights, check block-averaged residuals."""
    print("\n" + "=" * 70)
    print("PART C: Block-averaged visibility residual check")
    print("=" * 70)

    from casm_io.correlator.mapping import AntennaMapping
    from casm_io.correlator.formats import load_format
    from casm_vis_analysis.sources import find_transit_window

    from casm_calibrator.visibility import VisibilityLoader, VisibilityMatrix
    from casm_calibrator.fringe_stop import FringeStopMatrix

    # 1. Load antenna layout
    print("\n  Loading antenna layout...")
    mapping = AntennaMapping.load(LAYOUT)
    ant_ids = np.array(mapping.active_antennas(), dtype=int)
    n_ant = len(ant_ids)

    # 2. Load Mar11 visibilities
    print(f"\n  Loading Mar11 visibilities ({MAR11_OBS}, all 9 files)...")
    fmt = load_format(FORMAT)
    loader = VisibilityLoader(mapping)
    vis_matrix = loader.load(DATA_DIR, MAR11_OBS, fmt=fmt, nfiles=9)
    print(f"    vis shape: {vis_matrix.vis.shape}")

    # 3. Find transit window
    print("\n  Finding transit window (min_alt=5 deg)...")
    i_start, i_end = find_transit_window("sun", vis_matrix.time_unix, min_alt_deg=5.0)
    n_transit = i_end - i_start + 1
    print(f"    Transit: indices {i_start}-{i_end} ({n_transit} integrations)")

    vis_matrix = VisibilityMatrix(
        vis=vis_matrix.vis[i_start:i_end + 1],
        freq_mhz=vis_matrix.freq_mhz,
        time_unix=vis_matrix.time_unix[i_start:i_end + 1],
        ant_ids=vis_matrix.ant_ids,
        positions_enu=vis_matrix.positions_enu,
    )

    # 4. Fringe-stop
    print("\n  Fringe-stopping toward Sun...")
    vis_fs = FringeStopMatrix()(vis_matrix, "sun")

    # 5. Time-average
    vis_avg = np.mean(vis_fs.vis, axis=0)  # (F, n_ant, n_ant)
    n_chan = vis_avg.shape[0]
    freqs_mhz = vis_fs.freq_mhz
    freqs_hz = freqs_mhz * 1e6

    # 6. Load Mar10 calibration gains
    d10 = np.load(MAR10_NPZ, allow_pickle=True)
    cal_gains = d10["gains"]
    cal_freqs_hz = d10["freqs_hz"]
    ref_ant_id = int(d10["ref_ant_id"])
    ref_ant_idx = int(np.where(ant_ids == ref_ant_id)[0][0])
    if cal_freqs_hz[1] < cal_freqs_hz[0]:
        cal_freqs_hz = cal_freqs_hz[::-1]
        cal_gains = cal_gains[:, ::-1]

    # 7. Apply calibration: V_cal[f,i,j] = conj(g_i) * V[f,i,j] * g_j
    print("\n  Applying Mar10 calibration to Mar11 data...")
    cal_weights = np.conj(cal_gains)
    vis_cal = np.zeros_like(vis_avg)
    for f in range(n_chan):
        w = cal_weights[:, f]
        vis_cal[f] = w[:, np.newaxis] * vis_avg[f] * np.conj(w[np.newaxis, :])

    # 8. Block-average the calibrated visibilities
    # This gives phase jumps large enough for np.unwrap to work correctly.
    # Per-channel residual wrapping is too fine-grained (0.65 deg/chan for 60ns
    # drift) — np.unwrap can't detect wraps. Block-averaging gives ~21 deg/block.
    n_blocks = int(np.ceil(n_chan / BLOCK_SIZE))
    block_vis_cal = np.zeros((n_blocks, n_ant, n_ant), dtype=np.complex128)
    block_freqs_hz = np.zeros(n_blocks)
    block_good = np.zeros(n_blocks, dtype=bool)

    for b in range(n_blocks):
        ch_start = b * BLOCK_SIZE
        ch_end = min((b + 1) * BLOCK_SIZE, n_chan)
        block_vis_cal[b] = np.mean(vis_cal[ch_start:ch_end], axis=0)
        block_freqs_hz[b] = np.mean(freqs_hz[ch_start:ch_end])
        # Flag blocks in the 375-390 MHz RFI range
        block_freq_mhz = block_freqs_hz[b] / 1e6
        block_good[b] = block_freq_mhz >= 390.0

    n_good_blocks = block_good.sum()
    print(f"\n  Block-averaged: {n_blocks} blocks, {n_good_blocks} good (>390 MHz)")

    # 9. Extract per-antenna residual delay from block-averaged phases
    print(f"\n  Block-averaged residual phase vs ref ant {ref_ant_id}:")
    ant_residual_slope_ns = np.zeros(n_ant)
    ant_residual_rms = np.zeros(n_ant)
    ant_coherence = np.zeros(n_ant)

    freqs_good_blocks = block_freqs_hz[block_good]

    print(f"\n{'Ant':>4s}  {'Resid delay (ns)':>18s}  {'Phase RMS (deg)':>16s}  "
          f"{'Coherence':>10s}  {'Status':>10s}")
    print("-" * 70)

    for k in range(n_ant):
        if k == ref_ant_idx:
            print(f"{ant_ids[k]:4d}  {'0.00 (ref)':>18s}  {'0.00 (ref)':>16s}  "
                  f"{'--- (ref)':>10s}  {'REF':>10s}")
            continue

        i, j = min(k, ref_ant_idx), max(k, ref_ant_idx)
        bl_vis = block_vis_cal[block_good, i, j]

        # Coherence: |<V>| / <|V|> — measures phase stability
        coherence = np.abs(np.mean(bl_vis)) / np.mean(np.abs(bl_vis))
        ant_coherence[k] = coherence

        bl_phase = np.angle(bl_vis)
        if k > ref_ant_idx:
            bl_phase = -bl_phase

        # Phase RMS (wrapped)
        phase_rms_deg = np.degrees(np.sqrt(np.mean(bl_phase**2)))
        ant_residual_rms[k] = phase_rms_deg

        # Unwrap and fit
        phase_unwrap = np.unwrap(bl_phase)
        coeffs = np.polyfit(freqs_good_blocks, phase_unwrap, 1)
        slope_ns = coeffs[0] / (2 * np.pi) * 1e9
        ant_residual_slope_ns[k] = slope_ns

        status = "OK" if abs(slope_ns) < 2.0 else \
                 "DRIFT" if abs(slope_ns) < 5.0 else "*** BAD"
        print(f"{ant_ids[k]:4d}  {slope_ns:+18.2f}  {phase_rms_deg:16.2f}  "
              f"{coherence:10.4f}  {status:>10s}")

    non_ref = [k for k in range(n_ant) if k != ref_ant_idx]
    rms_all = np.sqrt(np.mean(ant_residual_slope_ns[non_ref]**2))
    max_slope = np.max(np.abs(ant_residual_slope_ns[non_ref]))
    worst_ant = ant_ids[non_ref[np.argmax(np.abs(ant_residual_slope_ns[non_ref]))]]
    mean_coh = np.mean(ant_coherence[non_ref])

    print(f"\n  RMS residual delay:   {rms_all:.2f} ns")
    print(f"  Max residual delay:   {max_slope:.2f} ns (Ant {worst_ant})")
    print(f"  Mean phase RMS:       {np.mean(ant_residual_rms[non_ref]):.2f} deg")
    print(f"  Mean coherence:       {mean_coh:.4f}")

    return (ant_residual_slope_ns, ant_residual_rms, ant_coherence,
            ant_ids, block_vis_cal, block_freqs_hz, block_good, ref_ant_idx)


# =========================================================================
# Plots
# =========================================================================

def make_plots(residual_slopes, residual_rms, ant_coherence, ant_ids,
               block_vis_cal, block_freqs_hz, block_good, ref_ant_idx,
               svd_diff, ratio_delays, all_delays):
    """Generate diagnostic plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    n_ant = len(ant_ids)
    plot_path = os.path.join(OUT_DIR, "mar10_verification.pdf")

    with PdfPages(plot_path) as pdf:
        # Page 1: Block-averaged residual phase vs frequency per antenna
        ncols = 4
        nrows = (n_ant + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
        axes = axes.flatten()

        freqs_good_mhz = block_freqs_hz[block_good] / 1e6

        for k, ax in enumerate(axes):
            if k >= n_ant:
                ax.set_visible(False)
                continue

            if k == ref_ant_idx:
                ax.text(0.5, 0.5, f"Ant {ant_ids[k]}\n(ref)", transform=ax.transAxes,
                        ha="center", va="center", fontsize=12)
                ax.set_title(f"Ant {ant_ids[k]} (REF)", fontsize=9)
                continue

            i, j = min(k, ref_ant_idx), max(k, ref_ant_idx)
            bl_phase = np.angle(block_vis_cal[block_good, i, j])
            if k > ref_ant_idx:
                bl_phase = -bl_phase

            bl_unwrap = np.unwrap(bl_phase)

            ax.plot(freqs_good_mhz, np.degrees(bl_unwrap),
                    lw=0.8, alpha=0.8, color="C0")
            ax.axhline(0, color="gray", lw=0.5, ls="--")
            ax.set_title(
                f"Ant {ant_ids[k]}  drift={residual_slopes[k]:+.1f}ns  "
                f"coh={ant_coherence[k]:.3f}",
                fontsize=7,
            )
            ax.set_xlabel("Freq (MHz)", fontsize=6)
            ax.set_ylabel("Unwrapped phase (deg)", fontsize=6)
            ax.tick_params(labelsize=6)

        fig.suptitle(
            "Block-Averaged Residual Phase (Mar10 cal on Mar11 data)\n"
            "Slope = delay drift in ns",
            fontsize=12,
        )
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Three methods compared
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(n_ant)
        width = 0.25

        ax.bar(x - width, svd_diff, width,
               label="SVD delay comparison", alpha=0.8, color="C0")
        ax.bar(x, ratio_delays, width,
               label="Gain ratio method", alpha=0.8, color="C1")
        ax.bar(x + width, residual_slopes, width,
               label="Visibility residual (block-avg)", alpha=0.8, color="C2")

        ax.set_xlabel("Antenna ID")
        ax.set_ylabel("Delay drift Mar11-Mar10 (ns)")
        ax.set_title("Mar11-Mar10 Delay Drift: Three Methods Compared")
        ax.set_xticks(x)
        ax.set_xticklabels([str(a) for a in ant_ids])
        ax.axhline(0, color="gray", lw=0.5)
        ax.axhline(2, color="orange", lw=0.5, ls="--")
        ax.axhline(-2, color="orange", lw=0.5, ls="--")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: Multi-day delay trend
        if len(all_delays) >= 2:
            labels = sorted(all_delays.keys())
            fig, ax = plt.subplots(figsize=(14, 6))
            colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
            bar_width = 0.8 / len(labels)

            for idx, label in enumerate(labels):
                offset = (idx - len(labels) / 2 + 0.5) * bar_width
                ax.bar(x + offset, all_delays[label], bar_width,
                       label=label, color=colors[idx], alpha=0.8)

            ax.set_xlabel("Antenna ID")
            ax.set_ylabel("Delay (ns, relative to ref ant 5)")
            ax.set_title("Per-Antenna Delays Across Days")
            ax.set_xticks(x)
            ax.set_xticklabels([str(a) for a in ant_ids])
            ax.axhline(0, color="gray", lw=0.5)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # Page 4: Coherence per antenna
        fig, ax = plt.subplots(figsize=(14, 5))
        colors = ["green" if ant_coherence[k] > 0.5 else
                  "orange" if ant_coherence[k] > 0.2 else
                  "red" for k in range(n_ant)]
        ax.bar(x, ant_coherence, color=colors, alpha=0.8)
        ax.set_xlabel("Antenna ID")
        ax.set_ylabel("Coherence |<V>|/<|V|>")
        ax.set_title("Post-Calibration Coherence (Mar10 cal on Mar11 vis)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(a) for a in ant_ids])
        ax.axhline(0.5, color="orange", lw=0.5, ls="--", label="Low coherence")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\n  Saved plots: {plot_path}")


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 70)
    print("TRACK 5: VERIFY MAR10 HDF5 CONSISTENCY WITH MAR11 DATA")
    print("=" * 70)

    # ── Part A: SVD on Mar11 + multi-day comparison ──
    mar11_npz = attempt_mar11_svd()
    if mar11_npz is None:
        print("\n  Mar11 SVD failed. Cannot proceed with delay comparison.")
        print("  Skipping to visibility residual check (Part C).")
        svd_diff = None
        svd_rms = None
        all_delays = {}
        ratio_delays = None
    else:
        svd_diff, svd_rms, ant_ids, all_delays = multi_day_delay_comparison(mar11_npz)

        # ── Part B: Gain-ratio cross-check ──
        ratio_delays, ratio_r2 = gain_ratio_check(mar11_npz)

    # ── Part C: Block-averaged visibility residual check ──
    (residual_slopes, residual_rms, ant_coherence, ant_ids,
     block_vis_cal, block_freqs_hz, block_good, ref_ant_idx
     ) = block_residual_phase_check()

    # ── Generate plots ──
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    if svd_diff is None:
        svd_diff = np.zeros(len(ant_ids))
    if ratio_delays is None:
        ratio_delays = np.zeros(len(ant_ids))

    make_plots(residual_slopes, residual_rms, ant_coherence, ant_ids,
               block_vis_cal, block_freqs_hz, block_good, ref_ant_idx,
               svd_diff, ratio_delays, all_delays)

    # ── Final decision ──
    non_ref_idx = [k for k in range(len(ant_ids)) if k != ref_ant_idx]

    print("\n" + "=" * 70)
    print("SUMMARY: ALL THREE METHODS")
    print("=" * 70)

    methods = [
        ("SVD delay comparison", svd_diff),
        ("Gain ratio method", ratio_delays),
        ("Visibility residual (block-avg)", residual_slopes),
    ]
    for name, vals in methods:
        rms = np.sqrt(np.mean(vals[non_ref_idx]**2))
        mx = np.max(np.abs(vals[non_ref_idx]))
        worst = ant_ids[non_ref_idx[np.argmax(np.abs(vals[non_ref_idx]))]]
        print(f"  {name:40s}  RMS={rms:6.2f} ns  Max={mx:6.2f} ns (Ant {worst})")

    # Use the SVD comparison as primary (most robust for block SVD data)
    primary_rms = svd_rms if svd_rms is not None else np.sqrt(
        np.mean(residual_slopes[non_ref_idx]**2))

    print("\n" + "=" * 70)
    print("FINAL DECISION")
    print("=" * 70)
    print(f"\n  Primary metric (SVD delay drift RMS): {primary_rms:.2f} ns")

    if primary_rms < 2.0:
        print(f"\n  RESULT: Mar10 weights are GOOD to deploy.")
        print(f"  Delays stable within {primary_rms:.1f} ns RMS over 1 day.")
    elif primary_rms < 5.0:
        print(f"\n  RESULT: Mar10 weights have MODERATE drift ({primary_rms:.1f} ns).")
        print(f"  Consider re-calibrating if beamforming SNR is critical.")
    else:
        print(f"\n  RESULT: Mar10 weights have SIGNIFICANT drift ({primary_rms:.1f} ns).")
        print(f"  Fresh calibration REQUIRED before deployment.")

    # Report per-antenna status
    drifted = [ant_ids[k] for k in non_ref_idx if abs(svd_diff[k]) > 2.0]
    if drifted:
        print(f"  Drifted antennas (>2 ns): {[int(a) for a in drifted]}")
    stable = [ant_ids[k] for k in non_ref_idx if abs(svd_diff[k]) <= 2.0]
    if stable:
        print(f"  Stable antennas (<2 ns):  {[int(a) for a in stable]}")

    print("\n" + "=" * 70)
    print("TRACK 5 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
