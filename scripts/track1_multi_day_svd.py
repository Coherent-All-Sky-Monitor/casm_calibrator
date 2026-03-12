#!/usr/bin/env python3
"""
Track 1: Multi-Day SVD Consistency Check.

Run SVD calibration on multiple days, extract per-antenna delays,
and compare across days to determine if delays are stable.

Reuses casm_calibrator CLI and compare_cal_delays.py::extract_delays().
"""

import os
import sys
import subprocess
import numpy as np

# Add scripts dir to path for extract_delays
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
from compare_cal_delays import extract_delays, print_delay_table, make_comparison_plot

# ── Config ──
DATA_DIR = "/mnt/nvme3/data/casm/visibilities_64ant"
LAYOUT = os.path.expanduser("~/software/dev/antenna_layouts/antenna_layout_current.csv")
FORMAT = "layout_64ant"
OUT_DIR = os.path.join(os.path.dirname(SCRIPTS_DIR), "results", "multi_day_svd")
os.makedirs(OUT_DIR, exist_ok=True)

VENV_PYTHON = os.path.expanduser(
    "~/software/dev/casm_venvs/casm_offline_env/bin/python"
)

# Observations to calibrate
OBSERVATIONS = [
    ("2026-02-19-05:22:55", 8),   # Feb 19 solar transit
    ("2026-03-06-05:36:03", 8),   # Mar 06 solar transit (74 files, use 8)
    ("2026-03-10-14:05:53", None), # Mar 10 solar transit (already done, 8 files)
    ("2026-03-11-12:17:47", None), # Mar 11 solar transit (today, 8 files)
]

# SVD parameters
SVD_ARGS = [
    "--svd-mode", "phase-only",
    "--block-size", "32",
    "--threshold", "2.0",
    "--fill-mode", "interpolate",
    "--rfi-mask-range", "375", "390",
    "--source", "sun",
    "--ref-ant", "5",
]


def run_calibration(obs_id, nfiles=None):
    """Run casm-svd-calibrate for one observation."""
    date_str = obs_id[:10]
    out_npz = os.path.join(OUT_DIR, f"svd_weights_{date_str}.npz")
    out_plots = os.path.join(OUT_DIR, f"svd_diag_{date_str}.pdf")

    if os.path.exists(out_npz):
        print(f"\n  {out_npz} already exists, skipping calibration.")
        return out_npz

    cmd = [
        VENV_PYTHON, "-m", "casm_calibrator.cli",
        "--data-dir", DATA_DIR,
        "--obs", obs_id,
        "--format", FORMAT,
        "--layout", LAYOUT,
        "--output", out_npz,
        "--plots", out_plots,
    ] + SVD_ARGS

    if nfiles is not None:
        cmd += ["--nfiles", str(nfiles)]

    print(f"\n  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  STDERR:\n{result.stderr}")
        print(f"  STDOUT:\n{result.stdout}")
        print(f"  Calibration FAILED for {obs_id}")
        return None
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    return out_npz


def main():
    print("=" * 70)
    print("TRACK 1: MULTI-DAY SVD CONSISTENCY CHECK")
    print("=" * 70)

    # ── Run calibrations ──
    npz_paths = {}
    for obs_id, nfiles in OBSERVATIONS:
        date_str = obs_id[:10]
        print(f"\n{'─'*60}")
        print(f"Processing {obs_id} (nfiles={nfiles})...")

        # Special case: Mar10 already done
        if date_str == "2026-03-10":
            existing = os.path.join(
                os.path.dirname(SCRIPTS_DIR), "results", "svd_weights_mar10.npz"
            )
            if os.path.exists(existing):
                # Copy to our output dir
                out_npz = os.path.join(OUT_DIR, f"svd_weights_{date_str}.npz")
                if not os.path.exists(out_npz):
                    import shutil
                    shutil.copy2(existing, out_npz)
                    print(f"  Copied existing: {existing} -> {out_npz}")
                npz_paths[date_str] = out_npz
                continue

        out_npz = run_calibration(obs_id, nfiles)
        if out_npz:
            npz_paths[date_str] = out_npz

    # ── Also include the old Feb19 NPZ from vishnu ──
    feb19_old = "/home/casm/software/vishnu/casm-voltage-analysis/correlator_analysis/results/multi_day_svd/svd_weights_feb19.npz"
    if os.path.exists(feb19_old):
        npz_paths["2026-02-19-old"] = feb19_old

    print(f"\n\n{'=' * 70}")
    print(f"DELAY EXTRACTION AND COMPARISON")
    print(f"{'=' * 70}")

    # ── Extract delays from all NPZs ──
    all_delays = {}
    all_r2 = {}
    ant_ids = None

    for label, npz_path in sorted(npz_paths.items()):
        print(f"\n{'─'*60}")
        print(f"Extracting delays: {label}")
        d = np.load(npz_path, allow_pickle=True)
        gains = d["gains"]
        flags = d["flags"]
        freqs_hz = d["freqs_hz"]
        if ant_ids is None:
            ant_ids = d["ant_ids"]

        # Ensure ascending
        if freqs_hz[1] < freqs_hz[0]:
            freqs_hz = freqs_hz[::-1]
            gains = gains[:, ::-1]
            flags = flags[::-1]

        ref_ant_id = int(d["ref_ant_id"])
        ref_ant_idx = int(np.where(d["ant_ids"] == ref_ant_id)[0][0])

        delays, offsets, r2 = extract_delays(gains, flags, freqs_hz, ref_ant_idx)
        all_delays[label] = delays
        all_r2[label] = r2
        print(f"  Ref ant: {ref_ant_id}, Delays (ns): {np.round(delays, 1)}")

    # ── Print comparison table ──
    if len(all_delays) < 2:
        print("  Not enough datasets to compare.")
        return

    labels = sorted(all_delays.keys())
    print(f"\n\n{'=' * 90}")
    print("MULTI-DAY DELAY TABLE (ns, relative to ref ant 5)")
    print("=" * 90)

    header = f"{'Ant':>4s}"
    for label in labels:
        header += f"  {label:>16s}"
    print(header)
    print("-" * len(header))

    n_ant = len(ant_ids)
    for k in range(n_ant):
        row = f"{ant_ids[k]:4d}"
        for label in labels:
            row += f"  {all_delays[label][k]:+16.2f}"
        if ant_ids[k] == 4:
            row += "  <-- ANT 4"
        print(row)

    # ── Delay stability analysis ──
    print(f"\n{'=' * 70}")
    print("DELAY STABILITY ANALYSIS")
    print("=" * 70)

    # Compare consecutive days
    for i in range(len(labels) - 1):
        l1, l2 = labels[i], labels[i + 1]
        d1, d2 = all_delays[l1], all_delays[l2]
        diff = d2 - d1
        print(f"\n  {l2} vs {l1}:")
        for k in range(n_ant):
            flag = " *** ANOMALOUS" if abs(diff[k]) > 5 else ""
            print(f"    Ant {ant_ids[k]:2d}: delta={diff[k]:+7.2f} ns{flag}")
        print(f"    RMS diff: {np.sqrt(np.mean(diff**2)):.2f} ns")
        print(f"    Max |diff|: {np.max(np.abs(diff)):.2f} ns (Ant {ant_ids[np.argmax(np.abs(diff))]})")

    # ── Decision criteria ──
    # Compare first and last
    first_label = labels[0]
    last_label = labels[-1]
    total_diff = all_delays[last_label] - all_delays[first_label]
    rms_total = np.sqrt(np.mean(total_diff**2))
    max_total = np.max(np.abs(total_diff))

    print(f"\n{'=' * 70}")
    print(f"DECISION: {last_label} vs {first_label}")
    print(f"  RMS delay change: {rms_total:.2f} ns")
    print(f"  Max delay change: {max_total:.2f} ns (Ant {ant_ids[np.argmax(np.abs(total_diff))]})")
    if rms_total < 2.0:
        print(f"  -> Delays STABLE (<2 ns RMS). SVD is correct, issue is downstream.")
    elif rms_total > 10.0:
        print(f"  -> Delays DRIFTING (>10 ns RMS). Hardware changes detected.")
    else:
        print(f"  -> Delays MODERATELY changing ({rms_total:.1f} ns). Some hardware drift.")

    # Ant 4 specific
    ant4_idx = np.where(ant_ids == 4)[0]
    if len(ant4_idx) > 0:
        k = ant4_idx[0]
        ant4_diff = total_diff[k]
        print(f"\n  Ant 4 specific: delay change = {ant4_diff:+.2f} ns")
        if abs(ant4_diff) > 10:
            print(f"  -> Ant 4 has LARGE delay drift. Likely hardware issue.")

    # ── Generate comparison plot ──
    plot_path = os.path.join(OUT_DIR, "multi_day_delay_comparison.pdf")
    print(f"\n  Generating comparison plot: {plot_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(plot_path) as pdf:
        # Page 1: Delay bar chart
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(n_ant)
        width = 0.8 / len(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

        for i, label in enumerate(labels):
            offset = (i - len(labels) / 2 + 0.5) * width
            ax.bar(x + offset, all_delays[label], width, label=label,
                   color=colors[i], alpha=0.8)

        ax.set_xlabel("Antenna ID")
        ax.set_ylabel("Delay (ns)")
        ax.set_title("Per-Antenna Delays Across Days (relative to ref ant 5)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(a) for a in ant_ids])
        ax.axhline(0, color="gray", lw=0.5)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Phase vs freq for each day (Ant 4 highlight)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        for idx, (label, npz_path) in enumerate(sorted(npz_paths.items())):
            if idx >= 4:
                break
            ax = axes[idx]
            d = np.load(npz_path, allow_pickle=True)
            gains = d["gains"]
            flags = d["flags"]
            freqs_hz = d["freqs_hz"]
            if freqs_hz[1] < freqs_hz[0]:
                freqs_hz = freqs_hz[::-1]
                gains = gains[:, ::-1]
                flags = flags[::-1]

            freqs_mhz = freqs_hz / 1e6
            for k in range(len(d["ant_ids"])):
                good = flags & (np.abs(gains[k]) > 0)
                if good.sum() < 10:
                    continue
                ph = np.unwrap(np.angle(gains[k, good]))
                lw = 2.0 if d["ant_ids"][k] == 4 else 0.4
                alpha = 1.0 if d["ant_ids"][k] == 4 else 0.5
                color = "red" if d["ant_ids"][k] == 4 else None
                ax.plot(freqs_mhz[good], np.degrees(ph),
                        lw=lw, alpha=alpha, color=color,
                        label=f"Ant {d['ant_ids'][k]}" if d["ant_ids"][k] == 4 else None)

            ax.set_title(label, fontsize=10)
            ax.set_xlabel("Freq (MHz)", fontsize=8)
            ax.set_ylabel("Phase (deg)", fontsize=8)
            ax.tick_params(labelsize=7)
            if idx == 0:
                ax.legend(fontsize=7)

        fig.suptitle("Gain Phase vs Frequency (Ant 4 in red)", fontsize=12)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"  Saved: {plot_path}")
    print(f"\n{'=' * 70}")
    print("TRACK 1 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
