#!/usr/bin/env python3
"""
compare_cal_delays.py
=====================
Compare per-antenna calibration delays between the Feb19 old SVD gains
(from correlator_analysis/svd_beamformer_weights_generator.py) and new
casm_calibrator runs on the same data.

Usage:
  python scripts/compare_cal_delays.py \\
      --old results/multi_day_svd/svd_weights_feb19.npz \\
      [--new <new_casm_calibrator_output.npz>] \\
      [--plot compare_delays.png]

If --new is omitted, only the old gains are analyzed (delay extraction + plot).

Delay extraction formula:
  phase(f) = 2*pi * f * delay + offset
  slope = d(phase)/d(freq_Hz)  [rad/Hz]
  delay_ns = slope / (2*pi) * 1e9
"""

import argparse
import numpy as np

# -------------------------------------------------------------------------
# Delay extraction
# -------------------------------------------------------------------------

def extract_delays(gains, flags, freqs_hz, ref_ant_idx):
    """Extract per-antenna delays via linear fit of unwrapped phase vs freq.

    Parameters
    ----------
    gains : ndarray, shape (n_ant, n_chan)
        Complex gains from SVD calibration.
    flags : ndarray, shape (n_chan,)
        True = good channel.
    freqs_hz : ndarray, shape (n_chan,)
        Frequencies in Hz (ascending).
    ref_ant_idx : int
        0-indexed reference antenna (should already be zeroed in gains).

    Returns
    -------
    delays_ns : ndarray, shape (n_ant,)
    phase_offsets_rad : ndarray, shape (n_ant,)
    r_squared : ndarray, shape (n_ant,)
    """
    n_ant, n_chan = gains.shape

    # Good channels: SVD passed AND gain non-zero
    has_gain = np.any(gains != 0, axis=0)
    good = flags & has_gain
    freqs_good = freqs_hz[good]

    print(f"  Delay fit: {good.sum()} good channels of {n_chan} total")
    print(f"  Freq range: {freqs_good[0]/1e6:.3f} – {freqs_good[-1]/1e6:.3f} MHz")

    if len(freqs_good) < 10:
        print(f"  WARNING: only {len(freqs_good)} good channels — delay fit unreliable")
        return (np.full(n_ant, np.nan),
                np.full(n_ant, np.nan),
                np.full(n_ant, np.nan))

    delays_ns = np.zeros(n_ant)
    phase_offsets = np.zeros(n_ant)
    r_squared = np.zeros(n_ant)

    for ant in range(n_ant):
        phase = np.angle(gains[ant, good])
        phase_unwrap = np.unwrap(phase)

        # Linear fit: phase [rad] = 2*pi * delay_s * freq_hz + offset
        # slope units: rad/Hz  ->  delay_s = slope / (2*pi)  ->  delay_ns = slope/(2*pi)*1e9
        coeffs = np.polyfit(freqs_good, phase_unwrap, 1)
        slope, offset = coeffs

        delays_ns[ant] = slope / (2.0 * np.pi) * 1e9
        phase_offsets[ant] = offset

        predicted = np.polyval(coeffs, freqs_good)
        ss_res = np.sum((phase_unwrap - predicted) ** 2)
        ss_tot = np.sum((phase_unwrap - np.mean(phase_unwrap)) ** 2)
        r_squared[ant] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Make relative to reference antenna
    delays_ns -= delays_ns[ref_ant_idx]
    phase_offsets -= phase_offsets[ref_ant_idx]

    return delays_ns, phase_offsets, r_squared


# -------------------------------------------------------------------------
# Print delay table
# -------------------------------------------------------------------------

DEFAULT_LABEL_OLD = "Old (Feb19 SVD)"
DEFAULT_LABEL_NEW = "New (casm_calibrator)"


def print_delay_table(ant_ids, delays_old, r2_old, delays_new=None, r2_new=None,
                      label_old=DEFAULT_LABEL_OLD, label_new=DEFAULT_LABEL_NEW):
    """Print a side-by-side delay comparison table."""
    n_ant = len(ant_ids)
    has_new = delays_new is not None

    print()
    print("=" * 80)
    print(f"DELAY COMPARISON TABLE (ns, relative to ref ant)")
    print("=" * 80)

    if has_new:
        hdr = (f"{'Ant':>4s}  {label_old:>18s}  {'R²':>5s}  "
               f"{label_new:>18s}  {'R²':>5s}  {'Diff(new-old)':>14s}")
    else:
        hdr = f"{'Ant':>4s}  {label_old:>18s}  {'R²':>5s}"
    print(hdr)
    print("-" * len(hdr))

    for k in range(n_ant):
        d_old = delays_old[k]
        r2_o = r2_old[k]
        if has_new:
            d_new = delays_new[k]
            r2_n = r2_new[k]
            diff = d_new - d_old
            print(f"{ant_ids[k]:4d}  {d_old:+18.3f}  {r2_o:5.3f}  "
                  f"{d_new:+18.3f}  {r2_n:5.3f}  {diff:+14.3f}")
        else:
            print(f"{ant_ids[k]:4d}  {d_old:+18.3f}  {r2_o:5.3f}")

    if has_new:
        finite = ~np.isnan(delays_old) & ~np.isnan(delays_new)
        if finite.sum() > 0:
            diffs = delays_new[finite] - delays_old[finite]
            print()
            print(f"  Diff (new-old) stats over {finite.sum()} antennas:")
            print(f"    mean = {np.mean(diffs):+.3f} ns")
            print(f"    std  = {np.std(diffs):.3f} ns")
            print(f"    max  = {np.max(np.abs(diffs)):.3f} ns  (ant {ant_ids[np.argmax(np.abs(diffs))]})")

    print("=" * 80)


# -------------------------------------------------------------------------
# Audit NPZ keys and metadata
# -------------------------------------------------------------------------

def audit_npz(path, label):
    """Print metadata audit of a calibration NPZ."""
    d = np.load(path, allow_pickle=True)
    print()
    print(f"--- NPZ AUDIT: {label} ---")
    print(f"  Path:       {path}")
    print(f"  Keys:       {sorted(d.keys())}")
    print(f"  weights:    {d['weights'].shape}  dtype={d['weights'].dtype}")
    print(f"  gains:      {d['gains'].shape}  dtype={d['gains'].dtype}")
    flags = d['flags']
    print(f"  flags:      {flags.sum()}/{len(flags)} good channels  "
          f"({100*flags.mean():.1f}%)")
    fmhz = d['freqs_mhz']
    print(f"  freqs_mhz:  {fmhz[0]:.3f} – {fmhz[-1]:.3f} MHz  "
          f"({'ascending' if fmhz[0]<fmhz[-1] else 'descending'})")
    print(f"  ant_ids:    {d['ant_ids']}")
    print(f"  ref_ant_id: {d['ref_ant_id']}")
    print(f"  source:     {d['source']}")
    print(f"  threshold:  {d['threshold']}")
    if 'n_time_averaged' in d:
        print(f"  n_time_avg: {d['n_time_averaged']}")
    if 'freq_order' in d:
        print(f"  freq_order: {d['freq_order']}")
    if 'block_size' in d:
        print(f"  block_size: {d['block_size']}")
    return d


# -------------------------------------------------------------------------
# Phase vs frequency plot
# -------------------------------------------------------------------------

def make_comparison_plot(freqs_hz_old, gains_old, flags_old, ant_ids,
                         delays_ns_old, r2_old,
                         freqs_hz_new=None, gains_new=None, flags_new=None,
                         delays_ns_new=None, r2_new=None,
                         out_path="compare_delays.png",
                         label_old=DEFAULT_LABEL_OLD,
                         label_new=DEFAULT_LABEL_NEW):
    """Plot phase vs frequency per antenna and delay bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    n_ant = len(ant_ids)
    has_new = freqs_hz_new is not None

    use_pdf = out_path.endswith(".pdf")

    # Color for old vs new
    col_old = "#1f77b4"
    col_new = "#d62728"

    # Pre-compute loop-invariant quantities
    good_old = flags_old
    freqs_mhz_old = freqs_hz_old / 1e6
    good_new = flags_new if has_new else None
    freqs_mhz_new = freqs_hz_new / 1e6 if has_new else None

    # ---- Page 1: Phase vs frequency, per antenna ----
    ncols = 4
    nrows = (n_ant + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
    axes = axes.flatten()

    for k, ax in enumerate(axes):
        if k >= n_ant:
            ax.set_visible(False)
            continue

        ph_unwrap_old = np.unwrap(np.angle(gains_old[k, good_old]))
        ax.plot(freqs_mhz_old[good_old], np.degrees(ph_unwrap_old),
                color=col_old, lw=0.5, alpha=0.7, label=label_old)

        if has_new:
            ph_unwrap_new = np.unwrap(np.angle(gains_new[k, good_new]))
            ax.plot(freqs_mhz_new[good_new], np.degrees(ph_unwrap_new),
                    color=col_new, lw=0.5, alpha=0.7, label=label_new)

        ax.set_title(f"Ant {ant_ids[k]}  delay={delays_ns_old[k]:+.1f}ns  R²={r2_old[k]:.2f}",
                     fontsize=7)
        ax.set_xlabel("Freq (MHz)", fontsize=6)
        ax.set_ylabel("Phase (deg)", fontsize=6)
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.legend(fontsize=5, loc="best")

    fig.suptitle("Gain Phase vs Frequency (unwrapped)", fontsize=13)
    fig.tight_layout()

    # ---- Page 2: Delay bar chart ----
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    x = np.arange(n_ant)
    width = 0.35

    ax2.bar(x - width/2 if has_new else x, delays_ns_old, width,
            label=label_old, color=col_old, alpha=0.8)
    if has_new and delays_ns_new is not None:
        ax2.bar(x + width/2, delays_ns_new, width,
                label=label_new, color=col_new, alpha=0.8)

    ax2.set_xlabel("Antenna", fontsize=11)
    ax2.set_ylabel("Delay (ns)", fontsize=11)
    ax2.set_title("Per-Antenna Cable Delays (relative to ref ant)", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(a) for a in ant_ids], fontsize=8)
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend()
    fig2.tight_layout()

    if use_pdf:
        with PdfPages(out_path) as pdf:
            pdf.savefig(fig)
            pdf.savefig(fig2)
        plt.close(fig)
        plt.close(fig2)
        print(f"  Saved: {out_path}")
    else:
        phase_path = out_path.replace(".png", "_phase.png")
        delay_path = out_path.replace(".png", "_delays.png")
        fig.savefig(phase_path, dpi=150, bbox_inches="tight")
        fig2.savefig(delay_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close(fig2)
        print(f"  Saved: {phase_path}")
        print(f"  Saved: {delay_path}")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare SVD calibration delays: old vs new casm_calibrator")
    parser.add_argument("--old", required=True,
                        help="Old SVD weights NPZ (e.g. svd_weights_feb19.npz)")
    parser.add_argument("--new", default=None,
                        help="New casm_calibrator NPZ (optional)")
    parser.add_argument("--plot", default=None,
                        help="Output plot path (.png or .pdf). "
                             "For PNG: two files _phase.png and _delays.png are written.")
    parser.add_argument("--old-label", default=DEFAULT_LABEL_OLD,
                        help="Label for old dataset")
    parser.add_argument("--new-label", default=DEFAULT_LABEL_NEW,
                        help="Label for new dataset")
    args = parser.parse_args()

    # ---- Load old NPZ ----
    d_old = audit_npz(args.old, args.old_label)
    gains_old = d_old["gains"]          # (n_ant, n_chan) complex64
    flags_old = d_old["flags"]          # (n_chan,) bool
    freqs_hz_old = d_old["freqs_hz"]    # (n_chan,) float64 — ascending
    ant_ids = d_old["ant_ids"]

    # Ensure ascending frequency order
    if freqs_hz_old[0] > freqs_hz_old[-1]:
        print("  [old] Reversing to ascending freq order...")
        freqs_hz_old = freqs_hz_old[::-1]
        gains_old = gains_old[:, ::-1]
        flags_old = flags_old[::-1]

    # Find ref ant index
    ref_ant_id = int(d_old["ref_ant_id"])
    ref_ant_idx_old = int(np.where(ant_ids == ref_ant_id)[0][0])
    print(f"\n  ref_ant_id={ref_ant_id}, ref_ant_idx={ref_ant_idx_old}")

    # ---- Extract delays: old ----
    print(f"\nExtracting delays from {args.old_label}...")
    delays_old, offsets_old, r2_old = extract_delays(
        gains_old, flags_old, freqs_hz_old, ref_ant_idx_old)

    # ---- Load new NPZ (optional) ----
    delays_new = None
    r2_new = None
    freqs_hz_new = None
    gains_new = None
    flags_new = None

    if args.new is not None:
        d_new = audit_npz(args.new, args.new_label)
        gains_new = d_new["gains"]
        flags_new = d_new["flags"]
        freqs_hz_new = d_new["freqs_hz"]

        if freqs_hz_new[0] > freqs_hz_new[-1]:
            print("  [new] Reversing to ascending freq order...")
            freqs_hz_new = freqs_hz_new[::-1]
            gains_new = gains_new[:, ::-1]
            flags_new = flags_new[::-1]

        ref_ant_id_new = int(d_new["ref_ant_id"])
        ref_ant_idx_new = int(np.where(d_new["ant_ids"] == ref_ant_id_new)[0][0])

        print(f"\nExtracting delays from {args.new_label}...")
        delays_new, offsets_new, r2_new = extract_delays(
            gains_new, flags_new, freqs_hz_new, ref_ant_idx_new)

    # ---- Print table ----
    print_delay_table(
        ant_ids, delays_old, r2_old,
        delays_new=delays_new, r2_new=r2_new,
        label_old=args.old_label, label_new=args.new_label
    )

    # ---- Print per-antenna phase RMS of old gains ----
    print()
    print("Per-antenna gain amplitude (median over good channels):")
    good = flags_old
    for k in range(len(ant_ids)):
        amp = np.abs(gains_old[k, good])
        print(f"  Ant {ant_ids[k]:2d}: |g| median={np.median(amp):.4f}  "
              f"min={amp.min():.4f}  max={amp.max():.4f}")

    # ---- Plots ----
    if args.plot is not None:
        print(f"\nGenerating plots -> {args.plot}...")
        make_comparison_plot(
            freqs_hz_old, gains_old, flags_old, ant_ids,
            delays_old, r2_old,
            freqs_hz_new=freqs_hz_new, gains_new=gains_new, flags_new=flags_new,
            delays_ns_new=delays_new, r2_new=r2_new,
            out_path=args.plot,
            label_old=args.old_label,
            label_new=args.new_label,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
