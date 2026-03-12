#!/usr/bin/env python3
"""
Verify calibration weights via visibility beamforming.

Load visibilities, fringe-stop, apply calibration gains, then beamform
toward the Sun and an off-source direction. The Sun/Off power ratio
indicates whether the calibration weights produce coherent beams.

Optionally generates a beam image (alt-az grid) to confirm a point
source at the Sun's position.
"""

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_sun
import astropy.units as u

from casm_io.correlator.mapping import AntennaMapping
from casm_io.constants import C_LIGHT_M_S
from casm_vis_analysis.sources import find_transit_window

from casm_calibrator.visibility import VisibilityLoader, VisibilityMatrix
from casm_calibrator.fringe_stop import FringeStopMatrix
from casm_calibrator.rfi import RFIMask

OVRO = EarthLocation(lat=37.2339 * u.deg, lon=-118.2821 * u.deg, height=1222 * u.m)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Verify calibration weights via visibility beamforming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-dir", required=True, help="Directory with .dat files")
    p.add_argument("--obs", required=True, help="Observation timestamp")
    p.add_argument(
        "--layout",
        default=os.path.expanduser(
            "~/software/dev/antenna_layouts/antenna_layout_current.csv"
        ),
        help="Antenna layout CSV",
    )
    p.add_argument(
        "--cal-weights", action="append", default=[],
        help="Calibration weights NPZ (repeatable for comparison)",
    )
    p.add_argument("--source", default="sun", help="Source name (default: sun)")
    p.add_argument("--nfiles", type=int, default=None, help="Number of files to read")
    p.add_argument(
        "--rfi-mask-range", nargs=2, type=float, action="append",
        metavar=("LO", "HI"), default=None,
        help="RFI range to mask in MHz (repeatable, default: 375 390)",
    )
    p.add_argument("--off-az-offset", type=float, default=10.0,
                    help="Off-source azimuth offset in degrees (default: 10)")
    p.add_argument("--make-image", action="store_true",
                    help="Generate beam image around the source")
    p.add_argument("--image-fov", type=float, default=15.0,
                    help="Image FoV in degrees (default: 15)")
    p.add_argument("--image-npix", type=int, default=51,
                    help="Image pixels per side (default: 51)")
    p.add_argument("--exclude-ants", type=int, nargs="+", default=[],
                    help="Antenna IDs to exclude (e.g. --exclude-ants 4)")
    p.add_argument("--output-dir", default=None,
                    help="Output directory (default: results/verify_vis_beamform/)")
    p.add_argument("--min-alt", type=float, default=10.0,
                    help="Min source altitude for transit window (degrees)")
    return p.parse_args(argv)


def load_cal_gains(npz_path, ant_ids, n_chan):
    """Load calibration gains from NPZ, matched to ant_ids ordering.

    Returns gains (n_ant, n_chan) and label string.
    """
    d = np.load(npz_path, allow_pickle=True)
    gains = d["gains"]  # (n_ant_cal, n_chan_cal)
    cal_ant_ids = d["ant_ids"]
    cal_freqs = d["freqs_hz"]

    # Ensure ascending freq order
    if cal_freqs[1] < cal_freqs[0]:
        gains = gains[:, ::-1]

    # Match antenna ordering
    cal_id_to_idx = {int(aid): i for i, aid in enumerate(cal_ant_ids)}
    n_ant = len(ant_ids)
    g_out = np.ones((n_ant, n_chan), dtype=np.complex64)
    for i, aid in enumerate(ant_ids):
        if int(aid) in cal_id_to_idx:
            g_out[i] = gains[cal_id_to_idx[int(aid)]]

    label = os.path.basename(npz_path).replace(".npz", "")
    return g_out, label


def sun_altaz(time_unix):
    """Get Sun alt/az at OVRO for given unix timestamp (mid-point)."""
    t = Time(time_unix, format="unix")
    frame = AltAz(obstime=t, location=OVRO)
    sun = get_sun(t).transform_to(frame)
    return sun.alt.rad, sun.az.rad


def altaz_to_enu(alt, az):
    """Alt-az to ENU direction cosines (l, m, n)."""
    l = np.cos(alt) * np.sin(az)
    m = np.cos(alt) * np.cos(az)
    n = np.sin(alt)
    return l, m, n


def compute_baseline_delay(pos_i, pos_j, l, m, n):
    """Geometric delay for baseline (i,j) toward direction (l,m,n).

    tau_bl = (pos_j - pos_i) . s_hat / c
    """
    b = pos_j - pos_i  # (3,)
    return (b[0] * l + b[1] * m + b[2] * n) / C_LIGHT_M_S


def visibility_beamform(vis_cal, freqs_hz, positions, l, m, n,
                        l_fs=None, m_fs=None, n_fs=None):
    """Beamform calibrated visibilities toward direction (l,m,n).

    If l_fs/m_fs/n_fs are given, the data is assumed to be fringe-stopped
    toward that direction. The beamforming phasor is then the *differential*
    delay: tau_target - tau_fringestop.

    Parameters
    ----------
    vis_cal : (n_chan, n_ant, n_ant) complex
        Calibrated, time-averaged visibilities.
    freqs_hz : (n_chan,) float
        Frequencies in Hz.
    positions : (n_ant, 3) float
        ENU positions.
    l, m, n : float
        Target direction cosines.
    l_fs, m_fs, n_fs : float, optional
        Fringe-stop direction cosines.

    Returns
    -------
    power : (n_chan,) float
        Beamformed power spectrum.
    """
    n_chan, n_ant, _ = vis_cal.shape
    power = np.zeros(n_chan, dtype=np.float64)

    # Cross-correlations only (autocorrelations are direction-independent
    # and would wash out the Sun/Off contrast)
    for i in range(n_ant):
        for j in range(i + 1, n_ant):
            tau_bl = compute_baseline_delay(positions[i], positions[j], l, m, n)
            if l_fs is not None:
                tau_fs = compute_baseline_delay(positions[i], positions[j],
                                                l_fs, m_fs, n_fs)
                tau_bl = tau_bl - tau_fs
            phasor = np.exp(-1j * 2 * np.pi * freqs_hz * tau_bl)
            power += 2.0 * np.real(vis_cal[:, i, j] * phasor)

    return power


def make_beam_image(vis_cal, freqs_hz, positions, rfi_mask,
                    center_alt, center_az, fov_deg, npix,
                    l_fs=None, m_fs=None, n_fs=None):
    """Create beam image by sweeping alt-az grid.

    Returns image (npix, npix) and axis arrays.
    """
    half_fov = np.radians(fov_deg / 2.0)
    alt_offsets = np.linspace(-half_fov, half_fov, npix)
    az_offsets = np.linspace(-half_fov, half_fov, npix)

    # Use only good channels for speed and quality
    good = rfi_mask
    vis_good = vis_cal[good]
    freqs_good = freqs_hz[good]

    image = np.zeros((npix, npix), dtype=np.float64)

    for ia, da in enumerate(alt_offsets):
        alt = center_alt + da
        if alt <= 0 or alt >= np.pi / 2:
            continue
        for iz, dz in enumerate(az_offsets):
            az = center_az + dz
            l, m, n = altaz_to_enu(alt, az)
            power = visibility_beamform(vis_good, freqs_good, positions, l, m, n,
                                        l_fs=l_fs, m_fs=m_fs, n_fs=n_fs)
            image[ia, iz] = np.mean(power)

    alt_axis = np.degrees(alt_offsets + center_alt)
    az_axis = np.degrees(az_offsets + center_az)
    return image, alt_axis, az_axis


def main(argv=None):
    args = parse_args(argv)

    # Defaults
    rfi_ranges = args.rfi_mask_range if args.rfi_mask_range else [(375.0, 390.0)]
    out_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "verify_vis_beamform",
    )
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("VISIBILITY BEAMFORMING VERIFICATION")
    print("=" * 70)

    # ── 1. Load visibilities ──
    print(f"\n1. Loading antenna layout: {args.layout}")
    mapping = AntennaMapping.load(args.layout)
    ant_ids = np.array(mapping.active_antennas(), dtype=int)
    print(f"   {len(ant_ids)} active antennas: {ant_ids}")

    print(f"\n2. Loading visibilities: {args.data_dir}, obs={args.obs}")
    loader = VisibilityLoader(mapping)
    vis_matrix = loader.load(args.data_dir, args.obs, nfiles=args.nfiles)
    print(f"   vis shape: {vis_matrix.vis.shape}")
    print(f"   freq range: {vis_matrix.freq_mhz[0]:.2f} - {vis_matrix.freq_mhz[-1]:.2f} MHz")
    print(f"   time range: {vis_matrix.time_unix[0]:.0f} - {vis_matrix.time_unix[-1]:.0f}")

    n_time, n_chan, n_ant, _ = vis_matrix.vis.shape

    # ── 2. Exclude antennas ──
    if args.exclude_ants:
        keep_mask = ~np.isin(ant_ids, args.exclude_ants)
        keep_idx = np.where(keep_mask)[0]
        ant_ids = ant_ids[keep_mask]
        positions_enu = vis_matrix.positions_enu[keep_idx]
        vis_data = vis_matrix.vis[:, :, keep_idx][:, :, :, keep_idx]
        n_ant = len(ant_ids)
        vis_matrix = VisibilityMatrix(
            vis=vis_data,
            freq_mhz=vis_matrix.freq_mhz,
            time_unix=vis_matrix.time_unix,
            ant_ids=ant_ids,
            positions_enu=positions_enu,
        )
        print(f"\n   After excluding ants {args.exclude_ants}: {n_ant} antennas")

    # ── 3. Find transit window ──
    source = args.source.lower()
    print(f"\n3. Finding transit window for {source} (min alt={args.min_alt}°)...")
    i_start, i_end = find_transit_window(source, vis_matrix.time_unix, min_alt_deg=args.min_alt)
    n_transit = i_end - i_start + 1
    print(f"   Transit: indices {i_start}-{i_end} ({n_transit} integrations)")

    vis_matrix = VisibilityMatrix(
        vis=vis_matrix.vis[i_start:i_end + 1],
        freq_mhz=vis_matrix.freq_mhz,
        time_unix=vis_matrix.time_unix[i_start:i_end + 1],
        ant_ids=vis_matrix.ant_ids,
        positions_enu=vis_matrix.positions_enu,
    )

    # ── 4. Fringe-stop ──
    print(f"\n4. Fringe-stopping toward {source}...")
    vis_fs = FringeStopMatrix()(vis_matrix, source)

    # ── 5. Time-average ──
    print(f"\n5. Time-averaging {vis_fs.vis.shape[0]} integrations...")
    vis_avg = np.mean(vis_fs.vis, axis=0)  # (F, n_ant, n_ant)
    print(f"   vis_avg shape: {vis_avg.shape}")

    # ── 6. RFI mask ──
    rfi_mask_obj = RFIMask(bad_ranges_mhz=rfi_ranges)
    rfi_mask = rfi_mask_obj(vis_fs.freq_mhz)
    n_good = int(np.sum(rfi_mask))
    print(f"\n6. RFI mask: {n_good}/{n_chan} channels good")

    # ── 7. Sun position at transit midpoint ──
    mid_time = np.mean(vis_fs.time_unix)
    sun_alt, sun_az = sun_altaz(mid_time)
    print(f"\n7. Sun position at transit midpoint:")
    print(f"   alt={np.degrees(sun_alt):.2f}°, az={np.degrees(sun_az):.2f}°")

    l_sun, m_sun, n_sun = altaz_to_enu(sun_alt, sun_az)

    # Off-source direction
    off_az = sun_az + np.radians(args.off_az_offset)
    l_off, m_off, n_off = altaz_to_enu(sun_alt, off_az)
    print(f"   Off-source: az offset = +{args.off_az_offset}°")

    freqs_hz = vis_fs.freq_mhz.astype(np.float64) * 1e6
    positions = vis_fs.positions_enu

    # ── 8. Beamform each weight set ──
    print(f"\n8. Beamforming...")
    results = {}

    # Build weight configurations
    cal_configs = []

    # No calibration (identity gains)
    cal_configs.append(("No cal", np.ones((n_ant, n_chan), dtype=np.complex64)))

    # Each cal weight file
    for npz_path in args.cal_weights:
        gains, label = load_cal_gains(npz_path, ant_ids, n_chan)
        cal_configs.append((label, gains))

    eps = 1e-30

    for cal_label, gains in cal_configs:
        # Apply calibration: V_cal[f,i,j] = conj(g[i,f]) * V[f,i,j] * g[j,f]
        g_conj = np.conj(gains)  # (n_ant, n_chan)
        vis_cal = vis_avg.copy()
        for f in range(n_chan):
            vis_cal[f] = g_conj[:, f:f+1] * vis_avg[f] * gains[:, f:f+1].T

        # Beamform toward Sun (data is fringe-stopped toward Sun, so
        # differential delay = 0 → phasors = 1 → just sums visibilities)
        power_sun = visibility_beamform(vis_cal, freqs_hz, positions,
                                        l_sun, m_sun, n_sun,
                                        l_fs=l_sun, m_fs=m_sun, n_fs=n_sun)

        # Beamform off-source (differential delay between off and Sun)
        power_off = visibility_beamform(vis_cal, freqs_hz, positions,
                                        l_off, m_off, n_off,
                                        l_fs=l_sun, m_fs=m_sun, n_fs=n_sun)

        # Masked means
        sun_mean = np.mean(power_sun[rfi_mask])
        off_mean = np.mean(power_off[rfi_mask])
        if abs(off_mean) > eps:
            ratio = sun_mean / abs(off_mean)
        else:
            ratio = np.inf if sun_mean > 0 else 0.0
        ratio_db = 10 * np.log10(max(abs(ratio), eps))

        results[cal_label] = {
            "power_sun": power_sun,
            "power_off": power_off,
            "sun_mean": sun_mean,
            "off_mean": off_mean,
            "ratio": ratio,
            "ratio_db": ratio_db,
            "gains": gains,
        }

        print(f"   {cal_label:30s}  Sun={sun_mean:.2e}  Off={off_mean:.2e}  "
              f"ratio={ratio:.2f} ({ratio_db:+.1f} dB)")

    # ── 9. Results table ──
    print(f"\n{'=' * 70}")
    print(f"RESULTS: Sun/Off Power Ratio")
    print(f"{'=' * 70}")
    print(f"{'Config':30s}  {'Sun':>12s}  {'Off':>12s}  {'Ratio':>8s}  {'dB':>8s}")
    print("-" * 75)
    for label, r in results.items():
        print(f"{label:30s}  {r['sun_mean']:12.4e}  {r['off_mean']:12.4e}  "
              f"{r['ratio']:8.2f}  {r['ratio_db']:+8.1f}")

    # ── 10. Optional beam image ──
    beam_image = None
    if args.make_image and len(args.cal_weights) > 0:
        # Use the last (presumably best) cal weight set
        best_label = list(results.keys())[-1]
        best_gains = results[best_label]["gains"]
        g_conj = np.conj(best_gains)
        vis_cal = vis_avg.copy()
        for f in range(n_chan):
            vis_cal[f] = g_conj[:, f:f+1] * vis_avg[f] * best_gains[:, f:f+1].T

        print(f"\n10. Making beam image ({args.image_npix}x{args.image_npix}, "
              f"FoV={args.image_fov}°) using '{best_label}'...")
        beam_image, alt_axis, az_axis = make_beam_image(
            vis_cal, freqs_hz, positions, rfi_mask,
            sun_alt, sun_az, args.image_fov, args.image_npix,
            l_fs=l_sun, m_fs=m_sun, n_fs=n_sun,
        )
        print(f"    Image peak at pixel: {np.unravel_index(np.argmax(beam_image), beam_image.shape)}")
        peak_alt_idx, peak_az_idx = np.unravel_index(np.argmax(beam_image), beam_image.shape)
        print(f"    Peak alt={alt_axis[peak_alt_idx]:.2f}°, az={az_axis[peak_az_idx]:.2f}°")
        print(f"    Sun  alt={np.degrees(sun_alt):.2f}°, az={np.degrees(sun_az):.2f}°")

    # ── 11. Plots ──
    print(f"\n11. Generating plots...")
    pdf_path = os.path.join(out_dir, f"verify_vis_beamform_{args.obs}.pdf")
    freqs_mhz = vis_fs.freq_mhz

    with PdfPages(pdf_path) as pdf:
        # Page 1: Power spectra (Sun and Off for each config)
        n_configs = len(results)
        fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_configs, 1)))

        ax = axes[0]
        for i, (label, r) in enumerate(results.items()):
            ax.plot(freqs_mhz, r["power_sun"], label=f"{label} (Sun)",
                    color=colors[i], alpha=0.8, lw=0.8)
            ax.plot(freqs_mhz, r["power_off"], label=f"{label} (Off)",
                    color=colors[i], alpha=0.4, lw=0.8, ls="--")
        # Shade RFI regions
        for lo, hi in rfi_ranges:
            ax.axvspan(lo, hi, color="gray", alpha=0.2)
        ax.set_ylabel("Beamformed Power")
        ax.set_title(f"Visibility Beamforming: {args.obs} / {source}")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

        # Per-channel Sun/Off ratio
        ax = axes[1]
        for i, (label, r) in enumerate(results.items()):
            ratio_chan = r["power_sun"] / np.where(
                np.abs(r["power_off"]) > eps, r["power_off"], eps)
            ax.plot(freqs_mhz, ratio_chan, label=f"{label} ({r['ratio']:.1f})",
                    color=colors[i], alpha=0.8, lw=0.8)
        for lo, hi in rfi_ranges:
            ax.axvspan(lo, hi, color="gray", alpha=0.2)
        ax.axhline(1.0, color="k", ls=":", lw=0.5, alpha=0.5)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Sun / Off Ratio")
        ax.set_ylim(-2, max(20, max(r["ratio"] for r in results.values()) * 1.5))
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Bar chart summary
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = list(results.keys())
        ratios = [results[l]["ratio"] for l in labels]
        bars = ax.bar(range(len(labels)), ratios, color=colors[:len(labels)])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Sun / Off Power Ratio")
        ax.set_title(f"Beamforming Verification: {args.obs}")
        ax.axhline(1.0, color="k", ls=":", lw=1, alpha=0.5, label="No coherence")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        for i, (bar, ratio) in enumerate(zip(bars, ratios)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{ratio:.1f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: Beam image (if generated)
        if beam_image is not None:
            fig, ax = plt.subplots(figsize=(8, 8))
            extent = [az_axis[0], az_axis[-1], alt_axis[0], alt_axis[-1]]
            im = ax.imshow(beam_image, origin="lower", extent=extent,
                           aspect="auto", cmap="inferno")
            ax.plot(np.degrees(sun_az), np.degrees(sun_alt), "w+",
                    markersize=15, markeredgewidth=2, label="Sun")
            ax.set_xlabel("Azimuth (°)")
            ax.set_ylabel("Altitude (°)")
            ax.set_title(f"Beam Image: {best_label}")
            ax.legend(loc="upper right")
            plt.colorbar(im, ax=ax, label="Beamformed Power")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"   Saved: {pdf_path}")

    # ── 12. Save NPZ ──
    npz_path = os.path.join(out_dir, f"verify_vis_beamform_{args.obs}.npz")
    save_dict = {
        "freqs_mhz": freqs_mhz,
        "freqs_hz": freqs_hz,
        "rfi_mask": rfi_mask,
        "ant_ids": ant_ids,
        "positions_enu": positions,
        "sun_alt_deg": np.degrees(sun_alt),
        "sun_az_deg": np.degrees(sun_az),
        "off_az_offset_deg": args.off_az_offset,
        "obs": args.obs,
        "source": source,
    }
    for label, r in results.items():
        key = label.replace(" ", "_").replace("/", "_").replace("+", "_")
        save_dict[f"power_sun_{key}"] = r["power_sun"]
        save_dict[f"power_off_{key}"] = r["power_off"]
        save_dict[f"ratio_{key}"] = r["ratio"]

    if beam_image is not None:
        save_dict["beam_image"] = beam_image
        save_dict["beam_alt_axis"] = alt_axis
        save_dict["beam_az_axis"] = az_axis

    np.savez_compressed(npz_path, **save_dict)
    print(f"   Saved: {npz_path}")

    print(f"\n{'=' * 70}")
    print("VISIBILITY BEAMFORMING VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
