#!/usr/bin/env python3
"""
Verify calibration weights via voltage coherent beamforming.

Load DADA voltage dumps, apply calibration + geometric weights, form
coherent beams toward the Sun and off-source. Compare CB/IB ratio
(ideal = N_ant on-source with perfect weights, ~1 off-source).

DADA I/O reused from track2_voltage_beamform.py.
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

OVRO = EarthLocation(lat=37.2339 * u.deg, lon=-118.2821 * u.deg, height=1222 * u.m)

# DADA format parameters
HEADER_SIZE = 4096
N_SNAPS = 11
N_ADC = 12
N_CHAN_PER_SUB = 1024
ACTIVE_SNAPS = [0, 2, 4]
CHAN_BW_MHZ = 125.0 / 4096
FREQ_START_MHZ = 375.0
C_M_S = 299792458.0

VOLTAGE_DIR = "/mnt/nvme3/data/casm/voltage_dumps"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Verify calibration weights via voltage beamforming (CB/IB ratio)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--voltage-dir", default=VOLTAGE_DIR,
                    help="Voltage dump directory")
    p.add_argument("--timestamp", required=True,
                    help="Voltage dump timestamp (e.g. 2026-02-17-21:10:43)")
    p.add_argument(
        "--layout",
        default=os.path.expanduser(
            "~/software/dev/antenna_layouts/antenna_layout_current.csv"
        ),
        help="Antenna layout CSV",
    )
    p.add_argument(
        "--cal-weights", action="append", default=[],
        help="Calibration weights NPZ (repeatable)",
    )
    p.add_argument("--freq-low", type=float, default=390.0,
                    help="Lower freq bound in MHz (default: 390)")
    p.add_argument("--freq-high", type=float, default=468.0,
                    help="Upper freq bound in MHz (default: 468)")
    p.add_argument("--n-time", type=int, default=65536,
                    help="Number of time samples to read (default: 65536)")
    p.add_argument("--off-az-offset", type=float, default=90.0,
                    help="Off-source azimuth offset in degrees (default: 90)")
    p.add_argument("--exclude-ants", type=int, nargs="+", default=[],
                    help="Antenna IDs to exclude")
    p.add_argument("--output-dir", default=None,
                    help="Output directory (default: results/verify_voltage_beamform/)")
    return p.parse_args(argv)


# ── DADA I/O (from track2_voltage_beamform.py) ──

def read_dada_header(filename):
    with open(filename, "rb") as f:
        hdr_bytes = f.read(HEADER_SIZE)
    hdr_text = hdr_bytes.decode("ascii", errors="ignore")
    hdr = {}
    for line in hdr_text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            hdr[parts[0]] = parts[1].strip()
    return hdr


def unpack_4bit(data):
    real = (data >> 4).astype(np.int8)
    imag = (data & 0x0F).astype(np.int8)
    real = np.where(real > 7, real - 16, real)
    imag = np.where(imag > 7, imag - 16, imag)
    return (real + 1j * imag).astype(np.complex64)


def read_subband(filename, n_time=None):
    header = read_dada_header(filename)
    file_size = os.path.getsize(filename)
    data_size = file_size - HEADER_SIZE
    bytes_per_time = N_SNAPS * N_CHAN_PER_SUB * N_ADC
    n_time_total = data_size // bytes_per_time
    if n_time is None:
        n_time = n_time_total
    else:
        n_time = min(n_time, n_time_total)
    print(f"  Reading {os.path.basename(filename)}: n_time={n_time}/{n_time_total}")
    with open(filename, "rb") as f:
        f.seek(HEADER_SIZE)
        raw = np.frombuffer(f.read(n_time * bytes_per_time), dtype=np.uint8)
    raw = raw.reshape(n_time, N_SNAPS, N_CHAN_PER_SUB, N_ADC)
    voltages = {}
    for s in ACTIVE_SNAPS:
        voltages[s] = unpack_4bit(raw[:, s, :, :])
    return voltages, header


def load_antenna_layout(csv_path):
    """Load antenna layout from casm_io-format CSV."""
    ant_data = np.genfromtxt(csv_path, delimiter=",", skip_header=1,
                              dtype=None, encoding="utf-8")
    ant_ids = np.array([r[0] for r in ant_data], dtype=int)
    ant_x = np.array([r[1] for r in ant_data], dtype=float)
    ant_y = np.array([r[2] for r in ant_data], dtype=float)
    ant_z = np.array([r[3] for r in ant_data], dtype=float)
    ant_snaps = np.array([r[4] for r in ant_data], dtype=int)
    ant_adcs = np.array([r[5] for r in ant_data], dtype=int)
    return ant_ids, ant_x, ant_y, ant_z, ant_snaps, ant_adcs


def find_voltage_file(voltage_dir, timestamp):
    """Find voltage dump files for all 3 subbands."""
    files = []
    for sub_dir in ["chan2048_3071", "chan1024_2047", "chan0_1023"]:
        sub_path = os.path.join(voltage_dir, sub_dir)
        if not os.path.isdir(sub_path):
            files.append(None)
            continue
        matches = [f for f in os.listdir(sub_path) if f.startswith(timestamp)]
        if matches:
            files.append(os.path.join(sub_path, matches[0]))
        else:
            files.append(None)
    return files


def geometric_weights(l, m, n, freqs_hz, ax, ay, az):
    """Geometric delay weights toward direction (l,m,n)."""
    tau_geom = -(ax * l + ay * m + az * n) / C_M_S
    w_geom = np.exp(-1j * 2 * np.pi * freqs_hz[None, :] * tau_geom[:, None])
    return w_geom.astype(np.complex64)


def load_cal_weights(npz_path, n_chan_full, band_mask, ant_ids_layout):
    """Load cal weights from NPZ, match to layout antenna ordering, restrict to band.

    Returns weights (n_ant, n_chan_band), flags (n_chan_band), label.
    """
    d = np.load(npz_path, allow_pickle=True)
    w = d["weights"]  # (n_ant_cal, n_chan_cal)
    flags = d["flags"]
    freqs = d["freqs_hz"]
    cal_ant_ids = d["ant_ids"]

    # Ensure ascending
    if freqs[1] < freqs[0]:
        w = w[:, ::-1]
        flags = flags[::-1]

    # Match to layout antenna ordering
    cal_id_to_idx = {int(aid): i for i, aid in enumerate(cal_ant_ids)}
    n_ant = len(ant_ids_layout)
    w_matched = np.ones((n_ant, w.shape[1]), dtype=np.complex64)
    for i, aid in enumerate(ant_ids_layout):
        if int(aid) in cal_id_to_idx:
            w_matched[i] = w[cal_id_to_idx[int(aid)]]

    # Restrict to band
    w_band = w_matched[:, band_mask]
    flags_band = flags[band_mask]

    # Fill zero-weight channels with identity
    zero_mask = np.all(w_band == 0, axis=0)
    w_band[:, zero_mask] = 1.0 + 0j

    label = os.path.basename(npz_path).replace(".npz", "")
    return w_band, flags_band, label


def main(argv=None):
    args = parse_args(argv)

    out_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "verify_voltage_beamform",
    )
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("VOLTAGE BEAMFORMING VERIFICATION (CB/IB)")
    print("=" * 70)

    # ── 1. Load antenna layout ──
    print(f"\n1. Loading antenna layout: {args.layout}")
    ant_ids, ant_x, ant_y, ant_z, ant_snaps, ant_adcs = load_antenna_layout(args.layout)
    n_ant = len(ant_ids)
    print(f"   {n_ant} antennas: {ant_ids}")

    # Exclude antennas
    if args.exclude_ants:
        keep = ~np.isin(ant_ids, args.exclude_ants)
        ant_ids = ant_ids[keep]
        ant_x, ant_y, ant_z = ant_x[keep], ant_y[keep], ant_z[keep]
        ant_snaps, ant_adcs = ant_snaps[keep], ant_adcs[keep]
        n_ant = len(ant_ids)
        print(f"   After excluding {args.exclude_ants}: {n_ant} antennas")

    # ── 2. Find and read voltage data ──
    print(f"\n2. Finding voltage files for {args.timestamp}...")
    sub_files = find_voltage_file(args.voltage_dir, args.timestamp)
    for i, sf in enumerate(sub_files):
        status = os.path.basename(sf) if sf else "NOT FOUND"
        print(f"   Subband {i}: {status}")

    print(f"\n3. Reading voltage data (n_time={args.n_time})...")
    ant_voltages = None
    hdr = None

    for sub_idx, sf in enumerate(sub_files):
        if sf is None:
            continue
        ch_start = (2 - sub_idx) * N_CHAN_PER_SUB
        ch_end = ch_start + N_CHAN_PER_SUB

        voltages_snap, hdr = read_subband(sf, n_time=args.n_time)

        if ant_voltages is None:
            n_time = list(voltages_snap.values())[0].shape[0]
            ant_voltages = np.zeros((n_time, 3072, n_ant), dtype=np.complex64)
            print(f"   Allocated: {ant_voltages.shape} "
                  f"({ant_voltages.nbytes / 1e9:.1f} GB)")

        for i in range(n_ant):
            snap = ant_snaps[i]
            adc = ant_adcs[i]
            if snap in voltages_snap:
                ant_voltages[:, ch_start:ch_end, i] = voltages_snap[snap][:, ::-1, adc]
        del voltages_snap

    if ant_voltages is None:
        print("   ERROR: No voltage data found!")
        sys.exit(1)

    # ── 3. Band-select ──
    freqs_mhz_full = FREQ_START_MHZ + np.arange(3072) * CHAN_BW_MHZ
    band_mask = (freqs_mhz_full >= args.freq_low) & (freqs_mhz_full <= args.freq_high)
    n_chan_band = int(np.sum(band_mask))
    freqs_mhz = freqs_mhz_full[band_mask]
    freqs_hz = freqs_mhz.astype(np.float64) * 1e6

    ant_voltages_band = ant_voltages[:, band_mask, :]
    del ant_voltages
    print(f"\n4. Band: {args.freq_low}-{args.freq_high} MHz, {n_chan_band} channels")
    print(f"   Voltage shape: {ant_voltages_band.shape} "
          f"({ant_voltages_band.nbytes / 1e9:.1f} GB)")

    # Per-antenna power
    print("\n   Per-antenna band power:")
    for i in range(n_ant):
        p = np.mean(np.abs(ant_voltages_band[:, :, i]) ** 2)
        print(f"   Ant {ant_ids[i]:2d}: power={p:.4f}")

    # ── 4. Compute Sun position ──
    print(f"\n5. Computing Sun position...")
    utc_start = hdr.get("UTC_START", args.timestamp) if hdr else args.timestamp
    utc_iso = utc_start[:10] + " " + utc_start[11:]
    obs_time = Time(utc_iso, format="iso", scale="utc")
    print(f"   Obs time: {obs_time.iso} UTC")

    altaz_frame = AltAz(obstime=obs_time, location=OVRO)
    sun_altaz = get_sun(obs_time).transform_to(altaz_frame)
    sun_alt = sun_altaz.alt.rad
    sun_az = sun_altaz.az.rad
    print(f"   Sun: alt={np.degrees(sun_alt):.2f}°, az={np.degrees(sun_az):.2f}°")

    l_sun = np.cos(sun_alt) * np.sin(sun_az)
    m_sun = np.cos(sun_alt) * np.cos(sun_az)
    n_sun = np.sin(sun_alt)

    off_az = sun_az + np.radians(args.off_az_offset)
    l_off = np.cos(sun_alt) * np.sin(off_az)
    m_off = np.cos(sun_alt) * np.cos(off_az)
    n_off = np.sin(sun_alt)

    # ── 5. Load cal weights and build weight sets ──
    print(f"\n6. Building weight sets...")
    w_geo_sun = geometric_weights(l_sun, m_sun, n_sun, freqs_hz, ant_x, ant_y, ant_z)
    w_geo_off = geometric_weights(l_off, m_off, n_off, freqs_hz, ant_x, ant_y, ant_z)
    amp = np.ones((n_ant, n_chan_band), dtype=np.float32) / n_ant

    weight_sets = {}

    # Geo-only toward Sun (no calibration baseline)
    weight_sets["Geo-only Sun"] = amp * w_geo_sun

    # Each cal weight set + geo toward Sun and off-Sun
    for npz_path in args.cal_weights:
        w_cal, flags_cal, label = load_cal_weights(
            npz_path, 3072, band_mask, ant_ids
        )
        weight_sets[f"{label} Sun"] = amp * w_cal * w_geo_sun
        weight_sets[f"{label} Off"] = amp * w_cal * w_geo_off

    # ── 6. Beamform ──
    print(f"\n7. Beamforming ({n_time} time × {n_chan_band} chan × {n_ant} ant)...")
    power_spectra = {}

    for name, w in weight_sets.items():
        # w is (n_ant, n_chan_band), voltages are (n_time, n_chan_band, n_ant)
        # CB: sum over antennas of w * v, then |.|^2, then time-average
        bf = np.sum(ant_voltages_band * w.T[None, :, :], axis=2)  # (n_time, n_chan)
        power = np.mean(np.abs(bf) ** 2, axis=0)  # (n_chan,)
        power_spectra[name] = power
        print(f"   {name:35s}: mean power = {np.mean(power):.6f}")
        del bf

    # Incoherent beam
    incoh_power = np.zeros(n_chan_band, dtype=np.float64)
    for i in range(n_ant):
        incoh_power += np.mean(np.abs(ant_voltages_band[:, :, i]) ** 2, axis=0)
    incoh_power /= n_ant
    power_spectra["Incoherent"] = incoh_power.astype(np.float32)
    print(f"   {'Incoherent':35s}: mean power = {np.mean(incoh_power):.6f}")

    del ant_voltages_band

    # ── 7. Compute CB/IB ratios ──
    p_incoh = np.mean(power_spectra["Incoherent"])
    eps = 1e-30

    print(f"\n{'=' * 70}")
    print(f"CB / IB RATIOS ({n_ant} antennas, ideal = {n_ant})")
    print(f"Band: {args.freq_low}-{args.freq_high} MHz ({n_chan_band} channels)")
    print(f"{'=' * 70}")
    print(f"{'Config':35s}  {'CB/IB':>8s}  {'dB':>8s}")
    print("-" * 55)

    ratios = {}
    for name in sorted(power_spectra.keys()):
        if name == "Incoherent":
            continue
        p = np.mean(power_spectra[name])
        ratio = p / max(p_incoh, eps)
        ratio_db = 10 * np.log10(max(ratio, eps))
        ratios[name] = ratio
        print(f"  {name:35s}  {ratio:8.3f}  {ratio_db:+8.2f} dB")

    # ── 8. Decision ──
    print(f"\n{'=' * 70}")
    print("DECISION:")
    best_label = max(ratios, key=ratios.get)
    best_ratio = ratios[best_label]

    if best_ratio > 0.5 * n_ant:
        print(f"  Coherence WORKING: best CB/IB = {best_ratio:.1f} ({best_label})")
    elif best_ratio < 2.0:
        print(f"  Coherence BROKEN: best CB/IB = {best_ratio:.1f}")
    else:
        print(f"  Coherence PARTIAL: best CB/IB = {best_ratio:.1f} ({best_label})")

    # Compare Sun vs Off for each cal set
    for npz_path in args.cal_weights:
        label = os.path.basename(npz_path).replace(".npz", "")
        sun_key = f"{label} Sun"
        off_key = f"{label} Off"
        if sun_key in ratios and off_key in ratios:
            print(f"\n  {label}: Sun={ratios[sun_key]:.2f}, Off={ratios[off_key]:.2f}, "
                  f"Sun/Off={ratios[sun_key]/max(ratios[off_key], eps):.2f}")

    # ── 9. Plots ──
    print(f"\n8. Generating plots...")
    pdf_path = os.path.join(out_dir, f"verify_voltage_beamform_{args.timestamp}.pdf")

    with PdfPages(pdf_path) as pdf:
        # Page 1: Power spectra
        fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        n_lines = len(power_spectra)
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_lines, 1)))

        ax = axes[0]
        for i, (name, power) in enumerate(sorted(power_spectra.items())):
            power_db = 10 * np.log10(np.maximum(power, eps))
            ls = "--" if name == "Incoherent" else "-"
            ax.plot(freqs_mhz, power_db, label=name, color=colors[i],
                    alpha=0.8, lw=0.8, ls=ls)
        ax.set_ylabel("Power (dB)")
        ax.set_title(f"Voltage Beamforming: {args.timestamp}")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

        # Per-channel CB/IB ratio
        ax = axes[1]
        for i, (name, power) in enumerate(sorted(power_spectra.items())):
            if name == "Incoherent":
                continue
            ratio_chan = power / np.maximum(power_spectra["Incoherent"], eps)
            ax.plot(freqs_mhz, ratio_chan, label=name, color=colors[i],
                    alpha=0.8, lw=0.8)
        ax.axhline(n_ant, color="red", ls="--", lw=1, alpha=0.7,
                    label=f"N={n_ant} (ideal)")
        ax.axhline(1.0, color="k", ls=":", lw=0.5, alpha=0.5)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("CB / IB Ratio")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Summary bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        labels_plot = list(ratios.keys())
        ratio_vals = [ratios[l] for l in labels_plot]
        bars = ax.bar(range(len(labels_plot)), ratio_vals,
                      color=plt.cm.tab10(np.linspace(0, 1, max(len(labels_plot), 1))))
        ax.set_xticks(range(len(labels_plot)))
        ax.set_xticklabels(labels_plot, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("CB / IB Ratio")
        ax.set_title(f"Voltage Beamforming Summary: {args.timestamp}")
        ax.axhline(n_ant, color="red", ls="--", lw=1, alpha=0.7,
                    label=f"N={n_ant} (ideal)")
        ax.axhline(1.0, color="k", ls=":", lw=1, alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        for bar, ratio in zip(bars, ratio_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{ratio:.1f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"   Saved: {pdf_path}")

    # ── 10. Save NPZ ──
    npz_path = os.path.join(out_dir, f"verify_voltage_beamform_{args.timestamp}.npz")
    save_dict = {
        "freqs_mhz": freqs_mhz,
        "n_ant": n_ant,
        "ant_ids": ant_ids,
        "timestamp": args.timestamp,
        "sun_alt_deg": np.degrees(sun_alt),
        "sun_az_deg": np.degrees(sun_az),
    }
    for name, power in power_spectra.items():
        key = name.replace(" ", "_").replace("/", "_").replace("+", "_")
        save_dict[f"power_{key}"] = power
    for name, ratio in ratios.items():
        key = name.replace(" ", "_").replace("/", "_").replace("+", "_")
        save_dict[f"ratio_{key}"] = ratio
    np.savez_compressed(npz_path, **save_dict)
    print(f"   Saved: {npz_path}")

    print(f"\n{'=' * 70}")
    print("VOLTAGE BEAMFORMING VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
