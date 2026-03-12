#!/usr/bin/env python3
"""
Track 2: Voltage Coherent Beamforming Test.

Load voltage dumps, form coherent beams with different weight combinations,
compare coherent/incoherent ratios to diagnose calibration issues.

Test cases:
  A) Geo-only toward Sun
  B) Feb19 cal + geo
  C) Mar10 cal + geo
  D) Mar10 cal + geo, Ant 4 excluded (weight=0)
  E) Off-source control (az + 90°)
  F) Incoherent baseline

Adapted from coherent_vs_incoherent.py
"""

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

# ── Config ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "results", "voltage_beamform")
os.makedirs(OUT_DIR, exist_ok=True)

VOLTAGE_DIR = "/mnt/nvme3/data/casm/voltage_dumps"
LAYOUT_CSV = os.path.expanduser("~/software/dev/antenna_layouts/antenna_layout_current.csv")
CAL_NPZ_FEB19 = "/home/casm/software/vishnu/casm-voltage-analysis/correlator_analysis/results/multi_day_svd/svd_weights_feb19.npz"
CAL_NPZ_MAR10 = "/home/casm/software/dev/casm_calibrator/results/svd_weights_mar10.npz"

# DADA format parameters
HEADER_SIZE = 4096
N_SNAPS = 11
N_ADC = 12
N_CHAN_PER_SUB = 1024
ACTIVE_SNAPS = [0, 2, 4]
CHAN_BW_MHZ = 125.0 / 4096
FREQ_START_MHZ = 375.0
C_M_S = 299792458.0

FREQ_LOW_MHZ = 430.0
FREQ_HIGH_MHZ = 460.0
N_TIME_READ = 131072

OVRO = EarthLocation(lat=37.2339 * u.deg, lon=-118.2821 * u.deg, height=1222 * u.m)


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


def geometric_weights(l, m, n, freqs_hz, ax, ay, az):
    tau_geom = -(ax * l + ay * m + az * n) / C_M_S
    w_geom = np.exp(-1j * 2 * np.pi * freqs_hz[None, :] * tau_geom[:, None])
    return w_geom.astype(np.complex64)


def load_cal_weights(npz_path, n_chan_band, band_mask):
    """Load and slice cal weights to band."""
    d = np.load(npz_path, allow_pickle=True)
    w = d["weights"]  # (16, 3072)
    flags = d["flags"]
    freqs = d["freqs_hz"]
    ant_ids = d["ant_ids"]

    # Ensure ascending
    if freqs[1] < freqs[0]:
        w = w[:, ::-1]
        flags = flags[::-1]

    # Restrict to band
    w_band = w[:, band_mask]
    flags_band = flags[band_mask]

    # Fill zero-weight channels with identity
    zero_mask = np.all(w_band == 0, axis=0)
    w_band[:, zero_mask] = 1.0 + 0j
    return w_band, flags_band, ant_ids


def find_voltage_file(timestamp):
    """Find voltage dump files for all 3 subbands."""
    files = []
    for sub_dir in ["chan2048_3071", "chan1024_2047", "chan0_1023"]:
        sub_path = os.path.join(VOLTAGE_DIR, sub_dir)
        matches = [f for f in os.listdir(sub_path) if f.startswith(timestamp)]
        if matches:
            files.append(os.path.join(sub_path, matches[0]))
        else:
            files.append(None)
    return files


def main():
    print("=" * 70)
    print("TRACK 2: VOLTAGE COHERENT BEAMFORMING TEST")
    print("=" * 70)

    # ── Find available voltage data ──
    print("\n1. Scanning voltage dumps...")
    available_timestamps = set()
    for sub_dir in ["chan0_1023", "chan1024_2047", "chan2048_3071"]:
        sub_path = os.path.join(VOLTAGE_DIR, sub_dir)
        if os.path.isdir(sub_path):
            for f in os.listdir(sub_path):
                ts = f.split("_0000")[0]
                available_timestamps.add(ts)
    print(f"   Available timestamps: {sorted(available_timestamps)}")

    # Use Feb 23 data (closest to Feb 19 cal)
    TIMESTAMP = "2026-02-23-23:05:01"
    if TIMESTAMP not in available_timestamps:
        # Try other timestamps
        for ts in sorted(available_timestamps, reverse=True):
            TIMESTAMP = ts
            break

    print(f"   Using: {TIMESTAMP}")

    # ── Load antenna layout ──
    print("\n2. Loading antenna layout...")
    ant_ids, ant_x, ant_y, ant_z, ant_snaps, ant_adcs = load_antenna_layout(LAYOUT_CSV)
    n_ant = len(ant_ids)
    print(f"   {n_ant} antennas loaded")

    # ── Read voltage data ──
    print("\n3. Reading voltage data...")
    sub_files = find_voltage_file(TIMESTAMP)
    for i, sf in enumerate(sub_files):
        if sf is None:
            print(f"   Subband {i}: NOT FOUND")
        else:
            print(f"   Subband {i}: {os.path.basename(sf)}")

    ant_voltages = None
    hdr = None

    for sub_idx, sf in enumerate(sub_files):
        if sf is None:
            continue
        ch_start = (2 - sub_idx) * N_CHAN_PER_SUB
        ch_end = ch_start + N_CHAN_PER_SUB

        voltages_snap, hdr = read_subband(sf, n_time=N_TIME_READ)

        if ant_voltages is None:
            n_time = list(voltages_snap.values())[0].shape[0]
            ant_voltages = np.zeros((n_time, 3072, n_ant), dtype=np.complex64)
            print(f"   Allocated: {ant_voltages.shape}")

        for i in range(n_ant):
            snap = ant_snaps[i]
            adc = ant_adcs[i]
            if snap in voltages_snap:
                ant_voltages[:, ch_start:ch_end, i] = voltages_snap[snap][:, ::-1, adc]
        del voltages_snap

    if ant_voltages is None:
        print("   ERROR: No voltage data found!")
        sys.exit(1)

    # ── Compute Sun position ──
    print("\n4. Computing Sun position...")
    utc_start = hdr.get("UTC_START", TIMESTAMP) if hdr else TIMESTAMP
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

    # Off-source: az + 90°
    off_az = sun_az + np.radians(90.0)
    l_off = np.cos(sun_alt) * np.sin(off_az)
    m_off = np.cos(sun_alt) * np.cos(off_az)
    n_off = np.sin(sun_alt)

    # ── Restrict to clean band ──
    freqs_mhz_full = FREQ_START_MHZ + np.arange(3072) * CHAN_BW_MHZ
    band_mask = (freqs_mhz_full >= FREQ_LOW_MHZ) & (freqs_mhz_full <= FREQ_HIGH_MHZ)
    n_chan_band = int(np.sum(band_mask))
    freqs_mhz = freqs_mhz_full[band_mask]
    freqs_hz = freqs_mhz.astype(np.float64) * 1e6

    ant_voltages_band = ant_voltages[:, band_mask, :]
    del ant_voltages
    print(f"\n5. Band: {FREQ_LOW_MHZ}-{FREQ_HIGH_MHZ} MHz, {n_chan_band} channels")
    print(f"   Voltage shape: {ant_voltages_band.shape}")

    # Power check
    print("\n   Per-antenna band power:")
    for i in range(n_ant):
        p = np.mean(np.abs(ant_voltages_band[:, :, i]) ** 2)
        flag = " <-- ANT 4" if ant_ids[i] == 4 else ""
        print(f"   Ant {ant_ids[i]:2d}: power={p:.4f}{flag}")

    # ── Load calibration weights ──
    print("\n6. Loading calibration weights...")
    w_cal_feb19, flags_feb19, _ = load_cal_weights(CAL_NPZ_FEB19, n_chan_band, band_mask)
    print(f"   Feb19: shape={w_cal_feb19.shape}")

    try:
        w_cal_mar10, flags_mar10, _ = load_cal_weights(CAL_NPZ_MAR10, n_chan_band, band_mask)
        print(f"   Mar10: shape={w_cal_mar10.shape}")
        has_mar10 = True
    except FileNotFoundError:
        print("   Mar10: NOT FOUND, skipping")
        has_mar10 = False

    # ── Generate geometric weights ──
    print("\n7. Computing geometric weights...")
    w_geo_sun = geometric_weights(l_sun, m_sun, n_sun, freqs_hz, ant_x, ant_y, ant_z)
    w_geo_off = geometric_weights(l_off, m_off, n_off, freqs_hz, ant_x, ant_y, ant_z)
    print(f"   Geo weights shape: {w_geo_sun.shape}")

    # Amplitude: 1/N for all antennas
    amp_weights = np.ones((n_ant, n_chan_band), dtype=np.float32) / n_ant

    # ── Build weight sets ──
    weight_sets = {}

    # A) Geo-only toward Sun
    weight_sets["A: Geo-only Sun"] = amp_weights * w_geo_sun

    # B) Feb19 cal + geo
    weight_sets["B: Feb19+geo Sun"] = amp_weights * w_cal_feb19 * w_geo_sun

    # C) Mar10 cal + geo
    if has_mar10:
        weight_sets["C: Mar10+geo Sun"] = amp_weights * w_cal_mar10 * w_geo_sun

    # D) Mar10 cal + geo, Ant 4 excluded
    if has_mar10:
        amp_no_ant4 = amp_weights.copy()
        ant4_mask = ant_ids == 4
        amp_no_ant4[ant4_mask, :] = 0.0
        # Renormalize
        n_active_no4 = n_ant - int(np.sum(ant4_mask))
        amp_no_ant4[~ant4_mask, :] = 1.0 / n_active_no4
        weight_sets["D: Mar10+geo no Ant4"] = amp_no_ant4 * w_cal_mar10 * w_geo_sun

    # E) Off-source control
    weight_sets["E: Feb19+geo Off"] = amp_weights * w_cal_feb19 * w_geo_off

    # ── Beamform ──
    print(f"\n8. Beamforming ({n_time} time × {n_chan_band} chan × {n_ant} ant)...")
    power_spectra = {}

    for name, w in weight_sets.items():
        bf = np.sum(ant_voltages_band * w.T[None, :, :], axis=2)
        power = np.mean(np.abs(bf) ** 2, axis=0)
        power_spectra[name] = power
        print(f"   {name}: mean power = {np.mean(power):.6f}")
        del bf

    # F) Incoherent
    incoh_power = np.zeros(n_chan_band, dtype=np.float64)
    for i in range(n_ant):
        incoh_power += np.mean(np.abs(ant_voltages_band[:, :, i]) ** 2, axis=0)
    incoh_power /= n_ant
    power_spectra["F: Incoherent"] = incoh_power.astype(np.float32)
    print(f"   F: Incoherent: mean power = {np.mean(incoh_power):.6f}")

    del ant_voltages_band

    # ── Compute and print ratios ──
    p_incoh = np.mean(power_spectra["F: Incoherent"])
    eps = 1e-30

    print(f"\n{'=' * 70}")
    print(f"COHERENT / INCOHERENT RATIOS ({n_ant} antennas)")
    print(f"Band: {FREQ_LOW_MHZ}-{FREQ_HIGH_MHZ} MHz ({n_chan_band} channels)")
    print(f"Ideal Coh/Incoh for {n_ant} ants = {n_ant}")
    print(f"{'=' * 70}")

    ratios = {}
    for name in sorted(power_spectra.keys()):
        if name.startswith("F:"):
            continue
        p = np.mean(power_spectra[name])
        ratio = p / max(p_incoh, eps)
        ratio_db = 10 * np.log10(max(ratio, eps))
        ratios[name] = ratio
        print(f"  {name:30s}  ratio={ratio:8.3f}  ({ratio_db:+6.2f} dB)")

    # ── Decision ──
    print(f"\n{'=' * 70}")
    print("DECISION:")

    best_label = max(ratios, key=ratios.get)
    best_ratio = ratios[best_label]

    if best_ratio > 0.5 * n_ant:
        print(f"  Coherence WORKING: best ratio = {best_ratio:.1f} ({best_label})")
        print(f"  If SNAP beamforming shows autocorrelations only, the issue is")
        print(f"  in the SNAP pipeline (weight loading, GPU kernel, or data path).")
    elif best_ratio < 2.0:
        print(f"  Coherence BROKEN: best ratio = {best_ratio:.1f}")
        print(f"  Calibration weights are not producing coherent beams.")
    else:
        print(f"  Coherence PARTIAL: best ratio = {best_ratio:.1f} ({best_label})")

    if has_mar10 and "D: Mar10+geo no Ant4" in ratios:
        r_with = ratios.get("C: Mar10+geo Sun", 0)
        r_without = ratios.get("D: Mar10+geo no Ant4", 0)
        if r_without > r_with * 1.3:
            print(f"\n  Excluding Ant 4 IMPROVES coherence: {r_with:.1f} -> {r_without:.1f}")
            print(f"  -> Ant 4 is degrading the beam. Recommend flagging.")
        else:
            print(f"\n  Ant 4 exclusion: {r_with:.1f} -> {r_without:.1f} (no significant improvement)")

    # ── Plot ──
    print(f"\n9. Generating plots...")
    plot_path = os.path.join(OUT_DIR, "coherent_vs_incoherent.pdf")
    with PdfPages(plot_path) as pdf:
        # Page 1: Power spectra
        fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        colors = plt.cm.tab10(np.linspace(0, 1, len(power_spectra)))

        ax = axes[0]
        for i, (name, power) in enumerate(sorted(power_spectra.items())):
            power_db = 10 * np.log10(np.maximum(power, eps))
            ax.plot(freqs_mhz, power_db, label=name, color=colors[i], alpha=0.8, linewidth=0.8)
        ax.set_ylabel("Power (dB)")
        ax.set_title(f"Coherent vs Incoherent: {TIMESTAMP}")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        for i, (name, power) in enumerate(sorted(power_spectra.items())):
            if name.startswith("F:"):
                continue
            ratio_per_chan = power / np.maximum(power_spectra["F: Incoherent"], eps)
            ax.plot(freqs_mhz, ratio_per_chan, label=name, color=colors[i], alpha=0.8, lw=0.8)
        ax.axhline(n_ant, color="red", ls="--", lw=1, label=f"N={n_ant} (ideal)", alpha=0.7)
        ax.axhline(1.0, color="k", ls=":", lw=0.5, alpha=0.5)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Coh/Incoh ratio")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

        axes[0].set_xlim(freqs_mhz[0], freqs_mhz[-1])
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"   Saved: {plot_path}")

    # Save data
    out_npz = os.path.join(OUT_DIR, "coherent_vs_incoherent.npz")
    save_dict = dict(
        freqs_mhz=freqs_mhz,
        ratios={k: v for k, v in ratios.items()},
        n_ant=n_ant,
        timestamp=TIMESTAMP,
        sun_alt_deg=np.degrees(sun_alt),
        sun_az_deg=np.degrees(sun_az),
    )
    for name, power in power_spectra.items():
        key = name.replace(" ", "_").replace(":", "").replace("+", "_")
        save_dict[f"power_{key}"] = power
    np.savez_compressed(out_npz, **save_dict)
    print(f"   Saved: {out_npz}")

    print(f"\n{'=' * 70}")
    print("TRACK 2 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
