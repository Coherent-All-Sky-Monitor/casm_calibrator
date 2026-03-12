#!/usr/bin/env python3
"""
Large-FoV beam image comparison: calibrated vs uncalibrated vs night.

Makes beam images over a large portion of the sky for:
  1. Mar10 transit data + Mar10 cal weights (should show Sun)
  2. Mar10 transit data + NO cal (null test — incoherent phases)
  3. Mar10 night data + Mar10 cal weights (no Sun above horizon)

All images use the same color scale for direct comparison.
Cross-correlations only (no autocorrelations).
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

from casm_io.correlator.mapping import AntennaMapping
from casm_io.constants import C_LIGHT_M_S
from casm_vis_analysis.sources import find_transit_window

from casm_calibrator.visibility import VisibilityLoader, VisibilityMatrix
from casm_calibrator.fringe_stop import FringeStopMatrix
from casm_calibrator.rfi import RFIMask

OVRO = EarthLocation(lat=37.2339 * u.deg, lon=-118.2821 * u.deg, height=1222 * u.m)

DATA_DIR = "/mnt/nvme3/data/casm/visibilities_64ant/"
LAYOUT = os.path.expanduser("~/software/dev/antenna_layouts/antenna_layout_current.csv")
CAL_NPZ = "results/svd_weights_mar10.npz"
EXCLUDE_ANTS = [4]
RFI_RANGES = [(375.0, 390.0)]

# Transit observation (Mar10 daytime)
OBS_TRANSIT = "2026-03-10-14:05:53"
# Night observation (Mar10 12:51 AM local — Sun well below horizon)
OBS_NIGHT = "2026-03-10-07:51:44"

# Image parameters
IMAGE_FOV = 90.0   # degrees — large portion of sky
IMAGE_NPIX = 91    # pixels per side
NFILES = 4         # fewer files for speed


def altaz_to_enu(alt, az):
    l = np.cos(alt) * np.sin(az)
    m = np.cos(alt) * np.cos(az)
    n = np.sin(alt)
    return l, m, n


def compute_baseline_delay(pos_i, pos_j, l, m, n):
    b = pos_j - pos_i
    return (b[0] * l + b[1] * m + b[2] * n) / C_LIGHT_M_S


def visibility_beamform_cross(vis_cal, freqs_hz, positions, l, m, n,
                               l_fs=None, m_fs=None, n_fs=None):
    """Cross-correlation-only beamforming."""
    n_chan, n_ant, _ = vis_cal.shape
    power = np.zeros(n_chan, dtype=np.float64)

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


def load_and_prepare(obs_id, mapping, ant_ids, exclude_ants, source,
                     cal_gains, rfi_ranges, nfiles, fringe_stop_source=True):
    """Load, optionally fringe-stop, time-average, and calibrate visibilities.

    If fringe_stop_source is True, fringe-stop toward the source.
    For night data where the source is below horizon, set to False
    (no fringe-stopping — just time-average directly).

    Returns vis_cal (n_chan, n_ant, n_ant), freqs_hz, positions, rfi_mask,
            mid_time, fringe_stop_dir (l,m,n) or None.
    """
    loader = VisibilityLoader(mapping)
    vis_matrix = loader.load(DATA_DIR, obs_id, nfiles=nfiles)
    print(f"   vis shape: {vis_matrix.vis.shape}")
    print(f"   freq: {vis_matrix.freq_mhz[0]:.2f} - {vis_matrix.freq_mhz[-1]:.2f} MHz")

    # Exclude antennas
    all_ids = vis_matrix.ant_ids
    keep_mask = ~np.isin(all_ids, exclude_ants)
    keep_idx = np.where(keep_mask)[0]
    kept_ids = all_ids[keep_mask]
    positions = vis_matrix.positions_enu[keep_idx]
    vis_data = vis_matrix.vis[:, :, keep_idx][:, :, :, keep_idx]
    n_ant = len(kept_ids)

    vis_matrix = VisibilityMatrix(
        vis=vis_data,
        freq_mhz=vis_matrix.freq_mhz,
        time_unix=vis_matrix.time_unix,
        ant_ids=kept_ids,
        positions_enu=positions,
    )
    print(f"   {n_ant} antennas after excluding {exclude_ants}")

    mid_time = np.mean(vis_matrix.time_unix)
    fs_dir = None

    if fringe_stop_source:
        # Find transit window and trim
        try:
            i_start, i_end = find_transit_window(source, vis_matrix.time_unix,
                                                  min_alt_deg=10.0)
            n_transit = i_end - i_start + 1
            print(f"   Transit window: {i_start}-{i_end} ({n_transit} integrations)")
            vis_matrix = VisibilityMatrix(
                vis=vis_matrix.vis[i_start:i_end + 1],
                freq_mhz=vis_matrix.freq_mhz,
                time_unix=vis_matrix.time_unix[i_start:i_end + 1],
                ant_ids=vis_matrix.ant_ids,
                positions_enu=vis_matrix.positions_enu,
            )
            mid_time = np.mean(vis_matrix.time_unix)
        except Exception as e:
            print(f"   No transit window found: {e}")
            print(f"   Using all integrations")

        # Fringe-stop
        print(f"   Fringe-stopping toward {source}...")
        vis_matrix = FringeStopMatrix()(vis_matrix, source)

        # Get fringe-stop direction at midpoint
        t = Time(mid_time, format="unix")
        frame = AltAz(obstime=t, location=OVRO)
        sun = get_sun(t).transform_to(frame)
        sun_alt, sun_az = sun.alt.rad, sun.az.rad
        fs_dir = altaz_to_enu(sun_alt, sun_az)
        print(f"   Sun at midpoint: alt={np.degrees(sun_alt):.1f}°, az={np.degrees(sun_az):.1f}°")
    else:
        print(f"   No fringe-stopping (night data)")

    # Time-average
    vis_avg = np.mean(vis_matrix.vis, axis=0)
    print(f"   Time-averaged: {vis_avg.shape}")

    # Apply calibration
    n_chan = vis_avg.shape[0]
    g_conj = np.conj(cal_gains)
    vis_cal = vis_avg.copy()
    for f in range(n_chan):
        vis_cal[f] = g_conj[:, f:f+1] * vis_avg[f] * cal_gains[:, f:f+1].T

    # RFI mask
    rfi_mask_obj = RFIMask(bad_ranges_mhz=rfi_ranges)
    rfi_mask = rfi_mask_obj(vis_matrix.freq_mhz)
    freqs_hz = vis_matrix.freq_mhz.astype(np.float64) * 1e6

    return vis_cal, freqs_hz, positions, rfi_mask, mid_time, fs_dir


def load_cal_gains(npz_path, ant_ids, n_chan):
    """Load gains matched to ant_ids ordering."""
    d = np.load(npz_path, allow_pickle=True)
    gains = d["gains"]
    cal_ant_ids = d["ant_ids"]
    cal_freqs = d["freqs_hz"]
    if cal_freqs[1] < cal_freqs[0]:
        gains = gains[:, ::-1]
    cal_id_to_idx = {int(aid): i for i, aid in enumerate(cal_ant_ids)}
    n_ant = len(ant_ids)
    g_out = np.ones((n_ant, n_chan), dtype=np.complex64)
    for i, aid in enumerate(ant_ids):
        if int(aid) in cal_id_to_idx:
            g_out[i] = gains[cal_id_to_idx[int(aid)]]
    return g_out


def make_beam_image(vis_cal, freqs_hz, positions, rfi_mask,
                    center_alt, center_az, fov_deg, npix,
                    l_fs=None, m_fs=None, n_fs=None):
    """Sweep alt-az grid and compute cross-only beamformed power."""
    half_fov = np.radians(fov_deg / 2.0)
    alt_offsets = np.linspace(-half_fov, half_fov, npix)
    az_offsets = np.linspace(-half_fov, half_fov, npix)

    good = rfi_mask
    vis_good = vis_cal[good]
    freqs_good = freqs_hz[good]
    n_good = int(np.sum(good))
    print(f"   Using {n_good} good channels for imaging")

    image = np.zeros((npix, npix), dtype=np.float64)
    total = npix * npix
    done = 0

    for ia, da in enumerate(alt_offsets):
        alt = center_alt + da
        if alt <= 0 or alt >= np.pi / 2:
            done += npix
            continue
        for iz, dz in enumerate(az_offsets):
            az = center_az + dz
            l, m, n = altaz_to_enu(alt, az)
            power = visibility_beamform_cross(
                vis_good, freqs_good, positions, l, m, n,
                l_fs=l_fs, m_fs=m_fs, n_fs=n_fs,
            )
            image[ia, iz] = np.mean(power)
            done += 1
        # Progress
        if (ia + 1) % 10 == 0:
            print(f"   Row {ia+1}/{npix} ({100*done/total:.0f}%)")

    alt_axis = np.degrees(alt_offsets + center_alt)
    az_axis = np.degrees(az_offsets + center_az)
    return image, alt_axis, az_axis


def main():
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "verify_vis_beamform",
    )
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("BEAM IMAGE COMPARISON: CAL vs NO-CAL vs NIGHT")
    print(f"FoV = {IMAGE_FOV}°, {IMAGE_NPIX}x{IMAGE_NPIX} pixels")
    print("=" * 70)

    # Load antenna mapping
    mapping = AntennaMapping.load(LAYOUT)
    all_ant_ids = np.array(mapping.active_antennas(), dtype=int)
    keep_mask = ~np.isin(all_ant_ids, EXCLUDE_ANTS)
    ant_ids = all_ant_ids[keep_mask]
    n_ant = len(ant_ids)
    n_chan = 3072
    print(f"\n{n_ant} antennas (excluding {EXCLUDE_ANTS})")

    # Load cal gains
    cal_gains = load_cal_gains(CAL_NPZ, ant_ids, n_chan)
    identity_gains = np.ones((n_ant, n_chan), dtype=np.complex64)
    print(f"Cal gains loaded: {cal_gains.shape}")

    # ── 1. Transit + calibration ──
    print(f"\n{'='*70}")
    print(f"1. TRANSIT + CAL: {OBS_TRANSIT}")
    print(f"{'='*70}")
    vis_cal_t, freqs_hz_t, pos_t, rfi_t, mid_t, fs_dir_t = load_and_prepare(
        OBS_TRANSIT, mapping, ant_ids, EXCLUDE_ANTS, "sun",
        cal_gains, RFI_RANGES, NFILES, fringe_stop_source=True,
    )

    # Sun position for image center
    t_transit = Time(mid_t, format="unix")
    frame_t = AltAz(obstime=t_transit, location=OVRO)
    sun_t = get_sun(t_transit).transform_to(frame_t)
    sun_alt_t, sun_az_t = sun_t.alt.rad, sun_t.az.rad

    print(f"\n   Making beam image (calibrated)...")
    image_cal, alt_ax, az_ax = make_beam_image(
        vis_cal_t, freqs_hz_t, pos_t, rfi_t,
        sun_alt_t, sun_az_t, IMAGE_FOV, IMAGE_NPIX,
        l_fs=fs_dir_t[0], m_fs=fs_dir_t[1], n_fs=fs_dir_t[2],
    )
    print(f"   Cal image: min={np.min(image_cal):.2e}, max={np.max(image_cal):.2e}")

    # ── 2. Transit + NO calibration (null test) ──
    print(f"\n{'='*70}")
    print(f"2. TRANSIT + NO CAL (null test): {OBS_TRANSIT}")
    print(f"{'='*70}")
    vis_nocal_t, _, _, _, _, _ = load_and_prepare(
        OBS_TRANSIT, mapping, ant_ids, EXCLUDE_ANTS, "sun",
        identity_gains, RFI_RANGES, NFILES, fringe_stop_source=True,
    )

    print(f"\n   Making beam image (no calibration)...")
    image_nocal, _, _ = make_beam_image(
        vis_nocal_t, freqs_hz_t, pos_t, rfi_t,
        sun_alt_t, sun_az_t, IMAGE_FOV, IMAGE_NPIX,
        l_fs=fs_dir_t[0], m_fs=fs_dir_t[1], n_fs=fs_dir_t[2],
    )
    print(f"   No-cal image: min={np.min(image_nocal):.2e}, max={np.max(image_nocal):.2e}")

    # ── 3. Night + calibration ──
    print(f"\n{'='*70}")
    print(f"3. NIGHT + CAL: {OBS_NIGHT}")
    print(f"{'='*70}")
    vis_cal_n, freqs_hz_n, pos_n, rfi_n, mid_n, _ = load_and_prepare(
        OBS_NIGHT, mapping, ant_ids, EXCLUDE_ANTS, "sun",
        cal_gains, RFI_RANGES, NFILES, fringe_stop_source=False,
    )

    t_night = Time(mid_n, format="unix")
    frame_n = AltAz(obstime=t_night, location=OVRO)
    sun_n = get_sun(t_night).transform_to(frame_n)
    sun_alt_n = sun_n.alt.deg
    print(f"   Sun altitude at night: {sun_alt_n:.1f}°")

    print(f"\n   Making beam image (night, calibrated)...")
    image_night, _, _ = make_beam_image(
        vis_cal_n, freqs_hz_n, pos_n, rfi_n,
        sun_alt_t, sun_az_t, IMAGE_FOV, IMAGE_NPIX,
        l_fs=None, m_fs=None, n_fs=None,
    )
    print(f"   Night image: min={np.min(image_night):.2e}, max={np.max(image_night):.2e}")

    # ── 4. Plot ──
    print(f"\n{'='*70}")
    print("4. PLOTTING")
    print(f"{'='*70}")

    # Common color scale from transit+cal image
    vmax_use = np.max(image_cal)
    vmin_use = min(np.min(image_cal), np.min(image_nocal), np.min(image_night), 0)

    extent = [az_ax[0], az_ax[-1], alt_ax[0], alt_ax[-1]]
    pdf_path = os.path.join(out_dir, "beam_image_comparison.pdf")

    with PdfPages(pdf_path) as pdf:
        # Page 1: 3-panel comparison (same color scale)
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))

        titles = [
            f"Transit + Cal\n{OBS_TRANSIT}",
            f"Transit + NO Cal (null)\n{OBS_TRANSIT}",
            f"Night + Cal\n{OBS_NIGHT}, Sun alt={sun_alt_n:.1f}°",
        ]
        images = [image_cal, image_nocal, image_night]

        for ax, img, title in zip(axes, images, titles):
            im = ax.imshow(img, origin="lower", extent=extent,
                           aspect="auto", cmap="inferno",
                           vmin=vmin_use, vmax=vmax_use)
            ax.plot(np.degrees(sun_az_t), np.degrees(sun_alt_t), "w+",
                    markersize=12, markeredgewidth=2)
            ax.set_xlabel("Azimuth (°)")
            ax.set_title(title, fontsize=11)

        axes[0].set_ylabel("Altitude (°)")
        fig.colorbar(im, ax=axes.tolist(), label="Beamformed Power (cross-only)",
                     shrink=0.8, pad=0.02)
        fig.suptitle(f"Beam Image Comparison — {IMAGE_FOV}° FoV, cross-correlations only",
                     fontsize=14, y=1.02)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: Transit + Cal full page
        fig, ax = plt.subplots(figsize=(10, 9))
        im = ax.imshow(image_cal, origin="lower", extent=extent,
                        aspect="auto", cmap="inferno")
        ax.plot(np.degrees(sun_az_t), np.degrees(sun_alt_t), "w+",
                markersize=20, markeredgewidth=3, label="Sun")
        ax.set_xlabel("Azimuth (°)", fontsize=12)
        ax.set_ylabel("Altitude (°)", fontsize=12)
        ax.set_title(f"Transit + Calibration ({IMAGE_FOV}° FoV)\n"
                      f"cross-corr only, peak={np.max(image_cal):.2e}",
                      fontsize=13)
        ax.legend(loc="upper right", fontsize=11)
        plt.colorbar(im, ax=ax, label="Beamformed Power", shrink=0.85)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: Transit + No Cal (null test, own scale)
        fig, ax = plt.subplots(figsize=(10, 9))
        im = ax.imshow(image_nocal, origin="lower", extent=extent,
                        aspect="auto", cmap="inferno")
        ax.plot(np.degrees(sun_az_t), np.degrees(sun_alt_t), "w+",
                markersize=20, markeredgewidth=3, label="Sun")
        ax.set_xlabel("Azimuth (°)", fontsize=12)
        ax.set_ylabel("Altitude (°)", fontsize=12)
        ax.set_title(f"Transit + NO Calibration — null test ({IMAGE_FOV}° FoV)\n"
                      f"cross-corr only, peak={np.max(image_nocal):.2e}",
                      fontsize=13)
        ax.legend(loc="upper right", fontsize=11)
        plt.colorbar(im, ax=ax, label="Beamformed Power", shrink=0.85)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: Night + Cal (own scale)
        fig, ax = plt.subplots(figsize=(10, 9))
        im = ax.imshow(image_night, origin="lower", extent=extent,
                        aspect="auto", cmap="inferno")
        ax.plot(np.degrees(sun_az_t), np.degrees(sun_alt_t), "w+",
                markersize=20, markeredgewidth=3, label="Sun (transit pos)")
        ax.set_xlabel("Azimuth (°)", fontsize=12)
        ax.set_ylabel("Altitude (°)", fontsize=12)
        ax.set_title(f"Night + Calibration ({IMAGE_FOV}° FoV, own scale)\n"
                      f"Sun alt={sun_alt_n:.1f}°, cross-corr only, "
                      f"peak={np.max(image_night):.2e}",
                      fontsize=13)
        ax.legend(loc="upper right", fontsize=11)
        plt.colorbar(im, ax=ax, label="Beamformed Power", shrink=0.85)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"   Saved: {pdf_path}")

    # Save data
    npz_path = os.path.join(out_dir, "beam_image_comparison.npz")
    np.savez_compressed(
        npz_path,
        image_cal=image_cal,
        image_nocal=image_nocal,
        image_night=image_night,
        alt_axis=alt_ax,
        az_axis=az_ax,
        sun_alt_deg=np.degrees(sun_alt_t),
        sun_az_deg=np.degrees(sun_az_t),
        sun_alt_night_deg=sun_alt_n,
        obs_transit=OBS_TRANSIT,
        obs_night=OBS_NIGHT,
        fov_deg=IMAGE_FOV,
        npix=IMAGE_NPIX,
    )
    print(f"   Saved: {npz_path}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Transit+Cal peak:    {np.max(image_cal):.3e}")
    print(f"Transit+NoCal peak:  {np.max(image_nocal):.3e}")
    print(f"Night+Cal peak:      {np.max(image_night):.3e}")
    print(f"Cal/NoCal ratio:     {np.max(image_cal)/max(np.max(np.abs(image_nocal)), 1e-30):.1f}")
    print(f"Cal/Night ratio:     {np.max(image_cal)/max(np.max(np.abs(image_night)), 1e-30):.1f}")


if __name__ == "__main__":
    main()
