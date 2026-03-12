#!/usr/bin/env python3
"""
Track 0: Quick Sanity Checks on int8 HDF5 and cal NPZ.

- Load int8 HDF5 and cal NPZ
- Verify frequency axis alignment, antenna mapping
- Reconstruct complex64 from int8 for beam 0, compare with w_cal * w_geo
- Print per-antenna weight stats — detect anomalous phase or amplitude
- Check Ant 4 specifically
"""

import sys
import numpy as np

# Paths
INT8_HDF5 = "/home/casm/software/vishnu/bf_weights_generator/weights_512beam/geo_fallback/int8_weights_512beam_feb19_svd.h5"
CAL_NPZ_FEB19 = "/home/casm/software/vishnu/casm-voltage-analysis/correlator_analysis/results/multi_day_svd/svd_weights_feb19.npz"
CAL_NPZ_MAR10 = "/home/casm/software/dev/casm_calibrator/results/svd_weights_mar10.npz"
LAYOUT_CSV = "/home/casm/software/dev/bf_weights_generator/casm_antenna_layout_current.csv"


def load_int8_hdf5(path):
    import h5py
    with h5py.File(path, "r") as f:
        weights_int8 = f["weights_int8"][:]
        freqs_hz = f["frequencies_hz"][:]
        scale_factor = f.attrs["scale_factor"]
        n_beams = f.attrs["n_beams"]

        ac = f["array_config"]
        positions_enu = ac["positions_enu"][:]
        active_mask = ac["active_mask"][:]
        snap_to_ant64 = ac["snap_to_ant64"][:]
        ant64_to_snap = ac["ant64_to_snap"][:]

        # Read pointings if available
        ptg = f["pointings"]
        alt_deg = ptg["alt_deg"][:] if "alt_deg" in ptg else None
        az_deg = ptg["az_deg"][:] if "az_deg" in ptg else None

    return dict(
        weights_int8=weights_int8,
        freqs_hz=freqs_hz,
        scale_factor=scale_factor,
        n_beams=n_beams,
        positions_enu=positions_enu,
        active_mask=active_mask,
        snap_to_ant64=snap_to_ant64,
        ant64_to_snap=ant64_to_snap,
        alt_deg=alt_deg,
        az_deg=az_deg,
    )


def int8_to_complex(weights_int8, scale_factor):
    """Reconstruct complex64 from int8: shape (2, n_chan, 2, n_beams, 64)
    -> (n_beams, 64, n_chan) complex64. Take pol 0."""
    real = weights_int8[0].astype(np.float32) / scale_factor
    imag = weights_int8[1].astype(np.float32) / scale_factor
    # Shape: (n_chan, 2_pol, n_beams, 64) -> take pol 0
    cx = real[:, 0, :, :] + 1j * imag[:, 0, :, :]
    # (n_chan, n_beams, 64) -> transpose to (n_beams, 64, n_chan)
    return cx.transpose(1, 2, 0)


def main():
    print("=" * 70)
    print("TRACK 0: QUICK SANITY CHECKS")
    print("=" * 70)

    # ── Load int8 HDF5 ──
    print(f"\n1. Loading int8 HDF5: {INT8_HDF5}")
    h5 = load_int8_hdf5(INT8_HDF5)
    print(f"   weights_int8 shape: {h5['weights_int8'].shape}")
    print(f"   scale_factor: {h5['scale_factor']}")
    print(f"   n_beams: {h5['n_beams']}")
    print(f"   freqs_hz: {h5['freqs_hz'][0]/1e6:.4f} – {h5['freqs_hz'][-1]/1e6:.4f} MHz")
    freq_order = "descending" if h5["freqs_hz"][0] > h5["freqs_hz"][-1] else "ascending"
    print(f"   freq order: {freq_order}")
    n_active = int(np.sum(h5["active_mask"]))
    active_indices = np.where(h5["active_mask"])[0]
    print(f"   active antennas: {n_active} at ant64 slots {list(active_indices)}")
    print(f"   ant_ids (1-indexed): {list(active_indices + 1)}")

    # ── Verify SNAP mapping ──
    print(f"\n2. SNAP mapping check")
    for ant64_idx in active_indices:
        snap_idx = h5["ant64_to_snap"][ant64_idx]
        reverse = h5["snap_to_ant64"][snap_idx] if snap_idx >= 0 else -1
        ok = "OK" if reverse == ant64_idx else "MISMATCH"
        print(f"   ant64={ant64_idx:2d} (Ant {ant64_idx+1:2d}) -> snap_input={snap_idx:2d} -> ant64={reverse:2d}  [{ok}]")

    # ── Reconstruct complex weights ──
    print(f"\n3. Reconstructing complex weights from int8")
    w_complex = int8_to_complex(h5["weights_int8"], h5["scale_factor"])
    print(f"   complex weights shape: {w_complex.shape}  (n_beams, 64, n_chan)")

    # ── Load cal NPZ (Feb19) ──
    print(f"\n4. Loading cal NPZ: {CAL_NPZ_FEB19}")
    cal = np.load(CAL_NPZ_FEB19, allow_pickle=True)
    cal_weights = cal["weights"]  # (16, 3072)
    cal_gains = cal["gains"]
    cal_flags = cal["flags"]
    cal_freqs_hz = cal["freqs_hz"]
    cal_ant_ids = cal["ant_ids"]
    cal_ref = int(cal["ref_ant_id"])
    print(f"   weights shape: {cal_weights.shape}")
    print(f"   ant_ids: {cal_ant_ids}")
    print(f"   ref_ant_id: {cal_ref}")
    print(f"   flags: {cal_flags.sum()}/{len(cal_flags)} good")
    print(f"   freqs: {cal_freqs_hz[0]/1e6:.4f} – {cal_freqs_hz[-1]/1e6:.4f} MHz")
    cal_forder = "ascending" if cal_freqs_hz[1] > cal_freqs_hz[0] else "descending"
    print(f"   freq order: {cal_forder}")

    # ── Channel count check ──
    n_chan_h5 = len(h5["freqs_hz"])
    n_chan_cal = len(cal_freqs_hz)
    print(f"\n5. Channel count: HDF5={n_chan_h5}, Cal={n_chan_cal}", end="")
    if n_chan_h5 == n_chan_cal:
        print("  [OK]")
    else:
        print(f"  [MISMATCH!]")
        sys.exit(1)

    # ── Frequency alignment ──
    # HDF5 is descending, cal is ascending. Compare after flipping.
    h5_freqs_asc = h5["freqs_hz"][::-1] if freq_order == "descending" else h5["freqs_hz"]
    cal_freqs_asc = cal_freqs_hz if cal_forder == "ascending" else cal_freqs_hz[::-1]
    freq_diff = np.abs(h5_freqs_asc - cal_freqs_asc)
    print(f"   Max freq diff (ascending): {freq_diff.max():.2f} Hz")
    if freq_diff.max() > 1.0:
        print("   WARNING: Frequency axis mismatch > 1 Hz!")
    else:
        print("   Frequency axes match  [OK]")

    # ── Per-antenna weight analysis (beam 0, zenith) ──
    print(f"\n6. Per-antenna weight analysis (beam 0)")
    if h5["alt_deg"] is not None:
        print(f"   Beam 0: alt={h5['alt_deg'][0]:.1f}°, az={h5['az_deg'][0]:.1f}°")

    # Get beam 0 weights in SNAP order: w_complex[0, :, :] = (64, n_chan)
    w_beam0_snap = w_complex[0]  # (64, n_chan) in SNAP order, descending freq

    # Convert to ant64 order
    w_beam0_ant64 = np.zeros((64, n_chan_h5), dtype=np.complex64)
    for snap_idx in range(64):
        ant64_idx = h5["snap_to_ant64"][snap_idx]
        if ant64_idx >= 0:
            w_beam0_ant64[ant64_idx] = w_beam0_snap[snap_idx]

    # Analyze active antennas
    print(f"\n   {'Ant':>4s}  {'|w| mean':>10s}  {'|w| std':>9s}  {'|w| min':>9s}  "
          f"{'|w| max':>9s}  {'phase std':>10s}  {'non-zero':>10s}")
    print(f"   {'-'*70}")

    for ant64_idx in active_indices:
        ant_id = ant64_idx + 1
        w = w_beam0_ant64[ant64_idx]
        amp = np.abs(w)
        nonzero = np.sum(amp > 0.01)
        if nonzero > 10:
            phase = np.angle(w[amp > 0.01])
            phase_std = np.std(np.unwrap(phase))
        else:
            phase_std = np.nan
        flag = " <-- ANT 4" if ant_id == 4 else ""
        print(f"   {ant_id:4d}  {amp.mean():10.4f}  {amp.std():9.4f}  {amp.min():9.4f}  "
              f"{amp.max():9.4f}  {np.degrees(phase_std):10.2f}°  "
              f"{nonzero:5d}/{n_chan_h5}{flag}")

    # ── Ant 4 deep-dive ──
    print(f"\n7. Ant 4 deep-dive")
    ant4_idx = 3  # 0-indexed
    w_ant4 = w_beam0_ant64[ant4_idx]
    amp4 = np.abs(w_ant4)
    print(f"   Ant 4 in int8 weights:")
    print(f"     Non-zero channels: {np.sum(amp4 > 0.01)}/{n_chan_h5}")
    print(f"     Mean amplitude: {amp4.mean():.4f}")

    # Check cal weights for Ant 4
    ant4_cal_idx = np.where(cal_ant_ids == 4)[0]
    if len(ant4_cal_idx) > 0:
        k = ant4_cal_idx[0]
        cal_w4 = cal_weights[k]
        cal_g4 = cal_gains[k]
        amp_cal4 = np.abs(cal_w4)
        print(f"   Ant 4 in cal NPZ:")
        print(f"     Non-zero channels: {np.sum(amp_cal4 > 0.001)}/{n_chan_cal}")
        print(f"     Mean amplitude (non-zero): {amp_cal4[amp_cal4>0.001].mean():.4f}")

        # Phase analysis
        good = cal_flags & (amp_cal4 > 0.001)
        if good.sum() > 10:
            phase4 = np.angle(cal_g4[good])
            phase4_unwrap = np.unwrap(phase4)
            # Fit delay
            freqs_good = cal_freqs_asc[good]
            coeffs = np.polyfit(freqs_good, phase4_unwrap, 1)
            delay_ns = coeffs[0] / (2 * np.pi) * 1e9
            residual = phase4_unwrap - np.polyval(coeffs, freqs_good)
            rms_deg = np.degrees(np.std(residual))
            print(f"     Delay: {delay_ns:.2f} ns")
            print(f"     Phase residual RMS (after delay removal): {rms_deg:.1f}°")
            print(f"     Phase range: {np.degrees(phase4_unwrap[0]):.1f}° to {np.degrees(phase4_unwrap[-1]):.1f}°")

    # ── Compare with Mar10 cal NPZ if available ──
    print(f"\n8. Cross-check with Mar10 cal weights")
    try:
        mar10 = np.load(CAL_NPZ_MAR10, allow_pickle=True)
        m_weights = mar10["weights"]
        m_gains = mar10["gains"]
        m_flags = mar10["flags"]
        m_freqs = mar10["freqs_hz"]
        m_ant_ids = mar10["ant_ids"]
        print(f"   Mar10 cal: {m_weights.shape}, {m_flags.sum()}/{len(m_flags)} good")

        # Compare Ant 4 between Feb19 and Mar10
        feb_idx = np.where(cal_ant_ids == 4)[0][0]
        mar_idx = np.where(m_ant_ids == 4)[0][0]

        # Use common good channels
        m_freqs_asc = m_freqs if m_freqs[1] > m_freqs[0] else m_freqs[::-1]
        common_good = cal_flags & m_flags
        if common_good.sum() > 10:
            feb_phase = np.unwrap(np.angle(cal_gains[feb_idx, common_good]))
            mar_phase = np.unwrap(np.angle(m_gains[mar_idx, common_good]))
            phase_diff = mar_phase - feb_phase
            # Remove mean offset
            phase_diff -= np.mean(phase_diff)
            print(f"   Ant 4 phase diff (Mar10 - Feb19) over {common_good.sum()} common channels:")
            print(f"     RMS: {np.degrees(np.std(phase_diff)):.1f}°")
            print(f"     Max: {np.degrees(np.max(np.abs(phase_diff))):.1f}°")

            # Fit delay diff
            freqs_common = cal_freqs_asc[common_good]
            coeffs_diff = np.polyfit(freqs_common, phase_diff, 1)
            delay_diff_ns = coeffs_diff[0] / (2 * np.pi) * 1e9
            print(f"     Delay shift: {delay_diff_ns:.2f} ns")

    except FileNotFoundError:
        print(f"   Mar10 cal NPZ not found, skipping.")

    # ── Independently compute w_cal * w_geo for beam 0 and compare ──
    print(f"\n9. Independent weight reconstruction (beam 0)")
    try:
        sys.path.insert(0, "/home/casm/software/dev/bf_weights_generator")
        from bf_weights_generator.snap_weights import Array64Config, SnapWeightsGenerator, load_calibration_weights
        from bf_weights_generator.weights import StationaryPointing, GeometricBeamformer, BeamMode
        from bf_weights_generator.config import FrequencyConfig

        array_config = Array64Config.from_csv(LAYOUT_CSV)
        print(f"   Array64Config: {array_config}")

        # Load cal weights
        cal_w = load_calibration_weights(CAL_NPZ_FEB19)
        print(f"   CalibrationWeights loaded: {cal_w.weights.shape}")

        # Compute geo weights for beam 0 (zenith)
        zenith = StationaryPointing(alt_deg=90.0, az_deg=0.0, name="zenith")
        freq_config = FrequencyConfig()
        beamformer = GeometricBeamformer(
            array_config=array_config.to_array_config(),
            freq_config=freq_config,
        )
        stationary = beamformer.compute_stationary_weights(
            pointings=[zenith],
            mode=BeamMode.COHERENT,
        )
        geo_weights = stationary.weights  # (1, n_active, n_chan) descending freq

        # Combine with cal
        gen = SnapWeightsGenerator(array_config)
        combined = gen._apply_calibration_weights(
            geo_weights, cal_w, geo_fallback=True,
        )

        # Expand to 64 slots in ant64 order
        w_recon_ant64 = np.zeros((64, freq_config.n_chan), dtype=np.complex64)
        for i, ant64_idx in enumerate(array_config.active_indices):
            w_recon_ant64[ant64_idx] = combined[0, i, :]
        # w_recon_ant64 is in descending freq, ant64 order

        # Compare with HDF5 beam 0 (also descending, now in ant64 order)
        print(f"\n   Comparing reconstructed vs HDF5 beam 0 weights:")
        for ant64_idx in active_indices:
            ant_id = ant64_idx + 1
            w_h5 = w_beam0_ant64[ant64_idx]     # descending
            w_rc = w_recon_ant64[ant64_idx]      # descending

            # Mask out zero channels
            mask = (np.abs(w_h5) > 0.01) & (np.abs(w_rc) > 0.001)
            if mask.sum() < 10:
                print(f"   Ant {ant_id:2d}: too few non-zero channels to compare ({mask.sum()})")
                continue

            # Phase difference
            phase_diff = np.angle(w_h5[mask] * np.conj(w_rc[mask]))
            amp_ratio = np.abs(w_h5[mask]) / np.abs(w_rc[mask])
            flag = " <-- ANT 4" if ant_id == 4 else ""
            print(f"   Ant {ant_id:2d}: phase_diff={np.degrees(np.mean(phase_diff)):+6.1f}° "
                  f"±{np.degrees(np.std(phase_diff)):5.1f}°  "
                  f"amp_ratio={np.mean(amp_ratio):.3f}±{np.std(amp_ratio):.3f}  "
                  f"({mask.sum()} chan){flag}")

    except Exception as e:
        print(f"   Independent reconstruction failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'=' * 70}")
    print("TRACK 0 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
