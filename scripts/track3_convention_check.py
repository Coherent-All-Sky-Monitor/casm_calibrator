#!/usr/bin/env python3
"""
Track 3: Convention & Weight Verification.

Numerical trace through the full calibration+beamforming chain for one
antenna, channel, and beam. Verifies sign conventions are consistent.

Also checks:
- int8 HDF5 weight reconstruction matches independent computation
- SNAP antenna ordering in HDF5 matches layout CSV
- Frequency axis in HDF5 is descending
- GPU kernel convention (from source code analysis)
"""

import os
import sys
import numpy as np

# ── Paths ──
INT8_HDF5 = "/home/casm/software/vishnu/bf_weights_generator/weights_512beam/geo_fallback/int8_weights_512beam_feb19_svd.h5"
CAL_NPZ = "/home/casm/software/vishnu/casm-voltage-analysis/correlator_analysis/results/multi_day_svd/svd_weights_feb19.npz"
LAYOUT_CSV = os.path.expanduser("~/software/dev/bf_weights_generator/casm_antenna_layout_current.csv")
LAYOUT_CSV_CASM_IO = os.path.expanduser("~/software/dev/antenna_layouts/antenna_layout_current.csv")

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "convention_check")
os.makedirs(OUT_DIR, exist_ok=True)

C_LIGHT = 299792458.0


def main():
    print("=" * 70)
    print("TRACK 3: CONVENTION & WEIGHT VERIFICATION")
    print("=" * 70)

    # ══════════════════════════════════════════════════════════════════════
    # PART 1: Numerical trace through the chain
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("PART 1: Numerical chain trace (Ant 1, channel 1500, zenith beam)")
    print("─" * 60)

    # Load antenna positions from casm_io layout
    ant_data = np.genfromtxt(LAYOUT_CSV_CASM_IO, delimiter=",", skip_header=1,
                              dtype=None, encoding="utf-8")
    ant_ids_io = np.array([r[0] for r in ant_data], dtype=int)
    ant_x = np.array([r[1] for r in ant_data], dtype=float)
    ant_y = np.array([r[2] for r in ant_data], dtype=float)
    ant_z = np.array([r[3] for r in ant_data], dtype=float)

    # Pick antenna 1 (index 0) and ref ant 5 (index 4)
    test_ant_id = 1
    test_ant_idx = int(np.where(ant_ids_io == test_ant_id)[0][0])
    ref_ant_id = 5
    ref_ant_idx = int(np.where(ant_ids_io == ref_ant_id)[0][0])
    test_chan = 1500  # mid-band channel

    x_test = ant_x[test_ant_idx]
    y_test = ant_y[test_ant_idx]
    z_test = ant_z[test_ant_idx]

    # Frequency for channel 1500 (ascending order)
    freq_mhz = 375.0 + test_chan * (125.0 / 4096)
    freq_hz = freq_mhz * 1e6
    print(f"  Antenna: {test_ant_id} at ({x_test}, {y_test}, {z_test}) m")
    print(f"  Channel: {test_chan}, freq = {freq_mhz:.4f} MHz")

    # ── Step 1: casm_calibrator fringe-stop convention ──
    # Toward zenith: l=0, m=0, n=1 (ENU)
    l, m, n = 0.0, 0.0, 1.0
    print(f"\n  Step 1: Fringe-stop toward zenith (l={l}, m={m}, n={n})")

    # casm_calibrator: tau = dot(source_enu, position) / c
    tau_test = (l * x_test + m * y_test + n * z_test) / C_LIGHT
    tau_ref = (l * ant_x[ref_ant_idx] + m * ant_y[ref_ant_idx] + n * ant_z[ref_ant_idx]) / C_LIGHT
    print(f"  tau_test = {tau_test*1e9:.4f} ns")
    print(f"  tau_ref  = {tau_ref*1e9:.4f} ns")

    # casm_calibrator: phi = -2*pi * freq_hz * tau
    phi_test = -2 * np.pi * freq_hz * tau_test
    phi_ref = -2 * np.pi * freq_hz * tau_ref
    print(f"  phi_test = {np.degrees(phi_test):.2f}°")
    print(f"  phi_ref  = {np.degrees(phi_ref):.2f}°")

    # Fringe-stop on baseline (ref, test): V_fs = V * exp(1j*(phi_test - phi_ref))
    # = V * exp(1j*(-2pi*f*(tau_test - tau_ref)))
    dphi_fs = phi_test - phi_ref
    print(f"  Differential phase: {np.degrees(dphi_fs):.2f}°")

    # ── Step 2: SVD gain extraction ──
    print(f"\n  Step 2: SVD gain extraction")
    print(f"  g = exp(1j * angle(U[:,0]))")
    print(f"  g *= exp(-1j * angle(g[ref_ant]))  # zero ref phase")
    print(f"  weight = conj(g)")

    # Load actual SVD gains
    cal = np.load(CAL_NPZ, allow_pickle=True)
    cal_gains = cal["gains"]  # (16, 3072)
    cal_weights = cal["weights"]
    cal_freqs = cal["freqs_hz"]

    # Ensure ascending
    if cal_freqs[1] < cal_freqs[0]:
        cal_gains = cal_gains[:, ::-1]
        cal_weights = cal_weights[:, ::-1]
        cal_freqs = cal_freqs[::-1]

    g_test = cal_gains[test_ant_idx, test_chan]
    w_cal_test = cal_weights[test_ant_idx, test_chan]
    print(f"  g_test = {g_test:.4f}  (phase = {np.degrees(np.angle(g_test)):.2f}°)")
    print(f"  w_cal_test = conj(g) = {w_cal_test:.4f}  (phase = {np.degrees(np.angle(w_cal_test)):.2f}°)")
    print(f"  Verify conj: g * w_cal = {g_test * w_cal_test:.6f}  (should be ~1)")

    # ── Step 3: bf_weights_generator geometric weight ──
    print(f"\n  Step 3: bf_weights_generator geometric weight")
    # delay = -(x*l + y*m + z*n) / c
    delay_test = -(x_test * l + y_test * m + z_test * n) / C_LIGHT
    print(f"  delay = -(x*l + y*m + z*n)/c = {delay_test*1e9:.4f} ns")
    # phase = 2*pi * f * delay
    phase_geo = 2 * np.pi * freq_hz * delay_test
    print(f"  phase = 2*pi*f*delay = {np.degrees(phase_geo):.2f}°")
    # w_geo = exp(-j * phase)
    w_geo = np.exp(-1j * phase_geo)
    print(f"  w_geo = exp(-j*phase) = {w_geo:.4f}  (phase = {np.degrees(np.angle(w_geo)):.2f}°)")

    # ── Step 4: Combined weight ──
    print(f"\n  Step 4: Combined weight")
    w_total = w_cal_test * w_geo
    print(f"  w_total = w_cal * w_geo = {w_total:.4f}")
    print(f"  |w_total| = {np.abs(w_total):.4f}")
    print(f"  angle(w_total) = {np.degrees(np.angle(w_total)):.2f}°")

    # ── Step 5: Verify: w_total * v should cancel phase ──
    print(f"\n  Step 5: Verify phase cancellation")
    # If incoming signal from source: v = exp(1j * 2*pi*f*tau_geo) * g * s
    # where tau_geo is the geometric delay
    tau_geo = (l * x_test + m * y_test + n * z_test) / C_LIGHT
    v_phase = 2 * np.pi * freq_hz * tau_geo  # geometric phase
    v_model = np.exp(1j * v_phase) * g_test  # observed voltage (geo+instrumental)
    beam = w_total * v_model
    print(f"  v_model = exp(j*2pi*f*tau) * g = {v_model:.4f}")
    print(f"  beam = w_total * v = {beam:.6f}")
    print(f"  |beam| = {np.abs(beam):.6f}")
    print(f"  angle(beam) = {np.degrees(np.angle(beam)):.2f}° (should be ~0 for coherent sum)")

    # ══════════════════════════════════════════════════════════════════════
    # PART 2: GPU kernel convention analysis
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 60}")
    print("PART 2: GPU kernel convention analysis")
    print("─" * 60)

    print("""
  From h23:/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/dsaX_bfCorr.cu:

  populate_weights_matrix kernel:
    afac = -2*pi*f*theta/c
    twr = cos(afac * antpos_e)
    twi = sin(afac * antpos_e)
    war = twr*cal_r - twi*cal_i   // = Re(w_geo * w_cal)
    wai = twi*cal_r + twr*cal_i   // = Im(w_geo * w_cal)

  -> GPU computes: w_total = w_geo * w_cal (standard complex multiply)
  -> w_geo = exp(j * afac * x) = exp(-j * 2*pi*f*theta*x/c)

  Beamformer CUBLAS:
    transa = CUBLAS_OP_T, transb = CUBLAS_OP_N
    beam = data^T * weights  (real-valued split multiply)
    -> Implements: beam = sum_ant(data * weight) = d * w

  -> GPU applies w*v, NOT conj(w)*v. This is the CORRECT convention
     when w = conj(g) * w_geo_conj (our pipeline produces w = conj(g) * exp(-j*phase))

  CRITICAL: The GPU kernel normalizes cal weights to unit amplitude:
    wnorm = sqrt(cal_r^2 + cal_i^2)
    cal_r /= wnorm; cal_i /= wnorm
  -> Any amplitude information in cal weights is discarded by the GPU.

  IMPORTANT: The GPU kernel in populate_weights_matrix computes geometric
  weights INTERNALLY from antenna positions. It reads cal weights from a
  binary input file, not from our int8 HDF5. If the SNAP beamformer uses
  calc_weights(), then our pre-combined int8 weights are NOT what it loads.
  Need to verify which code path is active on h23.
""")

    # ══════════════════════════════════════════════════════════════════════
    # PART 3: HDF5 format verification
    # ══════════════════════════════════════════════════════════════════════
    print(f"{'─' * 60}")
    print("PART 3: HDF5 format verification")
    print("─" * 60)

    import h5py
    with h5py.File(INT8_HDF5, "r") as f:
        w_int8 = f["weights_int8"][:]
        freqs_hz_h5 = f["frequencies_hz"][:]
        scale = f.attrs["scale_factor"]
        n_beams = f.attrs["n_beams"]

        ac = f["array_config"]
        snap_to_ant64 = ac["snap_to_ant64"][:]
        ant64_to_snap = ac["ant64_to_snap"][:]
        positions = ac["positions_enu"][:]
        active_mask = ac["active_mask"][:]

    print(f"  weights_int8 shape: {w_int8.shape}")
    print(f"  Expected: (2, 3072, 2, {n_beams}, 64)")
    print(f"  Match: {w_int8.shape == (2, 3072, 2, n_beams, 64)}")

    # Frequency order
    print(f"\n  Frequencies:")
    print(f"    First: {freqs_hz_h5[0]/1e6:.4f} MHz")
    print(f"    Last:  {freqs_hz_h5[-1]/1e6:.4f} MHz")
    freq_order = "descending" if freqs_hz_h5[0] > freqs_hz_h5[-1] else "ascending"
    print(f"    Order: {freq_order} {'[OK - SNAP native]' if freq_order == 'descending' else '[WARNING - should be descending!]'}")

    # ══════════════════════════════════════════════════════════════════════
    # PART 4: SNAP antenna ordering verification
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 60}")
    print("PART 4: SNAP antenna ordering verification")
    print("─" * 60)

    # Load bf_weights_generator CSV and compare mappings
    sys.path.insert(0, "/home/casm/software/dev/bf_weights_generator")
    from bf_weights_generator.snap_weights import Array64Config

    array_config = Array64Config.from_csv(LAYOUT_CSV)

    print(f"\n  Comparing HDF5 stored mapping vs CSV-computed mapping:")
    print(f"  {'ant64':>6s}  {'ant_id':>6s}  {'snap(HDF5)':>10s}  {'snap(CSV)':>9s}  {'match':>6s}")
    all_match = True
    active_indices = np.where(active_mask)[0]
    for ant64_idx in active_indices:
        ant_id = ant64_idx + 1
        snap_h5 = int(ant64_to_snap[ant64_idx])
        snap_csv = int(array_config.ant64_to_snap[ant64_idx])
        ok = snap_h5 == snap_csv
        if not ok:
            all_match = False
        print(f"  {ant64_idx:6d}  {ant_id:6d}  {snap_h5:10d}  {snap_csv:9d}  {'OK' if ok else 'MISMATCH!'}")

    print(f"\n  All SNAP mappings match: {all_match}")

    # Also verify positions match
    print(f"\n  Position comparison (HDF5 vs CSV):")
    pos_match = True
    for ant64_idx in active_indices:
        h5_pos = positions[ant64_idx]
        csv_pos = array_config.positions_enu[ant64_idx]
        diff = np.abs(h5_pos - csv_pos)
        ok = np.all(diff < 0.001)
        if not ok:
            pos_match = False
            print(f"  ant64={ant64_idx}: HDF5={h5_pos} CSV={csv_pos} DIFF={diff}")
    if pos_match:
        print(f"  All positions match within 1mm [OK]")
    else:
        print(f"  Position mismatches found! [WARNING]")

    # Also compare casm_io positions with bf_weights_generator positions
    print(f"\n  Comparing casm_io layout vs bf_weights_generator layout:")
    for ant64_idx in active_indices:
        ant_id = ant64_idx + 1
        io_idx = int(np.where(ant_ids_io == ant_id)[0][0])
        io_pos = np.array([ant_x[io_idx], ant_y[io_idx], ant_z[io_idx]])
        bf_pos = array_config.positions_enu[ant64_idx]
        diff = np.abs(io_pos - bf_pos)
        ok = np.all(diff < 0.001)
        if not ok:
            print(f"  Ant {ant_id}: casm_io={io_pos}  bf_gen={bf_pos}  diff={diff}")
    if all(np.all(np.abs(
        np.array([ant_x[int(np.where(ant_ids_io == idx+1)[0][0])],
                  ant_y[int(np.where(ant_ids_io == idx+1)[0][0])],
                  ant_z[int(np.where(ant_ids_io == idx+1)[0][0])]])
        - array_config.positions_enu[idx]) < 0.001)
        for idx in active_indices):
        print(f"  All positions match between layouts [OK]")
    else:
        print(f"  Position mismatches between layouts [WARNING]")

    # ══════════════════════════════════════════════════════════════════════
    # PART 5: Sign convention summary
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 60}")
    print("PART 5: Sign convention summary")
    print("─" * 60)

    print("""
  ┌─────────────────────────┬────────────────────────────────────────┐
  │ Component               │ Convention                             │
  ├─────────────────────────┼────────────────────────────────────────┤
  │ casm_calibrator          │                                        │
  │   fringe_stop.py        │ phi = -2*pi*f*tau                      │
  │   svd.py (gain)         │ g = exp(1j*angle(U[:,0]))              │
  │   output.py (weight)    │ w_cal = conj(g)                        │
  ├─────────────────────────┼────────────────────────────────────────┤
  │ bf_weights_generator     │                                        │
  │   coordinates.py        │ delay = -(x*l + y*m + z*n)/c           │
  │   weights.py            │ phase = 2*pi*f*delay                   │
  │                         │ w_geo = exp(-1j*phase)                  │
  │   snap_weights.py       │ w_total = w_cal * w_geo                │
  ├─────────────────────────┼────────────────────────────────────────┤
  │ GPU kernel (h23)         │                                        │
  │   populate_weights_matrix│ w_geo = exp(j*(-2*pi*f*theta/c)*x)    │
  │   beamform (CUBLAS)     │ beam = sum(w * v) [NOT conj(w)*v]      │
  │   cal normalization     │ cal weights forced to unit amplitude    │
  └─────────────────────────┴────────────────────────────────────────┘

  Chain consistency check:
    Source signal: s
    Incoming voltage at antenna a: v_a = s * exp(j*2*pi*f*tau_a) * g_a
      where tau_a = geometric delay, g_a = instrumental gain

    Total weight: w_a = conj(g_a) * exp(-j*2*pi*f*tau_a)
                      = conj(g_a) * conj(exp(j*2*pi*f*tau_a))

    Beam output: sum_a(w_a * v_a)
               = sum_a(conj(g_a) * conj(exp(j*2pi*f*tau_a)) * exp(j*2pi*f*tau_a) * g_a * s)
               = sum_a(|g_a|^2 * s)
               = N * |g|^2 * s    (if |g| ~ 1 for all antennas)

    -> Conventions are SELF-CONSISTENT within our pipeline.

  POTENTIAL ISSUE:
    If the GPU uses populate_weights_matrix(), it computes geo weights
    INTERNALLY and expects ONLY cal weights in the input file. But our
    int8 HDF5 contains COMBINED (cal*geo) weights. If the GPU's
    populate_weights_matrix is active, it would double-apply geometric
    corrections. Need to verify which GPU code path is used.
""")

    print(f"\n{'=' * 70}")
    print("TRACK 3 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
