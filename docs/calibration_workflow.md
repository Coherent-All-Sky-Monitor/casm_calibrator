# SVD Calibration Workflow — Full Math Reference

**Date**: 2026-03-11
**Package**: `casm_calibrator`
**Observatory**: OVRO (lat=37.2339°, lon=-118.2821°, elev=1222 m)
**Band**: 375–468.75 MHz, 3072 channels, channel BW = 0.030517578125 MHz

---

## Overview

```
Raw .dat files
    │
    ▼ Step 1 — Visibility Loading (visibility.py)
Flat baselines → NxN Hermitian matrix  (T, F, N, N)
    │
    ▼ Step 2 — Fringe Stopping (fringe_stop.py)
Remove geometric phase toward source  (T, F, N, N)
    │
    ▼ Step 3 — Transit Window + Time Average
Time mask → mean over T  (F, N, N)
    │
    ▼ Step 4 — SVD Calibration (svd.py)
Per-channel / per-block SVD → gains (N, F)
    │
    ▼ Step 5 — Output NPZ (output.py)
weights, gains, flags, freqs_hz, ant_ids → bf_weights_generator compatible
```

---

## Step 1 — Visibility Loading (`visibility.py`)

The CASM correlator outputs all baselines as a flat upper-triangular array.
Baseline order follows **packet index** (not antenna ID):

```
packet_index(i), packet_index(j)  with  packet_index(i) <= packet_index(j)
```

The flat index is computed via `casm_io.correlator.baselines.triu_flat_index(nsig, ii, jj)`.

### Hermitian assembly

For each antenna pair (i, j) in the active set:

```
pi = packet_index(ant_i)
pj = packet_index(ant_j)

ii = min(pi, pj),  jj = max(pi, pj)

flat_index(i,j) = triu_flat_index(nsig, ii, jj)

if pi > pj:
    # baseline stored as (j→i), so V[i,j] = conj(stored)
    bl_conj[i,j] = True

V_matrix[t, f, i, j] = V_flat[t, f, flat_index] [conjugated if bl_conj]
V_matrix[t, f, j, i] = conj(V_matrix[t, f, i, j])   # Hermitian symmetry
```

**Shape**: `(T, F, N, N)` complex64.
**Frequency order**: always reversed to **ascending** (375 → 468.75 MHz) during load.

---

## Step 2 — Fringe Stopping (`fringe_stop.py`)

Removes the rapidly rotating geometric phase from each baseline, phasing
all integrations toward a named source.

### Antenna positions

```
r_a  = ENU position of antenna a  [shape (N, 3), meters]
```

Loaded from antenna CSV via `casm_io.correlator.mapping.AntennaMapping`.

### Source direction

```
s_enu(t)  = ENU unit vector toward source at each timestamp t  [shape (T, 3)]
```

Computed by `casm_vis_analysis.sources.source_enu(source_name, time_unix)`.

### Per-antenna geometric delay

```
tau_a(t) = dot(s_enu(t), r_a) / c_light       [seconds, shape (T, N)]
```

### Per-antenna phase

```
phi_a(t, f) = -2*pi * f * tau_a(t)            [radians, shape (T, F, N)]
```

Sign convention: **negative** (matches `casm_vis_analysis.fringe_stop` with `sign=-1`).

### Differential application to visibilities

```
V_fs[t, f, i, j] = V[t, f, i, j] * exp(1j * (phi_j - phi_i))
                 = V[t, f, i, j] * exp(-1j * 2*pi * f * (tau_j - tau_i))
```

This is mathematically equivalent to the per-baseline formula:

```
tau_{ij}(t) = dot(s_enu(t), r_j - r_i) / c    [baseline delay]
V_fs[t, f, i, j] = V[t, f, i, j] * exp(-1j * 2*pi * f * tau_{ij}(t))
```

The old `correlator_analysis` code used the explicit per-baseline loop form;
`casm_calibrator` uses the vectorized per-antenna form — **mathematically identical**.

---

## Step 3 — Transit Window & Time Averaging

The transit window is determined by `casm_vis_analysis.sources.find_transit_window()`,
which finds the contiguous block of integrations where the source is above horizon
near culmination. The old code used hard-coded PST time strings (e.g. `"8:00"–"14:00"`).

```
# Restrict to transit window
V_transit[t, f, i, j] = V_fs[t in transit, f, i, j]   shape (T', F, N, N)

# Time average
V_avg[f, i, j] = mean_t(V_transit[t, f, i, j])          shape (F, N, N)
```

Number of integrations averaged = `n_time_averaged` (e.g. 137 for Feb19).

---

## Step 4 — SVD Calibration (`svd.py`)

### Matrix preparation

Three modes (selected via `--svd-mode`):

| Mode | Input matrix |
|------|-------------|
| `phase-only` | `M[i,j] = exp(1j * angle(V_avg[i,j]))` — unit amplitude |
| `cross-only` | `M = V_avg` with diagonal zeroed (no autocorrelations) |
| `raw` | `M = V_avg` — autocorrelation power dominates, worst |

**Recommended**: `phase-only` (best handles large auto-power variation, 30×).

### Per-channel SVD

For each frequency channel `f`:

```
M = prepare(V_avg[f])       # shape (N, N)

U, Σ, Vh = SVD(M)           # M = U Σ Vh

rank1_ratio[f] = Σ[0] / Σ[1]

if rank1_ratio[f] >= threshold:
    flags[f] = True          # good channel

    g_a = exp(1j * angle(U[a, 0]))    # phase of dominant left singular vector
    g *= exp(-1j * angle(g[ref_ant])) # zero reference antenna phase

    gains[:, f] = g
    weights[:, f] = conj(g)
```

Note: `U[:, 0]` is the **left** singular vector of M. Since M ≈ g g^H (rank-1),
`U[:, 0] ∝ g` — the column eigenvector equals the antenna gain vector.

### SVD convention note

`numpy.linalg.svd(M)` returns `U, Σ, Vh` such that `M = U @ diag(Σ) @ Vh`.
- `U[:, 0]` = first **left** singular vector (columns of U), shape `(N,)` per antenna
- `Vh[0, :]` = first **right** singular vector (rows of Vh), shape `(N,)` per antenna

For a Hermitian positive-semidefinite M: `U[:, 0] = conj(Vh[0, :])`.
Both old and new code extract `g = exp(1j * angle(U[:, 0]))` — **same convention**.

### Block SVD (optional)

When `--block-size B > 1`, average `B` adjacent channels before SVD to boost SNR:

```
V_block[b] = mean(V_avg[b*B : (b+1)*B], axis=0)    shape (N, N)
```

Failed blocks (rank1_ratio < threshold) are filled by linear interpolation
of unwrapped phase from good neighbors, then wrapped back to unit-amplitude phasors.
Block gains are then replicated to all per-channel slots.

### Threshold

| Workflow | Default threshold | Feb19 pass rate |
|----------|------------------|-----------------|
| Old (`correlator_analysis`) | **2.0** | 62% |
| New (`casm_calibrator`) | **4.0** | lower |

**Use `--threshold 2.0` for apples-to-apples comparison with old Feb19 results.**

---

## Step 5 — Output NPZ (`output.py`)

```
weights:    (N, F) complex64   = conj(gains)
gains:      (N, F) complex64   = exp(1j * angle(U[:,0]))
flags:      (F,)   bool        = True → good channel (SVD passed + not RFI)
freqs_hz:   (F,)   float64     ascending
freqs_mhz:  (F,)   float64     ascending
ant_ids:    (N,)   int         1-indexed
ref_ant_id: int
source:     str
```

Flagged channels: `weights[:, ~flags] = 0`.

---

## Step 6 — Beamforming (`bf_weights_generator`)

```
# Geometric delay for a given beam direction (l, m, n):
tau_geo(beam, ant) = -(x*l + y*m + z*n) / c

# Geometric weight:
w_geo(f, ant, beam) = exp(-2*pi*i * f * tau_geo(beam, ant))

# Total weight (calibration × geometry):
w_total(f, ant, beam) = w_cal(f, ant) * w_geo(f, ant, beam)
                      = conj(gain(f, ant)) * exp(-2*pi*i * f * tau_geo)
```

Final int8 beamformer weights shape: `(2, 3072, 2, 512, 64)` —
`(real/imag, chan, pol, beam, ant_snap_slot)`.

---

## Old vs New Comparison

### Key differences

| Parameter | Old (`correlator_analysis`) | New (`casm_calibrator`) |
|-----------|----------------------------|------------------------|
| SVD threshold default | **2.0** | **4.0** |
| Transit window | Hard-coded PST strings | `find_transit_window()` auto |
| Fringe stop | Per-baseline loop | Per-antenna vectorized (same math) |
| Phase sign | `-2π f τ` | `-2π f τ` (same) |
| Gain extraction | `exp(1j*angle(U[:,0]))` | `exp(1j*angle(U[:,0]))` (same) |
| Ref ant zeroing | Yes (ant_id=5) | Yes (`--ref-ant 5`) |
| `freq_order` key | Saved as `'ascending'` | Not saved |
| `ref_ant_idx` key | 0-indexed, saved | Not saved |
| RFI masking | Hard-coded ranges in script | CLI args only (no defaults) |
| Block SVD fill | Interpolate / nearest / zero | Interpolate / nearest / zero (same) |
| OVRO coordinates | lat=37.2342°, lon=-118.2834°, h=1207m | From `casm_vis_analysis` |

### Most likely source of discrepancy

1. **Threshold** (2.0 vs 4.0): more channels flagged at 4.0, more interpolation in block mode
2. **Transit window**: auto-detect may select slightly different time range than hard-coded strings
3. **OVRO coordinates**: small difference in lat/lon/height → tiny fringe-stop error (< 1 mm baseline equivalent)

---

## Delay Extraction

From the SVD gains (pure phase, unit amplitude):

```
phase_a(f) = angle(gain_a(f)) [radians]
phase_unwrapped = np.unwrap(phase_a)

# Linear fit: phase = 2*pi * f * tau + offset
slope, offset = np.polyfit(freqs_hz, phase_unwrapped, 1)

# slope has units rad/Hz
tau_seconds = slope / (2*pi)
tau_ns = tau_seconds * 1e9
```

All delays are measured **relative to the reference antenna** (ant_id=5),
which is zeroed by the gain extraction step.

The delay represents the residual cable + electronic delay after fringe-stopping
(i.e. any mismatch between assumed geometry and true geometry, plus cable lengths).

---

## Comparison Script

```bash
# Analyze old Feb19 gains only
python scripts/compare_cal_delays.py \
    --old /path/to/svd_weights_feb19.npz \
    --plot compare_feb19.pdf

# Compare old vs new
python scripts/compare_cal_delays.py \
    --old /path/to/svd_weights_feb19.npz \
    --new /path/to/new_casm_calibrator_output.npz \
    --plot compare_old_vs_new.pdf
```

---

## Verification Checklist

- [ ] Run `compare_cal_delays.py --old svd_weights_feb19.npz` → check delay table matches `delay_stability_table.txt`
- [ ] Run new calibrator on Feb19 data with `--threshold 2.0 --svd-mode phase-only` → compare delays
- [ ] Verify transit window: check `n_time_averaged` matches old (137 integrations for Feb19)
- [ ] Verify `sigma[0]/sigma[1]` distributions match (same data → same rank-1 ratios)
- [ ] Check phase sign: both use `-2π*f*tau` with ENU dot convention
- [ ] Check baseline conjugation: `bl_conj[i,j] = packet_index(i) > packet_index(j)`
