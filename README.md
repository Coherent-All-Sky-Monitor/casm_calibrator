# casm_calibrator

SVD-based beamformer delay calibration for CASM (Coherent All-Sky Monitor) at OVRO.

Fringe-stops correlator visibilities toward a bright source, runs per-channel or per-block SVD to extract per-antenna complex gains, and outputs calibration weights compatible with `bf_weights_generator`.

## Install

```bash
source ~/software/dev/casm_venvs/casm_offline_env/bin/activate
cd /home/casm/software/dev/casm_calibrator
pip install -e ".[dev]"
```

## Quick Start

### CLI: full pipeline

```bash
# Basic run — sun transit, per-channel SVD, phase-only mode
casm-svd-calibrate \
  --data-dir /mnt/nvme3/data/casm/visibilities_64ant/ \
  --obs "2026-03-10-14:05:53" \
  --source sun \
  --output cal_weights.npz \
  --plots diagnostics.pdf

# Block SVD (32 channels per block) with lower threshold
casm-svd-calibrate \
  --data-dir /mnt/nvme3/data/casm/visibilities_64ant/ \
  --obs "2026-03-10-14:05:53" \
  --source sun \
  --output cal_weights.npz \
  --plots diagnostics.pdf \
  --block-size 32 \
  --threshold 2.0

# With RFI masking
casm-svd-calibrate \
  --data-dir /mnt/nvme3/data/casm/visibilities_64ant/ \
  --obs "2026-03-10-14:05:53" \
  --source sun \
  --output cal_weights.npz \
  --rfi-mask-range 375 390 \
  --rfi-mask-range 450 452

# Cas A calibration with custom layout and reference antenna
casm-svd-calibrate \
  --data-dir /mnt/nvme3/data/casm/visibilities_64ant/ \
  --obs "2026-03-10-14:05:53" \
  --source cas_a \
  --layout ~/software/dev/antenna_layouts/antenna_layout_current.csv \
  --ref-ant 5 \
  --svd-mode cross-only \
  --output cas_a_weights.npz
```

### Python API (compose-friendly)

The high-level entry point is `svd_calibrate(fs, ant, *, data, config=...)`. It takes the fringe-stopped output from `casm_vis_analysis.fringe_stop` plus the full-triangle visibilities from `casm_io.read_visibilities` and returns a :class:`CalibrationResult` dict ready to write with :func:`save_calibration`.

```python
import numpy as np
from casm_io.correlator import read_visibilities
from casm_io.correlator.mapping import AntennaMapping
from casm_vis_analysis import run_fringe_stop
from casm_calibrator import (
    svd_calibrate, save_calibration,
    SVDConfig, SVDMode,
)

# 1. Load visibilities + run fringe-stop (gives us fs['time_mask'], freq_mhz, etc.)
ant  = AntennaMapping.load("antenna_layout_current.csv")
data = read_visibilities(time_start=..., time_end=..., layout="layout_64ant")
fs   = run_fringe_stop(
    data_dir=..., obs="2026-03-10-14:05:53", format="layout_64ant",
    layout="antenna_layout_current.csv", ref_ant=10, source="sun",
)

# 2. Configure SVD. svd_mode accepts either an enum or its string value.
config = SVDConfig(
    threshold=4.0,
    ref_ant_idx=0,          # remapped from fs['ref_ant'] internally
    svd_mode=SVDMode.PHASE_ONLY,   # or string "phase-only" — also: COMPLEX, CROSS_ONLY, RAW
    subband_size=16,        # optional: subband + smooth-fit path
    smooth_order=1,
)

# 3. Calibrate. `data` is keyword-only and required.
cal = svd_calibrate(fs, ant, data=data, config=config)
print(f"{int(np.sum(cal['flags']))}/{len(cal['flags'])} channels pass")

# 4. Persist. Refuses to overwrite by default — pass overwrite=True to replace.
save_calibration(cal, "cal_weights.h5", overwrite=False)
```

`CalibrationResult` is a `TypedDict` with `gains`, `weights`, `flags`,
`rank1_ratios`, `singular_values`, `freqs_mhz`, `ant_ids`,
`ref_ant_id`, `source`, `layout_version` — see the class docstring for
shapes.

### Load output with bf_weights_generator

```python
from bf_weights_generator.snap_weights import load_calibration_weights

cal = load_calibration_weights("cal_weights.h5")
print(cal.weights.shape)       # (n_ant, n_chan)
print(cal.frequencies_hz[:3])  # ascending Hz
print(cal.ant_ids)             # 1-indexed
```

### Low-level engine (advanced)

For one-shot SVD on an externally-built `(F, N, N)` Hermitian matrix,
the engine is exposed directly:

```python
from casm_calibrator import SVDCalibrator, SVDConfig, SVDMode

result = SVDCalibrator(SVDConfig(threshold=2.0,
                                  svd_mode=SVDMode.PHASE_ONLY)).calibrate(vis_avg)
# result is an SVDResult dataclass (gains, weights, flags, ...).
```

## CLI Reference

```
casm-svd-calibrate --help
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | required | Directory with .dat files |
| `--obs` | required | Observation base string (UTC timestamp) |
| `--source` | required | Source: sun, cas_a, tau_a, cyg_a |
| `--output` | required | Output NPZ path |
| `--layout` | `~/software/dev/antenna_layouts/antenna_layout_current.csv` | Antenna CSV |
| `--threshold` | 4.0 | sigma_1/sigma_2 cutoff |
| `--ref-ant` | 5 | 1-indexed reference antenna |
| `--svd-mode` | phase-only | `phase-only`, `cross-only`, `raw` |
| `--block-size` | 1 | Channels per SVD block (1 = per-channel) |
| `--fill-mode` | interpolate | Failed block fill: `interpolate`, `zero`, `nearest` |
| `--min-alt` | 10.0 | Minimum source altitude (deg) |
| `--plots` | none | Diagnostic PDF/PNG path |
| `--rfi-mask-range` | none | Repeatable: `START_MHZ END_MHZ` |
| `--nfiles` | all | Number of data files to read |
| `--verbose` | off | Extra output |

## SVD Modes

- **phase-only** — `exp(1j * angle(V))`: normalizes all baselines to unit amplitude. Best when antenna auto-power varies widely (e.g. 30x spread). Highest pass rate.
- **cross-only** — zero diagonal, keep raw cross amplitudes. Preserves natural SNR weighting but bright antennas dominate.
- **raw** — full matrix including autos. Autocorrelation power dominates; lowest pass rate.

## Architecture

```
src/casm_calibrator/
    __init__.py        # Public exports
    visibility.py      # VisibilityLoader: casm_io flat baselines -> NxN matrix
    fringe_stop.py     # FringeStopMatrix: vectorized per-antenna fringe stopping
    rfi.py             # RFIMask: user-configurable frequency masking
    svd.py             # SVDCalibrator: per-channel & per-block SVD engine
    diagnostics.py     # DiagnosticPlotter: PDF or PNG output
    output.py          # CalibrationWeightsWriter: NPZ serialization
    cli.py             # CLI: casm-svd-calibrate
```

## Dependencies

```
numpy, matplotlib, casm_io, casm_vis_analysis
```

Optional: `bf_weights_generator` (for output compatibility tests)

## Testing

```bash
python -m pytest tests/ -v
```
