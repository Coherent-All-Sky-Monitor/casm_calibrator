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
  --obs "2026-03-20-05:55:45" \
  --source sun \
  --output cal_weights.npz \
  --plots diagnostics.pdf

# Block SVD (32 channels per block) with lower threshold
casm-svd-calibrate \
  --data-dir /mnt/nvme3/data/casm/visibilities_64ant/ \
  --obs "2026-03-20-05:55:45" \
  --source sun \
  --output cal_weights.npz \
  --plots diagnostics.pdf \
  --block-size 32 \
  --threshold 2.0

# With RFI masking
casm-svd-calibrate \
  --data-dir /mnt/nvme3/data/casm/visibilities_64ant/ \
  --obs "2026-03-20-05:55:45" \
  --source sun \
  --output cal_weights.npz \
  --rfi-mask-range 375 390 \
  --rfi-mask-range 450 452

# Cas A calibration with custom layout and reference antenna
casm-svd-calibrate \
  --data-dir /mnt/nvme3/data/casm/visibilities_64ant/ \
  --obs "2026-03-20-05:55:45" \
  --source cas_a \
  --layout ~/software/dev/antenna_layouts/antenna_layout_mar21.csv \
  --ref-ant 3 \
  --svd-mode cross-only \
  --output cas_a_weights.npz
```

### Python API

```python
from casm_calibrator import (
    SVDCalibrator, SVDConfig, SVDMode,
    VisibilityLoader, CalibrationWeightsWriter,
)
from casm_calibrator.fringe_stop import FringeStopMatrix
from casm_calibrator.rfi import RFIMask
from casm_io.correlator.mapping import AntennaMapping
from casm_vis_analysis.sources import find_transit_window
import numpy as np

# Load data
mapping = AntennaMapping.load("antenna_layout_mar21.csv")
vis = VisibilityLoader(mapping).load("/data/visibilities_64ant", "2026-03-20-05:55:45")
print(vis.vis.shape)  # (T, 3072, 16, 16)

# Trim to transit
i_start, i_end = find_transit_window("sun", vis.time_unix, min_alt_deg=10.0)
from casm_calibrator import VisibilityMatrix
vis = VisibilityMatrix(
    vis=vis.vis[i_start:i_end+1], freq_mhz=vis.freq_mhz,
    time_unix=vis.time_unix[i_start:i_end+1],
    ant_ids=vis.ant_ids, positions_enu=vis.positions_enu,
)

# Fringe-stop and time-average
vis_fs = FringeStopMatrix()(vis, "sun")
vis_avg = np.mean(vis_fs.vis, axis=0)  # (F, n_ant, n_ant)

# SVD calibration
config = SVDConfig(threshold=2.0, ref_ant_idx=2, svd_mode=SVDMode.PHASE_ONLY)
result = SVDCalibrator(config).calibrate(vis_avg)
print(f"{np.sum(result.flags)}/{len(result.flags)} channels pass")

# Write output (compatible with bf_weights_generator.load_calibration_weights)
CalibrationWeightsWriter().write(
    "cal_weights.npz", result, vis.freq_mhz, vis.ant_ids,
    ref_ant_id=3, source="sun",
)
```

### Load output with bf_weights_generator

```python
from bf_weights_generator.snap_weights import load_calibration_weights

cal = load_calibration_weights("cal_weights.npz")
print(cal.weights.shape)       # (n_ant, n_chan)
print(cal.frequencies_hz[:3])  # ascending Hz
print(cal.ant_ids)             # 1-indexed
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
| `--layout` | `~/software/dev/antenna_layouts/antenna_layout_mar21.csv` | Antenna CSV |
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
