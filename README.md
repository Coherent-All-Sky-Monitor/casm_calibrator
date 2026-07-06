# casm_calibrator

SVD-based beamformer calibration for CASM. Takes fringe-stopped correlator
visibilities, builds a per-channel Hermitian visibility matrix, and extracts
per-antenna complex gains via rank-1 SVD decomposition. Outputs weights
consumed by `bf_weights_generator`.

## Install

```bash
source ~/software/dev/casm_venvs/casm_offline_env/bin/activate
cd /home/casm/software/dev/casm_calibrator
pip install -e ".[dev]"
```

## Usage

### Python API

```python
import numpy as np
from casm_io.correlator import read_visibilities, load_format, AntennaMapping
from casm_vis_analysis.fringe_stop import fringe_stop
from casm_calibrator import svd_calibrate, save_calibration, SVDConfig, SVDMode

# No path -> the canonical $CASM_LAYOUT_DIR/current layout.
ant = AntennaMapping.load().with_inactive([3])

# Read a solar-transit window (auto-discovers visibilities_* under /mnt).
data = read_visibilities(
    time_start="2026-06-28 12:00:00",
    time_end="2026-06-28 13:30:00",
    time_tz="America/Los_Angeles",
    data_root="/mnt",
    fmt=load_format("layout_64ant"),
)

# Fringe-stop toward the Sun; sign=-1 is the CASM convention.
fs = fringe_stop(data, ant, ref_ant=9, source="sun", sign=-1)

cfg = SVDConfig(
    threshold=5.0,                        # recommended for strong cal sources
    svd_mode=SVDMode.PHASE_ONLY,          # best pass rate on this array
    masked_band_strategy="zero",          # recommended; zeros RFI-flagged channels
)

# `data` is keyword-only and required
cal = svd_calibrate(fs, ant, data=data, config=cfg)
print(f"{int(np.sum(cal['flags']))}/{len(cal['flags'])} channels pass")

# .h5 preferred; raises FileExistsError unless overwrite=True
save_calibration(cal, "cal_weights.h5", n_time_averaged=len(data["time_unix"]))
```

See [docs/svd_calibration.md](docs/svd_calibration.md) for full `SVDConfig` and
`SVDMode` reference.

### CLI

```bash
casm-svd-calibrate \
  --data-dir /mnt/nvme3/data/casm/visibilities_64ant/ \
  --obs "2026-03-10-14:05:53" \
  --source sun \
  --output cal_weights.h5 \
  --threshold 5.0 \
  --svd-mode phase-only \
  --plots diagnostics.png \
  --rfi-mask-range 375 390 \
  --rfi-mask-range 450 452
```

| Flag | Default | Notes |
|------|---------|-------|
| `--data-dir` | required | Directory with `.dat` files |
| `--obs` | required | Observation start timestamp |
| `--source` | required | `sun`, `cas_a`, `tau_a`, `cyg_a` |
| `--output` | required | Output path; `.h5` or `.npz` |
| `--layout` | canonical layout (`$CASM_LAYOUT_CSV`, then `$CASM_LAYOUT_DIR/current`) | Antenna CSV |
| `--threshold` | `4.0` | sigma_1/sigma_2 cutoff (5.0 recommended for production) |
| `--ref-ant` | `5` | 1-indexed reference antenna ID |
| `--svd-mode` | `phase-only` | `phase-only` (recommended), `cross-only`, `raw` |
| `--block-size` | `1` | Channels per SVD block; 1 = per-channel |
| `--fill-mode` | `interpolate` | Failed-block fill: `interpolate`, `nearest`, `zero` |
| `--min-alt` | `10.0` | Source altitude cut (degrees) |
| `--plots` | none | Diagnostic PNG/PDF path |
| `--rfi-mask-range` | none | Repeatable: `START_MHZ END_MHZ` |
| `--nfiles` | all | Number of data files to read |
| `--amp-weighting` | `none` | `inverse-variance` downweights noisy antennas |
| `--verbose` | off | Print array shapes and extra diagnostics |

## Capabilities

- **Per-channel SVD** (`block_size=1`, default): highest frequency resolution.
- **Block SVD** (`block_size=N`): averages N channels before SVD; improves SNR on weaker sources.
- **Subband + smooth-fit** (`subband_size=N`): fits a per-antenna polynomial across valid subbands; see [docs/svd_calibration.md](docs/svd_calibration.md).
- **Diagnostic plots**: sigma_1/sigma_2 ratio, per-antenna gain phase, per-antenna gain amplitude.
- **Output formats**: `.h5` (preferred, consumed by `bf_weights_generator`) or `.npz` (legacy).

## Architecture

```
src/casm_calibrator/
    __init__.py      svd_calibrate, save_calibration, plot_calibration, CalibrationResult
    svd.py           SVDCalibrator, SVDConfig, SVDMode, SVDResult
    output.py        CalibrationWeightsWriter (legacy SVDResult writer)
    visibility.py    VisibilityLoader, VisibilityMatrix
    fringe_stop.py   FringeStopMatrix
    rfi.py           RFIMask (re-export from casm_vis_analysis)
    diagnostics.py   DiagnosticPlotter
    cli.py           casm-svd-calibrate entry point
```

## Detailed docs

- [SVD calibration reference](docs/svd_calibration.md): `SVDConfig`, `SVDMode`, recommended values, gotchas
- [Calibration I/O](docs/calibration_io.md): `save_calibration`, `plot_calibration`, downstream loading

## Testing

```bash
pytest tests/ -v
```

27 tests, all passing. Tests use synthetic fixtures; no real data required.
