# Calibration I/O

## save_calibration

```python
save_calibration(cal, path, *, n_time_averaged=0, rfi_mask=None, overwrite=False)
```

Saves a `CalibrationResult` dict. The output format is chosen by the file
extension.

| Extension | Format | Notes |
|-----------|--------|-------|
| `.h5` / `.hdf5` | HDF5 (preferred) | Consumed directly by `bf_weights_generator`. Supports gzip compression. |
| `.npz` | NumPy compressed archive | Legacy format. Same field set. |

HDF5 datasets written: `weights`, `gains`, `flags`, `freqs_hz`, `freqs_mhz`,
`ant_ids`, `rank1_ratios`.
HDF5 attributes written: `ref_ant_id`, `source`, `n_time_averaged`.

Frequencies are written in both MHz (`freqs_mhz`) and Hz (`freqs_hz`); the Hz
field is what `bf_weights_generator` reads.

### Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `cal` | required | `CalibrationResult` from `svd_calibrate` |
| `path` | required | Output path; parent directories are created automatically |
| `n_time_averaged` | `0` | Integration count that went into the time average; recorded as provenance |
| `rfi_mask` | `None` | Per-channel bool (True=good); ANDed with `cal['flags']` before writing. Channels where the combined mask is False have gains and weights zeroed. |
| `overwrite` | `False` | Raises `FileExistsError` if the target exists and this is False. Never silently overwrites. |

Weights are downcast to `complex64` before writing. The parent directory is
created with `mkdir -p` semantics.

### Example

```python
from casm_calibrator import save_calibration

# Basic save
save_calibration(cal, "cal_weights.h5", n_time_averaged=137)

# With RFI mask applied at write time
rfi_mask = np.ones(len(cal["freqs_mhz"]), dtype=bool)
rfi_mask[bad_channels] = False
save_calibration(
    cal,
    "cal_weights.h5",
    n_time_averaged=137,
    rfi_mask=rfi_mask,
    overwrite=True,
)
```

Note: `rfi_mask` at write time is in addition to any RFI mask already applied
during `svd_calibrate`. Applying it here is useful when you want to save
multiple files with different masks from the same `cal` dict without
rerunning SVD.

## Loading in bf_weights_generator

`load_calibration_weights` is defined in `bf_weights_generator`, not in
`casm_calibrator`. It reads the HDF5 or NPZ file written by `save_calibration`
and returns a `CalibrationWeights` dataclass.

```python
from bf_weights_generator.snap_weights import load_calibration_weights

cal = load_calibration_weights("cal_weights.h5")
print(cal.weights.shape)        # (n_ant, n_chan)
print(cal.frequencies_hz[:3])   # ascending Hz
print(cal.ant_ids)              # 1-indexed
```

`CalibrationWeights` is a dataclass defined in `bf_weights_generator`. It is
not exported from `casm_calibrator`.

## Legacy visibility loader

`VisibilityLoader` and `VisibilityMatrix` are exported from `casm_calibrator` but are not the recommended path. They wrap `casm_io.VisibilityReader` and reshape the flat upper-triangle visibility to an `(N, N)` Hermitian matrix per channel. The current preferred path is to pass the full-triangle `VisibilityResult` from `casm_io.read_visibilities` directly as the `data=` argument to `svd_calibrate`, which does the matrix build internally via `_build_hermitian_matrix`. Use `VisibilityLoader` only if you are maintaining code that already calls it.

## Legacy writer

`CalibrationWeightsWriter` in `output.py` is the older write path that takes
an `SVDResult` dataclass (returned by `SVDCalibrator.calibrate`) rather than
a `CalibrationResult` dict. It writes only NPZ. Use `save_calibration` for new
code.

```python
# Legacy path — prefer save_calibration for new code
from casm_calibrator import SVDCalibrator, SVDConfig, CalibrationWeightsWriter

svd_result = SVDCalibrator(SVDConfig()).calibrate(vis_avg)
CalibrationWeightsWriter().write(
    path="cal_weights.npz",
    svd_result=svd_result,
    freqs_mhz=freq_mhz,
    ant_ids=ant_ids,
    ref_ant_id=5,
    source="sun",
    n_time_averaged=137,
)
```

## plot_calibration

```python
from casm_calibrator import plot_calibration

figs = plot_calibration(
    cal,
    threshold=5.0,               # horizontal line on sigma_1/sigma_2 plot; inferred if None
    rfi_ranges=[(375, 390)],     # gray bands on all subplots
    output_path="cal.png",       # None = plt.show()
    ant=ant,                     # AntennaMapping; enables grid-position labels
)
# figs: [Figure_rank1, Figure_phase, Figure_amplitude]
```

When `output_path` is a path like `"cal.png"`, three files are saved:
`cal_rank1.png`, `cal_phase.png`, `cal_amp.png`. The figures are closed after
saving.

When `output_path` is `None`, all three figures are shown via `plt.show()` and
returned open.

The rank-1 plot colors good channels blue and failed channels red. Channels
where `flags=False` are shown with `np.nan` in the phase and amplitude plots
(no line drawn).
