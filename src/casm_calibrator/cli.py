"""CLI entry point: casm-svd-calibrate."""

import argparse
import os
import sys

import numpy as np

from casm_io.correlator.mapping import AntennaMapping
from casm_vis_analysis.sources import find_transit_window

from .diagnostics import DiagnosticPlotter
from .fringe_stop import FringeStopMatrix
from .output import CalibrationWeightsWriter
from .rfi import RFIMask
from .svd import SVDCalibrator, SVDConfig, SVDMode
from .visibility import VisibilityLoader


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SVD beamformer calibration for CASM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--data-dir", required=True, help="Directory with .dat files")
    parser.add_argument("--obs", required=True, help="Observation base string (UTC timestamp)")
    parser.add_argument("--format", default=None, help="Visibility format name (auto-detect if omitted)")
    parser.add_argument(
        "--layout",
        default=os.path.expanduser(
            "~/software/dev/antenna_layouts/antenna_layout_current.csv"
        ),
        help="Antenna layout CSV path",
    )
    parser.add_argument("--source", required=True, help="Source name: sun, cas_a, tau_a, cyg_a")
    parser.add_argument("--output", required=True, help="Output weights NPZ path")
    parser.add_argument("--threshold", type=float, default=4.0, help="sigma_1/sigma_2 threshold (default: 4.0)")
    parser.add_argument("--ref-ant", type=int, default=5, help="1-indexed reference antenna ID (default: 5)")
    parser.add_argument(
        "--svd-mode",
        default="phase-only",
        choices=["phase-only", "cross-only", "raw"],
        help="SVD input mode (default: phase-only)",
    )
    parser.add_argument("--block-size", type=int, default=1, help="Channels per SVD block (default: 1 = per-channel)")
    parser.add_argument(
        "--fill-mode",
        default="interpolate",
        choices=["interpolate", "zero", "nearest"],
        help="Failed block fill mode (default: interpolate)",
    )
    parser.add_argument("--min-alt", type=float, default=10.0, help="Minimum source altitude in degrees (default: 10)")
    parser.add_argument("--plots", default=None, help="Output diagnostic PDF path")
    parser.add_argument(
        "--rfi-mask-range",
        nargs=2,
        type=float,
        action="append",
        metavar=("START_MHZ", "END_MHZ"),
        help="RFI range to flag (repeatable)",
    )
    parser.add_argument("--nfiles", type=int, default=None, help="Number of files to read")
    parser.add_argument("--skip-nfiles", type=int, default=0,
                        help="Number of files to skip before reading (requires --nfiles)")
    parser.add_argument("--time-start", default=None,
                        help="Start time for data selection (ISO format)")
    parser.add_argument("--time-end", default=None,
                        help="End time for data selection (ISO format)")
    parser.add_argument("--time-tz", default="UTC",
                        help="Timezone for --time-start/--time-end (default: UTC)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--amp-weighting",
        default="none",
        choices=["none", "inverse-variance"],
        help="Amplitude weighting: 'inverse-variance' downweights noisy antennas (default: none).",
    )

    args = parser.parse_args(argv)

    # 1. Load antenna layout
    print(f"Loading antenna layout: {args.layout}")
    mapping = AntennaMapping.load(args.layout)
    ant_ids = np.array(mapping.active_antennas(), dtype=int)
    n_ant = len(ant_ids)
    print(f"  {n_ant} active antennas: {ant_ids}")

    # Find reference antenna index
    ref_ant_id = args.ref_ant
    ref_mask = ant_ids == ref_ant_id
    if not np.any(ref_mask):
        print(f"  ERROR: ref ant {ref_ant_id} not in active antennas: {ant_ids}", file=sys.stderr)
        sys.exit(1)
    ref_ant_idx = int(np.where(ref_mask)[0][0])
    print(f"  Reference antenna: Ant {ref_ant_id} (index {ref_ant_idx})")

    # 2. Load visibilities
    print(f"\nLoading visibilities from {args.data_dir}, obs={args.obs}")
    fmt = None
    if args.format:
        from casm_io.correlator.formats import load_format
        fmt = load_format(args.format)

    loader = VisibilityLoader(mapping)
    vis_matrix = loader.load(
        args.data_dir, args.obs, fmt=fmt,
        nfiles=args.nfiles,
        skip_nfiles=args.skip_nfiles,
        time_start=args.time_start,
        time_end=args.time_end,
        time_tz=args.time_tz,
    )
    print(f"  vis shape: {vis_matrix.vis.shape}")
    print(f"  freq range: {vis_matrix.freq_mhz[0]:.2f} - {vis_matrix.freq_mhz[-1]:.2f} MHz")

    # 3. Find transit window
    source = args.source.lower()
    print(f"\nFinding transit window for {source} (min alt={args.min_alt} deg)...")
    i_start, i_end = find_transit_window(source, vis_matrix.time_unix, min_alt_deg=args.min_alt)
    n_transit = i_end - i_start + 1
    print(f"  Transit: indices {i_start}-{i_end} ({n_transit} integrations)")

    # Trim to transit
    from .visibility import VisibilityMatrix
    vis_matrix = VisibilityMatrix(
        vis=vis_matrix.vis[i_start : i_end + 1],
        freq_mhz=vis_matrix.freq_mhz,
        time_unix=vis_matrix.time_unix[i_start : i_end + 1],
        ant_ids=vis_matrix.ant_ids,
        positions_enu=vis_matrix.positions_enu,
    )

    # 4. Fringe-stop
    print(f"\nFringe-stopping toward {source}...")
    vis_fs = FringeStopMatrix()(vis_matrix, source)

    # 5. Time-average
    print(f"\nTime-averaging {vis_fs.vis.shape[0]} integrations...")
    vis_avg = np.mean(vis_fs.vis, axis=0)  # (F, n_ant, n_ant)
    print(f"  vis_avg shape: {vis_avg.shape}")

    # 6. RFI mask
    rfi_ranges = args.rfi_mask_range or []
    rfi_mask_obj = RFIMask(bad_ranges_mhz=rfi_ranges)
    rfi_mask = rfi_mask_obj(vis_fs.freq_mhz)
    n_good_rfi = np.sum(rfi_mask)
    print(f"\nRFI mask: {n_good_rfi}/{len(rfi_mask)} channels good")

    # 7. SVD
    svd_mode = SVDMode(args.svd_mode)
    config = SVDConfig(
        threshold=args.threshold,
        ref_ant_idx=ref_ant_idx,
        svd_mode=svd_mode,
        block_size=args.block_size,
        fill_mode=args.fill_mode,
        amp_weighting=args.amp_weighting,
    )
    print(f"\nRunning SVD (mode={args.svd_mode}, threshold={args.threshold}, "
          f"block_size={args.block_size}, amp_weighting={args.amp_weighting})...")
    calibrator = SVDCalibrator(config)
    svd_result = calibrator.calibrate(vis_avg)

    n_svd_good = np.sum(svd_result.flags)
    flags_combined = svd_result.flags & rfi_mask
    n_combined = np.sum(flags_combined)
    print(f"  SVD: {n_svd_good}/{len(svd_result.flags)} channels pass threshold")
    print(f"  Combined (SVD + RFI): {n_combined}/{len(flags_combined)} channels good")

    # 8. Write output
    print(f"\nWriting weights to {args.output}...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    writer = CalibrationWeightsWriter()
    writer.write(
        path=args.output,
        svd_result=svd_result,
        freqs_mhz=vis_fs.freq_mhz,
        ant_ids=vis_fs.ant_ids,
        ref_ant_id=ref_ant_id,
        source=source,
        n_time_averaged=vis_fs.vis.shape[0],
        rfi_mask=rfi_mask,
    )
    print(f"  Done. Weights shape: {svd_result.weights.shape}")

    # 9. Diagnostic plots
    if args.plots:
        os.makedirs(os.path.dirname(os.path.abspath(args.plots)), exist_ok=True)
        print(f"\nGenerating diagnostic plots: {args.plots}")
        DiagnosticPlotter()(
            output_path=args.plots,
            freqs_mhz=vis_fs.freq_mhz,
            svd_result=svd_result,
            threshold=args.threshold,
            source_name=source,
            ant_ids=vis_fs.ant_ids,
            rfi_ranges=rfi_ranges,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
