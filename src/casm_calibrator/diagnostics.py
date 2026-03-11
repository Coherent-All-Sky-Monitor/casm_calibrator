"""Diagnostic plots for SVD calibration results."""

import os

import numpy as np

from .svd import SVDResult


class DiagnosticPlotter:
    """Generate diagnostic plots for SVD calibration."""

    def __call__(
        self,
        output_path: str,
        freqs_mhz: np.ndarray,
        svd_result: SVDResult,
        threshold: float,
        source_name: str,
        ant_ids: np.ndarray | None = None,
        rfi_ranges: list | None = None,
    ):
        """Generate diagnostic plots.

        Parameters
        ----------
        output_path : str
            Output path. If ends in .pdf, generates multi-page PDF.
            Otherwise treated as a base path for PNG files:
            {base}_rank1.png, {base}_phase.png, {base}_amplitude.png.
        freqs_mhz : ndarray, shape (n_chan,)
            Frequencies in MHz.
        svd_result : SVDResult
            SVD calibration results.
        threshold : float
            sigma_1/sigma_2 threshold used.
        source_name : str
            Source name for title.
        ant_ids : ndarray, optional
            Antenna IDs for labels.
        rfi_ranges : list of (float, float), optional
            RFI ranges to shade on plots.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        flags = svd_result.flags
        rank1_ratios = svd_result.rank1_ratios
        gains = svd_result.gains
        n_ant = gains.shape[0]

        if ant_ids is None:
            ant_ids = np.arange(1, n_ant + 1)
        rfi_ranges = rfi_ranges or []

        use_pdf = output_path.lower().endswith(".pdf")

        if use_pdf:
            from matplotlib.backends.backend_pdf import PdfPages
            pdf = PdfPages(output_path)

        def save_fig(fig, suffix):
            if use_pdf:
                pdf.savefig(fig)
            else:
                base, _ = os.path.splitext(output_path)
                png_path = f"{base}_{suffix}.png"
                fig.savefig(png_path, dpi=150, bbox_inches="tight")
                print(f"  Saved: {png_path}")
            plt.close(fig)

        # Page 1: sigma_1/sigma_2 vs frequency
        fig, ax = plt.subplots(figsize=(14, 5))
        ratios_plot = np.clip(rank1_ratios, 0, 50)
        colors = np.where(flags, "C0", "red")
        ax.scatter(freqs_mhz, ratios_plot, c=colors, s=2, alpha=0.5)
        ax.axhline(
            threshold, color="orange", ls="--", lw=1.5,
            label=f"threshold = {threshold}",
        )
        for lo, hi in rfi_ranges:
            ax.axvspan(lo, hi, alpha=0.1, color="gray")
        n_good = np.sum(flags)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel(r"$\sigma_1 / \sigma_2$")
        ax.set_title(
            f"Rank-1 Quality vs Frequency ({source_name}) — "
            f"{n_good}/{len(flags)} good channels"
        )
        ax.set_xlim(freqs_mhz[0], freqs_mhz[-1])
        ax.set_ylim(0, min(50, ratios_plot.max() * 1.1) if ratios_plot.max() > 0 else 50)
        ax.legend()
        fig.tight_layout()
        save_fig(fig, "rank1")

        # Page 2: per-antenna gain phase vs frequency
        fig, ax = plt.subplots(figsize=(14, 5))
        for k in range(n_ant):
            phase = np.angle(gains[k, :]).copy()
            phase[~flags] = np.nan
            ax.plot(freqs_mhz, np.degrees(phase), lw=0.8,
                    label=f"Ant {ant_ids[k]}")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Gain phase (deg)")
        ax.set_title(f"Per-antenna gain phase ({source_name})")
        ax.set_xlim(freqs_mhz[0], freqs_mhz[-1])
        ax.legend(fontsize=7, ncol=max(1, n_ant // 4))
        fig.tight_layout()
        save_fig(fig, "phase")

        # Page 3: per-antenna gain amplitude vs frequency
        fig, ax = plt.subplots(figsize=(14, 5))
        for k in range(n_ant):
            amp = np.abs(gains[k, :]).copy()
            amp[~flags] = np.nan
            ax.plot(freqs_mhz, amp, lw=0.8, label=f"Ant {ant_ids[k]}")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Gain amplitude")
        ax.set_title(f"Per-antenna gain amplitude ({source_name})")
        ax.set_xlim(freqs_mhz[0], freqs_mhz[-1])
        ax.legend(fontsize=7, ncol=max(1, n_ant // 4))
        fig.tight_layout()
        save_fig(fig, "amplitude")

        if use_pdf:
            pdf.close()
            print(f"Diagnostic PDF saved: {output_path}")
