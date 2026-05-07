"""Configurable RFI frequency masking.

Re-export of the canonical :class:`casm_vis_analysis.rfi.RFIMask`. Kept
here so existing callers (``from casm_calibrator.rfi import RFIMask``)
keep working; new code should import from ``casm_vis_analysis``.
"""

from casm_vis_analysis.rfi import RFIMask  # noqa: F401

__all__ = ["RFIMask"]
