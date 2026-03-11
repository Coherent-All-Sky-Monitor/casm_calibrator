"""Vectorized fringe-stopping for NxN visibility matrices."""

import numpy as np

from casm_io.constants import C_LIGHT_M_S
from casm_vis_analysis.sources import source_enu

from .visibility import VisibilityMatrix


class FringeStopMatrix:
    """Fringe-stop a full NxN visibility matrix toward a named source.

    Uses a per-antenna phase approach: compute delay per antenna, then
    apply differential phase (phi_j - phi_i) to each baseline (i, j).
    """

    def __call__(self, vis_matrix: VisibilityMatrix, source_name: str) -> VisibilityMatrix:
        """Apply fringe-stopping.

        Parameters
        ----------
        vis_matrix : VisibilityMatrix
            Input visibilities.
        source_name : str
            Source name (e.g. 'sun', 'cas_a').

        Returns
        -------
        VisibilityMatrix
            Fringe-stopped visibilities (new object, same metadata).
        """
        vis = vis_matrix.vis
        freqs_mhz = vis_matrix.freq_mhz
        time_unix = vis_matrix.time_unix
        positions = vis_matrix.positions_enu

        n_time, n_chan, n_ant, _ = vis.shape
        freqs_hz = freqs_mhz * 1e6  # (F,)

        # Source ENU direction: (T, 3)
        s_enu = source_enu(source_name, time_unix)

        # Per-antenna delay: tau_a = dot(source_enu, pos[a]) / c -> (T, n_ant)
        tau = s_enu @ positions.T / C_LIGHT_M_S  # (T, n_ant)

        # Per-antenna phase: phi_a = -2*pi * freq_hz * tau_a -> (T, F, n_ant)
        # tau[:, :, np.newaxis] is (T, n_ant, 1), freqs_hz is (F,)
        phi = -2.0 * np.pi * tau[:, np.newaxis, :] * freqs_hz[np.newaxis, :, np.newaxis]
        # phi shape: (T, F, n_ant)

        # Apply: vis_fs[t,f,i,j] = vis[t,f,i,j] * exp(1j*(phi_j - phi_i))
        # exp(1j*phi) shape (T, F, n_ant)
        exp_phi = np.exp(1j * phi)

        # vis_fs = vis * conj(exp_phi_i) * exp_phi_j
        # = vis * (exp_phi_j / exp_phi_i)
        # Using broadcasting: exp_phi[:,:,:,None] for j, conj(exp_phi[:,:,None,:]) for i
        vis_fs = vis * (
            np.conj(exp_phi[:, :, :, np.newaxis])  # conj(phi_i) for axis 2
            * exp_phi[:, :, np.newaxis, :]           # phi_j for axis 3
        )

        # Preserve diagonal (autos don't need fringe-stopping, but the
        # differential phase is zero for i==j so this is a no-op anyway)

        return VisibilityMatrix(
            vis=vis_fs.astype(np.complex64),
            freq_mhz=vis_matrix.freq_mhz,
            time_unix=vis_matrix.time_unix,
            ant_ids=vis_matrix.ant_ids,
            positions_enu=vis_matrix.positions_enu,
        )
