"""Visibility loading: casm_io flat baselines -> NxN Hermitian matrix."""

from dataclasses import dataclass

import numpy as np

from casm_io.correlator.baselines import triu_flat_index
from casm_io.correlator.mapping import AntennaMapping
from casm_io.correlator.reader import VisibilityReader


@dataclass
class VisibilityMatrix:
    """Container for NxN visibility matrix data.

    Attributes
    ----------
    vis : ndarray, shape (T, F, n_ant, n_ant)
        Complex visibility matrix, Hermitian.
    freq_mhz : ndarray, shape (F,)
        Frequencies in MHz, ascending.
    time_unix : ndarray, shape (T,)
        Unix timestamps per integration.
    ant_ids : ndarray, shape (n_ant,)
        1-indexed antenna IDs.
    positions_enu : ndarray, shape (n_ant, 3)
        ENU positions in meters.
    """

    vis: np.ndarray
    freq_mhz: np.ndarray
    time_unix: np.ndarray
    ant_ids: np.ndarray
    positions_enu: np.ndarray


class VisibilityLoader:
    """Load correlator data via casm_io and reshape to NxN matrix.

    Parameters
    ----------
    mapping : AntennaMapping
        Antenna hardware mapping (from CSV).
    """

    def __init__(self, mapping: AntennaMapping):
        self._mapping = mapping
        self._ant_ids = np.array(mapping.active_antennas(), dtype=int)
        self._packet_indices = np.array(
            [mapping.packet_index(a) for a in self._ant_ids], dtype=int
        )
        self._positions = mapping.get_positions()
        # Filter positions to active antennas only
        all_ids = mapping.dataframe["antenna_id"].values
        active_mask = np.isin(all_ids, self._ant_ids)
        # Reorder positions to match self._ant_ids order
        id_to_row = {int(aid): idx for idx, aid in enumerate(all_ids)}
        active_rows = [id_to_row[int(a)] for a in self._ant_ids]
        self._positions = self._positions[active_rows]

    def load(
        self,
        data_dir,
        obs_id,
        fmt=None,
        nfiles=None,
        skip_nfiles=0,
        time_start=None,
        time_end=None,
        time_tz="UTC",
    ) -> VisibilityMatrix:
        """Load visibilities and reshape to NxN matrix.

        Parameters
        ----------
        data_dir : str
            Directory containing .dat files.
        obs_id : str
            Observation base string (UTC timestamp).
        fmt : VisibilityFormat, optional
            Format config. None for auto-detect.
        nfiles : int, optional
            Number of files to read.
        skip_nfiles : int
            Number of files to skip before reading (requires nfiles).
        time_start : str | None
            Start time for data selection (ISO format).
        time_end : str | None
            End time for data selection (ISO format).
        time_tz : str
            Timezone for time_start/time_end (default: UTC).

        Returns
        -------
        VisibilityMatrix
        """
        reader = VisibilityReader(data_dir, obs_id, fmt=fmt)
        data = reader.read(
            nfiles=nfiles,
            skip_nfiles=skip_nfiles,
            time_start=time_start,
            time_end=time_end,
            time_tz=time_tz,
            freq_order="ascending",
            verbose=True,
        )

        vis_flat = data["vis"]  # (T, F, n_baselines)
        freq_mhz = data["freq_mhz"]
        time_unix = data["time_unix"]

        nsig = data["metadata"]["nsig"]
        vis_matrix = self._flat_to_matrix(vis_flat, nsig)

        return VisibilityMatrix(
            vis=vis_matrix,
            freq_mhz=freq_mhz,
            time_unix=time_unix,
            ant_ids=self._ant_ids.copy(),
            positions_enu=self._positions.copy(),
        )

    def _build_baseline_map(self, nsig):
        """Build flat-index and conjugation maps for active antennas.

        Returns
        -------
        bl_indices : ndarray, shape (n_ant, n_ant)
            Flat baseline index for each antenna pair.
        bl_conj : ndarray, shape (n_ant, n_ant)
            True where conjugation is needed (packet_index[i] > packet_index[j]).
        """
        n_ant = len(self._ant_ids)
        bl_indices = np.zeros((n_ant, n_ant), dtype=int)
        bl_conj = np.zeros((n_ant, n_ant), dtype=bool)

        for i in range(n_ant):
            for j in range(i, n_ant):
                pi, pj = self._packet_indices[i], self._packet_indices[j]
                ii, jj = min(pi, pj), max(pi, pj)
                bl_indices[i, j] = triu_flat_index(nsig, ii, jj)
                bl_indices[j, i] = bl_indices[i, j]
                bl_conj[i, j] = pi > pj
                bl_conj[j, i] = not bl_conj[i, j]

        return bl_indices, bl_conj

    def _flat_to_matrix(self, vis_flat, nsig):
        """Extract NxN Hermitian matrix from flat baselines.

        Parameters
        ----------
        vis_flat : ndarray, shape (T, F, n_baselines)
            Flat visibility data.
        nsig : int
            Number of signal inputs.

        Returns
        -------
        ndarray, shape (T, F, n_ant, n_ant)
            Hermitian visibility matrix.
        """
        bl_indices, bl_conj = self._build_baseline_map(nsig)
        n_ant = len(self._ant_ids)
        nt, nf = vis_flat.shape[0], vis_flat.shape[1]

        vis_matrix = np.zeros((nt, nf, n_ant, n_ant), dtype=np.complex64)
        for i in range(n_ant):
            for j in range(i, n_ant):
                v = vis_flat[:, :, bl_indices[i, j]]
                if bl_conj[i, j]:
                    v = np.conj(v)
                vis_matrix[:, :, i, j] = v
                if i != j:
                    vis_matrix[:, :, j, i] = np.conj(v)

        return vis_matrix
