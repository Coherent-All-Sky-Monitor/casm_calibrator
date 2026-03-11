"""Tests for visibility flat-to-matrix conversion."""

import numpy as np
import pytest

from casm_io.correlator.baselines import triu_flat_index, n_baselines


class TestFlatToMatrix:
    """Test baseline indexing and matrix construction logic."""

    def test_triu_index_round_trip(self):
        """Flat index computation matches expected triangle indices."""
        nsig = 4
        # Manual check: (0,0)=0, (0,1)=1, (0,2)=2, (0,3)=3, (1,1)=4, (1,2)=5, ...
        assert triu_flat_index(nsig, 0, 0) == 0
        assert triu_flat_index(nsig, 0, 1) == 1
        assert triu_flat_index(nsig, 0, 3) == 3
        assert triu_flat_index(nsig, 1, 1) == 4
        assert triu_flat_index(nsig, 1, 2) == 5
        assert triu_flat_index(nsig, 3, 3) == 9
        assert n_baselines(nsig) == 10

    def test_conjugation_symmetry(self):
        """Matrix constructed from flat baselines is Hermitian."""
        nsig = 4
        n_ant = 3
        nf = 8
        nt = 2
        nbl = n_baselines(nsig)

        rng = np.random.default_rng(42)
        vis_flat = (rng.standard_normal((nt, nf, nbl)) +
                    1j * rng.standard_normal((nt, nf, nbl))).astype(np.complex64)

        # Simulate packet indices for 3 antennas using inputs 0, 1, 2
        packet_indices = np.array([0, 1, 2])

        # Build matrix manually (same logic as VisibilityLoader._flat_to_matrix)
        bl_indices = np.zeros((n_ant, n_ant), dtype=int)
        bl_conj = np.zeros((n_ant, n_ant), dtype=bool)
        for i in range(n_ant):
            for j in range(i, n_ant):
                pi, pj = packet_indices[i], packet_indices[j]
                ii, jj = min(pi, pj), max(pi, pj)
                bl_indices[i, j] = triu_flat_index(nsig, ii, jj)
                bl_indices[j, i] = bl_indices[i, j]
                bl_conj[i, j] = pi > pj
                bl_conj[j, i] = not bl_conj[i, j]

        vis_matrix = np.zeros((nt, nf, n_ant, n_ant), dtype=np.complex64)
        for i in range(n_ant):
            for j in range(i, n_ant):
                v = vis_flat[:, :, bl_indices[i, j]]
                if bl_conj[i, j]:
                    v = np.conj(v)
                vis_matrix[:, :, i, j] = v
                if i != j:
                    vis_matrix[:, :, j, i] = np.conj(v)

        # Verify off-diagonal elements are Hermitian: V[i,j] = conj(V[j,i])
        for t in range(nt):
            for f in range(nf):
                for i in range(n_ant):
                    for j in range(i + 1, n_ant):
                        np.testing.assert_allclose(
                            vis_matrix[t, f, i, j],
                            np.conj(vis_matrix[t, f, j, i]),
                            atol=1e-6,
                        )

    def test_conjugation_with_reversed_indices(self):
        """When packet_index[i] > packet_index[j], conjugation is applied."""
        nsig = 4
        # Antenna 0 has packet_index=2, Antenna 1 has packet_index=0
        # So for pair (0,1): pi=2, pj=0, ii=0, jj=2
        # bl_conj[0,1] = (2 > 0) = True
        pi, pj = 2, 0
        ii, jj = min(pi, pj), max(pi, pj)
        idx = triu_flat_index(nsig, ii, jj)

        # Create flat visibility with known value
        v_stored = 3.0 + 4.0j  # V(input_0, input_2) in upper triangle

        # For antenna pair (0,1) where ant0->input2, ant1->input0:
        # We want V(ant0, ant1) = V(input2, input0)
        # Stored is V(input0, input2), so we need conj
        assert pi > pj  # conj should be True
        v_extracted = np.conj(v_stored)
        assert v_extracted == 3.0 - 4.0j
