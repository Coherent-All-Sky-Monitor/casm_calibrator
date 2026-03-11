"""Shared test fixtures for casm_calibrator tests."""

import numpy as np
import pytest


@pytest.fixture
def n_ant():
    return 4


@pytest.fixture
def n_chan():
    return 64


@pytest.fixture
def freq_mhz(n_chan):
    """Ascending frequency array spanning 375-468.75 MHz."""
    return np.linspace(375.0, 468.75, n_chan)


@pytest.fixture
def ant_ids(n_ant):
    """1-indexed antenna IDs."""
    return np.array([1, 3, 5, 7])


@pytest.fixture
def rank1_vis(n_ant, n_chan):
    """Synthetic rank-1 visibility matrix with known gains.

    Returns (vis_avg, true_gains) where vis_avg is (n_chan, n_ant, n_ant)
    and true_gains is (n_ant, n_chan) with phase-only gains referenced to ant 0.
    """
    rng = np.random.default_rng(42)

    # Random phase-only gains, ref ant 0 has phase=0
    phases = rng.uniform(-np.pi, np.pi, (n_ant, n_chan))
    phases[0, :] = 0.0  # reference antenna
    true_gains = np.exp(1j * phases)

    # Build rank-1 visibility: V[f,i,j] = g_i * conj(g_j)
    vis_avg = np.zeros((n_chan, n_ant, n_ant), dtype=np.complex128)
    for f in range(n_chan):
        g = true_gains[:, f]
        vis_avg[f] = np.outer(g, np.conj(g))

    return vis_avg, true_gains


@pytest.fixture
def noisy_rank1_vis(rank1_vis, n_ant, n_chan):
    """Rank-1 visibility with small noise added."""
    vis_avg, true_gains = rank1_vis
    rng = np.random.default_rng(123)
    noise = 0.01 * (rng.standard_normal(vis_avg.shape) + 1j * rng.standard_normal(vis_avg.shape))
    vis_noisy = vis_avg + noise
    # Make Hermitian
    for f in range(n_chan):
        vis_noisy[f] = 0.5 * (vis_noisy[f] + vis_noisy[f].conj().T)
    return vis_noisy, true_gains
