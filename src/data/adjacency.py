"""Adjacency matrix loading and normalization utilities."""

from __future__ import annotations

import pickle

import numpy as np


def load_adj(pkl_path: str) -> tuple[list, dict, np.ndarray]:
    """Load a pickled adjacency file in the METR-LA / PEMS-BAY format.

    Returns *(sensor_ids, sensor_id_to_ind, adj_mx)*.
    The pkl file is expected to contain a 3-tuple of those objects.
    """
    with open(pkl_path, "rb") as f:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding="latin1")
    return sensor_ids, sensor_id_to_ind, adj_mx


def sym_adj(adj: np.ndarray) -> np.ndarray:
    """Symmetric normalisation: D^{-1/2} A D^{-1/2}."""
    adj = np.array(adj, dtype=np.float32)
    d = np.array(adj.sum(axis=1), dtype=np.float32)
    d_inv_sqrt = np.power(np.where(d == 0, 1.0, d), -0.5)
    d_mat = np.diag(d_inv_sqrt)
    return d_mat @ adj @ d_mat


def asym_adj(adj: np.ndarray) -> np.ndarray:
    """Asymmetric normalisation: D^{-1} A (row-stochastic)."""
    adj = np.array(adj, dtype=np.float32)
    d = np.array(adj.sum(axis=1), dtype=np.float32)
    d_inv = np.power(np.where(d == 0, 1.0, d), -1.0)
    return np.diag(d_inv) @ adj


def build_supports(
    adj: np.ndarray,
    transition_type: str = "doubletransition",
) -> list[np.ndarray]:
    """Build support matrices for diffusion GCN.

    Parameters
    ----------
    adj:
        Raw adjacency matrix *(N, N)*.
    transition_type:
        ``"doubletransition"`` — forward + backward transition matrices (used by
        GWN, GWNv2, DSSA-TCN).
        ``"laplacian"`` — single symmetric normalised matrix.
        ``"single"`` — single asym normalised matrix.

    Returns
    -------
    List of numpy arrays, each *(N, N)*.
    """
    if transition_type == "doubletransition":
        return [asym_adj(adj), asym_adj(adj.T)]
    if transition_type == "laplacian":
        return [sym_adj(adj)]
    if transition_type == "single":
        return [asym_adj(adj)]
    raise ValueError(
        f"Unknown transition_type '{transition_type}'. "
        "Expected 'doubletransition', 'laplacian', or 'single'."
    )
