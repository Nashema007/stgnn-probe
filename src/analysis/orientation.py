"""Canonical edge orientation for learned dependency matrices.

Convention used by every structural comparison (Lens 2-3, AAS): ``W[i, j]`` is
the weight of **source i -> target j**, so column ``j`` holds target ``j``'s
incoming edges. This is the orientation of the Granger reference, whose
``fstats[i, j]`` tests "i Granger-causes j", and the top-k reduction
(``lens2_granger.build_gcg_topk``) ranks within columns.

Exporters save each model's matrix in its native orientation, which is not the
same across models:

* GWN v2 (``einsum('ncvl,vw->ncwl')``) and STAWnet (``x @ att``) aggregate along
  columns, so they are already canonical.
* DSSA-TCN (``attn_weights @ value``) and D2STGNN (``matmul(graph, X)``)
  aggregate along rows (row ``i`` is target ``i``), so they are transposed.
* GWN (tsl) applies its learned matrix in both directions (forward support and
  its transpose), and the STAEformer and BigST exports are similarity proxies
  that are never used to propagate, so they have no native direction and are
  read as exported.

Until the camera-ready revision, DSSA-TCN and D2STGNN were scored without the
transpose, which compared their outgoing edges with the reference's incoming
edges. Apply this function after any row normalisation, which assumes the
native orientation.
"""

from __future__ import annotations

import numpy as np

ROWS_ARE_TARGETS: frozenset[str] = frozenset({"dssa_tcn", "d2stgnn"})


def to_incoming_convention(matrix: np.ndarray, model_name: str) -> np.ndarray:
    """Return ``matrix`` oriented so that entry ``[i, j]`` is the weight of i -> j."""
    return np.asarray(matrix).T if model_name in ROWS_ARE_TARGETS else np.asarray(matrix)
