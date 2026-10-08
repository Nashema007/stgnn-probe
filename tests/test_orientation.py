"""Edge-orientation contract behind AAS: W[i, j] is the weight of source i -> target j.

The native-propagation tests pin which axis each model aggregates over, so a
model whose exported matrix is read in the wrong direction (the DSSA-TCN and
D2STGNN bug fixed for the camera-ready) fails here rather than silently.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from analysis.orientation import ROWS_ARE_TARGETS, to_incoming_convention
from models.d2stgnn.diffusion_block.dif_model import STLocalizedConv
from models.dssa_tcn import _AttentionLayer
from models.gwn_v2 import _nconv

N = 6


def test_only_row_target_models_are_transposed() -> None:
    m = np.arange(N * N, dtype=float).reshape(N, N)
    assert set(ROWS_ARE_TARGETS) == {"dssa_tcn", "d2stgnn"}
    for name in ("dssa_tcn", "d2stgnn"):
        np.testing.assert_array_equal(to_incoming_convention(m, name), m.T)
    for name in ("gwn", "gwn_v2", "stawnet", "staeformer", "bigst"):
        np.testing.assert_array_equal(to_incoming_convention(m, name), m)


def test_gwn_v2_aggregates_along_columns() -> None:
    # A single edge A[1, 0]: column-as-target means source 1 feeds target 0.
    a = torch.zeros(N, N)
    a[1, 0] = 1.0
    x = torch.arange(N, dtype=torch.float32).view(1, 1, N, 1)
    out = _nconv(x, a)
    assert out[0, 0, 0, 0] == x[0, 0, 1, 0]
    assert out[0, 0, 1, 0] == 0.0


def test_d2stgnn_aggregates_along_rows() -> None:
    # A single edge G[0, 1]: row-as-target means source 1 feeds target 0.
    g = torch.zeros(N, N)
    g[0, 1] = 1.0
    x = torch.arange(N, dtype=torch.float32).view(1, N, 1)
    identity = SimpleNamespace(gcn_updt=lambda t: t, dropout=lambda t: t)
    out = STLocalizedConv.gconv(identity, [g], x, torch.zeros_like(x))
    propagated = out[..., 1]
    assert propagated[0, 0] == x[0, 1, 0]
    assert propagated[0, 1] == 0.0


def test_dssa_tcn_attention_aggregates_along_rows() -> None:
    torch.manual_seed(0)
    layer = _AttentionLayer(model_dim=8, num_heads=1, use_topk=True)
    x = torch.randn(1, N, 8)
    value = x.clone().requires_grad_(True)
    out, weights = layer(x, x, value, return_attention=True)
    w = weights[0].detach()
    assert (w == 0).any(), "top-k should leave some edges exactly zero"
    for target in range(N):
        (grad,) = torch.autograd.grad(out[0, target].sum(), value, retain_graph=True)
        reaches = grad[0].abs().sum(dim=-1) > 0
        # Output row `target` depends on exactly the sources in weight ROW `target`.
        assert torch.equal(reaches, w[target] > 0)


def _load_finalize_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "finalize_paper_figures.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("finalize_paper_figures", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_aas_scores_row_target_exports_after_transposing(tmp_path: Path) -> None:
    fpf = _load_finalize_script()
    rng = np.random.default_rng(0)
    source = tmp_path / "granger_src"
    (source / "granger_cache").mkdir(parents=True)
    dirs = {"METR-LA": "metr_la", "PEMS-BAY": "pems_bay"}
    for ds, n in fpf.NUM_NODES.items():
        fstats = rng.random((n, n)) + 0.1
        np.savez(source / "granger_cache" / f"{ds}.npz", fstats=fstats)
        adj_dir = tmp_path / "data" / "probe_inputs" / dirs[ds] / "adjacency"
        adj_dir.mkdir(parents=True)
        for slug in fpf.MODEL_SLUGS:
            # Every model's export equals the reference in its own native
            # orientation, so a correct reading scores AAS = 1 for all of them.
            native = fstats.T if slug in ROWS_ARE_TARGETS else fstats
            for h in (6, 12, 42):
                np.save(adj_dir / f"{slug}_adjacency_h{h}_seeds.npy", native[None])
    aas, uniform = fpf._compute_aas(source, tmp_path)
    for ds in fpf.NUM_NODES:
        assert aas[ds] == pytest.approx([1.0] * len(fpf.MODEL_SLUGS))
        assert uniform[ds] == 0
