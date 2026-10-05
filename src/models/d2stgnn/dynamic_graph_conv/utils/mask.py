"""Predefined-graph masking used by the dynamic graph constructor.

Ported verbatim from D2STGNN (Shao et al., VLDB 2022),
``new_models/D2STGNN/models/dynamic_graph_conv/utils/mask.py``.

BUGFIX-2026-07-02 (tag: BUGFIX-2026-07-02): ``forward()`` unconditionally
indexes ``self.mask[0]``/``self.mask[1]`` (the 2 fixed distance modalities
from ``DistanceFunction``), which previously raised a cryptic ``IndexError``
if fewer than 2 predefined adjacency matrices were supplied. Added an
explicit precondition check in ``__init__`` instead, so it fails at
model-build time with an actionable message. No-op for the shipped
``doubletransition`` default (2 supports); only triggers for ``adj_type``
overrides that yield < 2 supports, which crashed regardless before this fix
(see the paired ``num_matric`` fix in ``diffusion_block/dif_model.py``, same
tag/date).
"""

import torch
import torch.nn as nn


class Mask(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.mask = model_args["adjs"]
        # BUGFIX-2026-07-02: see module docstring.
        if len(self.mask) < 2:
            raise ValueError(
                "D2STGNN's dynamic graph masking pairs each of the 2 fixed distance "
                f"modalities (see DistanceFunction) with its own predefined adjacency "
                f"matrix, requiring at least 2; got {len(self.mask)}. Use "
                "adj_type='doubletransition' (or otherwise supply >=2 adjacency "
                "matrices) instead of 'single'/'laplacian'."
            )

    def _mask(self, index, adj):
        mask = self.mask[index] + torch.ones_like(self.mask[index]) * 1e-7
        return mask.to(adj.device) * adj

    def forward(self, adj):
        result = []
        for index, _ in enumerate(adj):
            result.append(self._mask(index, _))
        return result
