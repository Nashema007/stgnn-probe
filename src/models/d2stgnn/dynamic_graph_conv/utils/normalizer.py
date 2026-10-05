"""Graph row-normalization and multi-order (k-hop) expansion.

Ported from D2STGNN (Shao et al., VLDB 2022),
``new_models/D2STGNN/models/dynamic_graph_conv/utils/normalizer.py``.

Change from the original: ``from utils.cal_adj import remove_nan_inf`` (an
absolute import into the original repo's standalone ``utils`` package, which
is not ported into this framework) is replaced with a local
``_remove_nan_inf`` helper with identical behaviour.
"""

import torch
import torch.nn as nn


def _remove_nan_inf(tensor: torch.Tensor) -> torch.Tensor:
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    return tensor


class Normalizer(nn.Module):
    def __init__(self):
        super().__init__()

    def _norm(self, graph):
        degree = torch.sum(graph, dim=2)
        degree = _remove_nan_inf(1 / degree)
        degree = torch.diag_embed(degree)
        normed_graph = torch.bmm(degree, graph)
        return normed_graph

    def forward(self, adj):
        return [self._norm(_) for _ in adj]


class MultiOrder(nn.Module):
    def __init__(self, order=2):
        super().__init__()
        self.order = order

    def _multi_order(self, graph):
        graph_ordered = []
        k_1_order = graph  # 1 order
        mask = torch.eye(graph.shape[1]).to(graph.device)
        mask = 1 - mask
        graph_ordered.append(k_1_order * mask)
        for _k in range(2, self.order + 1):  # e.g., order = 3, k=[2, 3]; order = 2, k=[2]
            k_1_order = torch.matmul(k_1_order, graph)
            graph_ordered.append(k_1_order * mask)
        return graph_ordered

    def forward(self, adj):
        return [self._multi_order(_) for _ in adj]
