"""Shared random-feature-map math for BigST and its long-term pretraining branch.

Ported from ``new_models/BigST/model.py`` (VLDB 2024). These helpers implement
a linear-complexity approximation of softmax attention via random feature
maps (performer-style), used both by the main ``BigST`` spatial convolution
(``src/models/bigst.py``) and by the long-term pretraining transformer
(``src/models/bigst_longterm.py``). Kept in one place so neither file
duplicates the math.

Bug fix vs. the original repo: ``create_products_of_givens_rotations`` used
``np.eye``/``np.random`` without ever importing numpy. Fixed here by adding
the import.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def create_products_of_givens_rotations(dim: int, seed: int) -> torch.Tensor:
    """Build a structured orthogonal matrix via products of Givens rotations.

    This faithfully preserves the upstream BigST branch, including its
    apparent rotation formula typo. It is only used when ``struct_mode=True``;
    no current framework path enables that mode.
    """
    nb_givens_rotations = dim * int(math.ceil(math.log(float(dim))))
    q = np.eye(dim, dim)
    np.random.seed(seed)
    for _ in range(nb_givens_rotations):
        random_angle = math.pi * np.random.uniform()
        random_indices = np.random.choice(dim, 2)
        index_i = min(random_indices[0], random_indices[1])
        index_j = max(random_indices[0], random_indices[1])
        slice_i = q[index_i]
        slice_j = q[index_j]
        new_slice_i = math.cos(random_angle) * slice_i + math.cos(random_angle) * slice_j
        new_slice_j = -math.sin(random_angle) * slice_i + math.cos(random_angle) * slice_j
        q[index_i] = new_slice_i
        q[index_j] = new_slice_j
    return torch.tensor(q, dtype=torch.float32)


def create_random_matrix(
    m: int,
    d: int,
    seed: int = 0,
    scaling: int = 0,
    struct_mode: bool = False,
) -> torch.Tensor:
    """Build an (m, d) random projection matrix from stacked orthogonal blocks.

    This mirrors upstream BigST by calling ``torch.manual_seed`` internally.
    Callers derive seeds from activations during forward passes, so the global
    RNG state is intentionally mutated as part of the faithful port.
    """
    nb_full_blocks = int(m / d)
    block_list = []
    current_seed = seed
    for _ in range(nb_full_blocks):
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))
            q, _ = torch.linalg.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q)
        current_seed += 1
    remaining_rows = m - nb_full_blocks * d
    if remaining_rows > 0:
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))
            q, _ = torch.linalg.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q[0:remaining_rows])
    final_matrix = torch.vstack(block_list)

    current_seed += 1
    torch.manual_seed(current_seed)
    if scaling == 0:
        multiplier = torch.norm(torch.randn((m, d)), dim=1)
    elif scaling == 1:
        multiplier = torch.sqrt(torch.tensor(float(d))) * torch.ones(m)
    else:
        raise ValueError(f"Scaling must be one of {{0, 1}}. Was {scaling}")

    return torch.matmul(torch.diag(multiplier), final_matrix)


def random_feature_map(
    data: torch.Tensor,
    is_query: bool,
    projection_matrix: torch.Tensor,
    numerical_stabilizer: float = 0.000001,
) -> torch.Tensor:
    """Performer-style random feature map approximating the softmax kernel."""
    last_dim = torch.tensor(data.shape[-1], dtype=torch.float32)
    data_normalizer = 1.0 / torch.sqrt(torch.sqrt(last_dim))
    data = data_normalizer * data
    ratio = 1.0 / torch.sqrt(torch.tensor(projection_matrix.shape[0], dtype=torch.float32))
    data_dash = torch.einsum("bnhd,md->bnhm", data, projection_matrix)
    diag_data = torch.square(data)
    diag_data = torch.sum(diag_data, dim=len(data.shape) - 1)
    diag_data = diag_data / 2.0
    diag_data = torch.unsqueeze(diag_data, dim=len(data.shape) - 1)
    last_dims_t = len(data_dash.shape) - 1
    attention_dims_t = len(data_dash.shape) - 3
    if is_query:
        max_data_dash = torch.max(data_dash, dim=last_dims_t, keepdim=True)[0]
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - max_data_dash) + numerical_stabilizer
        )
    else:
        data_dash = ratio * (
            torch.exp(
                data_dash
                - diag_data
                - torch.max(
                    torch.max(data_dash, dim=last_dims_t, keepdim=True)[0],
                    dim=attention_dims_t,
                    keepdim=True,
                )[0]
            )
            + numerical_stabilizer
        )
    return data_dash


# Alias matching the long-term branch's original naming
# (``new_models/BigST/preprocess/model.py::create_projection_matrix`` /
# ``softmax_kernel_transformation``) so both call sites can use whichever
# name reads better while sharing one implementation.
create_projection_matrix = create_random_matrix
softmax_kernel_transformation = random_feature_map


def numerator(qs: torch.Tensor, ks: torch.Tensor, vs: torch.Tensor) -> torch.Tensor:
    kvs = torch.einsum("nbhm,nbhd->bhmd", ks, vs)
    return torch.einsum("nbhm,bhmd->nbhd", qs, kvs)


def denominator(qs: torch.Tensor, ks: torch.Tensor) -> torch.Tensor:
    all_ones = torch.ones([ks.shape[0]]).to(qs.device)
    ks_sum = torch.einsum("nbhm,n->bhm", ks, all_ones)
    return torch.einsum("nbhm,bhm->nbh", qs, ks_sum)


def linearized_softmax(x: torch.Tensor, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    # x: [B, N, H, D] query: [B, N, H, m], key: [B, N, H, m]
    query = query.permute(1, 0, 2, 3)  # [N, B, H, m]
    key = key.permute(1, 0, 2, 3)  # [N, B, H, m]
    x = x.permute(1, 0, 2, 3)  # [N, B, H, D]

    z_num = numerator(query, key, x)  # [N, B, H, D]
    z_den = denominator(query, key)  # [N, H]

    z_num = z_num.permute(1, 0, 2, 3)  # [B, N, H, D]
    z_den = z_den.permute(1, 0, 2)
    z_den = torch.unsqueeze(z_den, len(z_den.shape))
    z_output = z_num / z_den  # [B, N, H, D]

    return z_output


def linear_kernel(
    x: torch.Tensor, node_vec1: torch.Tensor, node_vec2: torch.Tensor
) -> torch.Tensor:
    """Linear-complexity spatial-convolution kernel used by BigST's main model.

    x: [B, N, 1, nhid], node_vec1: [B, N, 1, r], node_vec2: [B, N, 1, r]
    """
    node_vec1 = node_vec1.permute(1, 0, 2, 3)  # [N, B, 1, r]
    node_vec2 = node_vec2.permute(1, 0, 2, 3)  # [N, B, 1, r]
    x = x.permute(1, 0, 2, 3)  # [N, B, 1, nhid]

    v2x = torch.einsum("nbhm,nbhd->bhmd", node_vec2, x)
    out1 = torch.einsum("nbhm,bhmd->nbhd", node_vec1, v2x)  # [N, B, 1, nhid]

    one_matrix = torch.ones([node_vec2.shape[0]]).to(node_vec1.device)
    node_vec2_sum = torch.einsum("nbhm,n->bhm", node_vec2, one_matrix)
    out2 = torch.einsum("nbhm,bhm->nbh", node_vec1, node_vec2_sum)  # [N, 1]

    out1 = out1.permute(1, 0, 2, 3)  # [B, N, 1, nhid]
    out2 = out2.permute(1, 0, 2)
    out2 = torch.unsqueeze(out2, len(out2.shape))
    out = out1 / out2  # [B, N, 1, nhid]

    return out


def spatial_loss(
    node_vec1: torch.Tensor,
    node_vec2: torch.Tensor,
    supports: list[torch.Tensor],
    edge_indices: torch.Tensor,
) -> torch.Tensor:
    """Spatial regularisation loss pulling the learned attention towards the graph."""
    B = node_vec1.size(0)
    node_vec1 = node_vec1.permute(1, 0, 2, 3)  # [N, B, 1, r]
    node_vec2 = node_vec2.permute(1, 0, 2, 3)  # [N, B, 1, r]

    node_vec1_end, node_vec2_start = (
        node_vec1[edge_indices[:, 0]],
        node_vec2[edge_indices[:, 1]],
    )  # [E, B, 1, r]
    attn1 = torch.einsum("ebhm,ebhm->ebh", node_vec1_end, node_vec2_start)  # [E, B, 1]
    attn1 = attn1.permute(1, 0, 2)  # [B, E, 1]

    one_matrix = torch.ones([node_vec2.shape[0]]).to(node_vec1.device)
    node_vec2_sum = torch.einsum("nbhm,n->bhm", node_vec2, one_matrix)
    attn_norm = torch.einsum("nbhm,bhm->nbh", node_vec1, node_vec2_sum)

    attn2 = attn_norm[edge_indices[:, 0]]  # [E, B, 1]
    attn2 = attn2.permute(1, 0, 2)  # [B, E, 1]
    attn_score = attn1 / attn2  # [B, E, 1]

    d_norm = supports[0][edge_indices[:, 0], edge_indices[:, 1]]
    d_norm = d_norm.reshape(1, -1, 1).repeat(B, 1, attn_score.shape[-1])
    loss = torch.mean(attn_score.log() * d_norm)

    return loss


class ConvApproximation(nn.Module):
    """Random-feature approximation of the linear_kernel attention (main model)."""

    def __init__(self, dropout: float, tau: float, random_feature_dim: int) -> None:
        super().__init__()
        self.tau = tau
        self.random_feature_dim = random_feature_dim
        self.activation = nn.ReLU()
        self.dropout = dropout

    def forward(self, x: torch.Tensor, node_vec1: torch.Tensor, node_vec2: torch.Tensor):
        dim = node_vec1.shape[-1]  # (N, 1, d)

        random_seed = int(torch.ceil(torch.abs(torch.sum(node_vec1) * 1e8)).item())
        random_matrix = create_random_matrix(self.random_feature_dim, dim, seed=random_seed).to(
            node_vec1.device
        )  # (d, r)

        node_vec1 = node_vec1 / math.sqrt(self.tau)
        node_vec2 = node_vec2 / math.sqrt(self.tau)
        node_vec1_prime = random_feature_map(node_vec1, True, random_matrix)  # [B, N, 1, r]
        node_vec2_prime = random_feature_map(node_vec2, False, random_matrix)  # [B, N, 1, r]

        x = linear_kernel(x, node_vec1_prime, node_vec2_prime)

        return x, node_vec1_prime, node_vec2_prime


class LinearizedConv(nn.Module):
    """Gated linear-complexity spatial convolution block used by BigST's main model."""

    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        dropout: float,
        tau: float = 1.0,
        random_feature_dim: int = 64,
    ) -> None:
        super().__init__()

        self.dropout = dropout
        self.tau = tau
        self.random_feature_dim = random_feature_dim

        self.input_fc = nn.Conv2d(
            in_channels=in_dim, out_channels=hid_dim, kernel_size=(1, 1), bias=True
        )
        self.output_fc = nn.Conv2d(
            in_channels=in_dim, out_channels=hid_dim, kernel_size=(1, 1), bias=True
        )
        self.activation = nn.Sigmoid()
        self.dropout_layer = nn.Dropout(p=dropout)

        self.conv_app_layer = ConvApproximation(self.dropout, self.tau, self.random_feature_dim)

    def forward(self, input_data: torch.Tensor, node_vec1: torch.Tensor, node_vec2: torch.Tensor):
        x = self.input_fc(input_data)
        x = self.activation(x) * self.output_fc(input_data)
        x = self.dropout_layer(x)

        x = x.permute(0, 2, 3, 1)  # (B, N, 1, dim*4)
        x, node_vec1_prime, node_vec2_prime = self.conv_app_layer(x, node_vec1, node_vec2)
        x = x.permute(0, 3, 1, 2)  # (B, dim*4, N, 1)

        return x, node_vec1_prime, node_vec2_prime


class LinearizedAttention(nn.Module):
    """Multi-head random-feature attention used by the long-term pretraining branch."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        dropout: float,
        random_feature_dim: int = 30,
        tau: float = 1.0,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.Wk = nn.Linear(c_in, c_out * num_heads)
        self.Wq = nn.Linear(c_in, c_out * num_heads)
        self.Wv = nn.Linear(c_in, c_out * num_heads)
        self.Wo = nn.Linear(c_out * num_heads, c_out)
        self.c_in = c_in
        self.c_out = c_out
        self.num_heads = num_heads
        self.tau = tau
        self.random_feature_dim = random_feature_dim
        self.activation = nn.ReLU
        self.dropout = dropout

    def reset_parameters(self) -> None:
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        self.Wv.reset_parameters()
        self.Wo.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)  # (B, T, D)
        query = self.Wq(x).reshape(-1, T, self.num_heads, self.c_out)  # (B, T, H, D)
        key = self.Wk(x).reshape(-1, T, self.num_heads, self.c_out)  # (B, T, H, D)
        x = self.Wv(x).reshape(-1, T, self.num_heads, self.c_out)  # (B, T, H, D)

        dim = query.shape[-1]
        seed = int(torch.ceil(torch.abs(torch.sum(query) * 1e8)).item())
        projection_matrix = create_random_matrix(self.random_feature_dim, dim, seed=seed).to(
            query.device
        )  # (d, m)
        query = query / math.sqrt(self.tau)
        key = key / math.sqrt(self.tau)
        query = random_feature_map(query, True, projection_matrix)  # [B, T, H, m]
        key = random_feature_map(key, False, projection_matrix)  # [B, T, H, m]

        x = linearized_softmax(x, query, key)

        x = self.Wo(x.flatten(-2, -1))  # (B, T, D)

        return x
