import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d, Conv2d, ModuleList, Parameter


def _nconv(x, A):
    return torch.einsum("ncvl,vw->ncwl", x, A).contiguous()


class _GraphConvNet(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super().__init__()
        self.final_conv = Conv2d((order * support_len + 1) * c_in, c_out, (1, 1), bias=True)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support: list):
        out = [x]
        for a in support:
            x1 = _nconv(x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x1 = _nconv(x1, a)
                out.append(x1)
        h = self.final_conv(torch.cat(out, dim=1))
        return F.dropout(h, self.dropout, training=self.training)


class GWNv2(nn.Module):
    """Graph WaveNet v2 with adaptive adjacency, optional feature concatenation,
    and a cleaner modular construction.

    Input shape:  (batch, in_dim, num_nodes, seq_len)
    Output shape: (batch, out_dim, num_nodes, 1)

    Reference: Shleifer et al., "incrementally Improving Graph WaveNet...", 2019.

    Deviation from the vendored original (new_models/Graph-WaveNet-v2/model.py):
    ``residual_convs``/``skip_convs`` are ``Conv2d`` here, where the original
    declares them as ``Conv1d(..., (1, 1))``. This isn't a behavior change —
    it's a forward-compatibility fix for a PyTorch regression. A 2-tuple
    kernel size on ``Conv1d`` has always produced a 4D ``(out, in, 1, 1)``
    weight (``_single`` never enforced tuple length, even in 1.3.1, this
    repo's declared minimum), and on PyTorch <=1.9.1 ``Conv1d.forward`` had
    no input-rank check, so feeding it the original's 4D ``(B, C, N, T)``
    tensor silently ran the exact same correlation a ``Conv2d`` with that
    weight would (verified: bit-identical output to ``Conv2d`` given the
    same weights, on torch 1.9.1). Somewhere between 1.9.1 and 1.13.1,
    PyTorch added an explicit ``input.dim() in {2, 3}`` guard to
    ``Conv1d.forward``, which rejects this same call on every version since.
    ``Conv2d`` reproduces the original's exact numerics on current PyTorch.
    """

    def __init__(
        self,
        device,
        num_nodes,
        dropout=0.3,
        supports=None,
        do_graph_conv=True,
        addaptadj=True,
        aptinit=None,
        in_dim=2,
        out_dim=12,
        residual_channels=32,
        dilation_channels=32,
        cat_feat_gc=False,
        skip_channels=256,
        end_channels=512,
        kernel_size=2,
        blocks=4,
        layers=2,
        apt_size=10,
    ):
        super().__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.do_graph_conv = do_graph_conv
        self.cat_feat_gc = cat_feat_gc
        self.addaptadj = addaptadj

        if self.cat_feat_gc:
            self.start_conv = Conv2d(1, residual_channels, (1, 1))
            self.cat_feature_conv = Conv2d(in_dim - 1, residual_channels, (1, 1))
        else:
            self.start_conv = Conv2d(in_dim, residual_channels, (1, 1))

        self.fixed_supports = supports or []
        self.supports_len = len(self.fixed_supports)

        if do_graph_conv and addaptadj:
            nodevecs = self._init_nodevecs(apt_size, aptinit, num_nodes)
            self.supports_len += 1
            self.nodevec1, self.nodevec2 = [Parameter(n.to(device)) for n in nodevecs]

        depth = list(range(blocks * layers))
        self.residual_convs = ModuleList(
            [Conv2d(dilation_channels, residual_channels, (1, 1)) for _ in depth]
        )
        self.skip_convs = ModuleList(
            [Conv2d(dilation_channels, skip_channels, (1, 1)) for _ in depth]
        )
        self.bn = ModuleList([BatchNorm2d(residual_channels) for _ in depth])
        self.graph_convs = ModuleList(
            [
                _GraphConvNet(
                    dilation_channels, residual_channels, dropout, support_len=self.supports_len
                )
                for _ in depth
            ]
        )

        self.filter_convs = ModuleList()
        self.gate_convs = ModuleList()
        receptive_field = 1
        for _ in range(blocks):
            additional_scope = kernel_size - 1
            D = 1
            for _ in range(layers):
                self.filter_convs.append(
                    Conv2d(residual_channels, dilation_channels, (1, kernel_size), dilation=D)
                )
                self.gate_convs.append(
                    Conv2d(residual_channels, dilation_channels, (1, kernel_size), dilation=D)
                )
                D *= 2
                receptive_field += additional_scope
                additional_scope *= 2
        self.receptive_field = receptive_field

        self.end_conv_1 = Conv2d(skip_channels, end_channels, (1, 1), bias=True)
        self.end_conv_2 = Conv2d(end_channels, out_dim, (1, 1), bias=True)

    @staticmethod
    def _init_nodevecs(apt_size, aptinit, num_nodes):
        if aptinit is None:
            return torch.randn(num_nodes, apt_size), torch.randn(apt_size, num_nodes)
        m, p, n = torch.svd(aptinit)
        nv1 = torch.mm(m[:, :apt_size], torch.diag(p[:apt_size] ** 0.5))
        nv2 = torch.mm(torch.diag(p[:apt_size] ** 0.5), n[:, :apt_size].t())
        return nv1, nv2

    def load_checkpoint(self, state_dict):
        """Load a checkpoint trained for a different seq_length (only end_conv_2 changes)."""
        bk, wk = "end_conv_2.bias", "end_conv_2.weight"
        b, w = state_dict.pop(bk), state_dict.pop(wk)
        self.load_state_dict(state_dict, strict=False)
        cur = self.state_dict()
        cur[bk][: b.shape[0]] = b
        cur[wk][: w.shape[0]] = w
        self.load_state_dict(cur)

    def forward(self, x):
        in_len = x.size(3)
        if in_len < self.receptive_field:
            x = F.pad(x, (self.receptive_field - in_len, 0, 0, 0))

        if self.cat_feat_gc:
            x = self.start_conv(x[:, [0]]) + F.leaky_relu(self.cat_feature_conv(x[:, 1:]))
        else:
            x = self.start_conv(x)

        skip = 0
        adj_matrices = self.fixed_supports
        if self.addaptadj:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            adj_matrices = self.fixed_supports + [adp]

        last = self.blocks * self.layers - 1
        for i in range(self.blocks * self.layers):
            residual = x
            x = torch.tanh(self.filter_convs[i](residual)) * torch.sigmoid(
                self.gate_convs[i](residual)
            )

            s = self.skip_convs[i](x)
            skip = s if isinstance(skip, int) else skip[:, :, :, -s.size(3) :] + s

            if i == last:
                break

            if self.do_graph_conv:
                gc_out = self.graph_convs[i](x, adj_matrices)
                x = x + gc_out if self.cat_feat_gc else gc_out
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3) :]
            x = self.bn[i](x)

        x = F.relu(self.end_conv_1(F.relu(skip)))
        return self.end_conv_2(x)
