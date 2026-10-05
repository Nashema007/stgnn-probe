"""D2STGNN: Decoupled Dynamic Spatial-Temporal Graph Neural Network.

Ported from the D2STGNN baseline (Shao et al., VLDB 2022):
    https://github.com/zezhishao/D2STGNN

Only the ``models/`` subpackage is ported (decouple/, diffusion_block/,
dynamic_graph_conv/, inherent_block/, model.py). The original repo's CLI
(main.py), dataloader/, configs/, datasets/, and training-loop utils/ are not
needed by this framework and were not ported.
"""

from .model import D2STGNN, DecoupleLayer, make_d2stgnn

__all__ = ["D2STGNN", "DecoupleLayer", "make_d2stgnn"]
