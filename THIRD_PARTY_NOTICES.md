# Third-party notices

The MIT license in `LICENSE` covers the original code in this repository. The model
implementations below are ports of third-party code; each module's docstring
documents what was changed from the original. All credit for the original
architectures belongs to their authors. Please cite the original papers.

| Module | Upstream | Upstream license |
| --- | --- | --- |
| `src/models/gwn_v2.py`, GWN via `tsl` | Graph WaveNet — https://github.com/nnzhan/Graph-WaveNet (Wu et al., IJCAI 2019) | MIT, Copyright (c) 2019 Zonghan Wu |
| `src/models/dssa_tcn.py` | BasicTS DSSA-TCN baseline — https://github.com/GestaltCogTeam/BasicTS | Apache-2.0 |
| `src/models/d2stgnn/` | D2STGNN — https://github.com/GestaltCogTeam/D2STGNN (Shao et al., VLDB 2022) | No license file published upstream |
| `src/models/stawnet.py` | STAWnet — https://github.com/CYBruce/STAWnet | No license file published upstream |
| `src/models/bigst.py`, `bigst_common.py`, `bigst_longterm.py` | BigST — https://github.com/usail-hkust/BigST (Han et al., VLDB 2024) | No license file published upstream |
| `src/models/staeformer.py` | STAEformer — https://github.com/XDZhelheim/STAEformer (Liu et al., CIKM 2023) | No license file published upstream |

Training and data pipelines depend on
[torch-spatiotemporal (tsl)](https://github.com/TorchSpatiotemporal/tsl) (MIT), which is
installed as a dependency and is not vendored here.

The METR-LA and PEMS-BAY datasets are not redistributed. They are downloaded by
`tsl` at run time, and you must follow their original terms of use.
