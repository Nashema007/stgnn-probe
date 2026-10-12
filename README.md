# STGNN-Probe: Analysing STGNNs for Long-Horizon Traffic Forecasting

Code for the paper **Analysing Spatio-Temporal Graph Neural Networks for Long-Horizon Traffic
Forecasting** by Ashley Nash C. Chakanyuka, Henok Tenaw Moges and Deshendran Moodley (Centre for
Artificial Intelligence Research, University of Cape Town), SACAIR 2026 (to appear).

The paper compares seven spatio-temporal graph neural networks (STGNNs) on METR-LA and PEMS-BAY at
30, 60 and 210 minutes, using two measures:

- **Spatial Gain Score (SGS):** each STGNN's forecasting gain over a per-sensor temporal baseline
  (a TCN that sees only that sensor's history).
- **Adjacency Alignment Score (AAS):** how closely each STGNN's strongest learned sensor
  dependencies overlap with a Granger predictive reference built from the observed traffic.

The repository has two parts:

| Part | Entry point | What it does |
|---|---|---|
| Training | `scripts/run_training.py` | Trains the STGNNs and baselines, one model per horizon and run, and exports test predictions and each model's learned N x N dependency matrix |
| STGNN-Probe | `run_probe.py` | Computes MAE, SGS, the Granger reference and AAS from those exports |

## Models

| Model | Paper |
|---|---|
| Graph WaveNet (GWN) | Wu et al., IJCAI 2019 |
| Graph WaveNet v2 (GWN v2) | Shleifer et al., arXiv 2019 |
| STAWnet | Tian and Chan, IET Intelligent Transport Systems 2021 |
| DSSA-TCN | Zhang et al., PLOS ONE 2025 |
| STAEformer | Liu et al., CIKM 2023 |
| D2STGNN | Shao et al., PVLDB 2022 |
| BigST | Han et al., PVLDB 2024 |
| Per-sensor TCN (baseline) | Bai et al., arXiv 2018 |
| ARIMA (baseline) | — |

Each model is a port of the original implementation, wrapped by an adapter in
`src/training/adapters.py`. Upstream sources and licences are listed in `THIRD_PARTY_NOTICES.md`.

## Setup

Python 3.12 with [`uv`](https://docs.astral.sh/uv/). `torch-scatter` and `torch-sparse` come from
PyG's wheel index, so they are installed separately:

```bash
uv sync
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.2+cpu.html
```

These two packages have no macOS wheel. On macOS, run every command through `./test.sh`, which
executes it inside a Linux Docker image (for example `./test.sh python run_probe.py ...`). Running
`./test.sh` with no arguments runs the linters, type checks and tests. For GPU builds and other
details, see [docs/development.md](docs/development.md).

## Reproducing the paper

Run these steps in order from the repository root.

| Step | Command | Output |
|---|---|---|
| 1. Data | Nothing to do: `tsl` downloads METR-LA and PEMS-BAY on first use | `data/tsl_cache/` |
| 2. Train | `python scripts/run_training.py --config scripts/configs/metr_la_training.yaml`, then the same with `pems_bay_training.yaml` | `checkpoints/`, `data/probe_inputs/` |
| 3. Analyse | `python run_probe.py --config scripts/configs/metr_la_probe.yaml --all --force-granger`, then the same with `pems_bay_probe.yaml` | `outputs/` |
| 4. Figures and AAS | `python scripts/finalize_paper_figures.py --aas-source outputs --no-manuscript-check` | `figures/` |

Step 2 is the expensive one: seven STGNNs, the per-sensor TCN bank and ARIMA, over three horizons
and three runs, on both datasets. Use a GPU. Retrained models will not match the paper to the third
decimal, which is why step 4 is run with `--no-manuscript-check`.

### Where each result comes from

| Paper item | File |
|---|---|
| Table 4 (test MAE) | `outputs/lens0_performance/<DATASET>/metrics_by_horizon.csv` |
| Table 5 (SGS and sensor classes) | `outputs/per_model/<model>_<DATASET>/summary.json` and `node_classification.csv` |
| Fig. 1 (SGS by horizon) | `figures/fig1_*` |
| Fig. 2 (AAS) | `figures/fig2_*`; the values are printed by `finalize_paper_figures.py` |
| Granger reference | `outputs/granger_cache/<DATASET>.npz` |

Take AAS from `finalize_paper_figures.py`, not from the probe's `alignment_scores.json`. The script
scores an exactly uniform learned matrix at the random-selection expectation `k/(N-1)`, as the paper
does; the probe breaks those ties arbitrarily.

The exact settings (input window, horizons, split, Granger lag, `k`), the edge-orientation convention
used for AAS, and which parts of the code the paper does and does not use are documented in
[docs/paper.md](docs/paper.md).

## Repository layout

```text
run_probe.py                  STGNN-Probe entry point
scripts/
  run_training.py             train one or all models on a dataset
  finalize_paper_figures.py   paper Figs. 1-2 and the reported AAS
  configs/                    experiment configs; models/<model>_base.yaml holds each model's settings
src/
  models/                     model implementations
  training/                   training loop and the per-model adapters
  analysis/                   STGNN-Probe (Lens 0 MAE, Lens 1 SGS, Lens 2 Granger, Lens 3 AAS)
  data/                       dataset loading through tsl
tests/                        test suite (./test.sh)
docs/                         detailed documentation (below)
```

## Documentation

- [docs/paper.md](docs/paper.md): paper configuration, edge convention, and what the paper uses
- [docs/training.md](docs/training.md): training options, configs and checkpoints
- [docs/probe.md](docs/probe.md): STGNN-Probe lenses, inputs, configuration and outputs
- [docs/development.md](docs/development.md): environment, Docker, GPU builds, checks and the full
  repository structure
- [src/analysis/README.md](src/analysis/README.md): the analysis lenses in detail

## Citation

```bibtex
@inproceedings{chakanyuka2026stgnnprobe,
  author    = {Chakanyuka, Ashley Nash C. and Moges, Henok Tenaw and Moodley, Deshendran},
  title     = {Analysing Spatio-Temporal Graph Neural Networks for Long-Horizon Traffic Forecasting},
  booktitle = {Southern African Conference for Artificial Intelligence Research (SACAIR 2026)},
  year      = {2026},
  note      = {To appear}
}
```

## License

MIT (see `LICENSE`). Ported model implementations remain credited to their original authors; see
`THIRD_PARTY_NOTICES.md`.
