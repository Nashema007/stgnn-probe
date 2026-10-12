# Development

## Environment Setup

This project uses `uv` for environment and dependency management. Use one local environment:
`.venv` on Python 3.12.

```bash
/usr/local/bin/uv venv .venv --python 3.12
```

Activate it:

```bash
source .venv/bin/activate
```

Install the full project dependency set:

```bash
/usr/local/bin/uv sync
```

`torch_scatter`/`torch_sparse` (required transitively by `torch-spatiotemporal`'s
`GraphWaveNetModel`/`TCNModel`, hence by `data.tsl_pipeline`) are binary extensions
distributed from PyG's own wheel index rather than PyPI, so `uv sync` cannot resolve
them and they are deliberately left out of `pyproject.toml`'s `dependencies`. Install
them as a required second step (same command CI and the Docker dev image use):

```bash
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
```

Plotly/Kaleido needs Chrome for static PNG export:

```bash
.venv/bin/plotly_get_chrome -y
```

If `plotly_get_chrome` fails with a local certificate error on macOS, retry with Python's certifi
bundle:

```bash
SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')" .venv/bin/plotly_get_chrome -y
```

## PyTorch Backend Support

The project pins a Python 3.12 CPU-compatible baseline (`torch==2.8.0`, `numpy>=1.26,<2`) so local
Apple Silicon macOS and Linux CI can share one dependency policy. Intel Mac (`darwin x86_64`) is not
supported — PyTorch stopped shipping wheels for that platform after `2.2.2`, and `torch>=2.7` is
required for NVIDIA Blackwell (sm_120) GPU support.

- CPU is always supported and is the default fallback.
- CUDA is supported at runtime on machines with a CUDA-capable PyTorch install.
- MPS is supported at runtime on Apple Silicon macOS when `torch.backends.mps.is_available()`.

**Running the test suite requires Linux** (CI or the `stgnn-tsl-dev` Docker image — see
`Dockerfile`) despite the platform list above. `src/evaluation/metrics.py` imports
`tsl.metrics.torch` at module level, and `src/training/__init__.py` imports `adapters.py`, which
imports `torch_geometric` (and transitively `tsl`) at module level — so `torch_scatter`/
`torch_sparse` (no prebuilt wheel for macOS + Python 3.12 + torch 2.8.0) are a hard prerequisite for
the entire suite, not just the tsl-pipeline-specific tests. This is enforced explicitly:
`tests/conftest.py` checks for both packages at the start of the pytest session and exits
immediately with installation instructions if either is missing, rather than letting collection
fail with a cryptic `ModuleNotFoundError` deep inside whichever test file happens to import `tsl`
first.

### Running lint/typecheck/tests

**macOS (or any platform without PyG wheels) — via Docker:** use `./test.sh`, which builds the
`stgnn-tsl-dev` image on first run (skips the build on later runs unless `FORCE_REBUILD=1` is set)
and runs commands inside it:

```bash
./test.sh                    # full verification: ruff check, ruff format --check, mypy, pytest
./test.sh pytest tests/ -v   # or any other command, run inside the container
./test.sh ruff check src/
```

**Native Linux:** the same `./test.sh` works unmodified (the image is Linux-based regardless of
host OS), or skip Docker entirely and install the PyG extensions straight into your `uv` venv,
matching what CI does:

```bash
/usr/local/bin/uv sync
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
uv run mypy src/ tests/
uv run pytest tests/
```

Training configs accept:

```yaml
device: auto  # cuda if available, then mps, then cpu
device: cpu
device: cuda
device: mps
```

Explicit `cuda` or `mps` requests fail clearly when that backend is unavailable. For NVIDIA GPU
training on Linux or Windows, install the CUDA-specific PyTorch wheel that matches the machine's
CUDA stack and the same `torch` version pinned in `pyproject.toml`, then rerun the same training
code. CI remains CPU-only.

On NVIDIA Blackwell GPUs (sm_120 — e.g. RTX PRO 6000), `torch` must be `>=2.7` with a CUDA 12.8+
wheel; earlier torch releases have no Blackwell kernels at all (`RuntimeError: CUDA error: no
kernel image is available for execution on the device`), regardless of CUDA toolkit version:

```bash
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

Verify PyTorch and backend availability:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.backends.mps.is_available())"
```

## Development Checks

Run smoke tests:

```bash
python -m pytest tests/ -v
```

Run analysis-only tests:

```bash
python -m pytest tests/test_analysis_lenses.py tests/test_analysis_io.py tests/test_analysis_config_runner.py -v
```

Run the full dummy end-to-end workflow test:

```bash
python -m pytest tests/test_e2e_dummy_workflow.py -v
```

Generate local dummy model/probe outputs and PNG/HTML reports for every supported model, ARIMA,
and all standard horizons:

```bash
python scripts/run_dummy_e2e.py --output-dir outputs/e2e_dummy --epochs 5
```

Run lint checks:

```bash
ruff check src/ tests/ scripts/
```

Format Python files:

```bash
ruff format src/ tests/
```

Run static type checks:

```bash
mypy src/ tests/ --ignore-missing-imports
```

## Full repository structure

```text
src/
  analysis/      # STGNN-Probe interpretability framework
  data/          # dataset preparation, loading, and feature generation
  evaluation/    # metrics and evaluation utilities
  models/        # STGNN model implementations
  training/      # training loop, scheduler dispatch, adapters, and config
scripts/
  configs/
    models/                    # per-model architecture + dataset-specific training hyperparameters
      gwn_base.yaml            #   loaded automatically at runtime; one file per model
      gwn_v2_base.yaml
      stawnet_base.yaml
      staeformer_base.yaml
      dssa_tcn_base.yaml
      d2stgnn_base.yaml
      bigst_base.yaml
      tcn_base.yaml
      arima_base.yaml
    sweeps/                    # per-model W&B Bayesian sweep search spaces (not used for the paper)
    metr_la_training.yaml      # experiment config: dataset paths, horizons, output dirs
    pems_bay_training.yaml     # experiment config: dataset paths, horizons, output dirs
    metr_la_smoke_new_models.yaml # low-cost D2STGNN/BigST integration smoke config
    metr_la_probe.yaml         # STGNN-Probe config for METR-LA (207 nodes)
    pems_bay_probe.yaml        # STGNN-Probe config for PEMS-BAY (325 nodes)
    metr_la_smoke_probe.yaml   # single-horizon probe config for the smoke run
  prepare_standard_splits.py   # convert raw HDF → pre-windowed train/val/test .npz splits
  prepare_walk_forward_data.py # convert raw HDF → raw (T, N, C) .npz for walk-forward folds
  preprocess_bigst_features.py # optional BigST long-term feature pretraining artifact
  run_training.py              # train one or all models on a dataset
  run_sweep.py                 # W&B hyperparameter sweep entry point
  run_walk_forward.py          # expanding-window walk-forward evaluation
  run_dummy_e2e.py             # deterministic smoke test (all models, CPU)
  prepare_probe_raw_data.py    # export STGNN-Probe inputs from raw data
  reextract_adjacency.py       # re-extract learned adjacency from saved checkpoints (no retraining)
  finalize_paper_figures.py    # paper Figs. 1-2; computes the reported AAS (uniform matrices at chance)
```

### Implemented models and classes

| Model | Class | Reference |
|---|---|---|
| Graph WaveNet | `GWN` | Wu et al., IJCAI 2019 |
| Graph WaveNet v2 | `GWNv2` | Shleifer et al., 2019 |
| STAWnet | `STAWnet` | Tian and Chan, IET Intelligent Transport Systems 2021 |
| DSSA-TCN | `DSSATCN` | Zhang et al., PLOS ONE 2025 |
| STAEformer | `STAEformer` | Liu et al., CIKM 2023 |
| D2STGNN | `D2STGNN` / `make_d2stgnn` | Shao et al., VLDB 2022 |
| BigST | `BigST` / `make_bigst` | Han et al., VLDB 2024 |
| TCN-PerNode (baseline) | `TCNModel` | — |
| ARIMA (baseline) | `RollingARIMA` | — |

## Notes

- The pinned CPU baseline is intentional; CUDA users should install the matching CUDA PyTorch
  build for their machine.
- `device: auto` is the portable default for training configs.
