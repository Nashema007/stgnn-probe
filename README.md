# STGNN Framework

A PyTorch framework for training and analysing spatio-temporal graph neural network (STGNN)
models on traffic forecasting benchmarks. It accompanies the paper
*Analysing Spatio-Temporal Graph Neural Networks for Long-Horizon Traffic Forecasting*.

The repository has two parts:

- **Training** (`scripts/run_training.py`, `src/training`, `src/models`): trains seven STGNNs and
  two temporal baselines on METR-LA and PEMS-BAY, one model per forecast horizon and seed. It
  exports each model's test predictions and its learned N x N dependency matrix.
- **STGNN-Probe** (`run_probe.py`, `src/analysis`): compares those exports with a per-sensor
  temporal baseline (the Spatial Gain Score, SGS) and with a Granger predictive reference (the
  Adjacency Alignment Score, AAS).

## Reproducing the paper

Run the steps below in order. All commands are run from the repository root. On macOS, prefix
each one with `./test.sh` so it runs inside the Linux Docker image (see
[Environment Setup](#environment-setup)).

| Step | Command | Writes |
|---|---|---|
| 0. Environment | `uv sync`, then install `torch-scatter`/`torch-sparse` ([details](#environment-setup)) | `.venv/` |
| 1. Data | nothing to do: `tsl` downloads METR-LA and PEMS-BAY on first use | `data/tsl_cache/` |
| 2. Train | `python scripts/run_training.py --config scripts/configs/metr_la_training.yaml`, then the same with `pems_bay_training.yaml` | `checkpoints/<dataset>/`, plus `data/probe_inputs/<dataset>/predictions/` and `adjacency/` |
| 3. Probe | `python run_probe.py --config scripts/configs/metr_la_probe.yaml --all --force-granger`, then the same with `pems_bay_probe.yaml` | `outputs/` |
| 4. Figures and AAS | `python scripts/finalize_paper_figures.py --aas-source outputs --no-manuscript-check` | `figures/` |

Step 2 is the expensive one: 7 STGNNs plus the per-sensor TCN bank and ARIMA, over 3 horizons and
3 seeds, on both datasets. Use a GPU. Retrained models will not match the paper's numbers to the
third decimal, which is why step 4 is run with `--no-manuscript-check`. Without that flag the
script asserts the 14 published AAS values and fails on any difference.

### Where each result in the paper comes from

| Paper item | Source |
|---|---|
| Table 4 (test MAE by horizon) | `outputs/lens0_performance/<DATASET>/metrics_by_horizon.csv` |
| Table 5 (SGS and sensor classes) | `outputs/per_model/<model>_<DATASET>/summary.json` and `node_classification.csv` |
| Fig. 1 (SGS by horizon) | `figures/fig1_*`, built from `outputs/lens0_performance/<DATASET>/figures/` |
| Fig. 2 (AAS) | `figures/fig2_*`. The values are printed by `finalize_paper_figures.py` |
| Granger reference | `outputs/granger_cache/<DATASET>.npz` (`fstats[i, j]` is the F-statistic for i -> j) |

Use `finalize_paper_figures.py` for AAS, not the probe's own `alignment_scores.json`. The probe
breaks ties arbitrarily when a learned matrix is exactly uniform. The paper scores such a matrix at
the chance level `k/(N-1)`, and the script implements that.

### Paper configuration

The shipped configs are the paper's configuration.

| Setting | Value | Where it is set |
|---|---|---|
| Datasets | METR-LA (207 sensors), PEMS-BAY (325 sensors), loaded through `tsl` | `scripts/configs/<dataset>_training.yaml` |
| Missing readings | METR-LA's zero-coded readings (~8%) are forward-filled by `tsl` during preprocessing, before the chronological split. The Granger reference uses the full preprocessed series | `tsl.datasets.MetrLA` default |
| Input window | 12 steps (60 min), features `[speed, time of day, day of week]` | `in_len: 12` |
| Forecast horizons | 6, 12 and 42 steps (30, 60 and 210 min). One model per horizon, evaluated at the final step | `horizons: [6, 12, 42]` |
| Runs | 3 seeds per model and horizon | `num_runs: 3` |
| Split | chronological 70/10/20, with the Z-score fitted on the training split | `val_len: 0.125`, `test_len: 0.2` |
| Models | GWN, GWN v2, STAWnet, DSSA-TCN, STAEformer, D2STGNN, BigST; per-sensor TCN and ARIMA baselines | `spatial_models`; `scripts/configs/models/*_base.yaml` |
| Hyperparameters | the published settings of each model | `scripts/configs/models/*_base.yaml` |
| SGS | relative gain over the per-sensor TCN; sensor classes use a 1% threshold | `sgs_rel_threshold: 0.01` |
| Granger reference | pairwise F-test at a **fixed lag p = 12** for every pair, fitted on the full series; top `k = 10` incoming sources per sensor | `granger.max_lag: 12`, `granger.gcg_top_k: 10` |
| AAS | overlap of each model's top-10 incoming edges with the reference, averaged over 3 horizons x 3 seeds. Exactly uniform matrices are scored at the chance level `k/(N-1)` | `scripts/finalize_paper_figures.py` |

### Edge convention

Every structural comparison reads a dependency matrix `W` with **`W[i, j]` = weight of source
`i` on target `j`**. Column `j` therefore holds target `j`'s incoming edges, which is also the
orientation of the Granger `fstats[i, j]`. Exported matrices are saved in each model's native
orientation. `src/analysis/orientation.py` puts them into the canonical one before any top-k
selection:

| Model | Native aggregation | Treatment |
|---|---|---|
| GWN v2 | `einsum('ncvl,vw->ncwl')`, columns are targets | as exported |
| STAWnet | `x @ attention`, columns are targets | as exported |
| DSSA-TCN | `attention @ value`, rows are targets | **transposed** |
| D2STGNN | `graph @ X`, rows are targets | **transposed** |
| GWN (tsl) | applies the matrix and its transpose | as exported |
| STAEformer, BigST | similarity proxies, never used to propagate | as exported |

`tests/test_orientation.py` checks the native aggregation axis of each propagating model, so a
model whose export is read in the wrong direction fails the test suite. When adding a model,
add it to `ROWS_ARE_TARGETS` if it aggregates along rows.

### What the paper does and does not use

| Used in the paper | Included but not used in the paper |
|---|---|
| `scripts/run_training.py`, `scripts/configs/<dataset>_training.yaml`, `scripts/configs/models/` | `scripts/run_sweep.py`, `scripts/configs/sweeps/` (W&B hyperparameter sweeps) |
| `run_probe.py`, `scripts/configs/<dataset>_probe.yaml` | `scripts/run_walk_forward.py`, `scripts/prepare_walk_forward_data.py` (walk-forward validation) |
| Lens 0 (performance), Lens 1 (SGS), Lens 2 (Granger), Lens 3 (AAS) | Lens 4 (communities), Lens 5 (horizon degradation), `src/analysis/explore.py` |
| `scripts/finalize_paper_figures.py`, `scripts/regenerate_figures.py`, `scripts/pdf_to_eps.py` | `scripts/png_to_eps.py` |
| `scripts/reextract_adjacency.py` (re-exports matrices from checkpoints) | `scripts/run_dummy_e2e.py`, `scripts/configs/*smoke*.yaml` (synthetic smoke tests) |
| `src/data/tsl_pipeline.py` | `src/data/raw_sources.py`, `scripts/prepare_standard_splits.py` (manually downloaded DCRNN files) |

`run_probe.py --all` also runs Lenses 4 and 5. Their outputs are written to `outputs/` but are
not reported in the paper. The modules that are not used carry a "Not used in the paper" note in
their docstrings.

## Structure

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

### Implemented models

| Model | Class | Reference |
|---|---|---|
| Graph WaveNet | `GWN` | Wu et al., IJCAI 2019 |
| Graph WaveNet v2 | `GWNv2` | Shleifer et al., 2019 |
| STAWnet | `STAWnet` | Tan et al., IET Intelligent Transport Systems 2021 |
| DSSA-TCN | `DSSATCN` | Ni et al., PLOS ONE 2025 |
| STAEformer | `STAEformer` | Liu et al., CIKM 2023 |
| D2STGNN | `D2STGNN` / `make_d2stgnn` | Shao et al., VLDB 2022 |
| BigST | `BigST` / `make_bigst` | Han et al., VLDB 2024 |
| TCN-PerNode (baseline) | `TCNModel` | — |
| ARIMA (baseline) | `RollingARIMA` | — |

## Model Training

### 1. Training data source

The current config-driven trainer uses `data.tsl_pipeline`, which loads
`tsl.datasets.MetrLA` and `tsl.datasets.PemsBay` directly. The tsl cache is written under
`data/tsl_cache/` on first use and is intentionally gitignored. The trainer derives the graph
from the tsl dataset connectivity and builds a common 3-channel input `[speed, tod, dow/7]`.
`tsl.datasets.MetrLA` forward-fills METR-LA's zero-coded missing readings, so training targets,
test targets and the Granger reference all use the imputed series. The masked loss
(`null_val: 0.0`) therefore only excludes the few zeros left in PEMS-BAY.

The older local preparation scripts are still available for manual fixed-split or walk-forward
experiments, but they are not required for the default `scripts/run_training.py` flow.

**Manual fixed-split evaluation** — produces pre-windowed `train.npz`, `val.npz`, `test.npz`
(70 / 10 / 20 split, matching the original Graph WaveNet setup):

```bash
python scripts/prepare_standard_splits.py \
  --raw-file src/data/raw-data/metr-la.h5 \
  --output-dir data/METR-LA/

python scripts/prepare_standard_splits.py \
  --raw-file src/data/raw-data/pems-bay.h5 \
  --output-dir data/PEMS-BAY/
```

Optional flags: `--seq-length-x` (default 12), `--seq-length-y` (default 12), `--y-start` (default 1), `--dow` (include day-of-week channel), `--train-ratio`, `--test-ratio`.

**Manual walk-forward data export** — produces a single raw `(T, N, C)` `.npz` for legacy
walk-forward helpers. The current `scripts/run_walk_forward.py` command uses
`build_tsl_walk_forward_pipeline()` and loads folds from `tsl.datasets` directly:

```bash
python scripts/prepare_walk_forward_data.py \
  --dataset METR-LA \
  --raw-file src/data/raw-data/metr-la.h5 \
  --output-dir data/

python scripts/prepare_walk_forward_data.py \
  --dataset PEMS-BAY \
  --raw-file src/data/raw-data/pems-bay.h5 \
  --output-dir data/
```

Optional flag: `--dow` (include day-of-week channel).

### 2. Standard training run

Train one model or all configured models using paper-default hyperparameters:

```bash
# Train a single model
python scripts/run_training.py \
  --config scripts/configs/metr_la_training.yaml \
  --model gwn

# Train all configured models and baselines sequentially
python scripts/run_training.py \
  --config scripts/configs/metr_la_training.yaml
```

Every run loads two config files:

- **`metr_la_training.yaml`** (passed on the CLI) — dataset paths, horizons, output
  directories, and which models to run. Dataset-specific; does not contain model hyperparameters.
- **`scripts/configs/models/<model>_base.yaml`** (loaded automatically) — architecture
  parameters and per-dataset training hyperparameters (`lr`, `epochs`, `scheduler`, etc.).
  Each file has a `datasets.METR-LA` and `datasets.PEMS-BAY` section so the right values
  are picked up depending on which experiment config is active.

To override a single parameter without editing the model YAML (e.g. for a one-off run),
add a `model_overrides` block to the experiment config — it is merged on top of the model
YAML with the highest precedence:

```yaml
# metr_la_training.yaml
model_overrides:
  gwn:
    dropout: 0.5
```

The production experiment configs currently run:

```text
gwn, gwn_v2, stawnet, staeformer, dssa_tcn, d2stgnn, bigst
```

Temporal baselines (`arima`, `tcn`) are added by the orchestrator. The smoke config
`scripts/configs/metr_la_smoke_new_models.yaml` is scoped to `d2stgnn` and `bigst`, one horizon,
one seed, and short training for integration checks.

#### Re-extracting learned adjacency from checkpoints

`scripts/reextract_adjacency.py` regenerates the exported `*_adjacency*.npy` for the
learned-structure lenses from existing `_best.pt` checkpoints, without retraining (predictions and
forecasting metrics are untouched). Use it after changing how a model's learned representation is
extracted, then re-run the probe's structural lenses:

```bash
python scripts/reextract_adjacency.py --config scripts/configs/metr_la_training.yaml
python run_probe.py --config scripts/configs/metr_la_probe.yaml --all --skip-performance
```

The TCN baseline uses the three canonical input features and trains a separate
parameter set for every sensor, so it never consumes another sensor's observations.

Both production configs set `num_runs: 3`. If `seeds.json` already contains a completed
single-seed experiment, rerunning the same config preserves that first seed and its cache, allocates
two new unique seeds, and trains only the missing jobs. Do not delete `outputs/<dataset>/seeds.json`
or `completed_runs.json` when extending an existing experiment: together they are what connect the
new runs to the completed first run.
Reducing `num_runs` below the number already recorded is rejected up front by
`validate_experiment_config`, before ARIMA is refit or any model trains, so completed run identities
are never silently dropped from the seed log.

**BigST long-term features** — `bigst_base.yaml` defaults to `use_long: false`. If an experiment
sets `use_long: true`, `run_training.py` checks for
`data/probe_inputs/<dataset>/bigst_long_term_features.npy` and automatically runs
`scripts/preprocess_bigst_features.py` to create it when missing. You do not need to invoke the
preprocess script manually for normal training runs.

**Progress output** — neural models (GWN, TCN, etc.) print one line per epoch with
train loss, val MAE, val RMSE, elapsed time, CPU usage, process RSS memory, and CUDA memory when
running on a GPU. ARIMA shows a `tqdm` progress bar over nodes with elapsed time, ETA, and a
`skipped` count for any nodes that fail to fit.

Generated training artifacts are written to the paths configured by the experiment YAML:

```text
checkpoints/<dataset>/              # best/resume checkpoints, uniquely named by horizon + seed
data/probe_inputs/<dataset>/        # combined (W, H, N, R) probe arrays and selected adjacency
outputs/<dataset>/
  seeds.json                        # stable seed allocation, extended when num_runs increases
  completed_runs.json               # resume/cache index
  resource_usage.csv                # one CPU/RSS/GPU/time row per trained job
  runs/<model>/h<horizon>/
    run_<index>_seed_<seed>/
      manifest.json                 # run identity, fingerprint, resolved model config, paths
      metrics.jsonl                 # epoch, test, table, and system telemetry events
      summary.json                  # validation, test, timing, CPU/RSS, and GPU summary
      predictions.npy               # this run only, never another seed's predictions
      adjacency.npy                 # this run only, for spatial models
      node_metrics.json             # per-node TCN only
```

The combined arrays under `data/probe_inputs` are regenerated from all three isolated run
directories for STGNN-Probe. Rewriting a combined array does not remove or replace the individual
run artifacts. When an older cache uses the legacy `checkpoints/*_preds.npy` layout, the next
orchestrator run materializes it into `outputs/<dataset>/runs/...` without retraining. If a cached
run names an adjacency file that has since been deleted, that run is retrained rather than accepted
without a graph — otherwise the model's global adjacency would quietly become the mean over fewer
horizons than configured.

A run directory outlives any single training attempt, since a crash-resume or a retrain after the
fingerprint changed reopens it. Every `metrics.jsonl` event therefore carries a 1-based `attempt`
number, delimited by a `run_started` event, so two attempts never read as one epoch series;
`summary.json` records the attempt it describes and never inherits keys from a discarded one.

These generated payloads are ignored by Git. The repository tracks `.gitkeep` stubs so the
expected directories exist without committing checkpoint data, probe reports, tsl cache files, or
large arrays.

### 3. Hyperparameter sweep (W&B, not used in the paper)

Run a Bayesian sweep over architecture and training hyperparameters:

```bash
python scripts/run_sweep.py \
  --config scripts/configs/metr_la_training.yaml \
  --sweep  scripts/configs/sweeps/gwn_sweep.yaml \
  --model  gwn \
  --horizon 12 \
  --count  30

# Resume an existing sweep
python scripts/run_sweep.py \
  --config scripts/configs/metr_la_training.yaml \
  --sweep  scripts/configs/sweeps/gwn_sweep.yaml \
  --model  gwn --sweep-id <existing-id> --count 10
```

Sweep search spaces are defined in `scripts/configs/sweeps/<model>_sweep.yaml`.
By default sweeps and runs log to W&B project `stgnn-framework`; W&B entity
selection comes from `WANDB_ENTITY` in `.env`, `--entity`, or `wandb_entity`. Start from the template:

```bash
cp .env.example .env
```

then set:

```dotenv
WANDB_ENTITY=your-wandb-team
```

```bash
python scripts/run_sweep.py \
  --config scripts/configs/metr_la_training.yaml \
  --sweep  scripts/configs/sweeps/gwn_sweep.yaml \
  --model  gwn \
  --horizon 12 \
  --entity your-wandb-team \
  --project stgnn-framework
```

For non-sweep training code paths, set `WANDB_ENTITY` in `.env` or provide
`wandb_entity` explicitly:

```yaml
use_wandb: true
wandb_entity: your-wandb-team
wandb_project: stgnn-framework
wandb_mode: online
```

Each `(model, horizon, seed)` job is a separate W&B run. Its config includes the seed and
1-based `run_index`; epoch train/validation metrics, test metrics, CPU percentage, process RSS,
current/peak CUDA allocation and reservation, parameter count, and elapsed time are mirrored to
the run's local `metrics.jsonl`/`summary.json`. The per-node TCN additionally logs a per-sensor
metrics table; the shared TCN uses the standard neural-model logger. ARIMA is deterministic, so it
is fitted once, tiled across the three probe run columns, and logged under
`outputs/<dataset>/runs/arima/deterministic`.

### 4. Walk-forward validation (not used in the paper)

Train and evaluate a model independently on each expanding-window fold:

```bash
python scripts/run_walk_forward.py \
  --config  scripts/configs/metr_la_training.yaml \
  --model   gwn \
  --dataset METR-LA
```

Reports per-fold MAE / MAPE / RMSE and a mean ± std summary across all folds.

### Feature convention

All models receive a common 3-channel input `[speed, tod, dow/7]`:

| Channel | Content | Range | Notes |
|---------|---------|-------|-------|
| 0 | Traffic speed | z-scored | Per-node `StandardScaler`; fitted on train only |
| 1 | Time of day | `[0, 1)` | Fractional seconds-in-day |
| 2 | Day of week | `[0, 6/7]` | Fractional; use `(x * 7).long()` for embeddings |

GWN-family models treat channels 1–2 as raw features. STAEformer, DSSA-TCN, D2STGNN, and
BigST convert day-of-week back to integer indices for embedding lookups inside their adapters.

## STGNN-Probe Analysis

`src/analysis/` contains STGNN-Probe, the analysis code behind the paper. For each model it reads
per-window forecast predictions and the model's exported `N x N` learned dependency matrix, and
compares them with a per-sensor temporal baseline and a Granger predictive reference. The
Granger reference is a linear statistical reference, not a physical or ground-truth graph.

Lens 0 (MAE), Lens 1 (SGS), Lens 2 (Granger reference) and Lens 3 (AAS) produce the paper's
results. Lenses 4 and 5 are additional diagnostics that the paper does not report:

1. Spatial Utility compares STGNN predictions against the per-sensor TCN and reports per-node
   Spatial Gain Scores (SGS).
2. The Granger reference keeps each sensor's top-k incoming sources by fixed-lag F-statistic.
3. Structural Alignment measures the overlap of each model's top-k edges with that reference (AAS).
4. Community Coherence checks learned graph communities against geographic structure (not in the paper).
5. Horizon Degradation tracks SGS and AAS across horizons (not in the paper).

Functionally, the current implementation covers the intended STGNN-Probe analysis methods:

- TCN differencing is Lens 1's Spatial Gain Score, computed from TCN predictions, model
  predictions, and `ground_truth.npy`.
- The Granger predictive reference is Lens 2's cached dataset-level graph (fixed lag 12, top-10 sources).
- Pearson correlation is Lens 2's lagged Pearson correlation tensor, saved with Granger outputs.
- Adjacency comparison is Lens 3's precision, recall, AAS/F1, weighted precision, threshold
  sweep, and TP/FP/FN edge matrix.
- Community detection is Lens 4's deterministic Louvain analysis.
- Centrality analysis is Lens 4's degree, betweenness, closeness, and eigenvector centrality.
- Geographic mapping is saved by Lens 1 and Lens 4 as spatial gain, community, and centrality
  plots.
- Pairwise/node degree comparison is saved per model by comparing learned adjacency degree against
  Granger graph degree with diagonal self-loops excluded.
- Horizon analysis is Lens 5's SGS and AAS degradation curves.

The analysis code is designed to fail loudly on invalid inputs:

- Lens 1 requires matching `(H, N)` prediction shapes for baseline and model outputs, both in the
  per-window `(W, H, N, R)` layout.
- Lens 2 validates Granger settings before multiprocessing and raises if pair failures exceed the
  configured internal failure threshold.
- Lens 3 and Lens 5 ignore diagonal self-loops when scoring graph alignment.
- Lens 4 supports deterministic Louvain runs through `CommunityConfig.random_seed`.
- JSON result output is strict: non-finite values are written as `null`, not `NaN` or `Infinity`.

Install the project environment before running analysis or visualization commands:

```bash
/usr/local/bin/uv sync
```

Plotly static PNG export uses Kaleido and requires Chrome; CI installs it with
`uv run plotly_get_chrome -y`.

### Local Raw Data

Download METR-LA and PEMS-BAY data from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) links provided by [DCRNN](https://github.com/liyaguang/DCRNN).


Large raw METR-LA and PEMS-BAY files belong in `src/data/raw-data/`. The directory is kept in git
with `.gitkeep`, but its payload is ignored so local data does not leak into commits.

Generated files under `data/` (`.npz`, `.npy`, `.h5`), tsl cache payloads under
`data/tsl_cache/`, probe fingerprint files, checkpoints, and run outputs are gitignored. They can
be large and are reproducible from the raw sources, tsl downloads, and training/probe scripts.
The expected `data/`, `checkpoints/`, and `outputs/` directory structure is preserved in git with
`.gitkeep` files.

Expected local filenames:

```text
src/data/raw-data/
  metr-la.h5
  pems-bay.h5
  adj_mx.pkl
  adj_mx_bay.pkl
  graph_sensor_locations.csv
  graph_sensor_locations_bay.csv
  graph_sensor_ids.txt
  distances_la_2012.csv
  distances_bay_2017.csv
```

**Training data** — produce the `.npz` files read by the training scripts (choose one per dataset):

```bash
# Standard fixed splits (train/val/test.npz)
python scripts/prepare_standard_splits.py \
  --raw-file src/data/raw-data/metr-la.h5 --output-dir data/METR-LA/

python scripts/prepare_standard_splits.py \
  --raw-file src/data/raw-data/pems-bay.h5 --output-dir data/PEMS-BAY/

# Walk-forward raw features (<dataset>.npz)
python scripts/prepare_walk_forward_data.py --dataset METR-LA \
  --raw-file src/data/raw-data/metr-la.h5 --output-dir data/

python scripts/prepare_walk_forward_data.py --dataset PEMS-BAY \
  --raw-file src/data/raw-data/pems-bay.h5 --output-dir data/
```

**Probe inputs** — produce standardized STGNN-Probe inputs (raw traffic arrays, coordinates,
adjacency). Two sources are supported:

```bash
# Default: sources from tsl's own auto-downloaded dataset cache (data/tsl_cache/) —
# no manual download needed, and guarantees the same road_adjacency every spatial
# model actually trains on (src/data/tsl_pipeline.py's export_probe_dataset_inputs_from_tsl).
python scripts/prepare_probe_raw_data.py --dataset METR-LA --output-dir data/probe_inputs/metr_la
python scripts/prepare_probe_raw_data.py --dataset PEMS-BAY --output-dir data/probe_inputs/pems_bay

# Legacy: sources from manually-downloaded DCRNN raw files (src/data/raw_sources.py adapter)
python scripts/prepare_probe_raw_data.py --source legacy \
  --dataset METR-LA --raw-dir src/data/raw-data --output-dir data/probe_inputs/metr_la
```

`--output-dir` must point at the same per-dataset subdirectory (`data/probe_inputs/metr_la/`,
`data/probe_inputs/pems_bay/`) that `run_training.py` writes `ground_truth.npy`,
`predictions_dir/`, and `adjacency_dir/` into — the probe config's `raw_data`/`coordinates`
paths expect everything to live together in that one directory.

Either path writes standardized raw traffic arrays, canonical coordinates, normalized distance
edges, and a reference road adjacency, under the same output filenames. Neither generates
`ground_truth.npy`, `{model}_predictions.npy`, or `{model}_adjacency.npy`; those remain outputs
from model evaluation workflows.

### STGNN-Probe Inputs

Shape notation used throughout STGNN-Probe:

```text
W  number of paired test windows (repo-default leading axis for predictions/ground truth)
N  number of graph nodes or sensors
H  number of forecast horizons
R  number of independent prediction runs, seeds, folds, or samples
T  number of raw traffic timesteps
L  number of lag values used for lagged correlation or Granger checks
```

The config-driven runner expects fixed dataset-level inputs plus two model-level inputs:

```text
Dataset inputs:
  raw_data              # .npy, shape (T, N), un-normalized traffic
  coordinates           # .csv, columns: node_id, latitude, longitude
  ground_truth          # .npy, shape (W, H, N)
  predictions_dir/
    tcn_predictions.npy        # per-node TCN, shape (W, H, N, R)

Per-model inputs:
  predictions_dir/{model}_predictions.npy # shape (W, H, N, R)
  adjacency_dir/{model}_adjacency.npy     # shape (N, N)
```

Adjacency matrices must be finite and non-negative. The runner row-normalizes them before lens
execution so each model is evaluated on the same adjacency contract.

`predictions_dir` and `adjacency_dir` are configurable. They can point to directories named
`model_predictions/`, `adjacency_matrix/`, or any other local layout as long as the standardized
file names above are present.

### STGNN-Probe Config

Two ready-made probe configs ship with the repo, one per dataset:

- `scripts/configs/metr_la_probe.yaml` — METR-LA (207 nodes)
- `scripts/configs/pems_bay_probe.yaml` — PEMS-BAY (325 nodes)

**`scripts/configs/metr_la_probe.yaml`**:

```yaml
datasets:
  - name: METR-LA
    num_nodes: 207
    raw_data: data/probe_inputs/metr_la/metr_la_raw.npy
    coordinates: data/probe_inputs/metr_la/metr_la_coords.csv
    ground_truth: data/probe_inputs/metr_la/ground_truth.npy
    predictions_dir: data/probe_inputs/metr_la/predictions
    adjacency_dir: data/probe_inputs/metr_la/adjacency
    horizons: [6, 12, 42]
    horizon_minutes: [30, 60, 210]

models:
  temporal_baselines: [arima, tcn]
  spatial_models: [gwn, gwn_v2, stawnet, staeformer, dssa_tcn, d2stgnn, bigst]

granger:
  # Fixed lag p = 12 for every ordered pair (no lag selection), matching the
  # 12-step model input window. The reference is fitted once on the full series.
  max_lag: 12
  significance: 0.05
  # GCG keeps each node's top-k strongest incoming edges by F-statistic
  # (density = k/(N-1) ~= 5% at k=10, N=207). Effect-size sparsification —
  # p-value significance is non-discriminative here. Re-derived from cached
  # F-stats, so changing this does NOT rerun the Granger tests.
  gcg_top_k: 10
  # -1 (all cores) oversubscribed RAM: each forked worker holds 2-3GB
  # resident (torch/numpy/statsmodels), so all-cores on this box pushed
  # combined RES past 128GB RAM and swap filled, grinding to a crawl.
  n_jobs: 16

community:
  algorithm: louvain
  num_runs: 10
  resolution: 1.0
  random_seed: 0

alignment:
  threshold: 0.1
  sweep_min: 0.0
  sweep_max: 1.0
  sweep_steps: 21

sgs_threshold: 0.1        # absolute-SGS reference (target units); reporting only
sgs_rel_threshold: 0.01   # relative-SGS threshold for node labels (1% of TCN MAE, scale-free)

performance:
  primary_metric: mae
  horizon_groups:
    short_range: [30]
    boundary: [60]
    long_range: [210]
  baselines:
    statistical: arima
    temporal: tcn
  model_groups:
    statistical_baseline: [arima]
    temporal_baseline: [tcn]
    graph_wavenet_based: [gwn, gwn_v2]
    attention_adaptive_stgnn: [stawnet, dssa_tcn, staeformer]
    decoupled_stgnn: [d2stgnn]
    linear_complexity_stgnn: [bigst]
  ranking_lower_is_better: true
```

**`scripts/configs/pems_bay_probe.yaml`** is identical except `name: PEMS-BAY`,
`num_nodes: 325`, all paths use `data/probe_inputs/pems_bay/`, and `n_jobs` differs.
`sgs_threshold` (absolute, in mph) is reported only; the paper's sensor classes use
`sgs_rel_threshold`. `alignment.threshold` is unused when `gcg_top_k` is set, which is the case for
the paper.

**`scripts/configs/metr_la_smoke_probe.yaml`** is a single-horizon companion config for
`scripts/configs/metr_la_smoke_new_models.yaml`. It points at
`data/probe_inputs/metr_la_smoke/` and only includes `tcn`, `d2stgnn`, and `bigst`.

### Running STGNN-Probe

All commands are run from the **repo root**. `--config` defaults to `config.yaml`, which this repo
does not provide — always pass the actual config path explicitly (e.g.
`scripts/configs/metr_la_probe.yaml`).

**Run one model on one dataset:**

```bash
# METR-LA
python run_probe.py \
  --config scripts/configs/metr_la_probe.yaml \
  --dataset METR-LA --model gwn

# PEMS-BAY
python run_probe.py \
  --config scripts/configs/pems_bay_probe.yaml \
  --dataset PEMS-BAY --model gwn
```

**Run every spatial model on every dataset configured in a probe config:**

```bash
python run_probe.py --config scripts/configs/metr_la_probe.yaml --all
python run_probe.py --config scripts/configs/pems_bay_probe.yaml --all
```

**Common options:**

```bash
# Skip figure generation (faster, useful for metrics-only runs)
python run_probe.py --config scripts/configs/metr_la_probe.yaml --all --no-figures

# Force Granger recomputation even if a cache already exists
python run_probe.py --config scripts/configs/metr_la_probe.yaml \
  --dataset METR-LA --model gwn --force-granger

# Write outputs to a custom directory instead of outputs/
python run_probe.py --config scripts/configs/metr_la_probe.yaml \
  --all --output-dir outputs/my_run
```

**Adding a new spatial model:**

1. Produce the two required artifact files:
   ```text
   data/probe_inputs/metr_la/predictions/{model}_predictions.npy  # shape (W, H, N, R)
   data/probe_inputs/metr_la/adjacency/{model}_adjacency.npy      # shape (N, N)
   ```
2. Add the model name to `models.spatial_models` in the probe config.
3. Run the probe:
   ```bash
   python run_probe.py \
     --config scripts/configs/metr_la_probe.yaml \
     --dataset METR-LA --model {model}
   ```

### STGNN-Probe Outputs

Per-model reports are written to:

```text
outputs/per_model/{model}_{dataset}/
  sgs_matrix.npy
  sgs_mean.npy
  node_classification.csv
  alignment_scores.json
  tp_fp_fn_matrix.npy
  community_assignments.csv
  centrality_metrics.csv
  degree_comparison.csv
  degree_comparison.json
  modularity_score.json
  gcs_per_community.json
  gcs_overall.json
  lens4_betweenness_centrality.npy
  lens4_closeness_centrality.npy
  lens4_eigenvector_centrality.npy
  degradation_rates.json
  spatial_gain_map.png
  sgs_distribution.png
  threshold_sweep.png
  degree_comparison.png
  community_map.png
  centrality_heatmap.png
  centrality_by_community.png
  adjacency_heatmap.png
  sgs_degradation.png
  aas_degradation.png
  correlation_plot.png
  combined_dashboard.png
```

Every saved figure also gets a same-name `.html` companion for interactive Plotly or Altair
inspection, for example `combined_dashboard.html` next to `combined_dashboard.png`.

Dataset-level cached Granger and ground-truth community outputs are written to:

```text
outputs/granger_cache/{dataset}.npz
outputs/dataset/{dataset}/granger/
  gcg_matrix.npy
  granger_pvalues.npy
  granger_fstats.npy
  pearson_correlations.npy
  gcg_heatmap.png
  gcg_network.png
  pvalue_histogram.png
outputs/dataset/{dataset}/gcg_communities/
  community_assignments.csv
  modularity_score.json
  gcs_per_community.json
  gcs_overall.json
```

The Granger figures also get `.html` companions: `gcg_heatmap.html`, `gcg_network.html`, and
`pvalue_histogram.html`.

Cross-model reports are written to:

```text
outputs/comparative/<dataset_slug>/     # e.g. metr_la, pems_bay — one dir per dataset
  comparative_summary.json
  lens3_alignment_comparison.png
  lens5_sgs_degradation.png
  lens5_aas_degradation.png
  lens5_combined_dashboard.png
  lens5_degradation_rate_correlation.png  # only when at least 3 models are available
```

Comparative figures are written as both PNG and HTML. PNGs are intended for publication/static
documents; HTML files preserve hover, zoom, and interactive legends.

### Interactive Notebook Exploration

For ad hoc notebook exploration before committing to a final plot, use PyGWalker:

```python
from analysis import explore

explore(result)
```

`explore(result)` opens one walker over a combined DataFrame with a `table` column that separates
node-level, horizon-level, and alignment-threshold rows. The helper functions
`explore_nodes(result)`, `explore_horizons(result)`, and `explore_alignment(result)` remain
available from `analysis.explore` for narrower views.

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

### PyTorch Backend Support

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

#### Running lint/typecheck/tests

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

## Notes

- The pinned CPU baseline is intentional; CUDA users should install the matching CUDA PyTorch
  build for their machine.
- `device: auto` is the portable default for training configs.

## License

This code is released under the MIT License (see `LICENSE`). Ported model implementations
remain credited to their original authors. `THIRD_PARTY_NOTICES.md` lists the upstream
sources and their licenses.
