# Model Training

## 1. Training data source

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

## 2. Standard training run

Train one model or all configured models using paper-default hyperparameters:

```bash
 Train a single model
python scripts/run_training.py \
  --config scripts/configs/metr_la_training.yaml \
  --model gwn

 Train all configured models and baselines sequentially
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
 metr_la_training.yaml
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

### Re-extracting learned adjacency from checkpoints

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

## 3. Hyperparameter sweep (W&B, not used in the paper)

Run a Bayesian sweep over architecture and training hyperparameters:

```bash
python scripts/run_sweep.py \
  --config scripts/configs/metr_la_training.yaml \
  --sweep  scripts/configs/sweeps/gwn_sweep.yaml \
  --model  gwn \
  --horizon 12 \
  --count  30

 Resume an existing sweep
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

## 4. Walk-forward validation (not used in the paper)

Train and evaluate a model independently on each expanding-window fold:

```bash
python scripts/run_walk_forward.py \
  --config  scripts/configs/metr_la_training.yaml \
  --model   gwn \
  --dataset METR-LA
```

Reports per-fold MAE / MAPE / RMSE and a mean ± std summary across all folds.

## Feature convention

All models receive a common 3-channel input `[speed, tod, dow/7]`:

| Channel | Content | Range | Notes |
|---------|---------|-------|-------|
| 0 | Traffic speed | z-scored | Per-node `StandardScaler`; fitted on train only |
| 1 | Time of day | `[0, 1)` | Fractional seconds-in-day |
| 2 | Day of week | `[0, 6/7]` | Fractional; use `(x * 7).long()` for embeddings |

GWN-family models treat channels 1–2 as raw features. STAEformer, DSSA-TCN, D2STGNN, and
BigST convert day-of-week back to integer indices for embedding lookups inside their adapters.
