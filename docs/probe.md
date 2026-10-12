# STGNN-Probe Analysis

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

## Local Raw Data

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
 Standard fixed splits (train/val/test.npz)
python scripts/prepare_standard_splits.py \
  --raw-file src/data/raw-data/metr-la.h5 --output-dir data/METR-LA/

python scripts/prepare_standard_splits.py \
  --raw-file src/data/raw-data/pems-bay.h5 --output-dir data/PEMS-BAY/

 Walk-forward raw features (<dataset>.npz)
python scripts/prepare_walk_forward_data.py --dataset METR-LA \
  --raw-file src/data/raw-data/metr-la.h5 --output-dir data/

python scripts/prepare_walk_forward_data.py --dataset PEMS-BAY \
  --raw-file src/data/raw-data/pems-bay.h5 --output-dir data/
```

**Probe inputs** — produce standardized STGNN-Probe inputs (raw traffic arrays, coordinates,
adjacency). Two sources are supported:

```bash
 Default: sources from tsl's own auto-downloaded dataset cache (data/tsl_cache/) —
 no manual download needed, and guarantees the same road_adjacency every spatial
 model actually trains on (src/data/tsl_pipeline.py's export_probe_dataset_inputs_from_tsl).
python scripts/prepare_probe_raw_data.py --dataset METR-LA --output-dir data/probe_inputs/metr_la
python scripts/prepare_probe_raw_data.py --dataset PEMS-BAY --output-dir data/probe_inputs/pems_bay

 Legacy: sources from manually-downloaded DCRNN raw files (src/data/raw_sources.py adapter)
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

## STGNN-Probe Inputs

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

## STGNN-Probe Config

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

## Running STGNN-Probe

All commands are run from the **repo root**. `--config` defaults to `config.yaml`, which this repo
does not provide — always pass the actual config path explicitly (e.g.
`scripts/configs/metr_la_probe.yaml`).

**Run one model on one dataset:**

```bash
 METR-LA
python run_probe.py \
  --config scripts/configs/metr_la_probe.yaml \
  --dataset METR-LA --model gwn

 PEMS-BAY
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
 Skip figure generation (faster, useful for metrics-only runs)
python run_probe.py --config scripts/configs/metr_la_probe.yaml --all --no-figures

 Force Granger recomputation even if a cache already exists
python run_probe.py --config scripts/configs/metr_la_probe.yaml \
  --dataset METR-LA --model gwn --force-granger

 Write outputs to a custom directory instead of outputs/
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

## STGNN-Probe Outputs

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

## Interactive Notebook Exploration

For ad hoc notebook exploration before committing to a final plot, use PyGWalker:

```python
from analysis import explore

explore(result)
```

`explore(result)` opens one walker over a combined DataFrame with a `table` column that separates
node-level, horizon-level, and alignment-threshold rows. The helper functions
`explore_nodes(result)`, `explore_horizons(result)`, and `explore_alignment(result)` remain
available from `analysis.explore` for narrower views.
