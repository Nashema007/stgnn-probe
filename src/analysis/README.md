# STGNN-Probe — Analysis Framework

**STGNN-Probe** is a six-lens interpretability framework for Spatio-Temporal Graph Neural Networks. It answers a single overarching question: *does the spatial graph an STGNN learns reflect Granger-inferred directional dependency structure, and does the quality of that alignment predict how well spatial context helps at inference?*

Lens 0 establishes the forecasting performance baseline before any spatial interpretation begins. Lenses 1–5 then interpret *why* models perform the way they do.

> **Scope of the paper.** The paper reports Lenses 0–3 only: MAE (Lens 0), SGS (Lens 1), the fixed-lag Granger reference (Lens 2) and AAS (Lens 3), with the settings in `scripts/configs/<dataset>_probe.yaml`. Lenses 4 and 5 and the absolute-threshold alignment modes described below are additional diagnostics. The paper's AAS scores exactly uniform learned matrices at the chance level `k/(N-1)` (see `scripts/finalize_paper_figures.py`); Lens 3 as shipped scores them by tie order. Before Lens 3, every learned matrix is put in the canonical orientation `W[i, j]` = source `i` -> target `j` by `src/analysis/orientation.py`. DSSA-TCN and D2STGNN aggregate along rows, so their exports are transposed (see the top-level README's *Edge convention*).

Each lens addresses one dimension of that question. Together they form a diagnostic pipeline that runs on any model that exposes per-node predictions and, for Lenses 1–5, a learned adjacency matrix.

---

## Contents

- [Architecture overview](#architecture-overview)
- [Running the probe](#running-the-probe)
- [Output directory layout](#output-directory-layout)
- [Lens 0 — Forecasting Performance Benchmark](#lens-0--forecasting-performance-benchmark)
- [Lens 1 — Spatial Utility Test](#lens-1--spatial-utility-test)
- [Lens 2 — Granger Predictive Reference](#lens-2--granger-predictive-reference)
- [Lens 3 — Structural Alignment Test](#lens-3--structural-alignment-test)
- [Lens 4 — Community Coherence Test](#lens-4--community-coherence-test)
- [Lens 5 — Horizon Degradation Test](#lens-5--horizon-degradation-test)
- [Cross-model comparative reports](#cross-model-comparative-reports)
- [Interactive exploration with PyGWalker](#interactive-exploration-with-pygwalker)
- [Interpreting the headline finding](#interpreting-the-headline-finding)
- [Configuration reference](#configuration-reference)

---

## Architecture overview

```
ProbeRunner
├── run_performance(dataset)                  ← Lens 0, standalone benchmark
│   └── run_all_performance_from_config()     ← all datasets
├── run_granger(dataset, raw_traffic)         ← Lens 2, cached per dataset
│   └── _save_granger_outputs()               ← GCG heatmap + network
├── run_from_config(dataset, model)           ← single model run (Lenses 1+3+4+5)
│   ├── run_model(...)                        ← Lens 1 + 3 + 4 + 5
│   └── save_figures(result, coords)          ← all per-model plots
└── run_all_from_config()                     ← all models × all datasets
    └── save_comparative_figures(results)     ← cross-model comparison plots
```

Each lens is also callable as a standalone function (`run_lens0`, `run_lens1`, `run_lens2`, …) so individual analyses can be embedded directly in notebooks without the full runner.

---

## Running the probe

### CLI

```bash
# One model/dataset pair
python run_probe.py --config scripts/configs/metr_la_probe.yaml \
                    --dataset METR-LA --model gwn

# All configured models and datasets
python run_probe.py --config scripts/configs/metr_la_probe.yaml --all

# Skip slow static PNG export (HTML figures still written)
python run_probe.py --config scripts/configs/metr_la_probe.yaml --all --no-figures

# Force Granger recomputation (otherwise cached)
python run_probe.py --config scripts/configs/metr_la_probe.yaml --all --force-granger
```

### Python API

```python
from analysis.config import load_config
from analysis.probe import ProbeRunner

config = load_config("scripts/configs/metr_la_probe.yaml")
runner = ProbeRunner(output_dir="outputs/", config=config)

# Granger is computed once and cached; subsequent calls load from disk
granger = runner.run_granger("METR-LA", raw_traffic)

result = runner.run_from_config("METR-LA", "gwn")
result.print_summary()
```

The `result.summary()` dict exposes scalar metrics suitable for logging or building comparison tables across models. For multi-seed runs it adds the mean-of-evaluations structural metrics (`aas_evaluation_grand_mean`, `community_modularity_evaluation_grand_mean`) with their seed vs horizon spreads; the per-horizon breakdown and pooled spreads live in `structure_seed_summary.json`. Consensus-graph diagnostics (`*_consensus_graph`) are deliberately named apart from the evaluation means so they are never plotted interchangeably.

---

## Output directory layout

```
outputs/
├── granger_cache/
│   └── METR-LA.npz                    ← serialised GrangerResult (reused across models)
├── dataset/
│   └── METR-LA/
│       ├── granger/
│       │   ├── gcg_matrix.npy
│       │   ├── granger_pvalues.npy
│       │   ├── granger_fstats.npy
│       │   ├── pearson_correlations.npy
│       │   ├── gcg_heatmap.{png,html}
│       │   └── gcg_network.{png,html}
│       └── gcg_communities/
│           ├── community_assignments.csv
│           ├── modularity_score.json
│           ├── gcs_per_community.json
│           └── gcs_overall.json
├── per_model/
│   └── gwn_METR-LA/
│       ├── summary.json
│       ├── full_metrics.json
│       ├── node_classification.csv
│       ├── lens1_sgs_matrix.npy
│       ├── lens1_sgs_mean.npy
│       ├── lens3_sweep_thresholds.npy
│       ├── lens3_sweep_f1.npy
│       ├── lens3_sweep_precision.npy
│       ├── lens3_sweep_recall.npy
│       ├── tp_fp_fn_matrix.npy
│       ├── alignment_scores.json
│       ├── lens3_per_horizon/
│       │   └── alignment_per_horizon.json
│       ├── lens4_community_assignments.npy
│       ├── lens4_degree_centrality.npy
│       ├── lens4_betweenness_centrality.npy
│       ├── lens4_closeness_centrality.npy
│       ├── lens4_eigenvector_centrality.npy
│       ├── centrality_metrics.csv
│       ├── community_assignments.csv
│       ├── degree_comparison.csv
│       ├── degree_comparison.json
│       ├── lens5_mean_sgs_per_horizon.npy
│       ├── lens5_aas_per_horizon.npy
│       ├── degradation_rates.json
│       └── figures/
│           ├── sgs_distribution.{png,html}
│           ├── spatial_gain_map.{png,html}
│           ├── lens1_sgs_horizon.{png,html}
│           ├── threshold_sweep.{png,html}
│           ├── community_map.{png,html}
│           ├── adjacency_heatmap.{png,html}
│           ├── centrality_heatmap.{png,html}
│           ├── centrality_by_community.{png,html}
│           ├── degree_comparison.{png,html}
│           ├── sgs_degradation.{png,html}
│           ├── aas_degradation.{png,html}
│           ├── correlation_plot.{png,html}
│           └── combined_dashboard.{png,html}
├── lens0_performance/
│   └── METR-LA/
│       ├── metrics_by_run.csv
│       ├── metrics_by_horizon.csv
│       ├── metrics_by_node.csv
│       ├── horizon_group_summary.csv
│       ├── baseline_improvements.csv
│       ├── model_rankings_by_horizon.csv
│       ├── model_rankings_by_group.csv
│       ├── degradation_rates.json
│       ├── model_registry.csv
│       ├── summary.json
│       └── figures/
│           ├── mae_by_horizon.{png,html}
│           ├── rmse_by_horizon.{png,html}
│           ├── mape_by_horizon.{png,html}
│           ├── short_boundary_long_mae.{png,html}
│           ├── improvement_over_tcn.{png,html}
│           ├── improvement_over_arima.{png,html}
│           ├── model_rank_heatmap.{png,html}
│           └── degradation_rate_bar.{png,html}
└── comparative/
    └── <dataset_slug>/                   # e.g. metr_la, pems_bay — one dir per dataset
        ├── comparative_summary.json
        ├── lens3_alignment_comparison.{png,html}
        ├── lens5_sgs_degradation.{png,html}
        ├── lens5_aas_degradation.{png,html}
        ├── lens5_combined_dashboard.{png,html}
        └── lens5_degradation_rate_correlation.{png,html}
```

Every figure is produced in two formats: a static `.png` (publication-ready, 2× resolution) and an interactive `.html` (hover, zoom, pan — no server required).

---

## Lens 0 — Forecasting Performance Benchmark

**File:** `lens0_performance.py`<br>
**Key outputs:** `metrics_by_horizon.csv`, `metrics_by_node.csv`, `baseline_improvements.csv`, `degradation_rates.json`

### What it measures

Lens 0 computes the forecasting performance benchmark before any spatial interpretation is performed. It evaluates each model across datasets, horizons, runs, and nodes using MAE, RMSE, and MAPE. It also computes baseline improvements, model rankings, and degradation rates. The node-level metrics produced by Lens 0 are used by Lens 1 to calculate TCN-based spatial utility.

Specifically, Lens 0 answers:

1. Which model predicts best overall?
2. At which horizon is each model strongest?
3. On which dataset does each model generalise better?
4. How do the STGNNs compare with ARIMA, per-node TCN, and shared channel-independent TCN?
5. How quickly does each model degrade as the prediction horizon increases?
6. What node-level errors are needed for Lens 1's spatial utility scoring?

### Input shapes

| Array | Expected shape | Notes |
|---|---|---|
| `ground_truth` | `(W, H, N)` | W paired test windows, H horizons, N nodes |
| model predictions | `(W, H, N, R)` | W paired test windows, H horizons, N nodes, R runs |

All models — including DSSA-TCN and STAEformer — use this same shape;
`run_training.py` saves every model's artifacts uniformly. Metrics are
`mean(|pred - true|)` etc. over the flattened `(window, node[, run])` pairs
— the average of the per-window error, not the error of the per-window
average.

### Model groups

| Model | Group |
|---|---|
| ARIMA | `statistical_baseline` |
| TCN-PerNode | `temporal_baseline` |
| GWN, GWN V2 | `graph_wavenet_based` |
| STAWnet, DSSA-TCN, STAEformer | `attention_adaptive_stgnn` |

### Output files

All written to `outputs/lens0_performance/{dataset}/`:

| File | Description |
|---|---|
| `metrics_by_run.csv` | MAE, RMSE, MAPE per model × run × horizon |
| `metrics_by_horizon.csv` | Mean ± std across runs |
| `metrics_by_node.csv` | Node-level metrics (consumed by Lens 1) |
| `horizon_group_summary.csv` | Short (30 min), boundary (60 min), long (90–210 min) |
| `baseline_improvements.csv` | % improvement over ARIMA and TCN |
| `model_rankings_by_horizon.csv` | Rank by MAE at each horizon |
| `model_rankings_by_group.csv` | Rank by MAE within horizon groups |
| `degradation_rates.json` | Linear slope of MAE vs. horizon per model |
| `model_registry.csv` | Static model metadata table |
| `summary.json` | Headline scalars (best model, best MAE, etc.) |
| `figures/mae_by_horizon.html` | Interactive MAE line chart |
| `figures/rmse_by_horizon.html` | RMSE line chart |
| `figures/mape_by_horizon.html` | MAPE line chart |
| `figures/short_boundary_long_mae.html` | Grouped bar by horizon range |
| `figures/improvement_over_tcn.html` | % improvement over TCN |
| `figures/improvement_over_arima.html` | % improvement over ARIMA |
| `figures/model_rank_heatmap.html` | Rank heatmap across horizons |
| `figures/degradation_rate_bar.html` | MAE slope per model |

PNG versions of all figures are also written when kaleido is available.

### How Lens 0 feeds Lens 1

`metrics_by_node.csv` contains per-node MAE for every model and horizon. Lens 1 uses TCN-PerNode
(`tcn`) as its node-level comparison baseline for the Spatial Gain Score (SGS). Rather than
re-computing errors from raw predictions, Lens 1 can load these pre-computed values directly.

### Python API

```python
from analysis.lens0_performance import run_lens0
import numpy as np

ground_truth = np.load("data/probe_inputs/metr_la/ground_truth.npy")  # (W, H, N)
predictions = {
    "arima": np.load("data/probe_inputs/metr_la/predictions/arima_predictions.npy"),
    "tcn": np.load("data/probe_inputs/metr_la/predictions/tcn_predictions.npy"),
    "gwn": np.load("data/probe_inputs/metr_la/predictions/gwn_predictions.npy"),
    "dssa_tcn": np.load("data/probe_inputs/metr_la/predictions/dssa_tcn_predictions.npy"),
}

result = run_lens0(
    dataset="METR-LA",
    ground_truth=ground_truth,
    predictions=predictions,
    horizons=[30, 60, 210],
    model_groups={
        "statistical_baseline": ["arima"],
        "temporal_baseline": ["tcn"],
        "graph_wavenet_based": ["gwn"],
        "attention_adaptive_stgnn": ["dssa_tcn"],
    },
    baselines={"statistical": "arima", "temporal": "tcn"},
    output_dir="outputs/",
    save_figures=True,  # set False to skip HTML/PNG export (e.g. in tests)
)

print(result.summary())
# {'best_model': 'gwn', 'best_mae': 2.31, ...}
```

### CLI (via ProbeRunner)

```python
from analysis import ProbeRunner, load_config

config = load_config("scripts/configs/metr_la_probe.yaml")
runner = ProbeRunner("outputs/", config=config)

# One dataset
lens0_result = runner.run_performance("METR-LA", save_figures=True)

# All datasets in config
all_results = runner.run_all_performance_from_config(save_figures=True)
```

---

## Lens 1 — Spatial Utility Test

**File:** `lens1_spatial_utility.py`<br>
**Key output:** `sgs_matrix` (N × H), `node_labels`

### What it measures

Lens 1 isolates *where* spatial context is helping by comparing per-node MAE between a purely temporal baseline (TCN) and the spatial model under test. For each node *i* and horizon *h*:

```
SGS(i, h) = MAE_TCN(i, h) − MAE_model(i, h)
```

- **Positive SGS** — the spatial model was more accurate at this node/horizon. Spatial context helped.
- **Negative SGS** — the spatial model was less accurate. The learned graph introduced noise.
- **Zero SGS** — spatial context had no effect.

The paper reports the **relative** gain, `SGS_rel(i, h) = SGS(i, h) / MAE_TCN(i, h)`. Node labels use the
mean of `SGS_rel(i, h)` over horizons: **beneficial** above `sgs_rel_threshold` (0.01), **harmful** below
−0.01, otherwise **neutral**. The headline `mean_sgs_rel` pools numerator and denominator,
`sum(SGS) / sum(MAE_TCN)`. SGS compares the complete STGNN with the temporal baseline; it does not
isolate the spatial mechanism, so "spatial context helped" below is shorthand for that comparison.

### Reports and how to read them

#### `sgs_distribution.html`
A horizontal bar chart of all nodes sorted descending by SGS_mean. Each bar is coloured green (beneficial), red (harmful), or blue (neutral). The two dashed vertical rules mark ±threshold.

*What to look for:* A strongly right-skewed distribution (most bars green, few red) indicates the model is generally leveraging its spatial graph well. A distribution clustered near zero or negative suggests the learned graph is adding noise rather than signal — the model may have overfit its adjacency to the training split.

#### `spatial_gain_map.html`
A geographic scatter of all sensor nodes coloured by SGS_mean on a Red–Yellow–Green diverging scale.

*What to look for:* Spatial clustering of colour is diagnostically important. If beneficial nodes (green) cluster in one district and harmful nodes (red) cluster in another, spatial heterogeneity is driving the effect — the graph may model inter-district correlations poorly. Randomly scattered colours with no geographic pattern suggest the effect is idiosyncratic to individual sensors rather than structural.

#### `lens1_sgs_horizon.html`
Two panels: mean SGS across all nodes at each prediction horizon (left) and the fraction of nodes with positive SGS at each horizon (right).

*What to look for:* A downward slope on both panels as horizon increases is the expected pattern — spatial correlations decay with temporal distance, so spatial benefit diminishes. A model that maintains a positive slope or a model whose beneficial-node fraction stays above 50% at long horizons (e.g. 180–210 minutes) is genuinely leveraging spatial structure in a way that persists beyond short-range correlations. A model that crosses zero at an early horizon indicates that its spatial graph captures only very short-range dependencies.

### Key scalars (`summary.json`)

| Key | Interpretation |
|---|---|
| `mean_sgs` | Overall mean SGS across all nodes and horizons. Positive = model benefits from spatial context on average. Target: > 0. |
| `pct_beneficial` | Fraction of nodes where SGS_mean > threshold. Target: > 0.5 for a well-functioning STGNN. |
| `pct_harmful` | Fraction of nodes where spatial context *hurt* accuracy. Values above ~0.2 warrant investigation. |

---

## Lens 2 — Granger Predictive Reference

**File:** `lens2_granger.py`<br>
**Key output:** `gcg_matrix` (N × N binary top-k reference), `granger_fstats` (N × N)

### What it measures

Lens 2 constructs a **statistical reference dependency graph** from the raw traffic time series using pairwise Granger causality tests. For every ordered pair (i, j), it asks: *does knowing the history of node i improve prediction of node j beyond its own history alone?*

Every ordered pair is tested with the same **fixed lag** `p = max_lag` (12 steps = 60 minutes, matching the model input window; there is no per-pair lag selection), so all F-statistics share their degrees of freedom and can be ranked against each other. For each target sensor j, the `gcg_top_k` (10) sources i with the largest F-statistic are kept. The resulting binary matrix is the reference graph (`gcg_matrix`, historically called the Granger Causality Graph, GCG). The reference is fitted once on the full series. p-values are kept only as a diagnostic: with series this long almost every pair is significant, so significance cannot control the graph's density.

> **On the word "causal":** Granger causality tests *predictive temporal dependence*, not true mechanistic causation. Traffic data is observational, and Granger tests can flag shared periodic patterns, confounders, and non-stationarity as apparent dependencies. The GCG is a principled statistical reference structure — not a ground-truth causal graph. Conclusions in this framework are framed as alignment with Granger-inferred dependencies, not proof of causation.

This step is expensive (O(N²) regressions) and is computed **once per dataset** then cached to `granger_cache/`. Subsequent model analyses load from cache automatically.

### Reports and how to read them

#### `gcg_heatmap.html`
An N×N binary heatmap. A blue cell at (i, j) means node i is among the top-k Granger sources of node j.

*What to look for:* The edge density is fixed by construction at k/(N−1) per target sensor (about 5% on METR-LA and 3% on PEMS-BAY at k = 10). Bands of dense rows/columns indicate highly influential or highly influenced sensors — these are candidate nodes to watch in Lens 1.

#### `gcg_network.html`
A force-directed network graph (spring layout, seed-0 for reproducibility) showing the directed Granger causality graph. Node hover shows out-degree.

*What to look for:* Hub nodes (high out-degree, drawn to the centre) are the sensors that most strongly Granger-predict others. These are typically high-traffic arterial intersections or on-ramps. If the learned adjacency from Lens 3 disagrees with these structural hubs, the model has learned a graph that diverges from the Granger reference structure.

### Key outputs

| Array | Shape | Description |
|---|---|---|
| `gcg_matrix.npy` | (N, N) uint8 | Binary Granger graph. |
| `granger_pvalues.npy` | (N, N) float64 | Raw p-values (diagonal = 1.0). |
| `granger_fstats.npy` | (N, N) float64 | F-statistics at the fixed lag. |
| `pearson_correlations.npy` | (N, N, L) float64 | Lagged Pearson correlations for all lags 1…L. |
| `optimal_lags.npy` | (N, N) int32 | Lag used per pair (always `max_lag`); 0 = the test failed. |

`optimal_lags` is kept for cache compatibility. Caches written by older versions of this code, which
used AIC lag selection up to `max_lag`, contain varying lags; check this array before reusing a
cache, and pass `--force-granger` after changing `max_lag` (the cache is keyed by dataset only).

---

## Lens 3 — Structural Alignment Test

**File:** `lens3_alignment.py`<br>
**Key output:** `AlignmentResult` with precision, recall, F1 (the **AAS**)

### What it measures

Lens 3 asks: *does the model's learned adjacency matrix align with the Granger causality graph?* It binarises the learned adjacency at a weight threshold and computes precision, recall, and F1 against the GCG as binary classification targets.

The **Adjacency Alignment Score (AAS) = F1** is the headline metric. It penalises both false edges (model weights Granger-unlinked pairs) and missed edges (model fails to weight Granger-linked pairs).

A threshold sweep from 0 to 1 in 21 steps reveals the trade-off between precision and recall as the binarisation cutoff varies, and shows the maximum achievable AAS across all thresholds.

### Reports and how to read them

#### `threshold_sweep.html`
Three lines (F1/AAS in blue, Precision in green, Recall in orange) plotted against edge-weight threshold. The vertical dashed rule marks the configured default threshold (0.1).

*What to look for:*
- **Peak F1 location:** If the peak is near threshold = 0.0, the model assigns uniformly low weights to all edges and has essentially learned a diffuse, uninformative graph. If the peak is near threshold = 1.0, the model is very sparse but precise.
- **Precision vs Recall trade-off:** A model with high precision but low recall has correctly identified a small subset of Granger-inferred dependencies but missed most. A model with high recall but low precision has recovered most Granger-linked pairs but at the cost of including many unlinked edges.
- **Recall near zero at all thresholds:** The learned graph has essentially no overlap with the Granger structure. The model may be using a fixed graph (road distance-based adjacency) rather than a learned one, or the learned graph has converged to a near-identity structure.

#### `threshold_sweep.html` — model comparison variant (`lens3_alignment_comparison.html`)
Grouped bar chart showing precision, recall, and AAS for every spatial model side-by-side.

*What to look for:* Which model best recovers the Granger-inferred dependency structure? Models that score well here are better candidates for interpretable deployment because their routing decisions align with statistically observed directional dependencies. A model that scores well on prediction accuracy (MAE) but poorly on AAS may be exploiting spatial correlations that are not directionally grounded in the Granger reference.

### Key scalars

| Key | Interpretation |
|---|---|
| `aas_consensus_graph` (F1) | AAS of the seed-mean *consensus* graph — a diagnostic of the representative graph, `F1(mean G)`. Not the primary metric. > 0.3 indicates meaningful alignment with the Granger graph. |
| `aas_evaluation_grand_mean` | **Primary** structural-alignment metric: mean over horizon×seed evaluations, `mean F1(G)` (present only for multi-seed runs). Reported per horizon as `per_horizon_aas_mean` ± `per_horizon_aas_seed_sd`. |
| `aas_seed_sd_pooled` / `aas_horizon_mean_sd` | Seed-stability spread (pooled within-horizon) vs across-horizon drift — kept separate so horizon degradation is never read as seed noise. |
| `aas_consensus_precision` | Of all consensus-graph edges above threshold, what fraction match a real Granger link? |
| `aas_consensus_recall` | Of all real Granger links, what fraction does the consensus graph recover? |
| `weighted_precision` | The fraction of total off-diagonal adjacency weight that falls on Granger-linked edges. Interpretable as "how much of the model's attention is Granger-justified?" |
| `tp`, `fp`, `fn` | Raw edge counts. Useful for understanding absolute scale. |

---

## Lens 4 — Community Coherence Test

**File:** `lens4_community.py`<br>
**Key output:** `CommunityResult` with community assignments, modularity, GCS

### What it measures

Lens 4 detects communities in the **learned adjacency matrix** using the Louvain algorithm (10 runs, best modularity partition retained) and asks whether those communities correspond to geographic clusters of sensors.

The **Geographic Coherence Score (GCS)** per community is:

```
GCS(c) = mean_inter_distance(c) / mean_intra_distance(c)
```

Values above 1 indicate that nodes within the community are closer to each other geographically than they are to nodes in other communities — the learned community is geographically coherent. Values below 1 indicate that the learned community spans geographic distances while grouping sensors that are geographically far apart.

The overall GCS is a community-size-weighted mean across all communities.

### Reports and how to read them

#### `community_map.html`
Geographic scatter of all nodes, coloured by Louvain community ID.

*What to look for:* Geographically contiguous colour regions indicate that the model has learned to attend to spatially proximate sensors within communities — consistent with physical traffic flow. Interspersed, non-contiguous colour regions indicate that the model is grouping sensors that are geographically distant, potentially overfitting to shared diurnal patterns (e.g., two residential areas with similar morning peak profiles) rather than physical road proximity.

#### `adjacency_heatmap.html`
The learned N×N adjacency matrix reordered by community, with cyan vertical and horizontal rules marking community boundaries.

*What to look for:* A block-diagonal structure (strong off-diagonal weights concentrated in the blocks) confirms that the model has learned intra-community attention. Uniform weights across blocks suggest the model is not exploiting community structure. Dense off-diagonal cross-community blocks with specific communities indicate a model that has learned inter-community bridges — potentially meaningful (highway on-ramps) or spurious.

#### `centrality_heatmap.html`
Geographic scatter of nodes coloured by degree centrality (sum of row weights in the learned adjacency).

*What to look for:* High-centrality sensors (warm colours) that align with known high-traffic arterials validate that the model has learned physically meaningful hub structure. High-centrality sensors in low-traffic areas suggest the model has over-attended to those nodes, possibly because they exhibit distinctive temporal patterns.

#### `centrality_by_community.html`
Box plot of chosen centrality measure by community.

*What to look for:* High variance within a community indicates heterogeneous node roles — some nodes are hubs, others are peripheral. Communities with uniformly high centrality are driving the global adjacency structure. Communities with near-zero centrality contribute little to inter-node attention and may be redundant in the learned graph.

### Key scalars

| Key | Interpretation |
|---|---|
| `community_modularity_consensus_graph` (Q) | Louvain modularity of the seed-mean consensus graph. Multi-seed runs also report `community_modularity_evaluation_grand_mean` (± `community_modularity_seed_sd_pooled`). Values > 0.3 indicate meaningful community structure. |
| `community_gcs` | Overall Geographic Coherence Score. > 1.5 indicates communities cluster geographically. < 1.0 suggests the model has learned non-geographic groupings. |
| `num_communities` | Number of Louvain communities. Compare across models — too few may indicate underfitting spatial structure, too many may indicate fragmentation. |

---

## Lens 5 — Horizon Degradation Test

**File:** `lens5_degradation.py`<br>
**Key output:** `DegradationResult` with per-horizon SGS and AAS curves, and degradation slopes

### What it measures

Lens 5 synthesises Lenses 1 and 3 across prediction horizons. It tracks two quantities as the horizon grows:

1. **Mean SGS per horizon** — how does the spatial benefit change as the prediction target moves further into the future?
2. **Horizon-specific AAS** — the model's *per-horizon* learned graph is aligned (matched top-k) against the fixed GCG at each horizon, asking: *does the graph the model uses at this horizon stay aligned with Granger structure?* (An earlier design restricted the GCG per horizon by optimal lag ≤ horizon, but the AIC-optimal lags cluster at long values, so short-horizon GCGs were empty and the AAS collapsed to 0 — the horizon signal now comes from the model's own per-horizon adjacencies.)

Both curves are fit with a linear regression. The **SGS rate** (slope) and **AAS rate** (slope) are the headline scalars. A steep negative SGS rate means spatial utility collapses quickly with horizon. A steep negative AAS rate means the model's graph drifts away from Granger structure at longer horizons.

The **cross-model correlation** between SGS rates and AAS rates tests the headline finding: *models whose learned graph maintains its Granger alignment as the horizon grows also maintain their spatial utility*.

### Reports and how to read them

#### `sgs_degradation.html`
Two-panel line chart: mean SGS per horizon (left) and fraction of beneficial nodes per horizon (right), one line per model.

*What to look for:*
- Models whose mean SGS stays positive across all horizons are genuinely leveraging spatial structure at long range.
- A model that crosses zero at H=12 (60 minutes) is effectively spatial only for short-range predictions.
- Compare the slope across models: a shallower negative slope means more robust spatial utility.
- Watch for models that improve at intermediate horizons before degrading — this can indicate that the learned graph captures medium-range temporal correlations specifically.

#### `aas_degradation.html`
Line chart of horizon-specific AAS per model across horizons.

*What to look for:* Each horizon uses that horizon's learned graph against the fixed GCG. An AAS that rises then falls suggests the model's mid-horizon graph best captures Granger structure. An AAS that falls monotonically indicates the graph drifts away from Granger structure as the horizon grows (structural degradation). A flat curve means the model reuses essentially one graph across horizons.

#### `combined_dashboard.html`
4-panel interactive dashboard combining both degradation curves (panels 1–3) with a scatter of SGS rate vs AAS rate across all models (panel 4 — the headline finding scatter).

*What to look for in the rate scatter (panel 4):* A positive correlation between AAS rate and SGS rate confirms the headline: models with better-preserved Granger alignment across horizons also preserve spatial utility across horizons. A model appearing in the top-right quadrant (both rates near zero or positive) is the theoretically ideal model — its graph stays Granger-aligned and its spatial utility does not decay. A model in the bottom-left quadrant (both rates strongly negative) degrades rapidly on both structural and performance dimensions.

#### `correlation_plot.html`
Standalone scatter of AAS rate vs SGS rate across models, with Spearman ρ and p-value in the title.

*What to look for:* A Spearman ρ > 0.6 with p < 0.05 provides statistical evidence for the headline finding. Given that this study uses six spatial models, note that power is limited — the correlation should be interpreted alongside the visualised pattern rather than relied on solely. A ρ near zero with a clear visual pattern may indicate a non-monotonic relationship.

### Key scalars

| Key | Interpretation |
|---|---|
| `sgs_rate` | Linear slope of mean SGS vs horizon index. Negative = degrading. Target: as close to zero or positive as possible. |
| `aas_rate` | Linear slope of AAS vs horizon index. Negative = structural alignment degrades with horizon. |
| `sgs_r2` | R² of the SGS linear fit. Values near 1 mean the degradation is uniformly linear; near 0 means the degradation is non-monotonic (investigate the raw curve). |
| `aas_r2` | R² of the AAS linear fit. |

---

## Cross-model comparative reports

When `run_all_from_config()` is called, an additional `comparative/<dataset_slug>/` directory (e.g. `comparative/metr_la/`) is written per dataset with figures spanning all models for that dataset:

| File | Content |
|---|---|
| `comparative_summary.json` | All ten summary scalars for every model, keyed by `{model}:{dataset}`. |
| `lens3_alignment_comparison.{png,html}` | Grouped bar: precision, recall, AAS per model. |
| `lens5_sgs_degradation.{png,html}` | Multi-model SGS degradation curves. |
| `lens5_aas_degradation.{png,html}` | Multi-model AAS degradation curves. |
| `lens5_combined_dashboard.{png,html}` | 4-panel dashboard for all models. |
| `lens5_degradation_rate_correlation.{png,html}` | Rate scatter with Spearman ρ (only written when ≥ 3 models are compared). |

The comparative figures are the primary evidence for cross-model ranking and the headline Spearman correlation claim.

---

## Interactive exploration with PyGWalker

For open-ended exploration in a Jupyter notebook, `explore.py` provides drag-and-drop DataFrames over the same probe results:

```python
from analysis.explore import explore, explore_nodes, explore_horizons, explore_alignment

# All metrics in one combined walker (nodes + horizons + alignment rows)
explore(result)

# Focused views
explore_nodes(result)  # node_id, sgs_mean, community_id, all centrality metrics, degree deltas
explore_horizons(result)  # horizon_step, mean_sgs, pct_beneficial, aas
explore_alignment(result)  # threshold, precision, recall, f1
```

Each function returns a PyGWalker `Walker` so the exploration spec can be exported to JSON and re-applied across models.

Typical notebook workflow:
1. Call `explore_nodes(result)` and drag `sgs_mean` to Y, `degree_centrality` to X — does centrality predict spatial benefit?
2. Drag `community_id` to Colour — do high-SGS nodes cluster within the same community?
3. Call `explore_horizons(result)` and plot `mean_sgs` vs `horizon_step` — at what horizon does the model become spatially neutral?

---

## Interpreting the headline finding

> This section describes an exploratory Lens 5 hypothesis from an earlier stage of the project. The paper does not test it: it reports SGS and AAS separately and finds that their model rankings differ.

The framework was originally designed to test one specific hypothesis:

> *An STGNN whose structural alignment with the Granger Causality Graph degrades more slowly across prediction horizons (AAS rate) will also exhibit slower spatial utility degradation (SGS rate).*

**Why rates, not absolute AAS?** The absolute AAS for each model is reported in `summary.json` and `comparative_summary.json` — it is the first-order quality check: does the learned graph overlap meaningfully with the Granger reference at all? The rate-vs-rate scatter tests a stronger, time-structured claim: models that *maintain* their Granger alignment as the horizon grows also *maintain* their spatial benefit. A model could have a respectable static AAS but a steep AAS rate, meaning its graph is only Granger-aligned at short lags and collapses at longer horizons. That model should — and under this framing will — show a correspondingly steep SGS rate.

### Step-by-step evaluation

1. **Check absolute AAS first** (`comparative_summary.json`, field `aas`). If all models cluster near 0, the learned graphs have no meaningful overlap with the Granger structure and the rate analysis is not meaningful — see failure modes below.
2. **Open** `comparative/lens5_degradation_rate_correlation.html`.
3. **A positive Spearman ρ** in the scatter (AAS rate on x, SGS rate on y) supports the hypothesis: models with a less-negative AAS rate also have a less-negative SGS rate. Both rates are expected to be negative (spatial utility and alignment generally decay with horizon); the correlation tests whether the *pace* of decay is shared.
4. **Individual outliers** — a model with a less-negative AAS rate but a strongly negative SGS rate — represents a case where maintained structural alignment has not translated into maintained spatial benefit. Investigate via Lens 1 (`pct_harmful`, `spatial_gain_map.html`) to identify whether harmful nodes are geographically concentrated, and via Lens 4 (`community_map.html`) to check whether those nodes share a community that the model has mis-weighted.

### Failure modes

| Observation | Meaning | Where to look |
|---|---|---|
| All models cluster at AAS ≈ 0 | Learned graphs have no overlap with Granger structure. Rate scatter is uninterpretable. | `gcg_network.html` edge density; `threshold_sweep.html` per model — check whether peak F1 is also near 0. |
| All AAS rates and SGS rates cluster near 0 | Neither alignment nor spatial utility degrades materially across the horizon range. The window may be too short to observe decay, or the dataset has strong long-range correlations. | Extend `horizons` beyond 42 steps if the dataset permits, then rerun. |
| SGS rate positive for one or more models | Spatial utility *improves* with horizon for those models. Unusual — likely indicates the model relies on long-lag graph edges that only become Granger-active at longer horizons. | Cross-reference `optimal_lags.npy` from Lens 2: are the high-lag pairs present in that model's learned adjacency? |
| Spearman ρ near 0 with a visible linear trend | N=6 models gives very low power for Spearman. Do not rely on the p-value alone. | Report ρ and its 95% bootstrap confidence interval. A visually clear trend with a wide CI is still evidence worth reporting. |

---

## Configuration reference

```yaml
# scripts/configs/metr_la_probe.yaml (the paper's configuration)
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

### Per-horizon adjacency (optional)

If a model exposes per-horizon learned graphs, place them at:
```
{adjacency_dir}/{model}_adjacency_h{step}.npy
```
for each step in `horizons`. When all per-horizon files are present, Lens 3 is computed against the horizon-specific graph at each step (stored in `lens3_per_horizon/`), and the global adjacency used for Lens 4 is the mean across horizons. When per-horizon files are absent, the single global adjacency is used everywhere.
