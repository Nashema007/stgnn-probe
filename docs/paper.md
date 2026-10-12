# Paper details

Settings, conventions and scope behind the results in the paper. See the [README](../README.md) for how to run everything.

## Paper configuration

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

## Edge convention

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

## What the paper does and does not use

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
