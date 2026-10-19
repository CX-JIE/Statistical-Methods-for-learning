# Statistical Methods for Learning (BIOSTAT 707)

This repository is for BIOSTAT 707 programming assignments. Each checkpoint lives in its own folder at the repo root.

```
.
├── README.md
├── checkpoint1/
└── checkpoint2/
```

Everything belonging to a checkpoint lives inside that checkpoint's folder: its code or notebook,
its `pixi.toml`/`pixi.lock`, and its own `output/` with the written report. Nothing outside the
folder is needed except the `data/` directory, which is not committed.

| Checkpoint | Folder | Written report | Command |
| --- | --- | --- | --- |
| 1 | `checkpoint1/` | `checkpoint1/output/` (`checkpoint1_writeup.md` plus tables) | `cd checkpoint1 && pixi run --locked checkpoint1` |
| 2 | `checkpoint2/` | `checkpoint2/output/checkpoint2.html` | `cd checkpoint2 && pixi run --locked checkpoint2` |

## Checkpoint 1

All Checkpoint 1 files are in `checkpoint1/`. From that folder, run:

```bash
cd checkpoint1
pixi run --locked checkpoint1
```

This requires `data/set-a/` (4000 records) and `data/Outcomes-a.txt` to already be in place inside `checkpoint1/`. The script writes `output/set-a_long.csv`, `output/set-a_wide.csv`, `output/table1.csv`, `output/outcomes.csv`, `output/missingness_map.png`, and `output/missingness_vs_death.csv`. Interpretation is in `checkpoint1/checkpoint1_writeup.md`.

## Checkpoint 2

All Checkpoint 2 files are in `checkpoint2/`. From that folder, run:

```bash
cd checkpoint2
pixi run --locked checkpoint2
```

This requires `data/set-a/` (4000 records) and `data/Outcomes-a.txt` to already be in place inside `checkpoint2/` (for example `cp -R ../checkpoint1/data ./data`). The pixi task executes `checkpoint2.ipynb` and writes the report to `output/checkpoint2.html`; the committed notebook has its outputs stripped, so the HTML is the readable version.

What the run writes under `checkpoint2/output/`:

* `checkpoint2.html` — the written report (sections follow the checklist).
* `cp2_final_model.joblib` — the final gradient-boosting model refitted on all of set-a, with its feature list and tuned configuration, reloaded and checked at the end of the notebook.
* `cp2_oof_predictions.csv` — out-of-fold predictions for Models 1, 2 and 3 plus SAPS-I and SOFA, with fold and outcome.
* `cp2_model_comparison.csv`, `cp2_modeling_table.csv`, `cp2_predictors.csv`, `cp2_dropped_variables.csv`, `cp2_folds.csv`, `cp2_lasso_coefficients.csv`, `cp2_lasso_stability.csv`, `cp2_gam_partial_effect_ors.csv` — tables behind the report.
* `cp2_lasso_coefficients.png`, `cp2_gam_partial_effects.png`, `cp2_roc_oof.png`, `cp2_fold_auc.png` — figures 1 to 4.

Runtime is a few minutes (roughly 4 to 7 minutes depending on the machine and load) on an Apple silicon laptop, most of it in the inner cross-validation loops for the three models. The code reads only set-a; set-b and set-c are never opened. `cp2_lib.py` holds the shared functions (data preparation, folds, models, plots) and `checkpoint2.ipynb` is the report that calls them.
