# Statistical Methods for Learning (BIOSTAT 707)

This repository is for BIOSTAT 707 programming assignments. Each checkpoint lives in its own folder at the repo root.

```
.
├── README.md
└── checkpoint1/
```

## Checkpoint 1

All Checkpoint 1 files are in `checkpoint1/`. From that folder, run:

```bash
cd checkpoint1
pixi run --locked checkpoint1
```

This requires `data/set-a/` (4000 records) and `data/Outcomes-a.txt` to already be in place inside `checkpoint1/`. The script writes `output/set-a_long.csv`, `output/set-a_wide.csv`, `output/table1.csv`, `output/outcomes.csv`, `output/missingness_map.png`, and `output/missingness_vs_death.csv`. Interpretation is in `checkpoint1/checkpoint1_writeup.md`.
