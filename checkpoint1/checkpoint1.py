"""BIOSTAT 707 Checkpoint 1: cohort characterization and EDA, set-a.

Run with:  pixi run --locked checkpoint1
No arguments. Writes everything to output/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
from tableone import TableOne

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
# Use data/ at the repository root when checkpoint1/data/ is not there.
if not (DATA / "Outcomes-a.txt").exists() and (ROOT.parent / "data" / "Outcomes-a.txt").exists():
    DATA = ROOT.parent / "data"
OUT = ROOT / "output"
EXPECTED_N = 4000

STATIC_VARS = ["RecordID", "Age", "Gender", "Height", "ICUType", "Weight"]
DESC_VARS = ["Age", "Gender", "Height", "ICUType", "Weight"]

# Used only for wide-table summaries. Long table keeps the raw Value.
PHYSIO_BOUNDS = {
    "Age": (15, 120),
    "Gender": (0, 1),
    "Height": (100, 250),
    "ICUType": (1, 4),
    "Weight": (20, 300),
    "HR": (20, 250),
    "Temp": (30, 45),
    "SysABP": (40, 300),
    "DiasABP": (15, 200),
    "MAP": (20, 250),
    "NISysABP": (40, 300),
    "NIDiasABP": (15, 200),
    "NIMAP": (20, 250),
    "RespRate": (4, 80),
    "GCS": (3, 15),
    "FiO2": (0.21, 1.0),
    "pH": (6.5, 8.0),
    "PaCO2": (5, 150),
    "PaO2": (15, 700),
    "SaO2": (40, 100),
    "Glucose": (10, 1000),
    "BUN": (1, 250),
    "Creatinine": (0.1, 30),
    "HCT": (5, 70),
    "HCO3": (5, 55),
    "K": (1.0, 10),
    "Na": (95, 180),
    "Mg": (0.2, 5),
    "Platelets": (5, 1500),
    "WBC": (0.1, 150),
    "Albumin": (0.5, 6.5),
    "ALP": (5, 3000),
    "ALT": (1, 20000),
    "AST": (1, 20000),
    "Bilirubin": (0.1, 50),
    "Lactate": (0.1, 30),
    "Cholesterol": (20, 600),
    "TroponinI": (0.0, 100),
    "TroponinT": (0.0, 50),
    "Urine": (0, 3000),
    "MechVent": (0, 1),
}


def check_set_a() -> None:
    """Confirm data/set-a/ and data/Outcomes-a.txt are already in place."""
    assert (DATA / "Outcomes-a.txt").exists(), "data/Outcomes-a.txt is missing"
    n = len(list((DATA / "set-a").glob("*.txt")))
    assert n == EXPECTED_N, f"data/set-a has {n} records, expected {EXPECTED_N}"


def load_set_a() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (long time-series table, wide general-descriptor table)."""
    frames = []
    for f in sorted((DATA / "set-a").glob("*.txt")):
        df = pd.read_csv(f)
        df["RecordID"] = int(f.stem)
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(OUT / "set-a_long.csv", index=False)

    static = (
        raw[(raw["Time"] == "00:00") & raw["Parameter"].isin(DESC_VARS)]
        .pivot_table(
            index="RecordID", columns="Parameter", values="Value", aggfunc="first"
        )
        .reset_index()
    )
    static = static.replace(-1, pd.NA)
    ts = raw[~raw["Parameter"].isin(STATIC_VARS)]
    return ts, static


def _parse_hours(series: pd.Series) -> pd.Series:
    parts = series.str.split(":", expand=True)
    return parts[0].astype(int) + parts[1].astype(int) / 60.0


def _clean_static(static: pd.DataFrame) -> pd.DataFrame:
    out = static.copy()
    for col in DESC_VARS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    h = out["Height"]
    h = h.mask(h.eq(0))
    h = h.mask(h.between(1.2, 2.5), h * 100.0)
    h = h.mask(h.between(40, 90), h * 2.54)
    out["Height"] = h.where(h.between(100, 250))
    w = out["Weight"]
    out["Weight"] = w.where(w.between(20, 300))
    out["Age"] = out["Age"].where(out["Age"].between(15, 120))
    return out


def _clean_ts(ts: pd.DataFrame) -> pd.DataFrame:
    df = ts.copy()
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["time_hours"] = _parse_hours(df["Time"])
    cleaned = df["Value"].mask(df["Value"] < 0)
    fio2 = df["Parameter"].eq("FiO2")
    cleaned = cleaned.mask(fio2 & cleaned.eq(0))
    percent = fio2 & cleaned.gt(1.0) & cleaned.le(100)
    cleaned = cleaned.mask(percent, cleaned / 100.0)
    for name, (lo, hi) in PHYSIO_BOUNDS.items():
        mask = df["Parameter"].eq(name) & cleaned.notna()
        cleaned = cleaned.mask(mask & ((cleaned < lo) | (cleaned > hi)))
    df["value_clean"] = cleaned
    return df


def _aggregate_window(df: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    valid = df.dropna(subset=["value_clean"]).sort_values(
        ["RecordID", "Parameter", "time_hours"]
    )
    g = valid.groupby(["RecordID", "Parameter"])["value_clean"].agg(
        n="count", first="first", last="last", min="min", max="max", mean="mean"
    )
    wide = g.unstack("Parameter")
    wide.columns = [f"{param}_{stat}{suffix}" for stat, param in wide.columns]
    urine = valid[valid["Parameter"].eq("Urine")]
    if not urine.empty:
        wide = wide.join(
            urine.groupby("RecordID")["value_clean"].sum().rename(f"Urine_sum{suffix}")
        )
    return wide


def _write_table1(static: pd.DataFrame, outcomes: pd.DataFrame) -> None:
    oc = outcomes.rename(
        columns={"SAPS-I": "SAPS_I", "In-hospital_death": "In_hospital_death"}
    )
    df = static.merge(oc, on="RecordID")
    df["Gender"] = df["Gender"].map({0: "female", 1: "male"})
    df["ICUType"] = df["ICUType"].map({1: "CCU", 2: "CSRU", 3: "MICU", 4: "SICU"})
    df["death"] = df["In_hospital_death"].map({0: "Survivor", 1: "Died"})
    t1 = TableOne(
        df,
        columns=[
            "Age",
            "Gender",
            "Height",
            "ICUType",
            "Weight",
            "SAPS_I",
            "SOFA",
            "Length_of_stay",
        ],
        categorical=["Gender", "ICUType"],
        groupby="death",
        pval=True,
        missing=True,
        overall=True,
    )
    t1.to_csv(OUT / "table1.csv")


def _write_outcomes(outcomes: pd.DataFrame) -> None:
    death = outcomes["In-hospital_death"]
    rows = [
        ("n", int(len(outcomes))),
        ("n_died", int(death.sum())),
        ("n_survived", int((death == 0).sum())),
        ("mortality", float(death.mean())),
        ("SAPS_I_mean", float(outcomes["SAPS-I"].mean())),
        ("SOFA_mean", float(outcomes["SOFA"].mean())),
        ("Length_of_stay_mean", float(outcomes["Length_of_stay"].mean())),
        ("SAPS_I_mean_died", float(outcomes.loc[death == 1, "SAPS-I"].mean())),
        ("SAPS_I_mean_survived", float(outcomes.loc[death == 0, "SAPS-I"].mean())),
        ("SOFA_mean_died", float(outcomes.loc[death == 1, "SOFA"].mean())),
        ("SOFA_mean_survived", float(outcomes.loc[death == 0, "SOFA"].mean())),
    ]
    pd.DataFrame(rows, columns=["item", "value"]).to_csv(
        OUT / "outcomes.csv", index=False
    )


def _write_missingness(
    ts: pd.DataFrame, static: pd.DataFrame, wide: pd.DataFrame
) -> None:
    desc = static.set_index("RecordID")[DESC_VARS]
    n_patients = int(wide.shape[0])
    observed = (
        ts.dropna(subset=["value_clean"])
        .groupby("Parameter")["RecordID"]
        .nunique()
    )
    miss = (
        observed.reindex(sorted(ts["Parameter"].unique()))
        .fillna(0)
        .astype(int)
        .rename("n_measured")
        .to_frame()
    )
    miss["n_unmeasured"] = n_patients - miss["n_measured"]
    miss["pct_never_measured"] = 100 * miss["n_unmeasured"] / n_patients
    miss = miss.sort_values("pct_never_measured")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 12), gridspec_kw={"height_ratios": [1.0, 1.7]}
    )
    msno.matrix(desc, sparkline=False, fontsize=9, ax=axes[0])
    axes[0].set_title("Admission descriptors (white = missing)")
    plot_df = miss.reset_index(names="variable")
    plot_df["band"] = np.where(plot_df["pct_never_measured"] >= 50, ">=50%", "<50%")
    sns.barplot(
        data=plot_df,
        x="pct_never_measured",
        y="variable",
        hue="band",
        palette={">=50%": "#E45756", "<50%": "#4C78A8"},
        dodge=False,
        ax=axes[1],
    )
    axes[1].axvline(50, color="#9E9E9E", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Never measured in the 48h window (% of stays)")
    axes[1].set_ylabel("")
    axes[1].set_title("Time-series variables: patient-level missingness")
    axes[1].set_xlim(0, 100)
    axes[1].legend(frameon=False, title="")
    fig.savefig(OUT / "missingness_map.png", dpi=160, bbox_inches="tight")
    plt.close("all")

    death = wide["In_hospital_death"]
    rows = []
    for param in miss.index:
        measured = ts.loc[
            ts["Parameter"].eq(param) & ts["value_clean"].notna(), "RecordID"
        ].drop_duplicates()
        flag = wide.index.isin(measured)
        n_m = int(flag.sum())
        n_u = int((~flag).sum())
        if n_m == 0 or n_u == 0:
            continue
        rows.append(
            {
                "variable": param,
                "n_measured": n_m,
                "n_unmeasured": n_u,
                "pct_unmeasured": 100 * n_u / n_patients,
                "mort_measured": float(death.loc[flag].mean() * 100),
                "mort_unmeasured": float(death.loc[~flag].mean() * 100),
            }
        )
    out = pd.DataFrame(rows)
    out["diff"] = out["mort_unmeasured"] - out["mort_measured"]
    out = out.sort_values("diff")
    out.to_csv(OUT / "missingness_vs_death.csv", index=False)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    check_set_a()
    ts, static = load_set_a()
    outcomes = pd.read_csv(DATA / "Outcomes-a.txt")
    assert len(outcomes) == EXPECTED_N

    static = _clean_static(static)
    ts = _clean_ts(ts)

    wide_48 = _aggregate_window(ts)
    counts = (
        ts.dropna(subset=["value_clean"])
        .groupby(["RecordID", "Parameter"])
        .size()
        .rename("n")
        .reset_index()
    )
    median_n = counts.groupby("Parameter")["n"].median()
    split_vars = median_n[median_n >= 4].index.tolist()
    ts_hi = ts[ts["Parameter"].isin(split_vars)]
    wide_024 = _aggregate_window(ts_hi[ts_hi["time_hours"] < 24], suffix="_0_24")
    wide_2448 = _aggregate_window(ts_hi[ts_hi["time_hours"] >= 24], suffix="_24_48")

    oc = outcomes.rename(
        columns={"SAPS-I": "SAPS_I", "In-hospital_death": "In_hospital_death"}
    )
    wide = (
        static.set_index("RecordID")
        .join(wide_48, how="outer")
        .join(wide_024, how="left")
        .join(wide_2448, how="left")
        .join(oc.set_index("RecordID"), how="left")
    )
    wide.index.name = "RecordID"
    if "MechVent_max" in wide.columns:
        wide["MechVent_any"] = wide["MechVent_max"]
    wide.to_csv(OUT / "set-a_wide.csv")

    _write_table1(static, outcomes)
    _write_outcomes(outcomes)
    _write_missingness(ts, static, wide)

    print("Checkpoint 1 complete. See output/.")


if __name__ == "__main__":
    main()
