"""Shared code for BIOSTAT 707 Checkpoint 2 (set-a model building and comparison).

The notebook `checkpoint2.ipynb` is the written report; it calls the functions in
this module.  Everything reads from `data/` and writes to `output/`; there are no
absolute paths.  Random seeds are fixed so folds and fits are reproducible.

The modelling table has one row per ICU stay, summarising hours 0-48 of set-a.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve  # noqa: F401  (used in CP3)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 707
N_FOLDS = 5
N_FOLDS_INNER = 4
DATA = Path("data")
OUT = Path("output")


def find_data_dir() -> Path:
    """checkpoint2/data/ if it holds set-a, otherwise data/ at the repository root."""
    for candidate in (Path("data"), Path("..") / "data"):
        if (candidate / "set-a").is_dir() and (candidate / "Outcomes-a.txt").is_file():
            return candidate
    raise FileNotFoundError(
        "set-a and Outcomes-a.txt not found: place them in checkpoint2/data/ or in data/ at the repository root"
    )

STATIC_VARS = ["RecordID", "Age", "Gender", "Height", "ICUType", "Weight"]

# Never predictors: they describe what happened after the 48 h window.
LEAKY_VARS = ["Length_of_stay", "Survival"]

# Same cleaning bounds as Checkpoint 1.
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

# Time-series variables kept as predictors: measured at least once in >= 50% of stays.
MIN_MEASURED_FRACTION = 0.50
TS_KEEP = [
    "HR",
    "Temp",
    "SysABP",
    "DiasABP",
    "MAP",
    "NISysABP",
    "NIDiasABP",
    "NIMAP",
    "GCS",
    "FiO2",
    "pH",
    "PaCO2",
    "PaO2",
    "Glucose",
    "BUN",
    "Creatinine",
    "HCT",
    "HCO3",
    "K",
    "Na",
    "Mg",
    "Platelets",
    "WBC",
    "Lactate",
]
# Variables whose trend over the 48 h window is clinically informative.
SLOPE_VARS = ["HR", "SysABP", "Temp", "GCS", "BUN", "Creatinine", "Lactate", "WBC"]
# Too sparse to model as values; only their "never measured" flag is kept.
RARE_TS = ["RespRate", "SaO2", "Albumin", "ALP", "ALT", "AST", "Bilirubin", "TroponinI", "TroponinT", "Cholesterol"]
LIVER_PANEL = ["ALP", "ALT", "AST", "Bilirubin"]

GROUP_REASON = {
    "Admission": "Known at ICU admission; age and sex set the baseline risk, weight and height proxy body habitus.",
    "Latest value": "Most recent value in hours 0-48: the current physiological state at the end of the prediction window.",
    "Minimum": "Worst low value in hours 0-48: captures hypotensive, hypoxic or haematological nadirs.",
    "Maximum": "Worst high value in hours 0-48: captures fever, tachycardia, hypertension and peaks of renal or liver markers.",
    "Trend": "Least-squares slope per hour over hours 0-48: direction of deterioration or recovery.",
    "Monitoring intensity": "Number of valid measurements in hours 0-48, or total urine output: how closely the patient was watched and how much urine was produced.",
    "Missingness": "Indicator that a variable was never measured, or that an admission value was not recorded: ordering a test is itself a clinical decision and is informative.",
}

MISSING_LABEL = {
    "Height_missing": "Height not recorded at admission (indicator)",
    "Weight_missing": "Admission weight not recorded (indicator)",
    "Gender_missing": "Sex not recorded (indicator)",
    "ICUType_missing": "ICU type not recorded (indicator)",
    "TroponinI_never": "Troponin I never measured in hours 0-48 (indicator)",
    "TroponinT_never": "Troponin T never measured in hours 0-48 (indicator)",
    "LiverPanel_never": "None of ALP, ALT, AST or bilirubin measured in hours 0-48 (indicator)",
    "Albumin_never": "Albumin never measured in hours 0-48 (indicator)",
    "RespRate_never": "Respiratory rate never measured in hours 0-48 (indicator)",
}

MEASURE_REASON = {
    "Age": "Age at admission (years)",
    "Gender": "Sex, 1 = male",
    "Height": "Height (cm) after unit harmonisation",
    "Weight": "Admission weight (kg)",
    "ICUType": "ICU type, one-hot with MICU as reference",
    "HR": "Heart rate (bpm)",
    "Temp": "Temperature (degrees C)",
    "SysABP": "Invasive systolic arterial pressure (mmHg)",
    "DiasABP": "Invasive diastolic arterial pressure (mmHg)",
    "MAP": "Invasive mean arterial pressure (mmHg)",
    "NISysABP": "Non-invasive systolic pressure (mmHg)",
    "NIDiasABP": "Non-invasive diastolic pressure (mmHg)",
    "NIMAP": "Non-invasive mean pressure (mmHg)",
    "GCS": "Glasgow Coma Scale (3-15)",
    "FiO2": "Inspired oxygen fraction (0.21-1.0)",
    "pH": "Arterial pH",
    "PaCO2": "Arterial CO2 partial pressure (mmHg)",
    "PaO2": "Arterial O2 partial pressure (mmHg)",
    "Glucose": "Blood glucose (mg/dL)",
    "BUN": "Blood urea nitrogen (mg/dL)",
    "Creatinine": "Serum creatinine (mg/dL)",
    "HCT": "Haematocrit (%)",
    "HCO3": "Serum bicarbonate (mmol/L)",
    "K": "Serum potassium (mmol/L)",
    "Na": "Serum sodium (mmol/L)",
    "Mg": "Serum magnesium (mmol/L)",
    "Platelets": "Platelet count (10^3/muL)",
    "WBC": "White blood cell count (10^3/muL)",
    "Lactate": "Serum lactate (mmol/L)",
    "Urine": "Hourly urine output (mL)",
    "MechVent": "Mechanical ventilation recorded (1 = ventilated)",
}

# ---------------------------------------------------------------------------
# Loading and cleaning
# ---------------------------------------------------------------------------


def _parse_hours(series: pd.Series) -> pd.Series:
    parts = series.str.split(":", expand=True)
    return parts[0].astype(int) + parts[1].astype(int) / 60.0


def load_set_a(data_dir: Path = DATA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the long time-series table for set-a and the outcome table."""
    files = sorted((data_dir / "set-a").glob("*.txt"))
    if len(files) != 4000:
        raise ValueError(f"expected 4000 records in data/set-a, found {len(files)}")
    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["RecordID"] = int(path.stem)
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw["time_hours"] = _parse_hours(raw["Time"])
    outcomes = pd.read_csv(data_dir / "Outcomes-a.txt")
    return raw, outcomes


def clean_values(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the Checkpoint 1 value cleaning: -1 codes, FiO2 units, plausibility bounds."""
    df = raw.copy()
    value = pd.to_numeric(df["Value"], errors="coerce")
    value = value.mask(value < 0)  # -1 means not measured
    fio2 = df["Parameter"].eq("FiO2")
    value = value.mask(fio2 & value.eq(0))
    value = value.mask(fio2 & value.gt(1.0) & value.le(100), value / 100.0)
    for name, (lo, hi) in PHYSIO_BOUNDS.items():
        mask = df["Parameter"].eq(name) & value.notna() & ((value < lo) | (value > hi))
        value = value.mask(mask)
    df["value_clean"] = value
    return df


def admission_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Admission descriptors, cleaned as in Checkpoint 1 (Height unit fixes, bounds)."""
    static = (
        raw[(raw["Time"] == "00:00") & raw["Parameter"].isin(["Age", "Gender", "Height", "ICUType", "Weight"])]
        .pivot_table(index="RecordID", columns="Parameter", values="Value", aggfunc="first")
        .reindex(columns=["Age", "Gender", "Height", "ICUType", "Weight"])
    )
    static = static.replace(-1, np.nan).apply(pd.to_numeric, errors="coerce")

    height = static["Height"].mask(static["Height"].eq(0))
    height = height.mask(height.between(1.2, 2.5), height * 100.0)  # metres
    height = height.mask(height.between(40, 90), height * 2.54)  # inches
    static["Height"] = height.where(height.between(100, 250))
    static["Weight"] = static["Weight"].where(static["Weight"].between(20, 300))
    static["Age"] = static["Age"].where(static["Age"].between(15, 120))
    static["Gender"] = static["Gender"].where(static["Gender"].isin([0, 1]))
    return static


def summarise_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """One row per stay with last/min/max/n (and slope for selected variables)."""
    valid = df.dropna(subset=["value_clean"]).sort_values(["RecordID", "Parameter", "time_hours"])

    keep = valid[valid["Parameter"].isin(TS_KEEP)]
    agg = keep.groupby(["RecordID", "Parameter"])["value_clean"].agg(
        last="last", min="min", max="max", n="count"
    )
    wide = agg.unstack("Parameter")
    wide.columns = [f"{param}_{stat}" for stat, param in wide.columns]

    # Slope per hour, computed with a vectorised least-squares formula.
    slope_df = valid[valid["Parameter"].isin(SLOPE_VARS)].copy()
    gb = slope_df.groupby(["RecordID", "Parameter"])
    slope_df["x_mean"] = gb["time_hours"].transform("mean")
    slope_df["y_mean"] = gb["value_clean"].transform("mean")
    slope_df["dx"] = slope_df["time_hours"] - slope_df["x_mean"]
    slope_df["cov"] = slope_df["dx"] * (slope_df["value_clean"] - slope_df["y_mean"])
    slope_df["dx2"] = slope_df["dx"] ** 2
    slope = slope_df.groupby(["RecordID", "Parameter"]).agg(
        cov=("cov", "sum"), var=("dx2", "sum"), n=("value_clean", "count")
    )
    slope = (slope["cov"] / slope["var"].where((slope["n"] >= 2) & slope["var"].gt(0))).unstack("Parameter")
    slope.columns = [f"{param}_slope" for param in slope.columns]
    wide = wide.join(slope)

    # Binary mechanical ventilation: any record during the window.
    vent = valid[valid["Parameter"].eq("MechVent")].groupby("RecordID")["value_clean"]
    wide["MechVent_any"] = vent.max()
    wide["MechVent_n"] = vent.count()

    # Urine: hourly volumes, so the 48 h total and the number of recorded hours.
    urine = valid[valid["Parameter"].eq("Urine")].groupby("RecordID")["value_clean"]
    wide["Urine_sum"] = urine.sum()
    wide["Urine_n"] = urine.count()

    # Number of valid measurements for the rare variables, used for missingness flags only.
    for name in RARE_TS:
        wide[f"{name}_n"] = valid[valid["Parameter"].eq(name)].groupby("RecordID").size()
    return wide


def build_modeling_table(raw: pd.DataFrame, outcomes: pd.DataFrame) -> dict:
    """Assemble the one-row-per-stay modelling table plus the predictor catalogue."""
    cleaned = clean_values(raw)
    admission = admission_table(raw)
    ts = summarise_time_series(cleaned)

    n_stays = admission.shape[0]
    for name in RARE_TS:
        column = f"{name}_n"
        ts[column] = ts[column].fillna(0)

    table = admission.join(ts, how="left")
    table["Height_missing"] = table["Height"].isna().astype(float)
    table["Weight_missing"] = table["Weight"].isna().astype(float)
    table["Gender_missing"] = table["Gender"].isna().astype(float)
    table["ICUType_missing"] = table["ICUType"].isna().astype(float)

    # Missingness predictors: never measured during hours 0-48.
    table["TroponinI_never"] = (table["TroponinI_n"] == 0).astype(float)
    table["TroponinT_never"] = (table["TroponinT_n"] == 0).astype(float)
    liver_n = table[[f"{name}_n" for name in LIVER_PANEL]].sum(axis=1)
    table["LiverPanel_never"] = (liver_n == 0).astype(float)
    table["Albumin_never"] = (table["Albumin_n"] == 0).astype(float)
    table["RespRate_never"] = (table["RespRate_n"] == 0).astype(float)

    # ICU type as categorical, MICU as the reference level.
    icu = table["ICUType"].map({1: "CCU", 2: "CSRU", 3: "MICU", 4: "SICU"})
    for level in ["CCU", "CSRU", "SICU"]:
        table[f"ICUType_{level}"] = (icu == level).astype(float)

    outcomes_clean = outcomes.rename(
        columns={"SAPS-I": "SAPS_I", "In-hospital_death": "In_hospital_death"}
    ).set_index("RecordID")
    for column in ["SAPS_I", "SOFA"]:
        outcomes_clean[column] = outcomes_clean[column].where(outcomes_clean[column] >= 0)
    table = table.join(outcomes_clean[["SAPS_I", "SOFA", "In_hospital_death"]], how="left")

    features = feature_list(table)
    table = table.set_index(table.index.rename("RecordID")).sort_index()
    model_matrix = table[["In_hospital_death", "SAPS_I", "SOFA"] + features].copy()
    model_matrix["In_hospital_death"] = model_matrix["In_hospital_death"].astype(int)

    if model_matrix.index.has_duplicates:
        raise ValueError("modelling table has duplicate RecordIDs")
    if model_matrix.shape[0] != n_stays:
        raise ValueError("modelling table does not have one row per stay")

    return {
        "table": model_matrix,
        "features": features,
        "catalogue": predictor_catalogue(features),
        "dropped": dropped_table(cleaned, n_stays),
        "n_stays": n_stays,
        "n_deaths": int(model_matrix["In_hospital_death"].sum()),
    }


def feature_list(table: pd.DataFrame) -> list[str]:
    """Predictor columns in a fixed order, excluding outcomes and benchmarks."""
    excluded = {"SAPS_I", "SOFA", "In_hospital_death", "Length_of_stay", "Survival"}
    features = []
    features += ["Age", "Gender", "Height", "Weight", "Height_missing", "Weight_missing", "Gender_missing"]
    features += ["ICUType_CCU", "ICUType_CSRU", "ICUType_SICU", "ICUType_missing"]
    for name in TS_KEEP:
        features += [f"{name}_last", f"{name}_min", f"{name}_max", f"{name}_n"]
        if name in SLOPE_VARS:
            features.append(f"{name}_slope")
    features += ["MechVent_any", "MechVent_n", "Urine_sum", "Urine_n"]
    features += [
        "TroponinI_never",
        "TroponinT_never",
        "LiverPanel_never",
        "Albumin_never",
        "RespRate_never",
    ]
    features = [f for f in features if f in table.columns]
    assert not (set(features) & excluded)
    return features


def predictor_catalogue(features: list[str]) -> pd.DataFrame:
    """Table of predictors with the group they belong to and the measurement behind them."""
    rows = []
    for name in features:
        if name in MISSING_LABEL:
            group = "Missingness"
        elif name in {"Age", "Gender", "Height", "Weight"}:
            group = "Admission"
        elif name.startswith("ICUType"):
            group = "Admission"
        elif name in {"MechVent_any", "MechVent_n", "Urine_sum", "Urine_n"}:
            group = "Monitoring intensity"
        elif name.endswith("_slope"):
            group = "Trend"
        elif name.endswith("_last"):
            group = "Latest value"
        elif name.endswith("_min"):
            group = "Minimum"
        elif name.endswith("_max"):
            group = "Maximum"
        elif name.endswith("_n"):
            group = "Monitoring intensity"
        else:
            group = "Other"
        source = name.rsplit("_", 1)[0]
        measure = MISSING_LABEL.get(name, MEASURE_REASON.get(source, MEASURE_REASON.get(name, "")))
        rows.append({"group": group, "predictor": name, "measurement": measure, "reason": GROUP_REASON[group]})
    catalogue = pd.DataFrame(rows)
    order = ["Admission", "Latest value", "Minimum", "Maximum", "Trend", "Monitoring intensity", "Missingness"]
    catalogue["group"] = pd.Categorical(catalogue["group"], categories=order, ordered=True)
    return catalogue.sort_values(["group", "predictor"]).reset_index(drop=True)


def dropped_table(cleaned: pd.DataFrame, n_stays: int) -> pd.DataFrame:
    """Time-series variables excluded as values, and why."""
    rows = []
    for name in RARE_TS:
        measured = cleaned.loc[cleaned["Parameter"].eq(name) & cleaned["value_clean"].notna(), "RecordID"].nunique()
        rows.append(
            {
                "variable": name,
                "stays_measured": int(measured),
                "pct_measured": 100 * measured / n_stays,
                "decision": "values dropped (too sparse); never-measured flag kept"
                if name in {"TroponinI", "TroponinT", "Albumin", "RespRate", "ALP", "ALT", "AST", "Bilirubin"}
                else "values dropped (too sparse), no flag kept",
            }
        )
    table = pd.DataFrame(rows).sort_values("pct_measured")
    return table


# ---------------------------------------------------------------------------
# Cross-validation design
# ---------------------------------------------------------------------------


def make_folds(y: np.ndarray, n_splits: int = N_FOLDS, seed: int = SEED) -> np.ndarray:
    """Stratified fold id per stay; one row per stay, so no grouping is needed."""
    folds = np.full(len(y), -1, dtype=int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, test_index) in enumerate(splitter.split(np.zeros(len(y)), y)):
        folds[test_index] = fold
    if (folds < 0).any():
        raise ValueError("some rows were not assigned to a fold")
    return folds


def _inner_folds(y: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=N_FOLDS_INNER, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y))


# ---------------------------------------------------------------------------
# Model 1: penalised logistic regression
# ---------------------------------------------------------------------------

LASSO_GRID = np.logspace(-4, 0, 9)


def _lasso_pipeline(penalty_strength: float) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    l1_ratio=1.0,
                    solver="liblinear",
                    C=penalty_strength,
                    max_iter=5000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def tune_lasso(X: pd.DataFrame, y: np.ndarray, grid: np.ndarray = LASSO_GRID, seed: int = SEED) -> dict:
    """Pick C by inner CV log loss, using only the rows passed in."""
    rows = []
    for C in grid:
        losses = []
        for train_idx, valid_idx in _inner_folds(y, seed):
            model = _lasso_pipeline(C).fit(X.iloc[train_idx], y[train_idx])
            prob = model.predict_proba(X.iloc[valid_idx])[:, 1]
            losses.append(log_loss(y[valid_idx], prob, labels=[0, 1]))
        rows.append({"C": float(C), "inner_log_loss": float(np.mean(losses))})
    tuning = pd.DataFrame(rows)
    best = tuning.loc[tuning["inner_log_loss"].idxmin(), "C"]
    return {"C": float(best), "tuning": tuning}


# ---------------------------------------------------------------------------
# Model 2: logistic GAM
# ---------------------------------------------------------------------------

GAM_SMOOTH = [
    "Age",
    "GCS_last",
    "HR_max",
    "SysABP_min",
    "Temp_min",
    "pH_min",
    "PaO2_min",
    "BUN_last",
    "Creatinine_max",
    "Na_min",
    "WBC_max",
    "Platelets_min",
    "Lactate_max",
    "Urine_sum",
]
GAM_LINEAR = [
    "ICUType_CCU",
    "ICUType_CSRU",
    "ICUType_SICU",
    "MechVent_any",
    "RespRate_never",
    "LiverPanel_never",
]
GAM_FEATURES = GAM_SMOOTH + GAM_LINEAR
GAM_LAM_GRID = [0.6, 1.0, 3.0, 10.0, 30.0]
N_SPLINES = 5


def _gam_terms(n_smooth: int, n_linear: int):
    from pygam import l, s

    terms = None
    for i in range(n_smooth):
        term = s(i, n_splines=N_SPLINES, spline_order=3)
        terms = term if terms is None else terms + term
    for j in range(n_smooth, n_smooth + n_linear):
        term = l(j)
        terms = term if terms is None else terms + term
    return terms


class GamRisk:
    """Median imputation, standardisation, then a logistic GAM with cubic splines."""

    def __init__(self, features: list[str], lam: float, n_splines: int = N_SPLINES):
        self.features = list(features)
        self.lam = lam
        self.n_splines = n_splines

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "GamRisk":
        from pygam import LogisticGAM

        self.imputer = SimpleImputer(strategy="median").fit(X[self.features])
        self.scaler = StandardScaler().fit(self.imputer.transform(X[self.features]))
        Z = self._prepare(X)
        n_smooth = len([f for f in self.features if f in GAM_SMOOTH])
        terms = _gam_terms(n_smooth, len(self.features) - n_smooth)
        self.gam = LogisticGAM(terms, lam=self.lam, max_iter=150)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gam.fit(Z, y)
        return self

    def _prepare(self, X: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(self.imputer.transform(X[self.features]))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        prob = np.asarray(self.gam.predict_proba(self._prepare(X)))
        return prob[:, 1] if prob.ndim == 2 else prob

    @property
    def term_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.features)}


def tune_gam(X: pd.DataFrame, y: np.ndarray, grid: list[float] = GAM_LAM_GRID, seed: int = SEED) -> dict:
    """Pick the smoothing strength by inner CV log loss inside the training rows."""
    rows = []
    for lam in grid:
        losses = []
        for train_idx, valid_idx in _inner_folds(y, seed):
            model = GamRisk(GAM_FEATURES, lam).fit(X.iloc[train_idx], y[train_idx])
            prob = model.predict_proba(X.iloc[valid_idx])
            losses.append(log_loss(y[valid_idx], prob, labels=[0, 1]))
        rows.append({"lam": float(lam), "inner_log_loss": float(np.mean(losses))})
    tuning = pd.DataFrame(rows)
    best = tuning.loc[tuning["inner_log_loss"].idxmin(), "lam"]
    return {"lam": float(best), "tuning": tuning}


# ---------------------------------------------------------------------------
# Model 3: gradient boosting
# ---------------------------------------------------------------------------

GBM_GRID = [
    {"learning_rate": lr, "max_leaf_nodes": leaves, "min_samples_leaf": msl}
    for lr in (0.05, 0.10)
    for leaves in (7, 15, 31)
    for msl in (20,)
]


def _gbm_model(params: dict) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        **params,
        l2_regularization=1.0,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=SEED,
    )


def tune_gbm(X: pd.DataFrame, y: np.ndarray, grid: list[dict] = GBM_GRID, seed: int = SEED) -> dict:
    """Small grid searched by inner 3-fold CV log loss; early stopping uses the training rows only."""
    rows = []
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    inner = list(splitter.split(np.zeros(len(y)), y))
    for params in grid:
        losses = []
        for train_idx, valid_idx in inner:
            model = _gbm_model(params).fit(X.iloc[train_idx], y[train_idx])
            prob = model.predict_proba(X.iloc[valid_idx])[:, 1]
            losses.append(log_loss(y[valid_idx], prob, labels=[0, 1]))
        rows.append({**params, "inner_log_loss": float(np.mean(losses))})
    tuning = pd.DataFrame(rows)
    best = tuning.loc[tuning["inner_log_loss"].idxmin()]
    chosen = {key: best[key] for key in ["learning_rate", "max_leaf_nodes", "min_samples_leaf"]}
    chosen = {
        "learning_rate": float(chosen["learning_rate"]),
        "max_leaf_nodes": int(chosen["max_leaf_nodes"]),
        "min_samples_leaf": int(chosen["min_samples_leaf"]),
    }
    return {"params": chosen, "tuning": tuning}


# ---------------------------------------------------------------------------
# Cross-validated comparison
# ---------------------------------------------------------------------------


def _fit_kind(kind: str, X: pd.DataFrame, y: np.ndarray, seed: int) -> tuple[object, dict, pd.DataFrame]:
    if kind == "lasso":
        tuned = tune_lasso(X, y, seed=seed)
        return _lasso_pipeline(tuned["C"]).fit(X, y), {"C": tuned["C"]}, tuned["tuning"]
    if kind == "gam":
        tuned = tune_gam(X, y, seed=seed)
        return GamRisk(GAM_FEATURES, tuned["lam"]).fit(X, y), {"lam": tuned["lam"]}, tuned["tuning"]
    if kind == "gbm":
        tuned = tune_gbm(X, y, seed=seed)
        return _gbm_model(tuned["params"]).fit(X, y), tuned["params"], tuned["tuning"]
    raise ValueError(kind)


def _predict(model, kind: str, X: pd.DataFrame) -> np.ndarray:
    prob = np.asarray(model.predict_proba(X))
    return prob[:, 1] if prob.ndim == 2 else prob


def cv_model(
    table: pd.DataFrame,
    folds: np.ndarray,
    kind: str,
    features: list[str],
    seed: int = SEED,
    n_splits: int = N_FOLDS,
) -> dict:
    """Fit each fold's model on its training rows only, and predict the held-out fold."""
    y = table["In_hospital_death"].to_numpy()
    X = table[features]
    oof = np.full(len(table), np.nan)
    fold_auc, fold_params, fold_models, fold_tuning = [], [], [], []
    for fold in range(n_splits):
        train = folds != fold
        test = folds == fold
        model, params, tuning = _fit_kind(kind, X[train], y[train], seed + fold)
        fold_tuning.append(tuning)
        prob = _predict(model, kind, X[test])
        oof[test] = prob
        fold_auc.append(roc_auc_score(y[test], prob))
        fold_params.append(params)
        fold_models.append(model)
    return {
        "kind": kind,
        "features": list(features),
        "oof": oof,
        "fold_auc": np.array(fold_auc),
        "fold_params": fold_params,
        "fold_tuning": fold_tuning,
        "fold_models": fold_models,
        "mean_auc": float(np.mean(fold_auc)),
        "sd_auc": float(np.std(fold_auc, ddof=1)),
        "log_loss": float(log_loss(y, oof, labels=[0, 1])),
        "brier": float(brier_score_loss(y, oof)),
    }


def benchmark_auc(table: pd.DataFrame, folds: np.ndarray, column: str, n_splits: int = N_FOLDS) -> dict:
    """Per-fold AUROC of a published score; stays with the -1 (missing) code are skipped."""
    y = table["In_hospital_death"].to_numpy()
    score = table[column].to_numpy()
    available = ~np.isnan(score)
    fold_auc = []
    for fold in range(n_splits):
        mask = (folds == fold) & available
        fold_auc.append(roc_auc_score(y[mask], score[mask]))
    return {
        "kind": column,
        "fold_auc": np.array(fold_auc),
        "mean_auc": float(np.mean(fold_auc)),
        "sd_auc": float(np.std(fold_auc, ddof=1)),
        "n_scored": int(available.sum()),
        "oof": score,
    }


def average_tuning(cv_result: dict) -> pd.DataFrame:
    """Mean inner-CV log loss of every candidate configuration across the outer folds."""
    frames = [frame.assign(fold=i) for i, frame in enumerate(cv_result["fold_tuning"], start=1)]
    all_tuning = pd.concat(frames, ignore_index=True)
    keys = [c for c in all_tuning.columns if c not in {"inner_log_loss", "fold"}]
    return (
        all_tuning.groupby(keys, as_index=False)["inner_log_loss"]
        .mean()
        .sort_values("inner_log_loss")
        .reset_index(drop=True)
    )


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        row = {
            "Model": name,
            "AUROC fold 1": result["fold_auc"][0],
            "AUROC fold 2": result["fold_auc"][1],
            "AUROC fold 3": result["fold_auc"][2],
            "AUROC fold 4": result["fold_auc"][3],
            "AUROC fold 5": result["fold_auc"][4],
            "Mean AUROC": result["mean_auc"],
            "SD": result["sd_auc"],
        }
        if "log_loss" in result:
            row["Log loss"] = result["log_loss"]
            row["Brier score"] = result["brier"]
        else:
            row["Log loss"] = np.nan
            row["Brier score"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Final model
# ---------------------------------------------------------------------------


def refit_final(table: pd.DataFrame, kind: str, features: list[str], seed: int = SEED) -> dict:
    """Re-tune and refit on all of set-a; returns the bundle saved for Checkpoint 3."""
    y = table["In_hospital_death"].to_numpy()
    X = table[features]
    model, params, _ = _fit_kind(kind, X, y, seed + 100)
    return {"kind": kind, "features": list(features), "params": params, "model": model}


def predict_risk(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """Apply a saved bundle to a table that has the same predictor columns."""
    prob = np.asarray(bundle["model"].predict_proba(X[bundle["features"]]))
    return prob[:, 1] if prob.ndim == 2 else prob


def lasso_coefficients(result: dict) -> pd.DataFrame:
    """Non-zero coefficients of the Model 1 refit, on the standardised scale."""
    model = result["model"]
    coefs = model.named_steps["model"].coef_[0]
    return (
        pd.DataFrame({"predictor": result["features"], "coefficient": coefs})
        .assign(abs_coef=lambda d: d["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns="abs_coef")
        .reset_index(drop=True)
    )


def lasso_stability(cv_result: dict) -> pd.DataFrame:
    """How often each predictor was retained across the five fold-specific LASSO fits."""
    counts: dict[str, int] = {}
    for model in cv_result["fold_models"]:
        coefs = model.named_steps["model"].coef_[0]
        for name, coef in zip(cv_result["features"], coefs):
            if abs(coef) > 1e-8:
                counts[name] = counts.get(name, 0) + 1
    table = pd.DataFrame({"predictor": list(counts.keys()), "folds_selected": list(counts.values())})
    table = table.sort_values(["folds_selected", "predictor"], ascending=[False, True]).reset_index(drop=True)
    table["selected_in_all_folds"] = table["folds_selected"] == N_FOLDS
    return table


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def plot_coefficients(coefs: pd.DataFrame, path: Path, top_n: int = 35) -> plt.Figure:
    data = coefs.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.22 * len(data) + 1.2))
    colors = np.where(data["coefficient"] > 0, "#C44E52", "#4C72B0")
    ax.barh(data["predictor"], data["coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LASSO coefficient (log-odds per 1 SD)")
    ax.set_title(f"Model 1: {len(coefs)} retained predictors (top {len(data)} shown)")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def plot_partial_effects(
    model: GamRisk, X: pd.DataFrame, predictors: list[str], path: Path, n_cols: int = 3
) -> plt.Figure:
    n_rows = int(np.ceil(len(predictors) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.3 * n_rows), squeeze=False)
    axes = axes.ravel()
    term_index = model.term_index
    for ax, name in zip(axes, predictors):
        j = model.features.index(name)
        grid = np.linspace(np.nanpercentile(X[name], 1), np.nanpercentile(X[name], 99), 60)
        Xg = pd.DataFrame(
            np.tile(np.nanmedian(X[model.features].to_numpy(), axis=0), (len(grid), 1)),
            columns=model.features,
        )
        Xg[name] = grid
        Zg = model._prepare(Xg)
        pdep, confi = model.gam.partial_dependence(term=term_index[name], X=Zg, width=0.95)
        pdep = np.asarray(pdep).ravel()
        confi = np.asarray(confi)
        if confi.shape[0] == 2:
            lower, upper = confi[0], confi[1]
        else:
            lower, upper = confi[:, 0], confi[:, 1]
        reference = np.interp(np.nanmedian(X[name]), grid, pdep)
        ax.plot(grid, pdep - reference, color="#333333")
        ax.fill_between(grid, lower - reference, upper - reference, color="#9E9E9E", alpha=0.35, linewidth=0)
        ax.axhline(0, color="#BDBDBD", linestyle="--", linewidth=0.8)
        ax.axvline(np.nanmedian(X[name]), color="#BDBDBD", linestyle=":", linewidth=0.8)
        ax.set_xlabel(name)
        ax.set_ylabel("log-odds vs median")
    for ax in axes[len(predictors):]:
        ax.axis("off")
    fig.suptitle("Model 2: partial effects with 95% confidence bands", y=1.00)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def partial_effect_or_table(
    model: GamRisk, X: pd.DataFrame, names: list[str], quantiles: tuple[float, float] = (0.10, 0.90)
) -> pd.DataFrame:
    """Odds ratio at the 10th and 90th percentile against the median, with a 95% interval."""
    reference_row = pd.DataFrame(
        np.nanmedian(X[model.features].to_numpy(), axis=0)[None, :], columns=model.features
    )
    rows = []
    for name in names:
        j = model.term_index[name]
        pdep_ref, _ = model.gam.partial_dependence(term=j, X=model._prepare(reference_row), width=0.95)
        pdep_ref = float(np.asarray(pdep_ref).ravel()[0])
        for q in quantiles:
            value = float(np.nanquantile(X[name].to_numpy(), q))
            row = reference_row.copy()
            row[name] = value
            pdep, confi = model.gam.partial_dependence(term=j, X=model._prepare(row), width=0.95)
            pdep = float(np.asarray(pdep).ravel()[0])
            confi = np.asarray(confi).ravel()
            low, high = (confi[0], confi[1]) if confi.size == 2 else (confi[0], confi[-1])
            rows.append(
                {
                    "predictor": name,
                    "percentile": int(q * 100),
                    "value": value,
                    "reference (median)": float(np.nanmedian(X[name].to_numpy())),
                    "OR vs median": float(np.exp(pdep - pdep_ref)),
                    "OR 2.5%": float(np.exp(low - pdep_ref)),
                    "OR 97.5%": float(np.exp(high - pdep_ref)),
                }
            )
    return pd.DataFrame(rows)


def plot_roc_curves(results: dict[str, dict], y: np.ndarray, path: Path) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    for name, result in results.items():
        score = result["oof"]
        available = ~np.isnan(score)
        fpr, tpr, _ = roc_curve(y[available], score[available])
        auc = roc_auc_score(y[available], score[available])
        ax.plot(fpr, tpr, linewidth=1.6, label=f"{name} (OOF AUROC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#BDBDBD", linestyle="--", linewidth=1)
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Out-of-fold ROC curves, set-a, same 5 folds")
    ax.legend(frameon=False, loc="lower right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def plot_fold_auc(results: dict[str, dict], path: Path) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    names = list(results.keys())
    values = [results[name]["fold_auc"] for name in names]
    ax.boxplot(values, tick_labels=names, widths=0.5, showmeans=False)
    for i, vals in enumerate(values, start=1):
        ax.scatter(np.full(len(vals), i) + np.linspace(-0.08, 0.08, len(vals)), vals, s=18, color="#4C72B0", zorder=3)
    ax.set_ylabel("AUROC in held-out fold")
    ax.set_title("Fold-to-fold variation, same folds for every model")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig
