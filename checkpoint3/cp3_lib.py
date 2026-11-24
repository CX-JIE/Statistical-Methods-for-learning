"""Shared code for BIOSTAT 707 Checkpoint 3 (evaluating the Checkpoint 2 final model).

The notebook `checkpoint3.ipynb` is the written report; it calls the functions here.

What is re-used from Checkpoint 2, and how:
  * `../checkpoint2/cp2_lib.py` is imported unchanged, so set-b and set-c are turned into a
    modelling table by exactly the same code that built the set-a table (same cleaning, same
    predictors, same order).
  * `../checkpoint2/output/cp2_final_model.joblib` is the final model fitted on set-a. It is only
    loaded and applied here; nothing in this checkpoint refits or re-tunes it.

Data are read from `data/` and results written to `output/`; all paths are relative.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit, logit
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from statsmodels.nonparametric.smoothers_lowess import lowess

CP2_DIR = Path("..") / "checkpoint2"
if str(CP2_DIR) not in sys.path:
    sys.path.insert(0, str(CP2_DIR))

import cp2_lib  # noqa: E402  (Checkpoint 2 code, imported unchanged)

SEED = 707
N_BOOT = 2000
OUT = Path("output")


def find_data_dir() -> Path:
    """The repository's data/ folder: checkpoint3/data/ if present, otherwise data/ at the repo root."""
    for candidate in (Path("data"), Path("..") / "data"):
        if all((candidate / f"set-{name}").is_dir() and (candidate / f"Outcomes-{name}.txt").is_file()
               for name in "abc"):
            return candidate
    raise FileNotFoundError(
        "set-a, set-b, set-c and Outcomes-{a,b,c}.txt not found: place them in checkpoint3/data/ "
        "or in data/ at the repository root"
    )


DATA = find_data_dir()
MODEL_PATH = CP2_DIR / "output" / "cp2_final_model.joblib"
CP2_COMPARISON = CP2_DIR / "output" / "cp2_model_comparison.csv"
CP2_OOF = CP2_DIR / "output" / "cp2_oof_predictions.csv"

ICU_LABEL = {1: "CCU", 2: "CSRU", 3: "MICU", 4: "SICU"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_set(name: str, data_dir: Path = DATA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long table and outcomes for set-a, set-b or set-c, read the same way as in Checkpoint 2."""
    folder = data_dir / f"set-{name}"
    files = sorted({p.stem: p for p in folder.rglob("*.txt") if p.stem.isdigit()}.values())
    if len(files) != 4000:
        raise ValueError(f"expected 4000 records in {folder}, found {len(files)}")
    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["RecordID"] = int(path.stem)
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw["time_hours"] = cp2_lib._parse_hours(raw["Time"])
    outcomes = pd.read_csv(data_dir / f"Outcomes-{name}.txt")
    return raw, outcomes


def build_set(name: str, features: list[str], data_dir: Path = DATA) -> dict:
    """Run the Checkpoint 2 table builder on one set and return predictors in the model's order."""
    raw, outcomes = load_set(name, data_dir)
    built = cp2_lib.build_modeling_table(raw, outcomes)
    table = built["table"]
    absent = [f for f in features if f not in table.columns]
    # A predictor can only be absent if no stay in the set ever had that measurement; it is then
    # all-missing, which is exactly what the model sees for an unmeasured variable.
    X = table.reindex(columns=features)
    admission = cp2_lib.admission_table(raw).reindex(table.index)
    if not table.index.is_unique or len(table) != 4000:
        raise ValueError(f"set-{name}: modelling table must have one row per stay")
    if list(X.columns) != list(features):
        raise ValueError(f"set-{name}: predictor order differs from the saved model")
    return {
        "name": name,
        "table": table,
        "X": X,
        "y": table["In_hospital_death"].to_numpy(),
        "admission": admission,
        "absent": absent,
        "n_raw": len(raw),
    }


def load_final_model() -> dict:
    bundle = joblib.load(MODEL_PATH)
    if bundle.get("kind") != "gbm":
        raise ValueError("the saved Checkpoint 2 model is not the gradient-boosting model")
    return bundle


def predict(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    return cp2_lib.predict_risk(bundle, X)


# ---------------------------------------------------------------------------
# Discrimination
# ---------------------------------------------------------------------------


def bootstrap_auc(y: np.ndarray, score: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """AUROC with a percentile bootstrap 95% interval; stays with a missing score are skipped."""
    keep = ~np.isnan(score)
    y, score = y[keep], score[keep]
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        while y[idx].min() == y[idx].max():
            idx = rng.integers(0, n, n)
        boots[b] = roc_auc_score(y[idx], score[idx])
    return {
        "auc": float(roc_auc_score(y, score)),
        "low": float(np.percentile(boots, 2.5)),
        "high": float(np.percentile(boots, 97.5)),
        "se": float(np.std(boots, ddof=1)),
        "n": int(n),
        "boots": boots,
    }


def plot_roc_bc(curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]], path: Path) -> plt.Figure:
    """ROC curves of the final model and the two scores, one panel per test set."""
    from sklearn.metrics import roc_curve

    colors = {"Final model": "#C44E52", "SAPS-I": "#55A868", "SOFA": "#4C72B0"}
    fig, axes = plt.subplots(1, len(curves), figsize=(4.6 * len(curves), 4.4), squeeze=False)
    for ax, (set_name, scores) in zip(axes[0], curves.items()):
        for name, (y, score) in scores.items():
            keep = ~np.isnan(score)
            fpr, tpr, _ = roc_curve(y[keep], score[keep])
            auc = roc_auc_score(y[keep], score[keep])
            ax.plot(fpr, tpr, color=colors.get(name), linewidth=1.6, label=f"{name} (AUROC {auc:.3f})")
        ax.plot([0, 1], [0, 1], color="#BDBDBD", linestyle="--", linewidth=1)
        ax.set_xlabel("1 - specificity")
        ax.set_ylabel("Sensitivity")
        ax.set_title(set_name)
        ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def plot_pr_curve(y: np.ndarray, risk: np.ndarray, label: str, path: Path) -> plt.Figure:
    precision, recall, _ = precision_recall_curve(y, risk)
    ap = average_precision_score(y, risk)
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.plot(recall, precision, color="#C44E52", linewidth=1.6, label=f"Final model (average precision {ap:.3f})")
    ax.axhline(y.mean(), color="#777777", linestyle="--", linewidth=1,
               label=f"Baseline = mortality {100 * y.mean():.1f}%")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision (positive predictive value)")
    ax.set_title(f"Precision-recall curve, {label}")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _logit_risk(risk: np.ndarray) -> np.ndarray:
    return logit(np.clip(risk, 1e-6, 1 - 1e-6))


def calibration_stats(y: np.ndarray, risk: np.ndarray) -> dict:
    """Calibration intercept (offset model), slope, Brier and scaled Brier."""
    lp = _logit_risk(risk)
    ones = np.ones((len(y), 1))
    fit_int = sm.GLM(y, ones, family=sm.families.Binomial(), offset=lp).fit()
    fit_slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
    ci_int = fit_int.conf_int()[0]
    ci_slope = fit_slope.conf_int()[1]
    brier = brier_score_loss(y, risk)
    prevalence = y.mean()
    return {
        "intercept": float(fit_int.params[0]),
        "intercept_low": float(ci_int[0]),
        "intercept_high": float(ci_int[1]),
        "slope": float(fit_slope.params[1]),
        "slope_low": float(ci_slope[0]),
        "slope_high": float(ci_slope[1]),
        "brier": float(brier),
        "brier_null": float(prevalence * (1 - prevalence)),
        "scaled_brier": float(1 - brier / (prevalence * (1 - prevalence))),
        "mean_risk": float(risk.mean()),
        "observed": float(prevalence),
    }


def risk_band_table(y: np.ndarray, risk: np.ndarray, edges=(0, 0.05, 0.10, 0.20, 0.40, 0.60, 1.0)) -> pd.DataFrame:
    """Mean predicted risk against observed mortality within fixed bands of predicted risk."""
    band = pd.cut(risk, bins=list(edges), include_lowest=True)
    frame = pd.DataFrame({"band": band, "y": y, "risk": risk})
    out = frame.groupby("band", observed=False).agg(
        stays=("y", "size"), deaths=("y", "sum"), predicted=("risk", "mean"), observed=("y", "mean")
    ).reset_index()
    out["band"] = [f"{100 * lo:.0f}% to {100 * hi:.0f}%" for lo, hi in zip(edges[:-1], edges[1:])]
    out["predicted"] *= 100
    out["observed"] *= 100
    return out


def lowess_curve(y: np.ndarray, risk: np.ndarray, grid: np.ndarray, frac: float = 0.4) -> np.ndarray:
    # it=0: no robustness iterations, which would treat the rare deaths as outliers.
    return lowess(y, risk, frac=frac, it=0, xvals=grid)


def flexible_calibration(y: np.ndarray, risk: np.ndarray, n_boot: int = 200, seed: int = SEED) -> pd.DataFrame:
    """Loess calibration curve on a grid of predicted risks, with a bootstrap 95% band."""
    upper = float(np.quantile(risk, 0.99))
    grid = np.linspace(float(np.quantile(risk, 0.005)), upper, 80)
    curve = lowess_curve(y, risk, grid)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, len(grid)))
    n = len(y)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boots[b] = lowess_curve(y[idx], risk[idx], grid)
    return pd.DataFrame(
        {
            "predicted": grid,
            "observed": np.clip(curve, 0, 1),
            "low": np.clip(np.nanpercentile(boots, 2.5, axis=0), 0, 1),
            "high": np.clip(np.nanpercentile(boots, 97.5, axis=0), 0, 1),
        }
    )


def plot_calibration(curves: dict[str, tuple[pd.DataFrame, np.ndarray, dict]], path: Path) -> plt.Figure:
    names = list(curves)
    fig, axes = plt.subplots(
        2, len(names), figsize=(5.0 * len(names), 5.6), sharex=True,
        gridspec_kw={"height_ratios": [4, 1]}, squeeze=False,
    )
    for col, name in enumerate(names):
        curve, risk, stats = curves[name]
        ax, ax_hist = axes[0, col], axes[1, col]
        ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1, label="Perfect calibration")
        ax.fill_between(curve["predicted"], curve["low"], curve["high"], color="#C44E52", alpha=0.2, linewidth=0,
                        label="Bootstrap 95% band")
        ax.plot(curve["predicted"], curve["observed"], color="#C44E52", linewidth=1.8, label="Loess calibration curve")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Observed mortality")
        ax.set_title(name)
        ax.text(
            0.03, 0.97,
            f"Intercept {stats['intercept']:+.2f} ({stats['intercept_low']:+.2f} to {stats['intercept_high']:+.2f})\n"
            f"Slope {stats['slope']:.2f} ({stats['slope_low']:.2f} to {stats['slope_high']:.2f})\n"
            f"Brier {stats['brier']:.3f}, scaled Brier {stats['scaled_brier']:.3f}",
            transform=ax.transAxes, va="top", fontsize=8,
        )
        ax.legend(frameon=False, loc="lower right", fontsize=7.5)
        ax_hist.hist(risk, bins=np.linspace(0, 1, 51), color="#777777")
        ax_hist.set_yscale("log")
        ax_hist.set_ylabel("Stays (log)")
        ax_hist.set_xlabel("Predicted risk of in-hospital death")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Clinical usefulness
# ---------------------------------------------------------------------------


def net_benefit(y: np.ndarray, risk: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    n = len(y)
    out = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        flagged = risk >= t
        tp = np.sum(flagged & (y == 1))
        fp = np.sum(flagged & (y == 0))
        out[i] = tp / n - fp / n * t / (1 - t)
    return out


def decision_curve(y: np.ndarray, risks: dict[str, np.ndarray], thresholds: np.ndarray) -> pd.DataFrame:
    prevalence = y.mean()
    frame = pd.DataFrame({"threshold": thresholds})
    for name, risk in risks.items():
        frame[name] = net_benefit(y, risk, thresholds)
    frame["Treat all"] = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
    frame["Treat none"] = 0.0
    return frame


def plot_decision_curve(dca: pd.DataFrame, chosen: float, path: Path, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    styles = {
        "Final model": dict(color="#C44E52", linewidth=1.9),
        "SOFA (recalibrated on set-a)": dict(color="#4C72B0", linewidth=1.5),
        "SAPS-I (recalibrated on set-a)": dict(color="#55A868", linewidth=1.5),
        "Treat all": dict(color="#777777", linewidth=1.1, linestyle="--"),
        "Treat none": dict(color="black", linewidth=1.1),
    }
    for name in [c for c in dca.columns if c != "threshold"]:
        ax.plot(dca["threshold"], dca[name], label=name, **styles.get(name, {}))
    top = float(np.nanmax(dca.drop(columns="threshold").to_numpy()))
    ax.set_ylim(-0.02, top * 1.1)
    ax.axvline(chosen, color="#BDBDBD", linestyle=":", linewidth=1)
    ax.text(chosen + 0.005, top * 0.55, f"chosen threshold {chosen:.2f}", fontsize=7.5, color="#555555")
    ax.set_xlim(dca["threshold"].min(), dca["threshold"].max())
    ax.set_xlabel("Threshold probability (risk above which a stay is flagged)")
    ax.set_ylabel("Net benefit")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def threshold_metrics(y: np.ndarray, risk: np.ndarray, threshold: float) -> dict:
    flagged = risk >= threshold
    tp = int(np.sum(flagged & (y == 1)))
    fp = int(np.sum(flagged & (y == 0)))
    fn = int(np.sum(~flagged & (y == 1)))
    tn = int(np.sum(~flagged & (y == 0)))
    sens = tp / (tp + fn) if tp + fn else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return {
        "threshold": threshold,
        "flagged": int(flagged.sum()),
        "flagged_per_100": 100 * flagged.mean(),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "sensitivity": sens,
        "specificity": spec,
        "PPV": ppv,
        "event1_score": float(np.nanmin([sens, ppv])),
    }


def recalibrate_score_on_a(score_a: np.ndarray, y_a: np.ndarray):
    """Logistic map from a published score to a probability, fitted on set-a only."""
    keep = ~np.isnan(score_a)
    fit = sm.GLM(y_a[keep], sm.add_constant(score_a[keep]), family=sm.families.Binomial()).fit()

    def to_risk(score: np.ndarray) -> np.ndarray:
        return expit(fit.params[0] + fit.params[1] * score)

    return to_risk, fit


# ---------------------------------------------------------------------------
# Subgroups
# ---------------------------------------------------------------------------


def age_group(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=[0, 50, 65, 80, 200], right=False,
                  labels=["< 50", "50 to 64", "65 to 79", ">= 80"])


def subgroup_table(y: np.ndarray, risk: np.ndarray, groups: dict[str, pd.Series], seed: int = SEED) -> pd.DataFrame:
    rows = []
    for variable, labels in groups.items():
        for level in labels.cat.categories if hasattr(labels, "cat") else sorted(labels.dropna().unique()):
            mask = (labels == level).to_numpy()
            yy, rr = y[mask], risk[mask]
            deaths = int(yy.sum())
            row = {
                "Subgroup": variable, "Level": str(level), "Stays": int(mask.sum()), "Deaths": deaths,
                "Observed mortality (%)": 100 * yy.mean() if mask.sum() else np.nan,
                "Mean predicted risk (%)": 100 * rr.mean() if mask.sum() else np.nan,
            }
            if deaths >= 5 and deaths < mask.sum():
                auc = bootstrap_auc(yy, rr, n_boot=1000, seed=seed)
                cal = calibration_stats(yy, rr)
                row |= {"AUROC": auc["auc"], "AUROC 2.5%": auc["low"], "AUROC 97.5%": auc["high"],
                        "Calibration intercept": cal["intercept"],
                        "Intercept 2.5%": cal["intercept_low"], "Intercept 97.5%": cal["intercept_high"]}
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------


def shap_values(model, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    import shap

    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(X))
    base = float(np.ravel(explainer.expected_value)[0])
    return values, base


def plot_shap_importance(importance: pd.DataFrame, path: Path, top_n: int = 20) -> plt.Figure:
    data = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 0.26 * len(data) + 1.2))
    ax.barh(data["predictor"], data["mean |SHAP|"], color="#C44E52")
    ax.set_xlabel("Mean |SHAP value| on set-b (log-odds)")
    ax.set_title(f"Final model: top {len(data)} predictors by mean |SHAP|, set-b")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def gam_partial_curve(gam_model, X_a: pd.DataFrame, name: str, grid: np.ndarray) -> pd.DataFrame:
    """Model 2 partial effect on the log-odds scale, centred at the set-a median (as in CP2 Fig. 2)."""
    medians = np.nanmedian(X_a[gam_model.features].to_numpy(), axis=0)
    Xg = pd.DataFrame(np.tile(medians, (len(grid), 1)), columns=gam_model.features)
    Xg[name] = grid
    pdep, confi = gam_model.gam.partial_dependence(term=gam_model.term_index[name], X=gam_model._prepare(Xg), width=0.95)
    pdep = np.asarray(pdep).ravel()
    confi = np.asarray(confi)
    lower, upper = (confi[0], confi[1]) if confi.shape[0] == 2 else (confi[:, 0], confi[:, 1])
    reference = np.interp(np.nanmedian(X_a[name]), grid, pdep)
    return pd.DataFrame({"x": grid, "effect": pdep - reference, "low": lower - reference, "high": upper - reference})


def gbm_partial_dependence(model, X_ref: pd.DataFrame, name: str, grid: np.ndarray, reference: float) -> np.ndarray:
    """Partial dependence of the final model on the log-odds scale, centred at `reference`."""
    out = np.empty(len(grid))
    Xc = X_ref.copy()
    for i, value in enumerate(grid):
        Xc[name] = value
        out[i] = model.decision_function(Xc).mean()
    Xc[name] = reference
    return out - model.decision_function(Xc).mean()


def plot_dependence_comparison(
    model, gam_model, X_a: pd.DataFrame, X_b: pd.DataFrame, shap_b: np.ndarray, features: list[str],
    names: list[str], path: Path, seed: int = SEED,
) -> tuple[plt.Figure, pd.DataFrame]:
    fig, axes = plt.subplots(1, len(names), figsize=(4.3 * len(names), 3.6), squeeze=False)
    rng = np.random.default_rng(seed)
    X_pd = X_b.iloc[rng.choice(len(X_b), size=min(800, len(X_b)), replace=False)]
    rows = []
    for ax, name in zip(axes[0], names):
        lo, hi = np.nanpercentile(X_a[name], [1, 99])
        grid = np.linspace(lo, hi, 40)
        median_a = float(np.nanmedian(X_a[name]))
        j = features.index(name)
        x = X_b[name].to_numpy()
        s = shap_b[:, j]
        observed = ~np.isnan(x)
        # Centre the SHAP cloud at the SHAP value of stays near the set-a median, so all three are
        # read as log-odds relative to a patient at the median.
        near = observed & (np.abs(x - median_a) <= np.nanstd(x) * 0.1)
        shift = float(np.median(s[near])) if near.sum() >= 10 else float(np.median(s[observed]))
        jitter = rng.uniform(-0.15, 0.15, observed.sum()) if name.startswith("GCS") else 0
        ax.scatter(x[observed] + jitter, s[observed] - shift, s=4, alpha=0.25, color="#C44E52",
                   label="Final model: SHAP value per set-b stay", rasterized=True)
        pdep = gbm_partial_dependence(model, X_pd, name, grid, median_a)
        ax.plot(grid, pdep, color="#8B0000", linewidth=1.8, label="Final model: partial dependence")
        gam = gam_partial_curve(gam_model, X_a, name, grid)
        ax.plot(gam["x"], gam["effect"], color="#333333", linewidth=1.6, linestyle="--",
                label="CP2 Model 2 (GAM) partial effect")
        ax.fill_between(gam["x"], gam["low"], gam["high"], color="#9E9E9E", alpha=0.3, linewidth=0,
                        label="GAM 95% band")
        ax.axhline(0, color="#BDBDBD", linewidth=0.8)
        ax.set_xlim(lo - (0.3 if name.startswith("GCS") else 0), hi + (0.3 if name.startswith("GCS") else 0))
        ax.set_xlabel(name)
        ax.set_ylabel("log-odds relative to set-a median")
        if not observed.all():
            ax.set_title(f"{name} ({100 * (~observed).mean():.0f}% of set-b stays missing)", fontsize=9)
        else:
            ax.set_title(name, fontsize=9)
        for q in [1, 10, 50, 90, 99]:
            value = float(np.nanpercentile(X_a[name], q))
            if any(r["predictor"] == name and r["value"] == value for r in rows):
                continue  # e.g. GCS: the median and the 90th and 99th percentiles are all 15
            pd_value = gbm_partial_dependence(model, X_pd, name, np.array([value]), median_a)[0]
            pair = gam_partial_curve(gam_model, X_a, name, np.array(sorted([value, median_a])))
            gam_value = float(np.interp(value, pair["x"], pair["effect"]))
            rows.append({"predictor": name, "set-a percentile": q, "value": value,
                         "Final model partial dependence (log-odds)": float(pd_value),
                         "GAM partial effect (log-odds)": gam_value})
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig, pd.DataFrame(rows)


def top_drivers(shap_row: np.ndarray, x_row: pd.Series, features: list[str], k: int = 4) -> str:
    order = np.argsort(-np.abs(shap_row))[:k]
    parts = []
    for j in order:
        value = x_row.iloc[j]
        text = "missing" if pd.isna(value) else (f"{value:.0f}" if abs(value) >= 10 else f"{value:.2g}")
        parts.append(f"{features[j]}={text} ({shap_row[j]:+.2f})")
    return "; ".join(parts)
