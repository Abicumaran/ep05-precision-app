import io
import zipfile
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import streamlit as st
import statsmodels.formula.api as smf

# -----------------------------
# Defaults that match your EP05 template + current CBC paired dataset
# -----------------------------
REQUIRED_BASE_COLS = ["batch_id", "Blood Sample ID", "Level", "Day", "Replicate", "Device"]
BASE_COL_ALIASES = {
    "Blood Sample ID": ["Blood Sample ID", "bloodSampleId", "blood_sample_id", "sample_id"],
    # Prefer the true device identifier over serial-number columns for UI filtering/grouping.
    "Device": ["deviceID", "deviceId", "device_id", "DeviceID", "DeviceId", "Device ID", "Device", "serialNumber", "serial_number"],
}

# User-specified device/reference pairs.  NEU is kept as the display analyte, while
# NEUT_2/NEUT_ref are accepted because that is how the uploaded dataset labels them.
DEFAULT_ANALYTE_PAIRS = {
    "RBC": ("RBC", "RBC_ref"),
    "WBC": ("WBC_2", "WBC_ref"),
    "PLT": ("PLT", "PLT_ref"),
    "HCT": ("HCT", "HCT_ref"),
    "HGB": ("HGB", "HGB_ref"),
    "MCV": ("MCV", "MCV_ref"),
    "RDW": ("RDW", "RDW_ref"),
    "MCH": ("MCH", "MCH_ref"),
    "MCHC": ("MCHC", "MCHC_ref"),
    "NEU": (["NEU_2", "NEUT_2", "NEU", "NEUT"], ["NEU_ref", "NEUT_ref"]),
    "LYMPH": ("LYMPH_2", "LYMPH_ref"),
    "MXD": ("MXD_2", "MXD_ref"),
}

DEFAULT_ANALYTES = list(DEFAULT_ANALYTE_PAIRS.keys())
DEFAULT_LEVELS = ["Low", "Mid", "High"]
DEFAULT_DAYS = ["D1", "D2", "D3", "D4", "D5"]
DEFAULT_REPLICATES = [1, 2, 3, 4, 5]
DEFAULT_DEVICES = []

NORMALIZATION_METHODS = [
    "Raw/no normalization",
    "Day 1 anchoring: reference only",
    "Day 1 anchoring: device and reference separately",
    "Per-level median centering: reference only",
    "Per-level median centering: device and reference separately",
    "Reference drift correction: day-wise reference factors applied to paired values",
    "Robust median/MAD z-score: reference only",
    "Robust median/MAD z-score: device and reference separately",
]

VALUE_OUTPUT_MODES = [
    "Device normalized value",
    "Reference normalized value",
    "Bias: device - reference",
    "%Bias: 100*(device-reference)/reference",
]

# -----------------------------
# Core config
# -----------------------------
@dataclass
class Config:
    analytes: List[str]
    levels: List[str]
    days: List[str]
    replicates: List[int]
    devices: List[str]
    gcrit: float
    expected_n: int
    device_mode: str
    outlier_method: str
    max_remove_per_group: int
    gcrit_mode: str
    gcrit_alpha: float
    gcrit_tail: str
    modified_z_threshold: float
    robust_interval_z: float
    do_bootstrap_ci: bool
    n_boot: int
    seed: int
    paired_analytes: List[str] = field(default_factory=list)
    normalization_method: str = "Raw/no normalization"
    value_output_modes: List[str] = field(default_factory=lambda: ["Device normalized value"])
    analyte_pair_map: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    # Optional per-analyte override, populated by the normality-guided recommender.
    # When present, each paired analyte can use its own best normalization method.
    analyte_normalization_map: Dict[str, str] = field(default_factory=dict)


# -----------------------------
# Utility helpers
# -----------------------------
def first_existing_column(df: pd.DataFrame, candidates) -> Optional[str]:
    if isinstance(candidates, str):
        candidates = [candidates]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def standardize_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Accept uploaded dataset aliases without requiring manual column edits.

    Device handling intentionally prefers deviceID/deviceId over serialNumber.
    Some source files contain both a serial-number column and a true deviceID column;
    the app should display/filter by the true device ID.
    """
    out = df.copy()
    for canonical, aliases in BASE_COL_ALIASES.items():
        found = first_existing_column(out, aliases)
        if found is not None and (canonical not in out.columns or canonical == "Device"):
            out[canonical] = out[found]

    # Explicit final override for common true device-ID spellings, so an existing
    # serial-based "Device" column cannot win over the true device identifier.
    device_id_col = first_existing_column(out, ["deviceID", "deviceId", "device_id", "DeviceID", "DeviceId", "Device ID"])
    if device_id_col is not None:
        out["Device"] = out[device_id_col].astype(str)
    elif "Device" in out.columns:
        out["Device"] = out["Device"].astype(str)
    return out


def resolve_analyte_pairs(df: pd.DataFrame, selected: Optional[List[str]] = None) -> Dict[str, Tuple[str, str]]:
    selected = selected or list(DEFAULT_ANALYTE_PAIRS.keys())
    pairs = {}
    for label in selected:
        if label not in DEFAULT_ANALYTE_PAIRS:
            continue
        dev_spec, ref_spec = DEFAULT_ANALYTE_PAIRS[label]
        dev_col = first_existing_column(df, dev_spec)
        ref_col = first_existing_column(df, ref_spec)
        if dev_col is not None and ref_col is not None:
            pairs[label] = (dev_col, ref_col)
    return pairs


def safe_divide(num, den):
    den = pd.to_numeric(den, errors="coerce")
    num = pd.to_numeric(num, errors="coerce")
    return num / den.replace(0, np.nan)


def robust_sd_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad


def robust_mad_series(s: pd.Series) -> float:
    vals = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
    return robust_sd_mad(vals)


def iqr(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def bootstrap_ci_mean_or_median(y: np.ndarray, use_median: bool, n_boot: int, seed: int) -> Tuple[float, float]:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(int(seed))
    vals = []
    for _ in range(int(max(200, n_boot))):
        b = rng.choice(y, size=len(y), replace=True)
        vals.append(np.median(b) if use_median else np.mean(b))
    return tuple(np.percentile(vals, [2.5, 97.5]).astype(float))


def grubbs_gcrit_auto(n: int, alpha: float = 0.01, tail: str = "Two-sided") -> float:
    n = int(n)
    if n < 3:
        return np.nan
    alpha = float(alpha)
    if str(tail).lower().startswith("one"):
        t_quant = 1.0 - alpha / n
    else:
        t_quant = 1.0 - alpha / (2.0 * n)
    t = stats.t.ppf(t_quant, df=n - 2)
    return float(((n - 1) / np.sqrt(n)) * np.sqrt((t * t) / (n - 2 + t * t)))


def current_gcrit(n: int, cfg: Config) -> float:
    if str(cfg.gcrit_mode).startswith("Automatic"):
        return grubbs_gcrit_auto(n=n, alpha=cfg.gcrit_alpha, tail=cfg.gcrit_tail)
    return float(cfg.gcrit)


# -----------------------------
# Normalization and paired metrics
# -----------------------------

def day1_anchor_by_level(df: pd.DataFrame, values: pd.Series) -> pd.Series:
    tmp = df.assign(_x=pd.to_numeric(values, errors="coerce"))
    day_mask = tmp["Day"].astype(str).isin(["D1", "1", "Day1"])
    anchors = tmp.loc[day_mask].groupby("Level")["_x"].median()
    fallback = tmp.groupby("Level")["_x"].median()
    anchors = fallback.combine_first(anchors) if anchors.empty else anchors.combine_first(fallback)
    return tmp["Level"].map(anchors).astype(float)

def group_day_medians(df: pd.DataFrame, col: str, group_cols: List[str]) -> pd.Series:
    return df.groupby(group_cols)[col].transform(lambda x: pd.to_numeric(x, errors="coerce").median())


def apply_normalization_for_pair(df: pd.DataFrame, label: str, dev_col: str, ref_col: str, method: str) -> pd.DataFrame:
    out = df.copy()
    dev = pd.to_numeric(out[dev_col], errors="coerce")
    ref = pd.to_numeric(out[ref_col], errors="coerce")
    norm_dev = dev.copy()
    norm_ref = ref.copy()

    # Anchors are estimated within Level because Low/Mid/High are true concentration strata.
    level_cols = ["Level"]
    level_day_cols = ["Level", "Day"]

    if method == "Day 1 anchoring: reference only":
        day_center = group_day_medians(out.assign(_ref=ref), "_ref", level_day_cols)
        anchor = day1_anchor_by_level(out, ref)
        norm_ref = ref * safe_divide(anchor, day_center)

    elif method == "Day 1 anchoring: device and reference separately":
        for raw, name in [(dev, "dev"), (ref, "ref")]:
            tmp = out.assign(_x=raw)
            day_center = group_day_medians(tmp, "_x", level_day_cols)
            anchor = day1_anchor_by_level(out, raw)
            adj = raw * safe_divide(anchor, day_center)
            if name == "dev":
                norm_dev = adj
            else:
                norm_ref = adj

    elif method == "Per-level median centering: reference only":
        level_center = group_day_medians(out.assign(_ref=ref), "_ref", level_cols)
        norm_ref = ref - level_center

    elif method == "Per-level median centering: device and reference separately":
        dev_center = group_day_medians(out.assign(_dev=dev), "_dev", level_cols)
        ref_center = group_day_medians(out.assign(_ref=ref), "_ref", level_cols)
        norm_dev = dev - dev_center
        norm_ref = ref - ref_center

    elif method == "Reference drift correction: day-wise reference factors applied to paired values":
        tmp = out.assign(_ref=ref)
        day_center = group_day_medians(tmp, "_ref", level_day_cols)
        anchor = day1_anchor_by_level(out, ref)
        factor = safe_divide(anchor, day_center)
        norm_dev = dev * factor
        norm_ref = ref * factor

    elif method == "Robust median/MAD z-score: reference only":
        med = group_day_medians(out.assign(_ref=ref), "_ref", level_cols)
        mad_sd = out.assign(_ref=ref).groupby("Level")["_ref"].transform(robust_mad_series).replace(0, np.nan)
        norm_ref = (ref - med) / mad_sd

    elif method == "Robust median/MAD z-score: device and reference separately":
        dev_med = group_day_medians(out.assign(_dev=dev), "_dev", level_cols)
        ref_med = group_day_medians(out.assign(_ref=ref), "_ref", level_cols)
        dev_mad = out.assign(_dev=dev).groupby("Level")["_dev"].transform(robust_mad_series).replace(0, np.nan)
        ref_mad = out.assign(_ref=ref).groupby("Level")["_ref"].transform(robust_mad_series).replace(0, np.nan)
        norm_dev = (dev - dev_med) / dev_mad
        norm_ref = (ref - ref_med) / ref_mad

    out[f"{label}__device_norm"] = norm_dev
    out[f"{label}__ref_norm"] = norm_ref
    out[f"{label}__bias"] = norm_dev - norm_ref
    out[f"{label}__pctbias"] = 100.0 * safe_divide(norm_dev - norm_ref, norm_ref)
    return out


def build_analysis_dataframe(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    work = standardize_base_columns(df)
    analysis_cols = []
    pair_rows = []
    for label, (dev_col, ref_col) in cfg.analyte_pair_map.items():
        method_for_label = cfg.analyte_normalization_map.get(label, cfg.normalization_method)
        work = apply_normalization_for_pair(work, label, dev_col, ref_col, method_for_label)
        pair_rows.append({
            "analyte": label,
            "device_column": dev_col,
            "reference_column": ref_col,
            "normalization_method_used": method_for_label,
        })
        for mode in cfg.value_output_modes:
            if mode == "Device normalized value":
                analysis_cols.append(f"{label}__device_norm")
            elif mode == "Reference normalized value":
                analysis_cols.append(f"{label}__ref_norm")
            elif mode == "Bias: device - reference":
                analysis_cols.append(f"{label}__bias")
            elif mode == "%Bias: 100*(device-reference)/reference":
                analysis_cols.append(f"{label}__pctbias")
    # Preserve manually selected raw columns, if any.
    for a in cfg.analytes:
        if a in work.columns and a not in analysis_cols:
            analysis_cols.append(a)
    return work, analysis_cols, pd.DataFrame(pair_rows)


def paired_pctbias_residual_normality(df_method: pd.DataFrame, label: str) -> Dict[str, object]:
    """Shapiro-Wilk normality check on paired %bias residuals after Level/Day adjustment.

    The adjustment keeps the recommendation from confusing true Low/Mid/High level
    differences or day blocks with residual distribution shape. If the OLS model fails
    or the design is too small, it falls back to median-centered %bias.
    """
    out = {
        "residual_shapiro_p": np.nan,
        "residual_normality_pass_0_05": False,
        "residual_n": 0,
        "residual_model": "not_tested",
    }
    col = f"{label}__pctbias"
    if col not in df_method.columns:
        return out
    dat = df_method[["Level", "Day", col]].copy()
    dat[col] = pd.to_numeric(dat[col], errors="coerce")
    dat = dat.replace([np.inf, -np.inf], np.nan).dropna(subset=[col])
    if len(dat) < 3:
        return out
    try:
        # Shapiro-Wilk in scipy is intended for n<=5000 for p-value accuracy.
        model_dat = dat.rename(columns={col: "pctbias"})
        if model_dat["Level"].nunique() >= 2 or model_dat["Day"].nunique() >= 2:
            res = smf.ols("pctbias ~ C(Level) + C(Day)", data=model_dat).fit()
            resid = np.asarray(res.resid, dtype=float)
            out["residual_model"] = "OLS residuals: pctbias ~ C(Level) + C(Day)"
        else:
            vals = model_dat["pctbias"].to_numpy(dtype=float)
            resid = vals - np.nanmedian(vals)
            out["residual_model"] = "median-centered pctbias"
    except Exception:
        vals = dat[col].to_numpy(dtype=float)
        resid = vals - np.nanmedian(vals)
        out["residual_model"] = "fallback median-centered pctbias"

    resid = resid[np.isfinite(resid)]
    out["residual_n"] = int(len(resid))
    if len(resid) >= 3:
        try:
            p = float(stats.shapiro(resid[:5000]).pvalue)
            out["residual_shapiro_p"] = p
            out["residual_normality_pass_0_05"] = bool(p >= 0.05)
        except Exception:
            pass
    return out


def make_normality_guided_recommendations(summary: pd.DataFrame) -> pd.DataFrame:
    """Add per-analyte normalization recommendations using score + residual normality.

    Rule: reference-drift correction is preferred when its paired-%bias residuals are
    approximately normal and its composite score is competitive. Robust MAD z-score is
    preferred when residuals are non-normal/outlier-heavy or when it clearly scores
    better. Otherwise, the lowest composite-score method is used.
    """
    if summary.empty:
        return summary
    reference_method = "Reference drift correction: day-wise reference factors applied to paired values"
    robust_method = "Robust median/MAD z-score: device and reference separately"
    summary = summary.copy()
    summary["normality_guided_recommended_for_analyte"] = False
    summary["per_analyte_recommended_method"] = ""
    summary["recommendation_reason"] = ""

    for label, sub in summary.groupby("analyte"):
        usable = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=["normalization_score_lower_is_better"])
        if usable.empty:
            continue
        best_idx = usable["normalization_score_lower_is_better"].idxmin()
        best_method = str(summary.loc[best_idx, "normalization_method"])
        best_score = float(summary.loc[best_idx, "normalization_score_lower_is_better"])

        ref_row = usable[usable["normalization_method"].eq(reference_method)]
        robust_row = usable[usable["normalization_method"].eq(robust_method)]
        chosen_idx = best_idx
        reason = "Lowest composite normalization score across drift/CV, %bias dispersion, and level-separation preservation."

        if not ref_row.empty:
            ref_idx = ref_row.index[0]
            ref_score = float(summary.loc[ref_idx, "normalization_score_lower_is_better"])
            ref_normal = bool(summary.loc[ref_idx, "residual_normality_pass_0_05"])
            ref_competitive = np.isfinite(ref_score) and np.isfinite(best_score) and ref_score <= 1.15 * best_score
            if ref_normal and ref_competitive:
                chosen_idx = ref_idx
                reason = "Reference-drift correction recommended: paired %bias residuals pass Shapiro-Wilk after Level/Day adjustment and the drift/CV score is competitive."
            elif (not ref_normal) and not robust_row.empty:
                robust_idx = robust_row.index[0]
                robust_score = float(summary.loc[robust_idx, "normalization_score_lower_is_better"])
                if np.isfinite(robust_score) and robust_score <= 1.30 * best_score:
                    chosen_idx = robust_idx
                    reason = "Robust MAD z-score recommended: reference-drift residuals fail Shapiro-Wilk, suggesting non-normal/outlier-heavy paired bias."

        chosen_method = str(summary.loc[chosen_idx, "normalization_method"])
        summary.loc[chosen_idx, "normality_guided_recommended_for_analyte"] = True
        summary.loc[summary["analyte"].eq(label), "per_analyte_recommended_method"] = chosen_method
        summary.loc[summary["analyte"].eq(label), "recommendation_reason"] = reason

    return summary

def evaluate_normalization_methods(df: pd.DataFrame, analyte_pair_map: Dict[str, Tuple[str, str]], methods: List[str]) -> pd.DataFrame:
    rows = []
    normality_rows = []
    if not analyte_pair_map:
        return pd.DataFrame()
    base = standardize_base_columns(df)
    for method in methods:
        tmp = base.copy()
        for label, (dev_col, ref_col) in analyte_pair_map.items():
            tmp = apply_normalization_for_pair(tmp, label, dev_col, ref_col, method)
            normality_rows.append({
                "normalization_method": method,
                "analyte": label,
                **paired_pctbias_residual_normality(tmp, label),
            })
            for level in sorted(tmp["Level"].dropna().astype(str).unique()):
                sub = tmp[tmp["Level"].astype(str) == level]
                if sub.empty:
                    continue
                ref_day_meds = sub.groupby("Day")[f"{label}__ref_norm"].median(numeric_only=True).to_numpy(dtype=float)
                dev_day_meds = sub.groupby("Day")[f"{label}__device_norm"].median(numeric_only=True).to_numpy(dtype=float)
                pctbias = pd.to_numeric(sub[f"{label}__pctbias"], errors="coerce").dropna().to_numpy(dtype=float)
                ref_cv = 100.0 * np.nanstd(ref_day_meds, ddof=1) / np.nanmean(np.abs(ref_day_meds)) if len(ref_day_meds) > 1 and np.nanmean(np.abs(ref_day_meds)) != 0 else np.nan
                dev_cv = 100.0 * np.nanstd(dev_day_meds, ddof=1) / np.nanmean(np.abs(dev_day_meds)) if len(dev_day_meds) > 1 and np.nanmean(np.abs(dev_day_meds)) != 0 else np.nan
                rows.append({
                    "normalization_method": method,
                    "analyte": label,
                    "Level": level,
                    "ref_day_median_CV_%": ref_cv,
                    "device_day_median_CV_%": dev_cv,
                    "median_%bias": float(np.nanmedian(pctbias)) if len(pctbias) else np.nan,
                    "abs_median_%bias": float(abs(np.nanmedian(pctbias))) if len(pctbias) else np.nan,
                    "IQR_%bias": iqr(pctbias),
                    "N": int(len(sub)),
                })

        # Level separation preservation is computed per analyte across all levels.
        for label in analyte_pair_map:
            if f"{label}__ref_norm" not in tmp.columns:
                continue
            centers = tmp.groupby("Level")[f"{label}__ref_norm"].median(numeric_only=True)
            if all(l in centers.index for l in ["Low", "Mid", "High"]):
                sep_min = min(abs(centers["Mid"] - centers["Low"]), abs(centers["High"] - centers["Mid"]))
            else:
                vals = centers.dropna().to_numpy(dtype=float)
                sep_min = float(np.nanmin(np.abs(np.diff(np.sort(vals))))) if len(vals) >= 2 else np.nan
            rows.append({
                "normalization_method": method,
                "analyte": label,
                "Level": "ALL_LEVEL_SEPARATION",
                "ref_day_median_CV_%": np.nan,
                "device_day_median_CV_%": np.nan,
                "median_%bias": np.nan,
                "abs_median_%bias": np.nan,
                "IQR_%bias": np.nan,
                "min_level_center_separation": sep_min,
                "N": int(len(tmp)),
            })

    comp = pd.DataFrame(rows)
    if comp.empty:
        return comp

    metric_rows = comp[comp["Level"] != "ALL_LEVEL_SEPARATION"].copy()
    sep_rows = comp[comp["Level"] == "ALL_LEVEL_SEPARATION"].copy()

    # Raw separation is the reference for distortion. Higher separation is better.
    raw_sep = sep_rows[sep_rows["normalization_method"] == "Raw/no normalization"].set_index("analyte")["min_level_center_separation"].to_dict() if not sep_rows.empty else {}
    sep_rows["separation_preservation_ratio"] = sep_rows.apply(
        lambda r: r.get("min_level_center_separation", np.nan) / raw_sep.get(r["analyte"], np.nan)
        if raw_sep.get(r["analyte"], np.nan) not in [0, np.nan] else np.nan,
        axis=1,
    )
    sep_summary = sep_rows.groupby(["normalization_method", "analyte"], as_index=False)["separation_preservation_ratio"].median()

    summary = metric_rows.groupby(["normalization_method", "analyte"], as_index=False).agg(
        mid_ref_day_CV_pct=("ref_day_median_CV_%", lambda x: np.nanmedian(x[metric_rows.loc[x.index, "Level"].astype(str).eq("Mid")]) if any(metric_rows.loc[x.index, "Level"].astype(str).eq("Mid")) else np.nan),
        all_ref_day_CV_pct=("ref_day_median_CV_%", "median"),
        all_device_day_CV_pct=("device_day_median_CV_%", "median"),
        abs_median_pctbias=("abs_median_%bias", "median"),
        iqr_pctbias=("IQR_%bias", "median"),
    )
    summary = summary.merge(sep_summary, on=["normalization_method", "analyte"], how="left")
    normality_summary = pd.DataFrame(normality_rows)
    if not normality_summary.empty:
        summary = summary.merge(normality_summary, on=["normalization_method", "analyte"], how="left")
    else:
        summary["residual_shapiro_p"] = np.nan
        summary["residual_normality_pass_0_05"] = False
        summary["residual_n"] = 0
        summary["residual_model"] = "not_tested"

    # Robust score: lower drift/CV/bias variation is better; preserving separation is rewarded.
    # IMPORTANT: when the user selects only one Level, Low/Mid/High separation is not estimable.
    # Older code propagated that NaN into the composite score, so every score could become NaN
    # and pandas idxmin() raised: ValueError("Encountered all NA values").
    # For single-level or otherwise non-estimable separation, use a neutral separation penalty of 0
    # rather than failing the app.
    sep_penalty = 100.0 * (1.0 - summary["separation_preservation_ratio"].clip(lower=0, upper=1))
    sep_penalty = sep_penalty.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _metric_component(col: str) -> pd.Series:
        vals = pd.to_numeric(summary[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        fallback = float(np.nanmedian(vals.to_numpy(dtype=float))) if vals.notna().any() else 0.0
        return vals.fillna(fallback)

    mid_or_all_ref_cv = pd.to_numeric(summary["mid_ref_day_CV_pct"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    mid_or_all_ref_cv = mid_or_all_ref_cv.fillna(pd.to_numeric(summary["all_ref_day_CV_pct"], errors="coerce"))
    if mid_or_all_ref_cv.notna().any():
        mid_or_all_ref_cv = mid_or_all_ref_cv.fillna(float(np.nanmedian(mid_or_all_ref_cv.to_numpy(dtype=float))))
    else:
        mid_or_all_ref_cv = mid_or_all_ref_cv.fillna(0.0)

    summary["normalization_score_lower_is_better"] = (
        0.40 * mid_or_all_ref_cv +
        0.20 * _metric_component("all_ref_day_CV_pct") +
        0.15 * _metric_component("all_device_day_CV_pct") +
        0.15 * _metric_component("iqr_pctbias") +
        0.10 * sep_penalty
    ).replace([np.inf, -np.inf], np.nan)

    summary["recommended_for_analyte"] = False
    for label, sub in summary.groupby("analyte"):
        score = pd.to_numeric(sub["normalization_score_lower_is_better"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if score.notna().any():
            idx = score.idxmin()
        else:
            # Last-resort fallback so the UI never crashes; raw is safest/most transparent.
            raw = sub[sub["normalization_method"].eq("Raw/no normalization")]
            idx = raw.index[0] if not raw.empty else sub.index[0]
        summary.loc[idx, "recommended_for_analyte"] = True

    overall = summary.groupby("normalization_method", as_index=False).agg(
        overall_score_lower_is_better=("normalization_score_lower_is_better", "median"),
        recommended_analyte_count=("recommended_for_analyte", "sum"),
    ).sort_values(["overall_score_lower_is_better", "recommended_analyte_count"], ascending=[True, False], na_position="last")
    summary = summary.merge(overall, on="normalization_method", how="left")
    summary = make_normality_guided_recommendations(summary)
    return summary.sort_values(["overall_score_lower_is_better", "analyte", "normalization_score_lower_is_better"], na_position="last")


# -----------------------------
# Outlier logic
# -----------------------------
def select_actual_outlier_method(method: str, vals: np.ndarray) -> str:
    if not str(method).startswith("Automatic"):
        return method
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) >= 3:
        try:
            p = stats.shapiro(vals).pvalue
            if p >= 0.05:
                return "Gcrit Grubbs-like: remove largest |value-mean|/SD if >= Gcrit"
        except Exception:
            pass
    return "Robust MAD modified-z: remove largest robust z if >= threshold"


def detect_outliers_one_group(df_group: pd.DataFrame, analyte: str, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    work = df_group.copy()
    work[analyte] = pd.to_numeric(work[analyte], errors="coerce")
    work["is_outlier"] = False
    work["outlier_order"] = np.nan
    work["outlier_method"] = ""
    work["outlier_metric"] = np.nan
    work["outlier_threshold"] = np.nan
    work["outlier_direction"] = ""
    work["outlier_details"] = ""
    work["gcrit_mode"] = cfg.gcrit_mode
    work["gcrit_alpha"] = cfg.gcrit_alpha
    work["gcrit_tail"] = cfg.gcrit_tail

    log_rows = []
    remaining = list(work.index[work[analyte].notna()])
    method = select_actual_outlier_method(str(cfg.outlier_method), work.loc[remaining, analyte].to_numpy(dtype=float))

    if method == "None" or int(cfg.max_remove_per_group) <= 0 or len(remaining) < 3:
        return work, pd.DataFrame(log_rows)

    for step in range(int(cfg.max_remove_per_group)):
        vals = work.loc[remaining, analyte].astype(float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 3:
            break
        x = vals.to_numpy(dtype=float)
        idxs = list(vals.index)

        chosen_idx = None
        metric = np.nan
        threshold = np.nan
        direction = ""
        details = ""

        if method.startswith("Gcrit"):
            mu = float(np.mean(x))
            sd = float(np.std(x, ddof=1))
            if not np.isfinite(sd) or sd == 0:
                break
            gvals = np.abs(x - mu) / sd
            k = int(np.argmax(gvals))
            threshold = current_gcrit(len(x), cfg)
            metric = float(gvals[k])
            if np.isfinite(threshold) and metric >= threshold:
                chosen_idx = idxs[k]
                direction = "high" if x[k] > mu else "low"
                details = f"G={metric:.4g}; mean={mu:.4g}; sd={sd:.4g}; n={len(x)}; Gcrit={threshold:.4g}"
            else:
                break

        elif method.startswith("Robust MAD"):
            med = float(np.median(x))
            mad = float(np.median(np.abs(x - med)))
            if not np.isfinite(mad) or mad == 0:
                break
            modz = 0.6745 * (x - med) / mad
            k = int(np.argmax(np.abs(modz)))
            metric = float(abs(modz[k]))
            threshold = float(cfg.modified_z_threshold)
            if metric >= threshold:
                chosen_idx = idxs[k]
                direction = "high" if x[k] > med else "low"
                details = f"modified_z={modz[k]:.4g}; median={med:.4g}; MAD={mad:.4g}; threshold={threshold:.4g}"
            else:
                break

        elif method.startswith("95% robust interval"):
            med = float(np.median(x))
            rsd = robust_sd_mad(x)
            if not np.isfinite(rsd) or rsd == 0:
                break
            lo = med - float(cfg.robust_interval_z) * rsd
            hi = med + float(cfg.robust_interval_z) * rsd
            distances = np.maximum(lo - x, x - hi)
            k = int(np.argmax(distances))
            metric = float(distances[k])
            threshold = 0.0
            if metric > 0:
                chosen_idx = idxs[k]
                direction = "high" if x[k] > hi else "low"
                details = f"value outside robust interval [{lo:.4g}, {hi:.4g}]; median={med:.4g}; robust_SD={rsd:.4g}; z={cfg.robust_interval_z}"
            else:
                break

        if chosen_idx is None:
            break

        work.loc[chosen_idx, "is_outlier"] = True
        work.loc[chosen_idx, "outlier_order"] = step + 1
        work.loc[chosen_idx, "outlier_method"] = method
        work.loc[chosen_idx, "outlier_metric"] = metric
        work.loc[chosen_idx, "outlier_threshold"] = threshold
        work.loc[chosen_idx, "outlier_direction"] = direction
        work.loc[chosen_idx, "outlier_details"] = details

        row = work.loc[chosen_idx]
        log_rows.append({
            "analyte": analyte,
            "Level": row.get("Level", ""),
            "Device": row.get("Device", ""),
            "batch_id": row.get("batch_id", ""),
            "Blood Sample ID": row.get("Blood Sample ID", ""),
            "Day": row.get("Day", ""),
            "Replicate": row.get("Replicate", ""),
            "removed_order": step + 1,
            "outlier_method_requested": cfg.outlier_method,
            "outlier_method_used": method,
            "value_removed": row[analyte],
            "direction": direction,
            "outlier_metric": metric,
            "outlier_threshold": threshold,
            "details": details,
            "gcrit_mode": cfg.gcrit_mode,
            "gcrit_alpha": cfg.gcrit_alpha,
            "gcrit_tail": cfg.gcrit_tail,
        })
        remaining.remove(chosen_idx)

    return work, pd.DataFrame(log_rows)


# -----------------------------
# Validation + statistics
# -----------------------------
def validate_and_standardize(df: pd.DataFrame, analytes: List[str]) -> Tuple[bool, str]:
    df_s = standardize_base_columns(df)
    missing = [c for c in REQUIRED_BASE_COLS if c not in df_s.columns]
    if missing:
        return False, f"Missing required columns after alias handling: {missing}"
    missing_analytes = [a for a in analytes if a not in df_s.columns]
    if missing_analytes:
        return False, f"Missing analyte columns: {missing_analytes}"
    return True, "OK"


def robust_precision_components(df_clean: pd.DataFrame, analyte: str, days: List[str]) -> Tuple[float, float, float, float, float, float, float]:
    y = pd.to_numeric(df_clean[analyte], errors="coerce").dropna().to_numpy(dtype=float)
    within_sds = []
    for d in days:
        vals = pd.to_numeric(df_clean.loc[df_clean["Day"].astype(str) == str(d), analyte], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) >= 2:
            within_sds.append(robust_sd_mad(vals))
        elif len(vals) == 1:
            within_sds.append(0.0)
    sd_repeat = float(np.median(within_sds)) if len(within_sds) else np.nan

    day_meds = []
    for d in days:
        vals = pd.to_numeric(df_clean.loc[df_clean["Day"].astype(str) == str(d), analyte], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) > 0:
            day_meds.append(np.median(vals))
    sd_day = float(robust_sd_mad(np.array(day_meds))) if len(day_meds) >= 2 else 0.0

    sd_total = float(np.sqrt(sd_repeat**2 + sd_day**2)) if np.isfinite(sd_repeat) and np.isfinite(sd_day) else np.nan
    center = float(np.median(y)) if len(y) else np.nan
    cv_repeat = 100.0 * sd_repeat / abs(center) if np.isfinite(center) and center != 0 else np.nan
    cv_day = 100.0 * sd_day / abs(center) if np.isfinite(center) and center != 0 else np.nan
    cv_total = 100.0 * sd_total / abs(center) if np.isfinite(center) and center != 0 else np.nan
    return sd_repeat, sd_day, sd_total, center, cv_repeat, cv_day, cv_total


def bootstrap_precision_ci(df_clean: pd.DataFrame, analyte: str, days: List[str], n_boot: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(int(seed))
    clean = df_clean.copy()
    clean[analyte] = pd.to_numeric(clean[analyte], errors="coerce")
    clean = clean.dropna(subset=[analyte])
    if len(clean) < 3:
        return {}

    cols = ["SD_repeat", "SD_between_day", "SD_total", "CV_repeat_%", "CV_between_day_%", "CV_total_%"]
    vals = {c: [] for c in cols}
    for _ in range(int(n_boot)):
        idx = rng.choice(clean.index.to_numpy(), size=len(clean), replace=True)
        boot = clean.loc[idx].copy()
        comps = robust_precision_components(boot, analyte, days)
        row = {
            "SD_repeat": comps[0],
            "SD_between_day": comps[1],
            "SD_total": comps[2],
            "CV_repeat_%": comps[4],
            "CV_between_day_%": comps[5],
            "CV_total_%": comps[6],
        }
        for c in cols:
            if np.isfinite(row[c]):
                vals[c].append(row[c])

    out = {}
    for c in cols:
        arr = np.asarray(vals[c], dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr):
            lo, hi = np.percentile(arr, [2.5, 97.5])
            out[f"{c}_95CI_low"] = float(lo)
            out[f"{c}_95CI_high"] = float(hi)
        else:
            out[f"{c}_95CI_low"] = np.nan
            out[f"{c}_95CI_high"] = np.nan
    return out


def compute_assumption_tests(df_clean: pd.DataFrame, analyte: str, days: List[str]) -> Dict[str, object]:
    y = pd.to_numeric(df_clean[analyte], errors="coerce").dropna().to_numpy(dtype=float)
    out = {
        "shapiro_p": np.nan,
        "normality_pass_0_05": False,
        "levene_p": np.nan,
        "variance_pass_0_05": False,
        "statistical_branch": "nonparametric_or_robust",
    }
    if len(y) >= 3:
        try:
            out["shapiro_p"] = float(stats.shapiro(y).pvalue)
            out["normality_pass_0_05"] = bool(out["shapiro_p"] >= 0.05)
        except Exception:
            pass
    groups = []
    for d in days:
        vals = pd.to_numeric(df_clean.loc[df_clean["Day"].astype(str) == str(d), analyte], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) >= 2:
            groups.append(vals)
    if len(groups) >= 2:
        try:
            out["levene_p"] = float(stats.levene(*groups).pvalue)
            out["variance_pass_0_05"] = bool(out["levene_p"] >= 0.05)
        except Exception:
            pass
    if out["normality_pass_0_05"] and (np.isnan(out["levene_p"]) or out["variance_pass_0_05"]):
        out["statistical_branch"] = "parametric"
    return out


def compute_descriptive_ci(y: np.ndarray, normality_pass: bool, cfg: Config) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    out = {
        "mean": float(np.mean(y)) if len(y) else np.nan,
        "sd_sample": float(np.std(y, ddof=1)) if len(y) >= 2 else np.nan,
        "median": float(np.median(y)) if len(y) else np.nan,
        "IQR": iqr(y),
        "center_95CI_low": np.nan,
        "center_95CI_high": np.nan,
        "center_CI_type": "",
    }
    if len(y) >= 2 and normality_pass:
        se = stats.sem(y, nan_policy="omit")
        tcrit = stats.t.ppf(0.975, df=len(y) - 1)
        out["center_95CI_low"] = float(out["mean"] - tcrit * se)
        out["center_95CI_high"] = float(out["mean"] + tcrit * se)
        out["center_CI_type"] = "t_95CI_for_mean"
    elif len(y) >= 2:
        lo, hi = bootstrap_ci_mean_or_median(y, use_median=True, n_boot=min(cfg.n_boot, 5000), seed=cfg.seed)
        out["center_95CI_low"] = lo
        out["center_95CI_high"] = hi
        out["center_CI_type"] = "bootstrap_95CI_for_median"
    return out


def normalization_method_for_analysis_column(analyte_col: str, cfg: Config) -> str:
    base_label = str(analyte_col).split("__", 1)[0]
    return cfg.analyte_normalization_map.get(base_label, cfg.normalization_method)


def compute_ep05_components(df_group: pd.DataFrame, analyte: str, cfg: Config, expected_n_scope: int) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = {"analyte": analyte}
    out["Level"] = str(df_group["Level"].iloc[0]) if len(df_group) else None
    out["Device"] = str(df_group["Device"].iloc[0]) if len(df_group) else None
    out["normalization_method"] = normalization_method_for_analysis_column(analyte, cfg)

    y_raw = pd.to_numeric(df_group[analyte], errors="coerce").dropna()
    out["N_raw"] = int(len(y_raw))

    df_with, outlier_log = detect_outliers_one_group(df_group, analyte, cfg)
    df_clean = df_with.loc[~df_with["is_outlier"]].copy()
    y = pd.to_numeric(df_clean[analyte], errors="coerce").dropna().to_numpy(dtype=float)

    out["N_clean"] = int(len(y))
    out["n_outliers"] = int(df_with["is_outlier"].sum())
    out["expected_n"] = int(expected_n_scope)
    out["outlier_method_requested"] = cfg.outlier_method
    used_methods = sorted([m for m in df_with.get("outlier_method", pd.Series(dtype=str)).astype(str).unique() if m])
    out["outlier_method_used"] = "; ".join(used_methods) if used_methods else ("None" if cfg.outlier_method == "None" else "No outlier removed")
    out["max_outliers_allowed"] = int(cfg.max_remove_per_group)
    out["gcrit"] = current_gcrit(len(y_raw), cfg) if cfg.outlier_method.startswith(("Gcrit", "Automatic")) else float(cfg.gcrit)
    out["gcrit_mode"] = cfg.gcrit_mode
    out["gcrit_alpha"] = cfg.gcrit_alpha
    out["gcrit_tail"] = cfg.gcrit_tail
    out["modified_z_threshold"] = cfg.modified_z_threshold
    out["robust_interval_z"] = cfg.robust_interval_z

    assumption = compute_assumption_tests(df_clean, analyte, cfg.days)
    out.update(assumption)
    out.update(compute_descriptive_ci(y, bool(out["normality_pass_0_05"]), cfg))

    out["method"] = "ROBUST_MAD"

    def robust_path():
        return robust_precision_components(df_clean, analyte, cfg.days)

    # Parametric path only if assumptions pass and the expected design is complete/no outliers.
    if out["n_outliers"] == 0 and out["N_clean"] == int(expected_n_scope) and out["statistical_branch"] == "parametric":
        try:
            df_m = df_clean.rename(columns={analyte: "value"}).copy()
            model = smf.mixedlm("value ~ 1", df_m, groups=df_m["Day"])
            res = model.fit(reml=True, method="lbfgs", disp=False)

            var_day = float(res.cov_re.iloc[0, 0]) if res.cov_re.shape == (1, 1) else 0.0
            var_within = float(res.scale)

            sd_repeat = float(np.sqrt(max(var_within, 0.0)))
            sd_day = float(np.sqrt(max(var_day, 0.0)))
            sd_total = float(np.sqrt(max(var_within + var_day, 0.0)))

            mean_val = float(np.mean(y))
            cv_repeat = 100.0 * sd_repeat / abs(mean_val) if mean_val != 0 else np.nan
            cv_day = 100.0 * sd_day / abs(mean_val) if mean_val != 0 else np.nan
            cv_total = 100.0 * sd_total / abs(mean_val) if mean_val != 0 else np.nan

            out["method"] = "MIXEDLM"
            out["center_used"] = "mean"
            out["center_value"] = mean_val
            out["SD_repeat"] = sd_repeat
            out["SD_between_day"] = sd_day
            out["SD_total"] = sd_total
            out["CV_repeat_%"] = cv_repeat
            out["CV_between_day_%"] = cv_day
            out["CV_total_%"] = cv_total

            if cfg.do_bootstrap_ci:
                out.update(bootstrap_precision_ci(df_clean, analyte, cfg.days, cfg.n_boot, cfg.seed))

            return out, df_with, df_clean, outlier_log
        except Exception:
            out["statistical_branch"] = "parametric_assumptions_passed_but_MixedLM_failed_used_robust"

    sd_repeat, sd_day, sd_total, center, cv_repeat, cv_day, cv_total = robust_path()
    out["center_used"] = "median"
    out["center_value"] = center
    out["SD_repeat"] = sd_repeat
    out["SD_between_day"] = sd_day
    out["SD_total"] = sd_total
    out["CV_repeat_%"] = cv_repeat
    out["CV_between_day_%"] = cv_day
    out["CV_total_%"] = cv_total

    if cfg.do_bootstrap_ci:
        out.update(bootstrap_precision_ci(df_clean, analyte, cfg.days, cfg.n_boot, cfg.seed))

    return out, df_with, df_clean, outlier_log


# -----------------------------
# Plotting and ZIP output
# -----------------------------
def make_histogram_png(values: np.ndarray, title: str) -> bytes:
    fig = plt.figure()
    plt.hist(values, bins=20)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Count")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def make_boxplot_png(values: np.ndarray, title: str) -> bytes:
    fig = plt.figure()
    plt.boxplot(values, vert=True)
    plt.title(title)
    plt.ylabel("Value")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def run_pipeline_to_zip(df: pd.DataFrame, cfg: Config) -> bytes:
    df_work, analysis_analytes, pair_table = build_analysis_dataframe(df, cfg)
    df_f = df_work.copy()
    df_f = df_f[df_f["Level"].astype(str).isin(cfg.levels)]
    df_f = df_f[df_f["Day"].astype(str).isin(cfg.days)]
    df_f = df_f[df_f["Device"].astype(str).isin(cfg.devices)]
    df_f = df_f[df_f["Replicate"].astype(int).isin(cfg.replicates)]

    df_f["Day"] = df_f["Day"].astype(str)
    df_f["Level"] = df_f["Level"].astype(str)
    df_f["Device"] = df_f["Device"].astype(str)
    df_f["Replicate"] = df_f["Replicate"].astype(int)

    all_rows = []
    all_outlier_logs = []
    zip_buf = io.BytesIO()

    def safe(s: str) -> str:
        return str(s).replace(" ", "_").replace("/", "_").replace("%", "pct").replace("*", "x")

    base_dir = "ep05_precision_results_NORMALIZATION_UPDATED"

    norm_comparison = evaluate_normalization_methods(df_f, cfg.analyte_pair_map, NORMALIZATION_METHODS)

    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{base_dir}/analyte_pair_mapping.csv", pair_table.to_csv(index=False).encode("utf-8"))
        zf.writestr(f"{base_dir}/normalization_method_comparison.csv", norm_comparison.to_csv(index=False).encode("utf-8"))

        if not norm_comparison.empty:
            best_method = norm_comparison.sort_values("overall_score_lower_is_better").iloc[0]["normalization_method"]
            per_analyte_recommendation = (
                norm_comparison.loc[norm_comparison["normality_guided_recommended_for_analyte"].fillna(False)]
                .sort_values("analyte")
                [[
                    "analyte",
                    "per_analyte_recommended_method",
                    "residual_shapiro_p",
                    "residual_normality_pass_0_05",
                    "normalization_score_lower_is_better",
                    "recommendation_reason",
                ]]
            )
            recommendation = pd.DataFrame([{
                "recommended_normalization_method_overall": best_method,
                "final_run_mode": "per-analyte map" if cfg.analyte_normalization_map else "single selected method",
                "selection_rule": "per-analyte recommendation uses composite drift/CV/%bias/level-separation score plus Shapiro-Wilk normality of Level/Day-adjusted paired %bias residuals; reference-drift correction is preferred when residuals are normal and score is competitive, robust MAD z-score when residuals are non-normal/outlier-heavy.",
            }])
            zf.writestr(f"{base_dir}/normalization_recommendation.csv", recommendation.to_csv(index=False).encode("utf-8"))
            zf.writestr(f"{base_dir}/normalization_recommendation_by_analyte.csv", per_analyte_recommendation.to_csv(index=False).encode("utf-8"))

        scope_tables = []
        if cfg.device_mode == "Pool all devices":
            pooled_df = df_f.copy()
            pooled_df["Device"] = "pooled_all_devices"
            scope_tables.append(("pooled_all_devices", pooled_df, len(cfg.days) * len(cfg.replicates) * len(cfg.devices)))
        else:
            pooled_df = df_f.copy()
            pooled_df["Device"] = "pooled_all_devices"
            scope_tables.append(("pooled_all_devices", pooled_df, len(cfg.days) * len(cfg.replicates) * len(cfg.devices)))
            for device in cfg.devices:
                scope_tables.append((str(device), df_f[df_f["Device"] == str(device)].copy(), len(cfg.days) * len(cfg.replicates)))

        for analyte in analysis_analytes:
            for level in cfg.levels:
                for scope_name, scope_df, expected_n_scope in scope_tables:
                    sub = scope_df[scope_df["Level"] == level].copy()
                    if sub.empty or analyte not in sub.columns:
                        continue

                    out, with_outliers, clean, outlier_log = compute_ep05_components(sub, analyte, cfg, expected_n_scope)
                    all_rows.append(out)
                    if outlier_log is not None and not outlier_log.empty:
                        all_outlier_logs.append(outlier_log)

                    stem = f"{safe(analyte)}__{safe(level)}__{safe(scope_name)}"

                    zf.writestr(
                        f"{base_dir}/{safe(analyte)}/{stem}_data_with_outliers.csv",
                        with_outliers.to_csv(index=False).encode("utf-8"),
                    )
                    zf.writestr(
                        f"{base_dir}/{safe(analyte)}/{stem}_EP05_precision_table.csv",
                        pd.DataFrame([out]).to_csv(index=False).encode("utf-8"),
                    )

                    vals = pd.to_numeric(clean[analyte], errors="coerce").dropna().to_numpy(dtype=float)
                    if len(vals) > 0:
                        title = f"{analyte} | {level} | {scope_name} | {out['method']}"
                        zf.writestr(f"{base_dir}/{safe(analyte)}/{stem}_histogram.png", make_histogram_png(vals, title))
                        zf.writestr(f"{base_dir}/{safe(analyte)}/{stem}_boxplot.png", make_boxplot_png(vals, title))

        summary = pd.DataFrame(all_rows)
        zf.writestr(f"{base_dir}/ALL_analytes_precision_summary.csv", summary.to_csv(index=False).encode("utf-8"))

        outlier_log_all = pd.concat(all_outlier_logs, ignore_index=True) if all_outlier_logs else pd.DataFrame(
            columns=["analyte", "Level", "Device", "batch_id", "Blood Sample ID", "Day", "Replicate", "removed_order",
                     "outlier_method_requested", "outlier_method_used", "value_removed", "direction", "outlier_metric",
                     "outlier_threshold", "details", "gcrit_mode", "gcrit_alpha", "gcrit_tail"]
        )
        zf.writestr(f"{base_dir}/ALL_outlier_log.csv", outlier_log_all.to_csv(index=False).encode("utf-8"))

        if not summary.empty:
            pooled = (
                summary.groupby(["analyte", "Level"], as_index=False)
                .agg(
                    mean=("mean", "mean"),
                    median=("median", "mean"),
                    center_95CI_low=("center_95CI_low", "mean"),
                    center_95CI_high=("center_95CI_high", "mean"),
                    SD_repeat=("SD_repeat", "mean"),
                    SD_between_day=("SD_between_day", "mean"),
                    SD_total=("SD_total", "mean"),
                    CV_repeat_pct=("CV_repeat_%", "mean"),
                    CV_between_day_pct=("CV_between_day_%", "mean"),
                    CV_total_pct=("CV_total_%", "mean"),
                    N_groups=("Device", "nunique"),
                    parametric_groups=("statistical_branch", lambda x: int(np.sum(pd.Series(x).astype(str).eq("parametric")))),
                )
            )
        else:
            pooled = pd.DataFrame()

        zf.writestr(f"{base_dir}/ALL_analytes_precision_pooled_by_analyte_level.csv", pooled.to_csv(index=False).encode("utf-8"))

    return zip_buf.getvalue()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="EP05 Precision App", layout="wide")
st.title("EP05 Long-Term Precision + Reference Drift Normalization")

st.markdown(
    """
**Steps**
1) Upload the XLSX  
2) Confirm/adjust Levels, Days, Replicates, Devices  
3) Select paired CBC analytes and compare normalization methods  
4) Choose the recommended method or manually select another  
5) Choose pooled/per-device handling and outlier settings  
6) Run → download results ZIP
"""
)

uploaded = st.file_uploader("Upload the XLSX", type=["xlsx"])
if uploaded is None:
    st.info("Upload your EP05-style XLSX to begin.")
    st.stop()

try:
    df_raw = pd.read_excel(uploaded, engine="openpyxl")
    df = standardize_base_columns(df_raw)
except Exception as e:
    st.error(f"Could not read Excel file: {e}")
    st.stop()

st.subheader("Preview")
st.dataframe(df.head(20), use_container_width=True)


def present_values(col: str, defaults: List[str]) -> Tuple[List[str], List[str]]:
    if col in df.columns:
        vals = sorted(list({str(x) for x in df[col].dropna().tolist()}))
        if len(defaults) == 0:
            return vals, vals
        if all(str(d) in vals for d in defaults):
            return [str(d) for d in defaults], vals
        return vals, vals
    return [str(d) for d in defaults], [str(d) for d in defaults]


def present_int_values(col: str, defaults: List[int]) -> Tuple[List[int], List[int]]:
    if col in df.columns:
        vals = sorted(list({int(x) for x in df[col].dropna().tolist()}))
        if all(int(d) in vals for d in defaults):
            return defaults, vals
        return vals, vals
    return defaults, defaults


levels_default, levels_all = present_values("Level", DEFAULT_LEVELS)
days_default, days_all = present_values("Day", DEFAULT_DAYS)
devices_default, devices_all = present_values("Device", DEFAULT_DEVICES)
rep_default, rep_all = present_int_values("Replicate", DEFAULT_REPLICATES)

st.subheader("Design selection")
c1, c2, c3, c4 = st.columns(4)
with c1:
    levels = st.multiselect("Levels", options=levels_all, default=levels_default)
with c2:
    days = st.multiselect("Days", options=days_all, default=days_default)
with c3:
    replicates = st.multiselect("Replicates", options=rep_all, default=rep_default)
with c4:
    devices = st.multiselect("Devices", options=devices_all, default=devices_default)

st.subheader("Paired analytes and normalization")
available_pairs = resolve_analyte_pairs(df)
missing_pairs = [a for a in DEFAULT_ANALYTES if a not in available_pairs]
if missing_pairs:
    st.warning(f"Some default paired analytes were not found and are hidden: {missing_pairs}")

paired_analytes = st.multiselect(
    "Select paired analytes to analyze",
    options=list(available_pairs.keys()),
    default=list(available_pairs.keys()),
)
selected_pair_map = resolve_analyte_pairs(df, paired_analytes)

if selected_pair_map:
    st.caption("Detected device/reference column mapping")
    st.dataframe(pd.DataFrame([{"Analyte": k, "Device column": v[0], "Reference column": v[1]} for k, v in selected_pair_map.items()]), use_container_width=True)

compare_now = st.checkbox("Show normalization comparison and recommendation", value=True)
recommended_method = "Raw/no normalization"
norm_comparison = pd.DataFrame()
per_analyte_recommended_map: Dict[str, str] = {}
if compare_now and selected_pair_map:
    with st.spinner("Comparing normalization methods and testing paired %bias residual normality..."):
        norm_comparison = evaluate_normalization_methods(df[df["Level"].astype(str).isin(levels)] if levels else df, selected_pair_map, NORMALIZATION_METHODS)
    if not norm_comparison.empty:
        recommended_method = str(norm_comparison.sort_values("overall_score_lower_is_better").iloc[0]["normalization_method"])
        per_analyte_recs = (
            norm_comparison.loc[norm_comparison["normality_guided_recommended_for_analyte"].fillna(False)]
            .sort_values("analyte")
            [[
                "analyte",
                "per_analyte_recommended_method",
                "residual_shapiro_p",
                "residual_normality_pass_0_05",
                "normalization_score_lower_is_better",
                "recommendation_reason",
            ]]
        )
        per_analyte_recommended_map = dict(zip(per_analyte_recs["analyte"], per_analyte_recs["per_analyte_recommended_method"]))
        st.success(f"Overall recommended normalization: {recommended_method}")
        st.caption("Per-analyte recommendation uses a composite drift/CV/%bias/level-separation score plus Shapiro-Wilk normality of Level/Day-adjusted paired %bias residuals.")
        st.dataframe(per_analyte_recs, use_container_width=True)
        with st.expander("Show full normalization comparison table"):
            st.dataframe(norm_comparison, use_container_width=True)
    else:
        st.info("Normalization comparison could not be computed; using raw/no normalization by default.")

use_per_analyte_normalization = False
if per_analyte_recommended_map:
    use_per_analyte_normalization = st.checkbox(
        "Use normality-guided recommended normalization separately for each paired analyte in final run",
        value=True,
    )

norm_default_index = NORMALIZATION_METHODS.index(recommended_method) if recommended_method in NORMALIZATION_METHODS else 0
normalization_method = st.selectbox(
    "Fallback/single normalization method for the final EP05 run",
    NORMALIZATION_METHODS,
    index=norm_default_index,
)

active_analyte_normalization_map = per_analyte_recommended_map if use_per_analyte_normalization else {}
if active_analyte_normalization_map:
    st.info("Final run will use the per-analyte recommended normalization map. The fallback/single method applies only to raw extra analytes or analytes without a recommendation.")

value_output_modes = st.multiselect(
    "Metrics to calculate/analyze from each device-reference pair",
    VALUE_OUTPUT_MODES,
    default=["Device normalized value", "%Bias: 100*(device-reference)/reference"],
)

st.subheader("Additional raw analyte columns, optional")
observed_candidate_cols = [c for c in df.columns if c not in REQUIRED_BASE_COLS]
extra_analytes = st.multiselect(
    "Optional: also analyze raw columns directly without paired normalization",
    options=sorted(observed_candidate_cols),
    default=[],
)

st.subheader("Analysis settings")
c1, c2, c3, c4 = st.columns(4)
with c1:
    device_mode = st.selectbox("Device handling", ["Pool all devices", "Analyze each device separately + pooled"])
with c2:
    do_bootstrap_ci = st.checkbox("Bootstrap 95% CIs", value=True)
with c3:
    n_boot = st.number_input("Bootstrap iterations", min_value=200, max_value=20000, value=2000, step=200)
with c4:
    seed = st.number_input("Random seed", min_value=1, max_value=999999, value=123, step=1)

st.subheader("Optional outlier detection")
outlier_method = st.selectbox(
    "Outlier method for cleaned/sensitivity results",
    [
        "None",
        "Automatic: Grubbs if Shapiro normal, otherwise Robust MAD",
        "Gcrit Grubbs-like: remove largest |value-mean|/SD if >= Gcrit",
        "Robust MAD modified-z: remove largest robust z if >= threshold",
        "95% robust interval: remove most extreme outside median ± z*MAD_SD",
    ],
    index=1,
)
c1, c2, c3, c4 = st.columns(4)
with c1:
    max_remove_per_group = st.selectbox("Max outliers to remove per analyte/level/scope", [0, 1, 2], index=1)
with c2:
    gcrit_mode = st.selectbox("Gcrit mode", ["Manual Gcrit value", "Automatic from n, alpha, tail"], index=1)
with c3:
    gcrit = st.number_input("Manual Gcrit value", min_value=0.0, value=3.135, step=0.001, format="%.3f")
with c4:
    gcrit_alpha = st.number_input("Automatic Gcrit alpha", min_value=0.0001, max_value=0.2, value=0.01, step=0.001, format="%.4f")

c1, c2, c3 = st.columns(3)
with c1:
    gcrit_tail = st.selectbox("Automatic Gcrit tail", ["Two-sided", "One-sided"], index=0)
with c2:
    modified_z_threshold = st.number_input("MAD modified-z threshold", min_value=0.1, value=3.5, step=0.1)
with c3:
    robust_interval_z = st.number_input("Robust interval z", min_value=0.5, value=1.96, step=0.01)

expected_n_single_device = len(days) * len(replicates)
expected_n_pooled = len(days) * len(replicates) * len(devices)
st.caption(
    f"Expected N per single device group = Days × Replicates = {len(days)} × {len(replicates)} = **{expected_n_single_device}**. "
    f"Pooled expected N across selected devices = {len(days)} × {len(replicates)} × {len(devices)} = **{expected_n_pooled}**."
)
st.markdown("""
**Assumption-aware testing:** each analyte/level/scope now receives Shapiro-Wilk normality testing and Levene variance testing.  
If assumptions pass, the app attempts the parametric MixedLM precision estimate. If assumptions fail or the model cannot fit, it uses robust MAD-based precision.  
**Outlier default:** automatic outlier handling uses Grubbs only when Shapiro-Wilk supports normality; otherwise it uses robust MAD.
""")

# Validation is run after normalized columns are generated.
placeholder_cfg = Config(
    analytes=extra_analytes,
    levels=levels,
    days=days,
    replicates=[int(r) for r in replicates],
    devices=devices,
    gcrit=float(gcrit),
    expected_n=int(expected_n_single_device),
    device_mode=device_mode,
    outlier_method=outlier_method,
    max_remove_per_group=int(max_remove_per_group),
    gcrit_mode=gcrit_mode,
    gcrit_alpha=float(gcrit_alpha),
    gcrit_tail=gcrit_tail,
    modified_z_threshold=float(modified_z_threshold),
    robust_interval_z=float(robust_interval_z),
    do_bootstrap_ci=bool(do_bootstrap_ci),
    n_boot=int(n_boot),
    seed=int(seed),
    paired_analytes=paired_analytes,
    normalization_method=normalization_method,
    value_output_modes=value_output_modes,
    analyte_pair_map=selected_pair_map,
    analyte_normalization_map=active_analyte_normalization_map,
)
try:
    df_check, analysis_analytes_check, _ = build_analysis_dataframe(df, placeholder_cfg)
    ok, msg = validate_and_standardize(df_check, analysis_analytes_check)
except Exception as e:
    ok, msg = False, str(e)

if not ok:
    st.error(msg)
    st.stop()

run_btn = st.button("Run EP05 analysis", type="primary")
if run_btn:
    if len(levels) == 0 or len(days) == 0 or len(replicates) == 0 or len(devices) == 0:
        st.error("Please select at least one Level, Day, Replicate, and Device.")
        st.stop()
    if len(selected_pair_map) == 0 and len(extra_analytes) == 0:
        st.error("Please select at least one paired analyte or raw analyte column.")
        st.stop()
    if len(value_output_modes) == 0 and len(selected_pair_map) > 0:
        st.error("Please select at least one paired metric to analyze.")
        st.stop()

    cfg = Config(
        analytes=extra_analytes,
        levels=levels,
        days=days,
        replicates=[int(r) for r in replicates],
        devices=devices,
        gcrit=float(gcrit),
        expected_n=int(expected_n_single_device),
        device_mode=device_mode,
        outlier_method=outlier_method,
        max_remove_per_group=int(max_remove_per_group),
        gcrit_mode=gcrit_mode,
        gcrit_alpha=float(gcrit_alpha),
        gcrit_tail=gcrit_tail,
        modified_z_threshold=float(modified_z_threshold),
        robust_interval_z=float(robust_interval_z),
        do_bootstrap_ci=bool(do_bootstrap_ci),
        n_boot=int(n_boot),
        seed=int(seed),
        paired_analytes=paired_analytes,
        normalization_method=normalization_method,
        value_output_modes=value_output_modes,
        analyte_pair_map=selected_pair_map,
        analyte_normalization_map=active_analyte_normalization_map,
    )

    with st.spinner("Running analysis..."):
        zip_bytes = run_pipeline_to_zip(df, cfg)

    st.success("Done. Download your results ZIP below.")
    st.download_button(
        label="Download results ZIP",
        data=zip_bytes,
        file_name="ep05_precision_results_normalization_updated.zip",
        mime="application/zip",
    )
