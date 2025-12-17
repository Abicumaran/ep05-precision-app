import io
import zipfile
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import streamlit as st
import statsmodels.formula.api as smf

# -----------------------------
# Defaults that match your EP05 template
# -----------------------------
REQUIRED_BASE_COLS = ["batch_id", "Blood Sample ID", "Level", "Day", "Replicate", "Device"]

DEFAULT_ANALYTES = ["HGB", "HCT", "RBC", "WBC", "PLT", "NEUT", "LYMPH", "MXD"]
DEFAULT_LEVELS = ["Low", "Mid", "High"]
DEFAULT_DAYS = ["D1", "D2", "D3", "D4", "D5"]
DEFAULT_REPLICATES = [1, 2, 3, 4, 5]
DEFAULT_DEVICES = ["Unit 9", "Unit 10", "Unit 12"]

# -----------------------------
# Core EP05 logic
# -----------------------------
@dataclass
class Config:
    analytes: List[str]
    levels: List[str]
    days: List[str]
    replicates: List[int]
    devices: List[str]
    gcrit: float
    expected_n: int  # typically days * replicates


def grubbs_single_pass_flag(series: pd.Series, gcrit: float) -> pd.Series:
    """Single-pass fixed-threshold 'Grubbs-like' flagging using user-provided Gcrit."""
    x = series.astype(float)
    mu = x.mean()
    sd = x.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return pd.Series([False] * len(x), index=x.index)
    g = (x - mu).abs() / sd
    return g >= gcrit


def robust_sd_mad(x: np.ndarray) -> float:
    """Robust SD estimate via MAD scaling."""
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad


def validate_and_standardize(df: pd.DataFrame, analytes: List[str]) -> Tuple[bool, str]:
    missing = [c for c in REQUIRED_BASE_COLS if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {missing}"
    missing_analytes = [a for a in analytes if a not in df.columns]
    if missing_analytes:
        return False, f"Missing analyte columns: {missing_analytes}"
    return True, "OK"


def compute_ep05_components(df_group: pd.DataFrame, analyte: str, cfg: Config) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """
    One (Analyte, Level, Device) group:
      - single-pass Grubbs (fixed Gcrit)
      - if no outliers AND N_clean == expected_n:
          Shapiro + Levene; if ok MixedLM (random intercept by Day), else robust MAD
        else:
          robust MAD
    """
    out = {"analyte": analyte}
    out["Level"] = str(df_group["Level"].iloc[0]) if len(df_group) else None
    out["Device"] = str(df_group["Device"].iloc[0]) if len(df_group) else None

    y_raw = df_group[analyte].astype(float)
    out["N_raw"] = int(len(y_raw))

    # Outliers
    is_outlier = grubbs_single_pass_flag(y_raw, cfg.gcrit)
    df_with = df_group.copy()
    df_with["is_outlier"] = is_outlier.values

    df_clean = df_with.loc[~df_with["is_outlier"]].copy()
    y = df_clean[analyte].astype(float).values

    out["N_clean"] = int(len(y))
    out["n_outliers"] = int(df_with["is_outlier"].sum())
    out["expected_n"] = int(cfg.expected_n)
    out["gcrit"] = float(cfg.gcrit)

    out["method"] = "ROBUST_MAD"
    out["shapiro_p"] = np.nan
    out["levene_p"] = np.nan

    def robust_path():
        # within-day: MAD SD per day then median
        within_sds = []
        for d in cfg.days:
            vals = df_clean.loc[df_clean["Day"] == d, analyte].astype(float).values
            if len(vals) >= 2:
                within_sds.append(robust_sd_mad(vals))
            elif len(vals) == 1:
                within_sds.append(0.0)
        sd_repeat = float(np.median(within_sds)) if len(within_sds) else np.nan

        # between-day: MAD SD across day medians
        day_meds = []
        for d in cfg.days:
            vals = df_clean.loc[df_clean["Day"] == d, analyte].astype(float).values
            if len(vals) > 0:
                day_meds.append(np.median(vals))
        sd_day = float(robust_sd_mad(np.array(day_meds))) if len(day_meds) >= 2 else 0.0

        sd_total = float(np.sqrt(sd_repeat**2 + sd_day**2)) if np.isfinite(sd_repeat) and np.isfinite(sd_day) else np.nan

        center = float(np.median(y)) if len(y) else np.nan
        cv_repeat = 100.0 * sd_repeat / center if np.isfinite(center) and center != 0 else np.nan
        cv_day = 100.0 * sd_day / center if np.isfinite(center) and center != 0 else np.nan
        cv_total = 100.0 * sd_total / center if np.isfinite(center) and center != 0 else np.nan
        return sd_repeat, sd_day, sd_total, center, cv_repeat, cv_day, cv_total

    # MixedLM attempt if pristine expected N and no outliers
    if out["n_outliers"] == 0 and out["N_clean"] == cfg.expected_n:
        try:
            out["shapiro_p"] = float(stats.shapiro(y).pvalue) if len(y) >= 3 else np.nan
        except Exception:
            out["shapiro_p"] = np.nan

        try:
            groups = []
            for d in cfg.days:
                vals = df_clean.loc[df_clean["Day"] == d, analyte].astype(float).values
                if len(vals) > 0:
                    groups.append(vals)
            out["levene_p"] = float(stats.levene(*groups).pvalue) if len(groups) >= 2 else np.nan
        except Exception:
            out["levene_p"] = np.nan

        if (np.isnan(out["shapiro_p"]) or out["shapiro_p"] >= 0.05) and (np.isnan(out["levene_p"]) or out["levene_p"] >= 0.05):
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
                cv_repeat = 100.0 * sd_repeat / mean_val if mean_val != 0 else np.nan
                cv_day = 100.0 * sd_day / mean_val if mean_val != 0 else np.nan
                cv_total = 100.0 * sd_total / mean_val if mean_val != 0 else np.nan

                out["method"] = "MIXEDLM"
                out["center_used"] = "mean"
                out["center_value"] = mean_val
                out["SD_repeat"] = sd_repeat
                out["SD_between_day"] = sd_day
                out["SD_total"] = sd_total
                out["CV_repeat_%"] = cv_repeat
                out["CV_between_day_%"] = cv_day
                out["CV_total_%"] = cv_total

                return out, df_with, df_clean
            except Exception:
                pass

    # Robust fallback
    sd_repeat, sd_day, sd_total, center, cv_repeat, cv_day, cv_total = robust_path()
    out["center_used"] = "median"
    out["center_value"] = center
    out["SD_repeat"] = sd_repeat
    out["SD_between_day"] = sd_day
    out["SD_total"] = sd_total
    out["CV_repeat_%"] = cv_repeat
    out["CV_between_day_%"] = cv_day
    out["CV_total_%"] = cv_total

    return out, df_with, df_clean


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
    """Run analysis and return a ZIP (bytes) with per-group outputs + 2 global CSVs."""
    df_f = df.copy()
    df_f = df_f[df_f["Level"].astype(str).isin(cfg.levels)]
    df_f = df_f[df_f["Day"].astype(str).isin(cfg.days)]
    df_f = df_f[df_f["Device"].astype(str).isin(cfg.devices)]
    df_f = df_f[df_f["Replicate"].astype(int).isin(cfg.replicates)]

    # Types
    df_f["Day"] = df_f["Day"].astype(str)
    df_f["Level"] = df_f["Level"].astype(str)
    df_f["Device"] = df_f["Device"].astype(str)
    df_f["Replicate"] = df_f["Replicate"].astype(int)

    all_rows = []
    zip_buf = io.BytesIO()

    def safe(s: str) -> str:
        return str(s).replace(" ", "_").replace("/", "_")

    base_dir = f"ep05_precision_results_ROBUST_Gcrit{cfg.gcrit}".replace(".", "")

    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for analyte in cfg.analytes:
            for level in cfg.levels:
                for device in cfg.devices:
                    sub = df_f[(df_f["Level"] == level) & (df_f["Device"] == device)].copy()
                    if sub.empty:
                        continue

                    out, with_outliers, clean = compute_ep05_components(sub, analyte, cfg)
                    all_rows.append(out)

                    stem = f"{safe(analyte)}__{safe(level)}__{safe(device)}"

                    zf.writestr(
                        f"{base_dir}/{safe(analyte)}/{stem}_data_with_outliers.csv",
                        with_outliers.to_csv(index=False).encode("utf-8"),
                    )
                    zf.writestr(
                        f"{base_dir}/{safe(analyte)}/{stem}_EP05_precision_table.csv",
                        pd.DataFrame([out]).to_csv(index=False).encode("utf-8"),
                    )

                    vals = clean[analyte].astype(float).values
                    if len(vals) > 0:
                        title = f"{analyte} | {level} | {device} | {out['method']}"
                        zf.writestr(
                            f"{base_dir}/{safe(analyte)}/{stem}_histogram.png",
                            make_histogram_png(vals, title),
                        )
                        zf.writestr(
                            f"{base_dir}/{safe(analyte)}/{stem}_boxplot.png",
                            make_boxplot_png(vals, title),
                        )

        summary = pd.DataFrame(all_rows)
        zf.writestr(
            f"{base_dir}/ALL_analytes_precision_summary_ROBUST.csv",
            summary.to_csv(index=False).encode("utf-8"),
        )

        if not summary.empty:
            pooled = (
                summary.groupby(["analyte", "Level"], as_index=False)
                .agg(
                    SD_repeat=("SD_repeat", "mean"),
                    SD_between_day=("SD_between_day", "mean"),
                    SD_total=("SD_total", "mean"),
                    CV_repeat_pct=("CV_repeat_%", "mean"),
                    CV_between_day_pct=("CV_between_day_%", "mean"),
                    CV_total_pct=("CV_total_%", "mean"),
                    N_groups=("Device", "nunique"),
                )
            )
        else:
            pooled = pd.DataFrame()

        zf.writestr(
            f"{base_dir}/ALL_analytes_precision_pooled_by_analyte_level.csv",
            pooled.to_csv(index=False).encode("utf-8"),
        )

    return zip_buf.getvalue()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="EP05 Precision App", layout="wide")
st.title("EP05 Long-Term Precision (Upload XLSX → Configure → Run → Download ZIP)")

st.markdown(
    """
**Steps**
1) Upload the XLSX  
2) Confirm/adjust Levels, Days, Replicates, Devices  
3) Select analytes (or add new analyte names)  
4) Enter **Gcrit**  
5) Run → download results ZIP
"""
)

uploaded = st.file_uploader("Upload the XLSX", type=["xlsx"])
if uploaded is None:
    st.info("Upload your EP05-style XLSX to begin.")
    st.stop()

try:
    df = pd.read_excel(uploaded, engine="openpyxl")
except Exception as e:
    st.error(f"Could not read Excel file: {e}")
    st.stop()

st.subheader("Preview")
st.dataframe(df.head(20), use_container_width=True)

def present_values(col: str, defaults: List[str]) -> Tuple[List[str], List[str]]:
    if col in df.columns:
        vals = sorted(list({str(x) for x in df[col].dropna().tolist()}))
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

st.subheader("Analytes")
observed_candidate_cols = [c for c in df.columns if c not in REQUIRED_BASE_COLS]
default_analytes = [a for a in DEFAULT_ANALYTES if a in df.columns]
if len(default_analytes) == 0:
    default_analytes = observed_candidate_cols[:8]

analytes = st.multiselect(
    "Select analytes to analyze",
    options=sorted(list(set(observed_candidate_cols + DEFAULT_ANALYTES))),
    default=default_analytes,
)

new_analyte = st.text_input("Create new analyte (type exact column name, press Enter)", value="")
if new_analyte.strip():
    if new_analyte.strip() not in analytes:
        analytes = analytes + [new_analyte.strip()]
        st.success(f"Added analyte: {new_analyte.strip()}")
    else:
        st.info("That analyte is already selected.")

st.subheader("Outlier threshold")
gcrit = st.number_input("Gcrit (manual entry)", min_value=0.0, value=3.135, step=0.001, format="%.3f")

expected_n = len(days) * len(replicates)
st.caption(f"Expected N per (Analyte, Level, Device) group = Days × Replicates = {len(days)} × {len(replicates)} = **{expected_n}**")

ok, msg = validate_and_standardize(df, analytes)
if not ok:
    st.error(msg)
    st.stop()

run_btn = st.button("Run EP05 analysis", type="primary")
if run_btn:
    if len(levels) == 0 or len(days) == 0 or len(replicates) == 0 or len(devices) == 0 or len(analytes) == 0:
        st.error("Please select at least one Level, Day, Replicate, Device, and Analyte.")
        st.stop()

    cfg = Config(
        analytes=analytes,
        levels=levels,
        days=days,
        replicates=[int(r) for r in replicates],
        devices=devices,
        gcrit=float(gcrit),
        expected_n=int(expected_n),
    )

    with st.spinner("Running analysis..."):
        zip_bytes = run_pipeline_to_zip(df, cfg)

    st.success("Done. Download your results ZIP below.")
    st.download_button(
        label="Download results ZIP",
        data=zip_bytes,
        file_name=f"ep05_precision_results_Gcrit{gcrit}.zip",
        mime="application/zip",
    )
