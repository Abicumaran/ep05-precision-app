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
    expected_n: int  # typically days * replicates; overwritten per pooled/device scope when needed
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


def robust_sd_mad(x: np.ndarray) -> float:
    """Robust SD estimate via MAD scaling."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad


def grubbs_gcrit_auto(n: int, alpha: float = 0.01, tail: str = "Two-sided") -> float:
    """Critical Grubbs G value from n, alpha, and one-/two-sided choice."""
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


def detect_outliers_one_group(df_group: pd.DataFrame, analyte: str, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match the interference app style:
      - None
      - Gcrit Grubbs-like largest |value-mean|/SD if >= Gcrit
      - Robust MAD modified-z
      - 95% robust interval median ± z*MAD_SD
    Removes sequentially up to cfg.max_remove_per_group within one analyte/level/device-scope group.
    """
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
    method = str(cfg.outlier_method)

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
            "outlier_method": method,
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


def validate_and_standardize(df: pd.DataFrame, analytes: List[str]) -> Tuple[bool, str]:
    missing = [c for c in REQUIRED_BASE_COLS if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {missing}"
    missing_analytes = [a for a in analytes if a not in df.columns]
    if missing_analytes:
        return False, f"Missing analyte columns: {missing_analytes}"
    return True, "OK"


def robust_precision_components(df_clean: pd.DataFrame, analyte: str, days: List[str]) -> Tuple[float, float, float, float, float, float, float]:
    y = pd.to_numeric(df_clean[analyte], errors="coerce").dropna().to_numpy(dtype=float)
    # within-day: MAD SD per day then median
    within_sds = []
    for d in days:
        vals = pd.to_numeric(df_clean.loc[df_clean["Day"] == d, analyte], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) >= 2:
            within_sds.append(robust_sd_mad(vals))
        elif len(vals) == 1:
            within_sds.append(0.0)
    sd_repeat = float(np.median(within_sds)) if len(within_sds) else np.nan

    # between-day: MAD SD across day medians
    day_meds = []
    for d in days:
        vals = pd.to_numeric(df_clean.loc[df_clean["Day"] == d, analyte], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) > 0:
            day_meds.append(np.median(vals))
    sd_day = float(robust_sd_mad(np.array(day_meds))) if len(day_meds) >= 2 else 0.0

    sd_total = float(np.sqrt(sd_repeat**2 + sd_day**2)) if np.isfinite(sd_repeat) and np.isfinite(sd_day) else np.nan
    center = float(np.median(y)) if len(y) else np.nan
    cv_repeat = 100.0 * sd_repeat / center if np.isfinite(center) and center != 0 else np.nan
    cv_day = 100.0 * sd_day / center if np.isfinite(center) and center != 0 else np.nan
    cv_total = 100.0 * sd_total / center if np.isfinite(center) and center != 0 else np.nan
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


def compute_ep05_components(df_group: pd.DataFrame, analyte: str, cfg: Config, expected_n_scope: int) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    One (Analyte, Level, Device/scope) group:
      - optional sequential outlier removal using Gcrit, robust MAD z, or robust interval
      - if no outliers AND N_clean == expected_n:
          Shapiro + Levene; if ok MixedLM (random intercept by Day), else robust MAD
        else:
          robust MAD
      - optional bootstrap 95% CIs for robust precision components
    """
    out = {"analyte": analyte}
    out["Level"] = str(df_group["Level"].iloc[0]) if len(df_group) else None
    out["Device"] = str(df_group["Device"].iloc[0]) if len(df_group) else None

    y_raw = pd.to_numeric(df_group[analyte], errors="coerce").dropna()
    out["N_raw"] = int(len(y_raw))

    df_with, outlier_log = detect_outliers_one_group(df_group, analyte, cfg)
    df_clean = df_with.loc[~df_with["is_outlier"]].copy()
    y = pd.to_numeric(df_clean[analyte], errors="coerce").dropna().to_numpy(dtype=float)

    out["N_clean"] = int(len(y))
    out["n_outliers"] = int(df_with["is_outlier"].sum())
    out["expected_n"] = int(expected_n_scope)
    out["outlier_method"] = cfg.outlier_method
    out["max_outliers_allowed"] = int(cfg.max_remove_per_group)
    out["gcrit"] = current_gcrit(len(y_raw), cfg) if cfg.outlier_method.startswith("Gcrit") else float(cfg.gcrit)
    out["gcrit_mode"] = cfg.gcrit_mode
    out["gcrit_alpha"] = cfg.gcrit_alpha
    out["gcrit_tail"] = cfg.gcrit_tail
    out["modified_z_threshold"] = cfg.modified_z_threshold
    out["robust_interval_z"] = cfg.robust_interval_z

    out["method"] = "ROBUST_MAD"
    out["shapiro_p"] = np.nan
    out["levene_p"] = np.nan

    def robust_path():
        return robust_precision_components(df_clean, analyte, cfg.days)

    # MixedLM attempt if pristine expected N and no outliers
    if out["n_outliers"] == 0 and out["N_clean"] == int(expected_n_scope):
        try:
            out["shapiro_p"] = float(stats.shapiro(y).pvalue) if len(y) >= 3 else np.nan
        except Exception:
            out["shapiro_p"] = np.nan

        try:
            groups = []
            for d in cfg.days:
                vals = pd.to_numeric(df_clean.loc[df_clean["Day"] == d, analyte], errors="coerce").dropna().to_numpy(dtype=float)
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

                if cfg.do_bootstrap_ci:
                    out.update(bootstrap_precision_ci(df_clean, analyte, cfg.days, cfg.n_boot, cfg.seed))

                return out, df_with, df_clean, outlier_log
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

    if cfg.do_bootstrap_ci:
        out.update(bootstrap_precision_ci(df_clean, analyte, cfg.days, cfg.n_boot, cfg.seed))

    return out, df_with, df_clean, outlier_log


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
    """Run analysis and return a ZIP (bytes) with per-group outputs + global CSVs."""
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
    all_outlier_logs = []
    zip_buf = io.BytesIO()

    def safe(s: str) -> str:
        return str(s).replace(" ", "_").replace("/", "_")

    base_dir = "ep05_precision_results_OUTLIER_UPDATED"

    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
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

        for analyte in cfg.analytes:
            for level in cfg.levels:
                for scope_name, scope_df, expected_n_scope in scope_tables:
                    sub = scope_df[scope_df["Level"] == level].copy()
                    if sub.empty:
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
            f"{base_dir}/ALL_analytes_precision_summary.csv",
            summary.to_csv(index=False).encode("utf-8"),
        )

        outlier_log_all = pd.concat(all_outlier_logs, ignore_index=True) if all_outlier_logs else pd.DataFrame(
            columns=["analyte", "Level", "Device", "batch_id", "Blood Sample ID", "Day", "Replicate", "removed_order",
                     "outlier_method", "value_removed", "direction", "outlier_metric", "outlier_threshold",
                     "details", "gcrit_mode", "gcrit_alpha", "gcrit_tail"]
        )
        zf.writestr(
            f"{base_dir}/ALL_outlier_log.csv",
            outlier_log_all.to_csv(index=False).encode("utf-8"),
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
4) Choose pooled/per-device handling and outlier settings  
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
        "Gcrit Grubbs-like: remove largest |value-mean|/SD if >= Gcrit",
        "Robust MAD modified-z: remove largest robust z if >= threshold",
        "95% robust interval: remove most extreme outside median ± z*MAD_SD",
    ],
)
c1, c2, c3, c4 = st.columns(4)
with c1:
    max_remove_per_group = st.selectbox("Max outliers to remove per analyte/level/scope", [0, 1, 2], index=1)
with c2:
    gcrit_mode = st.selectbox("Gcrit mode", ["Manual Gcrit value", "Automatic from n, alpha, tail"])
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
**Outlier options match the interference app:** raw data are retained in the output, and selected outliers are logged.  
**Gcrit automatic default:** two-sided, alpha = 0.01.  
**Robust alternatives:** MAD modified-z and robust interval are better when the data are skewed or not normally distributed.
""")

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
    )

    with st.spinner("Running analysis..."):
        zip_bytes = run_pipeline_to_zip(df, cfg)

    st.success("Done. Download your results ZIP below.")
    st.download_button(
        label="Download results ZIP",
        data=zip_bytes,
        file_name="ep05_precision_results_updated_outliers.zip",
        mime="application/zip",
    )
