"""
================================================================================
XGBOOST + SHAP + SPATIAL VALIDATION PIPELINE
Urban Thermal Environment Analysis (Bamako, 2014–2024)
================================================================================
Author  : Adama KONATE
Version : 3.2 (LCZ mapping + diagnostic safeguards)
--------------------------------------------------------------------------------
PURPOSE
    Integrated pipeline that computes, for each year (2014, 2018, 2024) and each
    LCZ class:
        • RTI  : Relative Thermal Inequity (mean-of-class-means baseline)
        • Pearson correlations between LST and each driver
        • Random 5-fold CV          (reference, reproducible)
        • Spatial block CV          (KMeans-based, spatially disjoint)
        • XGBoost feature importance (corrected indexing)
        • SHAP values (global, per year)
        • SHAP values (per LCZ class)
        • Wilcoxon test: Random vs Spatial R² distributions
        • Delta analysis (drivers of decadal LST change)

--------------------------------------------------------------------------------
FIXES v3.2
--------------------------------------------------------------------------------
1. LCZ code mapping integrated (MAPPING_2014, MAPPING_2018_2024) applied
   immediately after loading, before any analysis.
2. formal_classes aligned with standard mappings: LCZ56 instead of LCZ5+6.
3. Diagnostic safeguards:
     - Input file existence check with explicit error message.
     - stdout line-buffering forced for PyCharm compatibility.
     - Global try/except with traceback around run_pipeline().
================================================================================
"""

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

# --- Force stdout to be line-buffered (PyCharm compatibility) -----------------
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, wilcoxon
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

import shap

warnings.filterwarnings("ignore", category=FutureWarning)


# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
CONFIG = {
    # --- Paths ---------------------------------------------------------------
    "input_path": r"C:\Users\ADAMA KONATE\Downloads\Donnes\Rands.xls",
    "output_dir": r"C:\Users\ADAMA KONATE\Downloads\Donnes\FINAL_PUBLICATION_SET_SCS",

    # --- Typography & export -------------------------------------------------
    "font_family": "serif",
    "font_serif": "Times New Roman",
    "font_size": 11,
    "dpi": 600,
    "pdf_fonttype": 42,

    # --- Analysis scope ------------------------------------------------------
    "years": ["2014", "2018", "2024"],
    "drivers": ["NDVI", "MNDWI", "Albedo", "Slope", "Dist_Water",
                "NDBI", "PD", "Ta", "NTL", "Wind"],

    # --- Formal residential baseline (mean-of-class-means, Table 5 logic) ----
    # Aligned with standard mappings: 'LCZ56' is the hybrid label produced
    # by both MAPPING_2014 (code 6) and MAPPING_2018_2024 (code 7).
    "formal_classes": ["LCZ5", "LCZ6", "LCZ56"],

    # --- XGBoost hyperparameters (Section 3.4) ------------------------------
    "xgb_params": {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.05,
        "random_state": 42,
        "n_jobs": -1,
    },

    # --- Validation ----------------------------------------------------------
    "random_cv_folds": 5,
    "spatial_blocks": 5,
    "min_samples_per_class": 30,
    "min_samples_per_spatial_block": 30,

    # --- Anomalous years (2018 is NOT flagged anymore) ----------------------
    "anomalous_years": [],

    # --- Coordinate columns --------------------------------------------------
    "coord_cols": ("POINT_X", "POINT_Y"),
}

plt.rcParams.update({
    "font.family": CONFIG["font_family"],
    "font.serif": [CONFIG["font_serif"]],
    "font.size": CONFIG["font_size"],
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": CONFIG["dpi"],
    "savefig.dpi": CONFIG["dpi"],
    "savefig.bbox": "tight",
    "pdf.fonttype": CONFIG["pdf_fonttype"],
    "ps.fonttype": CONFIG["pdf_fonttype"],
})
sns.set_style("ticks")

OUTPUT_DIR = Path(CONFIG["output_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 0bis. LCZ CODE MAPPING (aligns with the eleven previous scripts)
# ==============================================================================
# Mapping for 2014
MAPPING_2014 = {
    1: 'LCZ10', 3: 'LCZ3', 4: 'LCZ37', 5: 'LCZ5', 6: 'LCZ56',
    7: 'LCZ6', 8: 'LCZ7', 9: 'LCZ9', 10: 'LCZA', 11: 'LCZB',
    12: 'LCZZ', 13: 'LCZD', 14: 'LCZE', 15: 'LCZF', 16: 'LCZG'
}

# Mapping for 2018 and 2024
MAPPING_2018_2024 = {
    1: 'LCZ10', 4: 'LCZ3', 5: 'LCZ37', 6: 'LCZ5', 7: 'LCZ56',
    8: 'LCZ6', 9: 'LCZ7', 10: 'LCZ9', 11: 'LCZA', 12: 'LCZB',
    13: 'LCZZ', 14: 'LCZD', 15: 'LCZE', 16: 'LCZF', 17: 'LCZG'
}


def map_lcz_column(series, mapping):
    """
    Maps an LCZ column: preserves existing 'LCZ*' labels, otherwise
    converts numeric codes to LCZ labels using the supplied mapping.
    """
    def _map(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val).strip().upper()
        # Preserve values that are already LCZ labels
        if val_str.startswith('LCZ'):
            return val_str
        # Convert numeric codes
        try:
            code = int(float(val_str))
            return mapping.get(code, np.nan)
        except (ValueError, TypeError):
            return np.nan

    return series.apply(_map)


def apply_lcz_mapping(df, year_cols=('LCZ_2014', 'LCZ_2018', 'LCZ_2024')):
    """
    Applies the correct LCZ code-to-label mapping on the given DataFrame.
    Returns the DataFrame with mapped LCZ columns (in place).
    """
    for col in year_cols:
        if col in df.columns:
            yr = col.split('_')[-1]
            mapping = MAPPING_2014 if yr == '2014' else MAPPING_2018_2024
            # Clean strings first, then map
            df[col] = df[col].astype(str).str.replace('.0', '', regex=False).str.upper()
            df[col] = map_lcz_column(df[col], mapping)
    return df


# ==============================================================================
# 1. HELPER FUNCTIONS
# ==============================================================================
def _spatial_blocks(coords: np.ndarray, n_blocks: int, random_state: int = 42) -> np.ndarray:
    """
    Assign each pixel to a spatial block using KMeans.
    Falls back to a regular grid if any KMeans cluster is too small.
    """
    kmeans = KMeans(n_clusters=n_blocks, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(coords)

    counts = pd.Series(labels).value_counts()
    if counts.min() >= CONFIG["min_samples_per_spatial_block"]:
        return labels

    # --- Grid fallback -------------------------------------------------------
    x_edges = np.linspace(coords[:, 0].min(), coords[:, 0].max(), n_blocks + 1)
    y_edges = np.linspace(coords[:, 1].min(), coords[:, 1].max(), n_blocks + 1)
    x_bin = np.clip(np.digitize(coords[:, 0], x_edges[1:-1]), 0, n_blocks - 1)
    y_bin = np.clip(np.digitize(coords[:, 1], y_edges[1:-1]), 0, n_blocks - 1)
    return x_bin * n_blocks + y_bin


def _random_cv_per_fold(X: np.ndarray, y: np.ndarray):
    """
    Run 5-fold random CV and return:
        - aggregated predictions (for scatter plots)
        - per-fold R², RMSE, MAE (for statistical tests)
    """
    kf = KFold(n_splits=CONFIG["random_cv_folds"], shuffle=True, random_state=42)
    preds = np.full_like(y, fill_value=np.nan, dtype=float)
    r2_folds, rmse_folds, mae_folds = [], [], []

    for train_idx, test_idx in kf.split(X):
        model = XGBRegressor(**CONFIG["xgb_params"])
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        preds[test_idx] = y_pred
        r2_folds.append(r2_score(y[test_idx], y_pred))
        rmse_folds.append(np.sqrt(mean_squared_error(y[test_idx], y_pred)))
        mae_folds.append(mean_absolute_error(y[test_idx], y_pred))

    return preds, r2_folds, rmse_folds, mae_folds


def _spatial_cv_per_block(X: np.ndarray, y: np.ndarray, coords: np.ndarray):
    """
    Run spatial block CV and return:
        - aggregated predictions
        - per-block R², RMSE, MAE
        - per-block test sample sizes
    """
    blocks = _spatial_blocks(coords, CONFIG["spatial_blocks"])
    preds = np.full_like(y, fill_value=np.nan, dtype=float)
    r2_blocks, rmse_blocks, mae_blocks, n_blocks = [], [], [], []

    for block_id in np.unique(blocks):
        test_mask = blocks == block_id
        train_mask = ~test_mask
        if test_mask.sum() < CONFIG["min_samples_per_spatial_block"]:
            continue
        model = XGBRegressor(**CONFIG["xgb_params"])
        model.fit(X[train_mask], y[train_mask])
        y_pred = model.predict(X[test_mask])
        preds[test_mask] = y_pred
        r2_blocks.append(r2_score(y[test_mask], y_pred))
        rmse_blocks.append(np.sqrt(mean_squared_error(y[test_mask], y_pred)))
        mae_blocks.append(mean_absolute_error(y[test_mask], y_pred))
        n_blocks.append(int(test_mask.sum()))

    return preds, r2_blocks, rmse_blocks, mae_blocks, n_blocks


def _compute_shap(model: XGBRegressor, X: np.ndarray, feature_names: list) -> pd.DataFrame:
    """Compute SHAP values robustly (TreeExplainer fallback)."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
    except Exception:
        explainer = shap.Explainer(model.predict, X)
        shap_vals = explainer(X).values
    return pd.DataFrame(shap_vals, columns=feature_names)


# ==============================================================================
# 2. MAIN PIPELINE
# ==============================================================================
def run_pipeline() -> None:
    print("=" * 78)
    print(" XGBOOST + SHAP + SPATIAL VALIDATION PIPELINE — START")
    print("=" * 78)

    # --- [DIAGNOSTIC] Environment info --------------------------------------
    print(f"[DEBUG] Python executable : {sys.executable}")
    print(f"[DEBUG] Working directory : {os.getcwd()}")
    print(f"[DEBUG] Output directory  : {OUTPUT_DIR}")

    # --- [DIAGNOSTIC] Input file existence ----------------------------------
    input_path = Path(CONFIG["input_path"])
    print(f"[DEBUG] Input path        : {input_path}")
    print(f"[DEBUG] Input exists      : {input_path.exists()}")
    if not input_path.exists():
        raise FileNotFoundError(
            f"\n❌ INPUT FILE NOT FOUND: {input_path}\n"
            f"   Please verify the exact file name and location.\n"
        )
    print(f"[DEBUG] Input size        : {input_path.stat().st_size / 1024:.1f} KB")

    # --- Load ----------------------------------------------------------------
    print("[LOAD] Reading Excel file...")
    df = pd.read_excel(input_path, engine="xlrd")
    df.columns = df.columns.str.strip()
    print(f"[LOAD] Loaded shape       : {df.shape}")
    print(f"[LOAD] First 15 columns   : {list(df.columns)[:15]}")

    # --- APPLY LCZ MAPPING BEFORE ANY ANALYSIS ------------------------------
    print("[LCZ]  Applying LCZ code-to-label mapping...")
    df = apply_lcz_mapping(df)

    # Redundant cleaning loop (idempotent after mapping) - kept as in original
    for yr in CONFIG["years"]:
        col = f"LCZ_{yr}"
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                       .str.replace(".0", "", regex=False)
                       .str.upper()
            )
            df.loc[~df[col].str.startswith("LCZ", na=False), col] = np.nan

    for yr in CONFIG["years"]:
        col = f"LCZ_{yr}"
        if col in df.columns:
            uniq = sorted(df[col].dropna().unique())
            print(f"[LCZ]  {col} unique classes ({len(uniq)}): {uniq[:12]}"
                  f"{'...' if len(uniq) > 12 else ''}")

    # --- Delta variables -----------------------------------------------------
    delta_vars = []
    if {"LST_2014", "LST_2024"}.issubset(df.columns):
        df["delta_LST"] = df["LST_2024"] - df["LST_2014"]
        for v in CONFIG["drivers"]:
            v14, v24 = f"{v}_2014", f"{v}_2024"
            if v14 in df.columns and v24 in df.columns:
                df[f"delta_{v}"] = df[v24] - df[v14]
                delta_vars.append(f"delta_{v}")
    print(f"[DELTA] Delta variables computed: {len(delta_vars)}")

    # --- Accumulators --------------------------------------------------------
    rti_records = []
    corr_records = []
    random_records = []
    spatial_records = []
    importance_records = []
    preds_scatter = []
    shap_global = {}          # {year: shap_df}
    shap_per_class = {}       # {(year, lcz): mean_abs_series}
    wilcoxon_records = []

    # =========================================================================
    # MAIN LOOP
    # =========================================================================
    for yr in CONFIG["years"]:
        lst_col, lcz_col = f"LST_{yr}", f"LCZ_{yr}"
        if lst_col not in df.columns or lcz_col not in df.columns:
            print(f"[SKIP] Year {yr}: missing {lst_col} or {lcz_col}.")
            continue

        # RTI baseline: MEAN OF CLASS MEANS of LCZ 5, LCZ 6 (and LCZ56 hybrid)
        formal_df = df[df[lcz_col].isin(CONFIG["formal_classes"])]
        class_means = formal_df.groupby(lcz_col)[lst_col].mean()
        baseline = class_means.mean() if not class_means.empty else np.nanmean(df[lst_col])
        print(f"\n[{yr}] RTI baseline (mean of class means LCZ5/6/56) = {baseline:.3f} °C")
        print(f"[{yr}] Formal class means:\n{class_means.round(3).to_string()}")

        # Global SHAP accumulator for this year
        year_X_list, year_y_list = [], []

        for lcz in sorted(df[lcz_col].dropna().unique()):
            sub = df[df[lcz_col] == lcz].copy()
            if len(sub) < CONFIG["min_samples_per_class"]:
                continue

            # --- RTI ---------------------------------------------------------
            rti_records.append({
                "Year": yr, "LCZ": lcz,
                "RTI": np.nanmean(sub[lst_col]) - baseline,
                "N": len(sub),
            })

            # --- Pearson correlations ----------------------------------------
            row_c = {"LCZ": lcz, "Year": yr}
            for v in CONFIG["drivers"]:
                v_col = f"{v}_{yr}"
                if v_col in sub.columns:
                    valid = sub[[v_col, lst_col]].dropna()
                    if len(valid) > 10:
                        row_c[v] = pearsonr(valid[v_col], valid[lst_col])[0]
            corr_records.append(row_c)

            # --- Build design matrix (with coordinates) ----------------------
            X_cols = [f"{v}_{yr}" for v in CONFIG["drivers"] if f"{v}_{yr}" in sub.columns]
            clean_names = [c.replace(f"_{yr}", "") for c in X_cols]

            cx, cy = CONFIG["coord_cols"]
            needed = X_cols + [lst_col, cx, cy]
            if not all(c in sub.columns for c in needed):
                print(f"  [SKIP] {yr}/{lcz}: missing coordinates or LST.")
                continue

            data_ml = sub[needed].dropna()
            if len(data_ml) < CONFIG["min_samples_per_class"]:
                continue

            X = data_ml[X_cols].to_numpy()
            y = data_ml[lst_col].to_numpy()
            coords = data_ml[[cx, cy]].to_numpy()

            # --- Random CV ---------------------------------------------------
            preds_rand, r2_rand, rmse_rand, mae_rand = _random_cv_per_fold(X, y)
            random_records.append({
                "Year": yr, "LCZ": lcz, "N": len(y),
                "R2_mean": float(np.mean(r2_rand)),
                "R2_std":  float(np.std(r2_rand)),
                "RMSE_mean": float(np.mean(rmse_rand)),
                "MAE_mean":  float(np.mean(mae_rand)),
            })

            # --- Spatial block CV --------------------------------------------
            preds_spat, r2_spat, rmse_spat, mae_spat, n_blocks = _spatial_cv_per_block(X, y, coords)
            if len(r2_spat) >= 2:
                spatial_records.append({
                    "Year": yr, "LCZ": lcz, "N": len(y),
                    "R2_mean": float(np.mean(r2_spat)),
                    "R2_std":  float(np.std(r2_spat)),
                    "RMSE_mean": float(np.mean(rmse_spat)),
                    "MAE_mean":  float(np.mean(mae_spat)),
                    "N_blocks": len(r2_spat),
                })

                # --- Wilcoxon test -------------------------------------------
                if len(r2_rand) >= 2 and len(r2_spat) >= 2:
                    try:
                        stat, p = wilcoxon(r2_rand, r2_spat)
                        wilcoxon_records.append({
                            "Year": yr, "LCZ": lcz,
                            "R2_random_mean": float(np.mean(r2_rand)),
                            "R2_spatial_mean": float(np.mean(r2_spat)),
                            "Delta_R2": float(np.mean(r2_rand) - np.mean(r2_spat)),
                            "Wilcoxon_stat": float(stat),
                            "p_value": float(p),
                        })
                    except Exception:
                        pass

            # --- XGBoost feature importance (FIXED indexing) -----------------
            final_model = XGBRegressor(**CONFIG["xgb_params"])
            final_model.fit(X, y)
            for i, gain in enumerate(final_model.feature_importances_):
                importance_records.append({
                    "Year": yr, "LCZ": lcz,
                    "Driver": clean_names[i],
                    "Gain": float(gain),
                })

            # --- Store for SHAP (global per year) ----------------------------
            year_X_list.append(pd.DataFrame(X, columns=clean_names))
            year_y_list.append(pd.Series(y))

            # --- Scatter data ------------------------------------------------
            for obs, pr in zip(y, preds_rand):
                preds_scatter.append({"Year": yr, "LCZ": lcz, "Obs": obs, "Pred": pr})

            print(f"  [{yr}/{lcz}] N={len(y):5d}  "
                  f"R²_rand={np.mean(r2_rand):.3f}  "
                  f"R²_spat={(np.mean(r2_spat) if r2_spat else float('nan')):.3f}")

        # --- Global SHAP for the year ----------------------------------------
        if year_X_list:
            X_year = pd.concat(year_X_list, ignore_index=True)
            y_year = pd.concat(year_y_list, ignore_index=True).to_numpy()

            model_global = XGBRegressor(**CONFIG["xgb_params"])
            model_global.fit(X_year.to_numpy(), y_year)

            shap_df = _compute_shap(model_global, X_year.to_numpy(), list(X_year.columns))
            shap_global[yr] = shap_df
            print(f"[{yr}] Global SHAP computed on {len(X_year)} samples.")

            # --- SHAP per LCZ class (reproducibility of §4.5.2) --------------
            for lcz in sorted(df[lcz_col].dropna().unique()):
                sub = df[df[lcz_col] == lcz].copy()
                if len(sub) < CONFIG["min_samples_per_class"]:
                    continue
                X_cols = [f"{v}_{yr}" for v in CONFIG["drivers"] if f"{v}_{yr}" in sub.columns]
                needed = X_cols + [lst_col, cx, cy]
                if not all(c in sub.columns for c in needed):
                    continue
                data_ml = sub[needed].dropna()
                if len(data_ml) < CONFIG["min_samples_per_class"]:
                    continue

                X_c = data_ml[X_cols].to_numpy()
                y_c = data_ml[lst_col].to_numpy()
                model_c = XGBRegressor(**CONFIG["xgb_params"])
                model_c.fit(X_c, y_c)
                shap_c = _compute_shap(model_c, X_c, X_cols)

                shap_per_class[(yr, lcz)] = (
                    shap_c.abs().mean().rename(lambda x: x.replace(f"_{yr}", ""))
                )

    # --- Delta analysis (drivers of decadal change) --------------------------
    if "delta_LST" in df.columns and delta_vars:
        print("\n[DELTA] Computing drivers of decadal LST change...")
        for lcz in sorted(df["LCZ_2024"].dropna().unique()):
            sub_d = df[df["LCZ_2024"] == lcz].copy()
            data_d = sub_d[delta_vars + ["delta_LST"]].dropna()
            if len(data_d) >= CONFIG["min_samples_per_class"]:
                model_d = XGBRegressor(**CONFIG["xgb_params"])
                model_d.fit(data_d[delta_vars].to_numpy(), data_d["delta_LST"].to_numpy())
                for i, gain in enumerate(model_d.feature_importances_):
                    importance_records.append({
                        "Year": "delta", "LCZ": lcz,
                        "Driver": delta_vars[i].replace("delta_", ""),
                        "Gain": float(gain),
                    })

    # =========================================================================
    # EXPORT
    # =========================================================================
    print("\n[EXPORT] Writing Excel workbook, report, and figures...")

    rti_df = pd.DataFrame(rti_records)
    corr_df = pd.DataFrame(corr_records)
    random_df = pd.DataFrame(random_records)
    spatial_df = pd.DataFrame(spatial_records)
    importance_df = pd.DataFrame(importance_records)
    wilcoxon_df = pd.DataFrame(wilcoxon_records)
    scatter_df = pd.DataFrame(preds_scatter)

    # --- SHAP long-format tables --------------------------------------------
    shap_global_long = []
    for yr, sdf in shap_global.items():
        for col in sdf.columns:
            shap_global_long.append({
                "Year": yr, "Driver": col,
                "Mean_SHAP": float(sdf[col].mean()),
                "Mean_Abs_SHAP": float(sdf[col].abs().mean()),
                "Std_SHAP": float(sdf[col].std()),
            })
    shap_global_long_df = pd.DataFrame(shap_global_long)

    shap_per_class_records = []
    for (yr, lcz), series in shap_per_class.items():
        for driver, val in series.items():
            shap_per_class_records.append({
                "Year": yr, "LCZ": lcz,
                "Driver": driver,
                "Mean_Abs_SHAP": float(val),
            })
    shap_per_class_df = pd.DataFrame(shap_per_class_records)

    # --- Excel ---------------------------------------------------------------
    excel_path = OUTPUT_DIR / "XGBoost_SHAP_Results.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        rti_df.to_excel(writer, sheet_name="RTI", index=False)
        corr_df.to_excel(writer, sheet_name="Pearson", index=False)
        random_df.to_excel(writer, sheet_name="RandomCV", index=False)
        spatial_df.to_excel(writer, sheet_name="SpatialCV", index=False)
        wilcoxon_df.to_excel(writer, sheet_name="Wilcoxon", index=False)
        importance_df.to_excel(writer, sheet_name="FeatureImportance", index=False)
        shap_global_long_df.to_excel(writer, sheet_name="SHAP_global", index=False)
        shap_per_class_df.to_excel(writer, sheet_name="SHAP_per_LCZ", index=False)

    # =========================================================================
    # FIGURES
    # =========================================================================
    # --- Fig 1: RTI evolution ------------------------------------------------
    if not rti_df.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=rti_df, x="LCZ", y="RTI", hue="Year",
                    palette="magma", edgecolor="black")
        plt.axhline(0, color="black", lw=1.2, ls="--")
        plt.title("Evolution of Relative Thermal Inequity (RTI)")
        plt.ylabel("RTI (°C)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "Fig1_RTI_Evolution.png")
        plt.savefig(OUTPUT_DIR / "Fig1_RTI_Evolution.pdf")
        plt.close()

    # --- Fig 2: Random vs Spatial R² ----------------------------------------
    if not random_df.empty and not spatial_df.empty:
        rand_long = random_df.assign(Validation="Random")
        spat_long = spatial_df.assign(Validation="Spatial")
        comp = pd.concat([rand_long, spat_long], ignore_index=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.boxplot(data=comp, x="Year", y="R2_mean", hue="Validation",
                    palette="Set2", ax=axes[0])
        axes[0].axhline(0, color="black", ls="--", lw=1)
        axes[0].set_title("R² — Random vs Spatial Block CV", fontweight="bold")
        axes[0].set_ylabel("R²")

        sns.boxplot(data=comp, x="Year", y="RMSE_mean", hue="Validation",
                    palette="Set2", ax=axes[1])
        axes[1].set_title("RMSE — Random vs Spatial Block CV", fontweight="bold")
        axes[1].set_ylabel("RMSE (°C)")

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "Fig2_Random_vs_Spatial.png")
        plt.savefig(OUTPUT_DIR / "Fig2_Random_vs_Spatial.pdf")
        plt.close()

    # --- Fig 3: SHAP temporal comparison ------------------------------------
    if not shap_global_long_df.empty:
        plt.figure(figsize=(14, 7))
        sns.barplot(data=shap_global_long_df, x="Driver", y="Mean_Abs_SHAP",
                    hue="Year", palette="magma", edgecolor="black")
        plt.title("Temporal Evolution of SHAP Importance (2014–2024)")
        plt.ylabel("Mean |SHAP value|")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "Fig3_SHAP_Temporal.png")
        plt.savefig(OUTPUT_DIR / "Fig3_SHAP_Temporal.pdf")
        plt.close()

    # --- Fig 4: SHAP heatmap per LCZ ----------------------------------------
    if not shap_per_class_df.empty:
        for yr in CONFIG["years"]:
            sub = shap_per_class_df[shap_per_class_df["Year"] == yr]
            if sub.empty:
                continue
            pivot = sub.pivot(index="LCZ", columns="Driver",
                              values="Mean_Abs_SHAP").fillna(0)
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".2f",
                        linewidths=0.5,
                        cbar_kws={"label": "Mean |SHAP|"})
            plt.title(f"SHAP Importance per LCZ — {yr}")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"Fig4_SHAP_per_LCZ_{yr}.png")
            plt.savefig(OUTPUT_DIR / f"Fig4_SHAP_per_LCZ_{yr}.pdf")
            plt.close()

    # --- Fig 5: XGBoost feature importance heatmap --------------------------
    if not importance_df.empty:
        for yr in CONFIG["years"]:
            sub = importance_df[importance_df["Year"] == yr]
            if sub.empty:
                continue
            pivot = sub.pivot(index="LCZ", columns="Driver", values="Gain").fillna(0)
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".2f",
                        linewidths=0.5,
                        cbar_kws={"label": "Gain"})
            plt.title(f"XGBoost Feature Importance — {yr}")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"Fig5_XGB_Importance_{yr}.png")
            plt.savefig(OUTPUT_DIR / f"Fig5_XGB_Importance_{yr}.pdf")
            plt.close()

    # --- Fig 6: Observed vs Predicted (random CV) ---------------------------
    if not scatter_df.empty:
        plt.figure(figsize=(7, 7))
        sns.scatterplot(data=scatter_df, x="Obs", y="Pred", hue="Year",
                        alpha=0.35, edgecolor=None)
        lim = [scatter_df["Obs"].min(), scatter_df["Obs"].max()]
        plt.plot(lim, lim, "k--", lw=1.5)
        plt.title("XGBoost Validation (Random 5-fold CV)")
        plt.xlabel("Observed LST (°C)")
        plt.ylabel("Predicted LST (°C)")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "Fig6_Validation_Scatter.png")
        plt.savefig(OUTPUT_DIR / "Fig6_Validation_Scatter.pdf")
        plt.close()

    # =========================================================================
    # NARRATIVE REPORT
    # =========================================================================
    report_path = OUTPUT_DIR / "Scientific_Report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("XGBOOST + SHAP + SPATIAL VALIDATION REPORT\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n")
        f.write("=" * 80 + "\n\n")

        # --- 1. Random CV ----------------------------------------------------
        f.write("1. RANDOM 5-FOLD CROSS-VALIDATION\n")
        f.write("-" * 80 + "\n")
        f.write(random_df.round(3).to_string(index=False) if not random_df.empty else "No data")
        f.write("\n\n")

        # --- 2. Spatial block CV --------------------------------------------
        f.write("2. SPATIAL BLOCK VALIDATION (KMeans, 5 blocks)\n")
        f.write("-" * 80 + "\n")
        f.write(spatial_df.round(3).to_string(index=False) if not spatial_df.empty else "No data")
        f.write("\n\n")

        # --- 3. Wilcoxon test -----------------------------------------------
        f.write("3. WILCOXON TEST — RANDOM vs SPATIAL R² DISTRIBUTIONS\n")
        f.write("-" * 80 + "\n")
        if not wilcoxon_df.empty:
            f.write(wilcoxon_df.round(4).to_string(index=False))
            f.write("\n\n")
            f.write("INTERPRETATION:\n")
            f.write("  • p < 0.05 indicates that the random CV systematically\n")
            f.write("    overestimates performance relative to spatial CV.\n")
            f.write("  • Negative ΔR² indicates spatial autocorrelation-driven\n")
            f.write("    inflation of the random-CV estimate.\n\n")
        else:
            f.write("Insufficient folds for the test.\n\n")

        # --- 4. Per-year comparison -----------------------------------------
        f.write("4. PER-YEAR SUMMARY: RANDOM vs SPATIAL\n")
        f.write("-" * 80 + "\n")
        if not random_df.empty and not spatial_df.empty:
            r_mean = random_df.groupby("Year")[["R2_mean", "RMSE_mean"]].mean()
            s_mean = spatial_df.groupby("Year")[["R2_mean", "RMSE_mean"]].mean()
            comp = pd.DataFrame({
                "R2_random": r_mean["R2_mean"],
                "R2_spatial": s_mean["R2_mean"],
                "Delta_R2": r_mean["R2_mean"] - s_mean["R2_mean"],
                "RMSE_random": r_mean["RMSE_mean"],
                "RMSE_spatial": s_mean["RMSE_mean"],
            })
            f.write(comp.round(3).to_string())
            f.write("\n\n")
            f.write("INTERPRETATION:\n")
            f.write("  • ΔR² < 0.05  → no substantial spatial inflation.\n")
            f.write("  • ΔR² > 0.30  → severe spatial inflation; the model does\n")
            f.write("                  not generalise to spatially disjoint areas.\n")
            f.write("  • Negative R²_spatial → the model performs worse than a\n")
            f.write("                  mean-only benchmark on unseen regions.\n\n")

        # --- 5. SHAP global --------------------------------------------------
        f.write("5. SHAP GLOBAL IMPORTANCE (per year)\n")
        f.write("-" * 80 + "\n")
        if not shap_global_long_df.empty:
            for yr in CONFIG["years"]:
                sub = shap_global_long_df[shap_global_long_df["Year"] == yr]
                if sub.empty:
                    continue
                sub = sub.sort_values("Mean_Abs_SHAP", ascending=False)
                f.write(f"\nYEAR {yr}:\n")
                for _, row in sub.iterrows():
                    f.write(f"  {row['Driver']:12s}  "
                            f"Mean SHAP = {row['Mean_SHAP']:+.3f}  "
                            f"|Mean SHAP| = {row['Mean_Abs_SHAP']:.3f}\n")
        f.write("\n")

        # --- 6. SHAP per LCZ -------------------------------------------------
        f.write("6. SHAP IMPORTANCE PER LCZ (mean |SHAP|)\n")
        f.write("-" * 80 + "\n")
        if not shap_per_class_df.empty:
            pivot = shap_per_class_df.pivot_table(
                index=["Year", "LCZ"], columns="Driver",
                values="Mean_Abs_SHAP"
            ).round(3)
            f.write(pivot.to_string())
        f.write("\n\n")

        # --- 7. Feature importance ------------------------------------------
        f.write("7. XGBOOST FEATURE IMPORTANCE (Gain)\n")
        f.write("-" * 80 + "\n")
        if not importance_df.empty:
            pivot = importance_df.pivot_table(
                index=["Year", "LCZ"], columns="Driver",
                values="Gain"
            ).round(3)
            f.write(pivot.to_string())
        f.write("\n\n")

        # --- 8. Methodological note -----------------------------------------
        f.write("8. METHODOLOGICAL NOTE\n")
        f.write("-" * 80 + "\n")
        f.write("• LCZ codes mapped to labels via MAPPING_2014 and MAPPING_2018_2024.\n")
        f.write("• RTI baseline = mean of class means of LCZ 5, LCZ 6, and LCZ56 "
                "(Table 5 logic).\n")
        f.write("• Random CV = KFold(5, shuffle=True, random_state=42).\n")
        f.write("• Spatial CV = KMeans-based 5-block partition of "
                "POINT_X / POINT_Y.\n")
        f.write("• XGBoost hyperparameters: n_estimators=100, max_depth=5, "
                "learning_rate=0.05.\n")
        f.write("• SHAP values computed via TreeExplainer on the final "
                "XGBoost model.\n")
        f.write("• Feature importance indexing uses the filtered X_cols "
                "list, not the full driver list.\n")
        f.write("• Anomalous years flagged: none (2018 is not anomalous).\n")

    print(f"\n[EXPORT] Report written to : {report_path}")
    print(f"[EXPORT] Excel workbook    : {excel_path}")
    print(f"[EXPORT] Figures directory : {OUTPUT_DIR}")
    print("\n" + "=" * 78)
    print(" PIPELINE COMPLETED")
    print("=" * 78)


# ==============================================================================
# 3. ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        import traceback
        print("\n" + "=" * 78)
        print(" ❌ FATAL ERROR DURING EXECUTION")
        print("=" * 78)
        traceback.print_exc()
        sys.exit(1)
