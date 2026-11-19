"""
================================================================================
MULTISCALE GEOGRAPHICALLY WEIGHTED REGRESSION (MGWR) PIPELINE — STABLE v5.4
Urban Thermal Environment Analysis - Bamako (2014, 2018, 2024)
================================================================================
FIXES v5.4:
    • SE/t/p extracted only if they exist (MGWRResults lacks them with multi=True)
    • Manual MGWR predictions and local R²
    • Automatic fallback GWR if MGWR fails
================================================================================
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import scipy.linalg as la
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW

# ---------------------------------------------------------------------------
# 0. CONFIGURATION
# ---------------------------------------------------------------------------
matplotlib.use('Agg')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
warnings.filterwarnings('ignore')

FILE_PATH        = r"C:\Users\ADAMA KONATE\Downloads\Donnes\Rands.xls"
SHAPEFILE_PATH   = r"C:\Users\ADAMA KONATE\Desktop\New folder (3)\Bamako_Project\Limite_Bamako.shp"
OUTPUT_FOLDER    = r"C:\Users\ADAMA KONATE\Downloads\Donnes\Publication_Final_MGWR"

CORE_FEATURES    = ['NDVI', 'MNDWI', 'Albedo', 'Slope', 'Dist_Water',
                    'NDBI', 'PD', 'NTL', 'Ta', 'Wind']
YEARS            = ['2014', '2018', '2024']

VIF_THRESHOLD    = 5.0
MAX_FEATURES     = 5
MIN_SAMPLES      = 100
MAX_SAMPLES_MGWR = 5000
BW_MIN           = 100
BW_MAX_RATIO     = 0.5
KERNEL           = 'gaussian'
RIDGE_EPS        = 1e-6
N_SPATIAL_BLOCKS = 5
RANDOM_STATE     = 42
N_JOBS           = 1

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ---------------------------------------------------------------------------
# 0bis. LCZ CODE MAPPING
# ---------------------------------------------------------------------------
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
    If a year-specific column is not present, the function silently skips it.
    """
    for col in year_cols:
        if col in df.columns:
            yr = col.split('_')[-1]
            mapping = MAPPING_2014 if yr == '2014' else MAPPING_2018_2024
            # Clean strings first, then map
            df[col] = df[col].astype(str).str.replace('.0', '', regex=False).str.upper()
            df[col] = map_lcz_column(df[col], mapping)
    return df


# ---------------------------------------------------------------------------
# 1. HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def backward_vif_selection(df, features, threshold=VIF_THRESHOLD):
    current = list(features)
    while True:
        X = df[current].dropna()
        if X.empty or len(current) == 0:
            break
        X_int = add_constant(X)
        vifs = []
        for i, feat in enumerate(current):
            try:
                v = variance_inflation_factor(X_int.values, i + 1)
            except Exception:
                v = float('inf')
            vifs.append((feat, v))
        max_feat, max_vif = max(vifs, key=lambda x: x[1])
        if max_vif > threshold:
            logging.warning(f"VIF elimination: '{max_feat}' (VIF = {max_vif:.2f})")
            current.remove(max_feat)
        else:
            break
    return current


def compute_condition_numbers(coords, X, bw, kernel=KERNEL):
    logging.info("Computing local condition numbers via SVD...")
    n = coords.shape[0]
    cn = np.zeros(n)
    X_const = np.column_stack([np.ones(n), X])
    for i in range(n):
        dists = la.norm(coords - coords[i], axis=1)
        sorted_d = np.sort(dists)
        idx = min(int(bw) - 1, n - 1)
        max_d = sorted_d[idx] if sorted_d[idx] > 0 else 1e-6
        w = np.zeros(n)
        mask = dists <= max_d
        if kernel == 'gaussian':
            sigma = max_d / 2.0
            w[mask] = np.exp(-0.5 * (dists[mask] / sigma) ** 2)
        else:
            w[mask] = (1.0 - (dists[mask] / max_d) ** 2) ** 2
        xw = X_const * np.sqrt(w.reshape(-1, 1))
        try:
            _, s, _ = la.svd(xw, full_matrices=False)
            cn[i] = s[0] / s[-1] if s[-1] > 1e-12 else 999.0
        except la.LinAlgError:
            cn[i] = 999.0
    return cn


def compute_local_r2_manual(coords, y, y_pred, bw, kernel=KERNEL):
    n = len(y)
    y = y.flatten()
    y_pred = y_pred.flatten()
    residuals = y - y_pred
    local_r2 = np.zeros(n)

    for i in range(n):
        dists = la.norm(coords - coords[i], axis=1)
        sorted_d = np.sort(dists)
        idx = min(int(bw) - 1, n - 1)
        max_d = sorted_d[idx] if sorted_d[idx] > 0 else 1e-6

        w = np.zeros(n)
        mask = dists <= max_d
        if kernel == 'gaussian':
            sigma = max_d / 2.0
            w[mask] = np.exp(-0.5 * (dists[mask] / sigma) ** 2)
        else:
            w[mask] = (1.0 - (dists[mask] / max_d) ** 2) ** 2

        w_sum = np.sum(w)
        if w_sum == 0:
            local_r2[i] = 0.0
            continue

        y_wmean = np.sum(w * y) / w_sum
        sst = np.sum(w * (y - y_wmean) ** 2)
        ssr = np.sum(w * residuals ** 2)
        local_r2[i] = 1.0 - (ssr / sst) if sst > 0 else 0.0

    return local_r2


def spatial_block_cv(coords, n_blocks=N_SPATIAL_BLOCKS, random_state=RANDOM_STATE):
    km = KMeans(n_clusters=n_blocks, n_init=10, random_state=random_state)
    return km.fit_predict(coords)


def safe_gwr_fit(coords, y, X, bw, kernel=KERNEL):
    try:
        return GWR(coords, y, X, bw, kernel=kernel, fixed=False).fit()
    except Exception as e:
        new_bw = max(int(bw * 2), 300)
        logging.warning(f"GWR fit failed at bw={int(bw)} ({e}). Retrying at bw={new_bw}.")
        try:
            return GWR(coords, y, X, new_bw, kernel=kernel, fixed=False).fit()
        except Exception as e2:
            logging.error(f"GWR fit failed again: {e2}")
            return None


def random_cv_gwr(coords, y, X, bw, n_folds=5, random_state=RANDOM_STATE):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    r2_list = []
    for train_idx, test_idx in kf.split(X):
        model = safe_gwr_fit(coords[train_idx], y[train_idx], X[train_idx], bw)
        if model is None:
            continue
        try:
            preds = model.predict(coords[test_idx], X[test_idx])[0].flatten()
            r2_list.append(r2_score(y[test_idx].flatten(), preds))
        except Exception as e:
            logging.warning(f"CV fold failed: {e}")
    return r2_list


def spatial_cv_gwr(coords, y, X, bw, blocks):
    r2_list = []
    for block in np.unique(blocks):
        test = blocks == block
        train = ~test
        if test.sum() < 30 or train.sum() < 30:
            continue
        model = safe_gwr_fit(coords[train], y[train], X[train], bw)
        if model is None:
            continue
        try:
            preds = model.predict(coords[test], X[test])[0].flatten()
            r2_list.append(r2_score(y[test].flatten(), preds))
        except Exception as e:
            logging.warning(f"Block CV failed: {e}")
    return r2_list


def try_extract_attribute(model, attr_name, col_index):
    """Safely extract an attribute (SE, tvalues, pvalues) if it exists."""
    if not hasattr(model, attr_name):
        return np.nan
    arr = getattr(model, attr_name, None)
    if arr is None:
        return np.nan
    try:
        return arr[:, col_index].flatten()
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# 2. MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    logging.info("=" * 78)
    logging.info(" MGWR PIPELINE — STABLE v5.4")
    logging.info("=" * 78)

    df_raw = pd.read_excel(FILE_PATH, engine='xlrd')
    df_raw.columns = df_raw.columns.str.strip()

    # --- APPLY LCZ MAPPING BEFORE ANY GEOSPATIAL PROCESSING ---
    # Ensures all LCZ columns (LCZ_2014, LCZ_2018, LCZ_2024) are correctly
    # labeled before aggregation by LCZ, before export, and before mapping.
    df_raw = apply_lcz_mapping(df_raw)

    lcz_cols = [c for c in df_raw.columns if 'LCZ' in str(c).upper()]
    lcz_col = lcz_cols[0] if lcz_cols else None

    df_raw['POINT_X'] = pd.to_numeric(df_raw['POINT_X'], errors='coerce')
    df_raw['POINT_Y'] = pd.to_numeric(df_raw['POINT_Y'], errors='coerce')
    df_clean = (df_raw
                .drop_duplicates(subset=['POINT_X', 'POINT_Y'])
                .dropna(subset=['POINT_X', 'POINT_Y']))

    geometry = [Point(xy) for xy in zip(df_clean['POINT_X'], df_clean['POINT_Y'])]
    gdf = gpd.GeoDataFrame(df_clean, geometry=geometry, crs="EPSG:4326").to_crs("EPSG:32629")
    logging.info(f"Geometry validated: {len(gdf)} unique points.")

    report_path = os.path.join(OUTPUT_FOLDER, 'MGWR_SCS_Report.txt')
    summary_rows = []

    with open(report_path, 'w', encoding='utf-8') as report:
        report.write("=" * 80 + "\n")
        report.write(" MGWR ANALYSIS REPORT — STABLE v5.4\n")
        report.write("=" * 80 + "\n\n")

        for yr in YEARS:
            logging.info(f"\n--- Processing MGWR: {yr} ---")
            target = f'LST_{yr}'
            features = [f'{v}_{yr}' for v in CORE_FEATURES if f'{v}_{yr}' in gdf.columns]

            cols_needed = features + [target]
            if lcz_col:
                cols_needed.append(lcz_col)

            data = gdf.dropna(subset=cols_needed).copy()
            if len(data) < MIN_SAMPLES:
                logging.warning(f"Insufficient samples for {yr}. Skipping.")
                continue

            if len(data) > MAX_SAMPLES_MGWR:
                logging.info(f"Subsampling {MAX_SAMPLES_MGWR} of {len(data)} points.")
                data = data.sample(n=MAX_SAMPLES_MGWR, random_state=RANDOM_STATE).copy()

            # Step 1: VIF
            valid_features = backward_vif_selection(data, features)
            if len(valid_features) > MAX_FEATURES:
                X_temp = add_constant(data[valid_features].dropna())
                vif_pairs = []
                for i, feat in enumerate(valid_features):
                    try:
                        v = variance_inflation_factor(X_temp.values, i + 1)
                    except Exception:
                        v = float('inf')
                    vif_pairs.append((feat, v))
                vif_pairs.sort(key=lambda x: x[1])
                valid_features = [f for f, _ in vif_pairs[:MAX_FEATURES]]
            logging.info(f"Retained features ({yr}): {valid_features}")

            if len(valid_features) < 2:
                logging.warning(f"Too few features for {yr}. Skipping.")
                continue

            # Step 2: Standardization
            coords = np.array(list(zip(data.geometry.x, data.geometry.y)))
            X_raw = data[valid_features].values
            y_raw = data[target].values.reshape(-1, 1)

            scaler = StandardScaler()
            X_std = scaler.fit_transform(X_raw)
            X_std += np.random.normal(0, RIDGE_EPS, X_std.shape)

            # Step 3: GWR bandwidth
            bw_max = int(len(coords) * BW_MAX_RATIO)
            logging.info(f"Selecting GWR bandwidth (min={BW_MIN}, max={bw_max})...")
            sel_gwr = Sel_BW(coords, y_raw, X_std,
                             kernel=KERNEL, fixed=False, n_jobs=N_JOBS)
            try:
                bw_gwr = sel_gwr.search(criterion='AICc',
                                        bw_min=BW_MIN, bw_max=bw_max)
            except Exception as e:
                logging.warning(f"Bandwidth search failed: {e}. Using bw={BW_MIN*3}.")
                bw_gwr = BW_MIN * 3

            gwr_model = safe_gwr_fit(coords, y_raw, X_std, bw_gwr)
            if gwr_model is None:
                logging.error(f"GWR failed for {yr}. Skipping.")
                continue

            logging.info(f"GWR: bw={int(bw_gwr)}, adj. R²={gwr_model.adj_R2:.4f}, "
                         f"AICc={gwr_model.aicc:.2f}")

            # Step 4: MGWR
            mgwr_model = None
            bw_mgwr = None
            try:
                sel_mgwr = Sel_BW(coords, y_raw, X_std,
                                  kernel=KERNEL, fixed=False,
                                  multi=True, n_jobs=N_JOBS)
                bw_mgwr = sel_mgwr.search(criterion='AICc',
                                          bw_min=BW_MIN, bw_max=bw_max)
                mgwr_model = MGWR(coords, y_raw, X_std, sel_mgwr,
                                  kernel=KERNEL, fixed=False).fit()
                logging.info(f"MGWR: adj. R²={mgwr_model.adj_R2:.4f}, "
                             f"AICc={mgwr_model.aicc:.2f}")
            except Exception as e:
                logging.warning(f"MGWR failed for {yr} ({e}). Using GWR fallback.")
                mgwr_model = None

            # Step 5: Active model
            if mgwr_model is not None:
                active_model = mgwr_model
                model_label = 'MGWR'
                bw_used = bw_mgwr
            else:
                active_model = gwr_model
                model_label = 'GWR (MGWR fallback)'
                bw_used = bw_gwr

            # Step 6: Predictions + Local R² + Residuals
            if mgwr_model is not None:
                median_bw_val = (int(np.median(bw_mgwr))
                                 if np.ndim(bw_mgwr) > 0 else int(bw_mgwr))

                logging.info("Computing MGWR predictions manually...")
                X_with_intercept = np.column_stack([np.ones(len(X_std)), X_std])
                y_pred_mgwr = np.sum(X_with_intercept * mgwr_model.params, axis=1)

                logging.info("Computing MGWR local R² manually...")
                data['Local_R2'] = compute_local_r2_manual(
                    coords, y_raw, y_pred_mgwr, median_bw_val, KERNEL)
                residuals = y_raw.flatten() - y_pred_mgwr
            else:
                data['Local_R2'] = gwr_model.localR2.flatten()
                residuals = y_raw.flatten() - gwr_model.predictions.flatten()

            data['Residuals'] = residuals

            # Step 7: Beta coefficients + safe SE/t/p extraction
            beta_cols = []
            for i, var in enumerate(valid_features):
                short = var.replace(f'_{yr}', '')
                data[f'Beta_{short}'] = active_model.params[:, i + 1].flatten()
                beta_cols.append(f'Beta_{short}')

                data[f'SE_{short}']   = try_extract_attribute(active_model, 'se', i + 1)
                data[f'tval_{short}'] = try_extract_attribute(active_model, 'tvalues', i + 1)
                data[f'pval_{short}'] = try_extract_attribute(active_model, 'pvalues', i + 1)

            # Step 8: Condition numbers
            median_bw = (int(np.median(bw_used))
                         if np.ndim(bw_used) > 0 else int(bw_used))
            data['Condition_Number'] = compute_condition_numbers(
                coords, X_std, median_bw, KERNEL)

            # Step 9: Moran's I
            moran_I, moran_p = np.nan, np.nan
            try:
                from esda.moran import Moran
                w_geom = gpd.GeoDataFrame(geometry=data.geometry)
                moran = Moran(residuals, w_geom)
                moran_I, moran_p = moran.I, moran.p_sim
            except Exception:
                pass

            # Step 10: CV
            logging.info("Running random 5-fold CV...")
            r2_rand = random_cv_gwr(coords, y_raw, X_std, bw_gwr)

            logging.info("Running spatial block CV...")
            blocks = spatial_block_cv(coords)
            r2_spat = spatial_cv_gwr(coords, y_raw, X_std, bw_gwr, blocks)

            # Step 11: Report
            report.write("-" * 80 + "\n")
            report.write(f"YEAR: {yr}\n")
            report.write("-" * 80 + "\n")
            report.write(f"Sample size (n)                    : {len(data)}\n")
            report.write(f"Retained predictors                : {', '.join(valid_features)}\n")
            report.write(f"Model used                         : {model_label}\n")
            report.write(f"GWR bandwidth (neighbors)          : {int(bw_gwr)}\n")
            if bw_mgwr is not None:
                report.write(f"MGWR bandwidths (per variable)     : "
                             f"{list(np.round(bw_mgwr, 0).astype(int))}\n")
            report.write(f"GWR adjusted R²                    : {gwr_model.adj_R2:.4f}\n")
            report.write(f"GWR AICc                           : {gwr_model.aicc:.2f}\n")
            if mgwr_model is not None:
                report.write(f"MGWR adjusted R²                   : {mgwr_model.adj_R2:.4f}\n")
                report.write(f"MGWR AICc                          : {mgwr_model.aicc:.2f}\n")
                report.write(f"AICc improvement (GWR - MGWR)      : "
                             f"{gwr_model.aicc - mgwr_model.aicc:.2f}\n")
            report.write(f"Mean local R²                      : {data['Local_R2'].mean():.4f}\n")
            report.write(f"Mean condition number              : "
                         f"{data[data['Condition_Number']<999]['Condition_Number'].mean():.2f}\n")
            report.write(f"Singular zones (stabilized)        : "
                         f"{(data['Condition_Number']>=999).sum()}\n")
            report.write(f"Moran's I (residuals)              : "
                         f"{moran_I:.4f} (p = {moran_p:.4f})\n")
            if r2_rand:
                report.write(f"Random 5-fold CV R² (mean)         : {np.mean(r2_rand):.3f}\n")
            if r2_spat:
                report.write(f"Spatial block CV R² (mean)         : {np.mean(r2_spat):.3f}\n")
            report.write("\n")

            if lcz_col and lcz_col in data.columns:
                lcz_stats = data.groupby(lcz_col)[['Local_R2'] + beta_cols].mean().round(4)
                report.write("Local statistics by LCZ:\n")
                report.write(lcz_stats.to_string())
                report.write("\n\n")

            # Step 12: Exports
            csv_path = os.path.join(OUTPUT_FOLDER, f'MGWR_Coefficients_{yr}.csv')
            data[['POINT_X', 'POINT_Y', lcz_col, 'Local_R2',
                  'Condition_Number', 'Residuals'] + beta_cols].to_csv(
                csv_path, index=False)

            try:
                data.to_file(os.path.join(OUTPUT_FOLDER, f'MGWR_Spatial_{yr}.gpkg'),
                             driver="GPKG")
            except Exception as e:
                logging.warning(f"GeoPackage export failed: {e}")

            # Step 13: Map
            base_map = (gpd.read_file(SHAPEFILE_PATH).to_crs("EPSG:32629")
                        if os.path.exists(SHAPEFILE_PATH) else None)
            fig, ax = plt.subplots(figsize=(10, 8))
            if base_map is not None:
                base_map.plot(ax=ax, color='#f0f0f0',
                              edgecolor='#666666', linewidth=0.8, zorder=1)
            sc = ax.scatter(data.geometry.x, data.geometry.y,
                            c=data['Local_R2'], cmap='viridis',
                            s=8, zorder=2, vmin=0, vmax=1)
            cb = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.03)
            cb.set_label('Local $R^2$', fontsize=12)
            ax.set_title(f'{model_label} Local $R^2$ — {yr}', fontsize=14, pad=15)
            ax.axis('off')
            plt.savefig(os.path.join(OUTPUT_FOLDER, f'MGWR_LocalR2_{yr}.png'))
            plt.close()

            summary_rows.append({
                'Year': yr,
                'n': len(data),
                'Model': model_label,
                'GWR_bw': int(bw_gwr),
                'GWR_adjR2': round(gwr_model.adj_R2, 4),
                'MGWR_adjR2': (round(mgwr_model.adj_R2, 4)
                               if mgwr_model is not None else None),
                'GWR_AICc': round(gwr_model.aicc, 2),
                'MGWR_AICc': (round(mgwr_model.aicc, 2)
                              if mgwr_model is not None else None),
                'Mean_LocalR2': round(data['Local_R2'].mean(), 4),
                'Moran_I': (round(moran_I, 4)
                            if not np.isnan(moran_I) else None),
                'CV_random_R2': (round(np.mean(r2_rand), 3)
                                 if r2_rand else None),
                'CV_spatial_R2': (round(np.mean(r2_spat), 3)
                                  if r2_spat else None),
            })

        report.write("\n" + "=" * 80 + "\n")
        report.write("SUMMARY ACROSS YEARS\n")
        report.write("=" * 80 + "\n")
        report.write(pd.DataFrame(summary_rows).to_string(index=False))
        report.write("\n")

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(OUTPUT_FOLDER, 'MGWR_Summary.csv'), index=False)

    logging.info(f"\n✅ MGWR pipeline completed. Outputs at: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
