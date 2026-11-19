"""
================================================================================
LCZ-LST-DHS HEAT VULNERABILITY WORKFLOW — BAMAKO (2018 & 2024)
Combined multi-year pipeline:
  - Per-year: DHS socio-economic + LCZ + LST → SVI/TSI/CHVI
  - Multi-radius sensitivity (250/500/1000/1500/2000 m) for MAUP
  - Bootstrap CIs for mean CHVI by LCZ
  - Weight sensitivity (Equal / TSI-dom / SVI-dom / PCA)
  - Multi-variable RASTER MAPS (SVI, TSI, CHVI) at 500 m + GeoTIFF exports
  - Inter-annual comparison report (2018 vs 2024)
================================================================================
"""

import os
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.interpolate import griddata
from scipy.stats import spearmanr
from shapely.geometry import Point
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.path import Path as MplPath
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# =============================================================================
# 0. HELPERS
# =============================================================================
def load_stata_safe(filepath, columns=None):
    """Reads Stata (.DTA) files using native Pandas. Falls back to pyreadstat."""
    try:
        return pd.read_stata(filepath, columns=columns, convert_categoricals=False)
    except Exception as e1:
        try:
            import pyreadstat
            df, _ = pyreadstat.read_dta(filepath, usecols=columns)
            return df
        except Exception as e2:
            raise ImportError(
                f"Failed to read {filepath}.\nPandas error: {e1}\nPyreadstat error: {e2}"
            )


MAPPING_2014 = {
    1: 'LCZ10', 3: 'LCZ3', 4: 'LCZ37', 5: 'LCZ5', 6: 'LCZ56',
    7: 'LCZ6', 8: 'LCZ7', 9: 'LCZ9', 10: 'LCZA', 11: 'LCZB',
    12: 'LCZZ', 13: 'LCZD', 14: 'LCZE', 15: 'LCZF', 16: 'LCZG'
}

MAPPING_2018_2024 = {
    1: 'LCZ10', 4: 'LCZ3', 5: 'LCZ37', 6: 'LCZ5', 7: 'LCZ56',
    8: 'LCZ6', 9: 'LCZ7', 10: 'LCZ9', 11: 'LCZA', 12: 'LCZB',
    13: 'LCZZ', 14: 'LCZD', 15: 'LCZE', 16: 'LCZF', 17: 'LCZG'
}


def map_lcz_column(series, mapping):
    def _map(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val).strip().upper()
        if val_str.startswith('LCZ'):
            return val_str
        try:
            code = int(float(val_str))
            return mapping.get(code, np.nan)
        except (ValueError, TypeError):
            return np.nan
    return series.apply(_map)


def apply_lcz_mapping(df, year_cols=('LCZ_2014', 'LCZ_2018', 'LCZ_2024')):
    for col in year_cols:
        if col in df.columns:
            yr = col.split('_')[-1]
            mapping = MAPPING_2014 if yr == '2014' else MAPPING_2018_2024
            df[col] = df[col].astype(str).str.replace('.0', '', regex=False).str.upper()
            df[col] = map_lcz_column(df[col], mapping)
    return df


def min_max_scale(series, invert=False):
    s_min, s_max = series.min(), series.max()
    if s_max == s_min:
        return np.zeros_like(series)
    scaled = (series - s_min) / (s_max - s_min)
    return 1 - scaled if invert else scaled


def get_lcz_mode(series):
    m = series.mode()
    return m.iloc[0] if not m.empty else np.nan


def bootstrap_mean_ci(values, n_boot, ci=95, rng=None):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan, 0, np.nan
    if n == 1:
        return values[0], values[0], values[0], 1, np.nan
    boot_means = np.array([
        rng.choice(values, size=n, replace=True).mean()
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boot_means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return values.mean(), lo, hi, n, boot_means.std(ddof=1)


# =============================================================================
# 1. CONFIGURATION (per-year paths & columns)
# =============================================================================
OUTPUT_DIR = r"C:\Users\ADAMA KONATE\Downloads\Donnes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BOUNDARY_SHP = r"D:\GCN_projet\New folder (3)\Bamako_Project\Limite_Bamako.shp"
COMMUNES_SHP = r"D:\GCN_projet\New folder (3)\Bamako_Project\Communes_Bamako.shp" # <-- À adapter

BUFFER_RADIUS_METERS = 500
SENSITIVITY_RADII = [250, 500, 1000, 1500, 2000]
REFERENCE_RADIUS = 500

N_BOOTSTRAP = 2000
N_BOOTSTRAP_PCA = 500
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)


YEAR_CONFIG = {
    2018: {
        'excel_env':   r"C:\Users\ADAMA KONATE\Downloads\Donnes\Rands.xls",
        'dhs_shp':     r"C:\Users\ADAMA KONATE\Downloads\MLGE7AFL\MLGE7AFL_Clip1_ExportFeature.shp",
        'dhs_hr_dta':  r"C:\Users\ADAMA KONATE\Downloads\MLHR7AFL\MLHR7AFL.DTA",
        'dhs_pr_dta':  r"C:\Users\ADAMA KONATE\Downloads\MLPR7AFL\MLPR7AFL.DTA",
        'lst_col':     'LST_2018', 'ndvi_col': 'NDVI_2018', 'ndbi_col': 'NDBI_2018',
        'mndwi_col':   'MNDWI_2018', 'albedo_col': 'Albedo_2018', 'pd_col': 'PD_2018',
        'lcz_col':     'LCZ_2018', 'radius_suffix': '',
    },
    2024: {
        'excel_env':   r"C:\Users\ADAMA KONATE\Downloads\Donnes\Rand.xls",
        'dhs_shp':     r"C:\Users\ADAMA KONATE\Downloads\Donnes\MLGE8AFL_Clip1_ExportFeature.shp",
        'dhs_hr_dta':  r"C:\Users\ADAMA KONATE\Downloads\Donnes\MLHR8AFL.dta",
        'dhs_pr_dta':  r"C:\Users\ADAMA KONATE\Downloads\Donnes\MLPR8AFL.dta",
        'lst_col':     'LST_2024', 'ndvi_col': 'NDVI_2024', 'ndbi_col': 'NDBI_2024',
        'mndwi_col':   'MNDWI_2024', 'albedo_col': 'Albedo_2024', 'pd_col': 'PD_2024',
        'lcz_col':     'LCZ_2024', 'radius_suffix': 'm',
    },
}


# =============================================================================
# 2. PER-YEAR WORKFLOW
# =============================================================================
def run_year_workflow(year, cfg, output_dir):
    print("\n" + "#" * 78)
    print(f"#  PROCESSING YEAR {year}")
    print("#" * 78)

    suffix = cfg['radius_suffix']
    tag = str(year)

    # -------------------------------------------------------------------------
    # STEP 1: DHS socio-economic indicators
    # -------------------------------------------------------------------------
    print(f"\n[{year}] Step 1/6: Processing DHS Stata files...")
    df_hr = load_stata_safe(cfg['dhs_hr_dta'])
    df_hr.columns = [c.lower() for c in df_hr.columns]
    cluster_col_hr = 'hv001' if 'hv001' in df_hr.columns else 'dhsclust'
    wealth_col = 'hv270' if 'hv270' in df_hr.columns else 'hv271'

    df_hr['electricity_dep'] = df_hr['hv206'].apply(lambda x: 1 if x == 0 else 0) if 'hv206' in df_hr.columns else 0
    df_hr['floor_dep'] = df_hr['hv213'].apply(lambda x: 1 if x in [11, 12, 13, 96] else 0) if 'hv213' in df_hr.columns else 0

    hr_clust = df_hr.groupby(cluster_col_hr).agg({
        wealth_col: 'mean', 'electricity_dep': 'mean', 'floor_dep': 'mean'
    }).reset_index()

    hr_clust.rename(columns={
        cluster_col_hr: 'DHSCLUST', wealth_col: 'Wealth_Index',
        'electricity_dep': 'Elec_Deprivation', 'floor_dep': 'Floor_Deprivation'
    }, inplace=True)

    hr_clust['Housing_Deprivation_Index'] = (hr_clust['Elec_Deprivation'] + hr_clust['Floor_Deprivation']) / 2.0

    df_pr = load_stata_safe(cfg['dhs_pr_dta'])
    df_pr.columns = [c.lower() for c in df_pr.columns]
    cluster_col_pr = 'hv001' if 'hv001' in df_pr.columns else 'dhsclust'
    age_col = 'hv105' if 'hv105' in df_pr.columns else 'hv104'

    df_pr['is_age_vulnerable'] = df_pr[age_col].apply(lambda x: 1 if (x < 5 or x >= 65) else 0)

    pr_clust = df_pr.groupby(cluster_col_pr).agg({'is_age_vulnerable': 'mean'}).reset_index()
    pr_clust.rename(columns={cluster_col_pr: 'DHSCLUST', 'is_age_vulnerable': 'Age_Vulnerability_Pct'}, inplace=True)

    dhs_socio = pd.merge(hr_clust, pr_clust, on='DHSCLUST', how='inner')
    print(f"   -> {len(dhs_socio)} DHS clusters with socio-economic data.")

    # -------------------------------------------------------------------------
    # STEP 2: Merge with DHS shapefile
    # -------------------------------------------------------------------------
    print(f"[{year}] Step 2/6: Merging with DHS spatial layer...")
    gdf_dhs = gpd.read_file(cfg['dhs_shp'])

    shp_clust_col = [c for c in ['DHSCLUST', 'DHSCLUST_', 'DHSID', 'cluster'] if c in gdf_dhs.columns]
    if shp_clust_col:
        gdf_dhs.rename(columns={shp_clust_col[0]: 'DHSCLUST'}, inplace=True)

    gdf_dhs['DHSCLUST'] = gdf_dhs['DHSCLUST'].astype(dhs_socio['DHSCLUST'].dtype)
    gdf_dhs = gdf_dhs.merge(dhs_socio, on='DHSCLUST', how='inner')

    gdf_dhs = gdf_dhs.set_crs(epsg=4326) if gdf_dhs.crs is None else gdf_dhs.to_crs(epsg=4326)
    gdf_dhs['LONGITUDE'] = gdf_dhs.geometry.x
    gdf_dhs['LATITUDE'] = gdf_dhs.geometry.y

    # -------------------------------------------------------------------------
    # STEP 3: Buffer-based environmental extraction
    # -------------------------------------------------------------------------
    print(f"[{year}] Step 3/6: Extracting environmental variables...")
    df_env = pd.read_excel(cfg['excel_env'])
    df_env = apply_lcz_mapping(df_env)

    num_cols = ['POINT_X', 'POINT_Y', cfg['lst_col'], cfg['ndvi_col'], cfg['ndbi_col'],
                cfg['mndwi_col'], cfg['albedo_col'], cfg['pd_col']]
    for col in num_cols:
        if col in df_env.columns:
            df_env[col] = pd.to_numeric(df_env[col], errors='coerce')

    df_env = df_env.dropna(subset=['POINT_X', 'POINT_Y', cfg['lst_col']])

    gdf_env = gpd.GeoDataFrame(
        df_env, geometry=gpd.points_from_xy(df_env['POINT_X'], df_env['POINT_Y']), crs="EPSG:4326"
    ).to_crs(epsg=32629)

    gdf_dhs_utm = gdf_dhs.to_crs(epsg=32629)

    def extract_buffer_stats(radius_m):
        buffers = gdf_dhs_utm.copy()
        buffers['geometry'] = gdf_dhs_utm.geometry.buffer(radius_m)
        joined = gpd.sjoin(gdf_env, buffers[['DHSCLUST', 'geometry']], how='inner', predicate='within')
        stats = joined.groupby('DHSCLUST').agg({
            cfg['lcz_col']: get_lcz_mode, cfg['lst_col']: 'mean',
            cfg['ndvi_col']: 'mean', cfg['ndbi_col']: 'mean',
            cfg['mndwi_col']: 'mean', cfg['albedo_col']: 'mean', cfg['pd_col']: 'mean',
        }).reset_index()
        stats = stats.rename(columns={
            cfg['lcz_col']: f'LCZ_{radius_m}{suffix}', cfg['lst_col']: f'LST_{radius_m}{suffix}',
            cfg['ndvi_col']: f'NDVI_{radius_m}{suffix}', cfg['ndbi_col']: f'NDBI_{radius_m}{suffix}',
            cfg['mndwi_col']: f'MNDWI_{radius_m}{suffix}', cfg['albedo_col']: f'Albedo_{radius_m}{suffix}',
            cfg['pd_col']: f'PD_{radius_m}{suffix}',
        })
        return stats

    # Main extraction (500 m)
    stats_main = extract_buffer_stats(BUFFER_RADIUS_METERS)
    gdf_dhs = gdf_dhs.merge(stats_main, on='DHSCLUST', how='left')

    gdf_dhs['LCZ'] = gdf_dhs[f'LCZ_{BUFFER_RADIUS_METERS}{suffix}']
    gdf_dhs['LST'] = gdf_dhs[f'LST_{BUFFER_RADIUS_METERS}{suffix}']
    gdf_dhs['NDVI'] = gdf_dhs[f'NDVI_{BUFFER_RADIUS_METERS}{suffix}']
    gdf_dhs['NDBI'] = gdf_dhs[f'NDBI_{BUFFER_RADIUS_METERS}{suffix}']

    print(f"   -> Main extraction (500 m) complete: {len(gdf_dhs)} clusters.")

    # Sensitivity extractions
    sensitivity_radii_to_merge = [r for r in SENSITIVITY_RADII if r != BUFFER_RADIUS_METERS]
    print(f"[{year}] Step 3bis: Multi-radius sensitivity {sensitivity_radii_to_merge} m...")
    for r in sensitivity_radii_to_merge:
        stats_r = extract_buffer_stats(r)
        stats_r['DHSCLUST'] = stats_r['DHSCLUST'].astype(gdf_dhs['DHSCLUST'].dtype)
        gdf_dhs = gdf_dhs.merge(stats_r, on='DHSCLUST', how='left')
        print(f"      Buffer {r:>5}m -> {len(stats_r)} clusters matched")

    # -------------------------------------------------------------------------
    # STEP 4: Indices (main + sensitivity)
    # -------------------------------------------------------------------------
    print(f"[{year}] Step 4/6: Computing SVI / TSI / CHVI...")

    def compute_chvi_components(df, radius_m):
        lst_col  = f'LST_{radius_m}{suffix}'
        ndbi_col = f'NDBI_{radius_m}{suffix}'
        ndvi_col = f'NDVI_{radius_m}{suffix}'

        norm_lst  = min_max_scale(df[lst_col])
        norm_ndbi = min_max_scale(df[ndbi_col])
        norm_ndvi = min_max_scale(df[ndvi_col])
        tsi = min_max_scale(norm_lst + norm_ndbi - norm_ndvi)

        norm_wealth_inv = min_max_scale(df['Wealth_Index'], invert=True)
        norm_age_vuln   = min_max_scale(df['Age_Vulnerability_Pct'])
        norm_housing    = min_max_scale(df['Housing_Deprivation_Index'])

        svi = (norm_wealth_inv + norm_age_vuln + norm_housing) / 3.0
        chvi = (svi + tsi) / 2.0
        return svi, tsi, chvi

    gdf_dhs['LST_Anomaly'] = gdf_dhs['LST'] - gdf_dhs['LST'].mean()

    gdf_dhs['SVI'], gdf_dhs['Thermal_Stress_Index'], gdf_dhs['CHVI'] = \
        compute_chvi_components(gdf_dhs, BUFFER_RADIUS_METERS)

    for r in SENSITIVITY_RADII:
        svi_r, tsi_r, chvi_r = compute_chvi_components(gdf_dhs, r)
        gdf_dhs[f'SVI_{r}{suffix}']  = svi_r
        gdf_dhs[f'TSI_{r}{suffix}']  = tsi_r
        gdf_dhs[f'CHVI_{r}{suffix}'] = chvi_r

    # Canonical CHVI 500 m column
    gdf_dhs['CHVI_500'] = gdf_dhs[f'CHVI_{BUFFER_RADIUS_METERS}{suffix}']
    gdf_dhs['SVI_500'] = gdf_dhs[f'SVI_{BUFFER_RADIUS_METERS}{suffix}']
    gdf_dhs['TSI_500'] = gdf_dhs[f'TSI_{BUFFER_RADIUS_METERS}{suffix}']

    # -------------------------------------------------------------------------
    # STEP 4bis: Bootstrap CIs + weight sensitivity
    # -------------------------------------------------------------------------
    print(f"[{year}] Step 4bis: Bootstrap CIs & weight sensitivity...")

    boot_rows = []
    for lcz, grp in gdf_dhs.dropna(subset=['LCZ', 'CHVI']).groupby('LCZ'):
        m, lo, hi, n, se = bootstrap_mean_ci(grp['CHVI'].values, N_BOOTSTRAP, rng=rng)
        boot_rows.append({
            'LCZ': lcz, 'n_clusters': n, 'Mean_CHVI': round(m, 3),
            'CI95_low': round(lo, 3), 'CI95_high': round(hi, 3),
            'CI95_width': round(hi - lo, 3),
            'Bootstrap_SE': round(se, 4) if not np.isnan(se) else np.nan,
            'Reliability_flag': (
                'HIGH (n>=15)' if n >= 15 else 'MODERATE (5<=n<15)' if n >= 5 else
                'LOW (2<=n<5)' if n >= 2 else 'SINGLETON (n=1) — no CI'
            )
        })
    boot_df = pd.DataFrame(boot_rows).sort_values('Mean_CHVI', ascending=False)
    print("\n   Bootstrap 95% CIs of mean CHVI by LCZ:")
    print(boot_df.to_string(index=False))

    def compute_chvi_weighted(svi, tsi, w_svi=0.5):
        return w_svi * svi + (1 - w_svi) * tsi

    WEIGHT_SCENARIOS = {'Equal': 0.50, 'TSI_dominant': 0.30, 'SVI_dominant': 0.70}
    for scenario, w_svi in WEIGHT_SCENARIOS.items():
        gdf_dhs[f'CHVI_w_{scenario}'] = compute_chvi_weighted(gdf_dhs['SVI'], gdf_dhs['Thermal_Stress_Index'], w_svi=w_svi)

    # PCA weights
    pca_input = gdf_dhs[['SVI', 'Thermal_Stress_Index']].dropna()
    scaler = StandardScaler()
    Z = scaler.fit_transform(pca_input)
    pca = PCA(n_components=1, random_state=RANDOM_STATE).fit(Z)
    loadings = np.abs(pca.components_[0])
    w_pca = loadings / loadings.sum()
    print(f"\n   PCA weights: w_SVI={w_pca[0]:.3f}, w_TSI={w_pca[1]:.3f}")

    gdf_dhs['CHVI_w_PCA'] = compute_chvi_weighted(gdf_dhs['SVI'], gdf_dhs['Thermal_Stress_Index'], w_svi=w_pca[0])

    weight_cols = ['CHVI'] + [f'CHVI_w_{s}' for s in WEIGHT_SCENARIOS] + ['CHVI_w_PCA']

    rank_stability_rows = []
    for lcz, grp in gdf_dhs.dropna(subset=['LCZ']).groupby('LCZ'):
        row = {'LCZ': lcz, 'n_clusters': len(grp)}
        for col in weight_cols:
            row[f'mean_{col}'] = grp[col].mean() if col in grp else np.nan
        rank_stability_rows.append(row)

    rank_stability_df = pd.DataFrame(rank_stability_rows)
    for col in weight_cols:
        rank_stability_df[f'rank_{col}'] = rank_stability_df[f'mean_{col}'].rank(ascending=False, method='min')

    weight_rank_corr = []
    ref_rank = rank_stability_df['rank_CHVI']
    for col in weight_cols[1:]:
        alt_rank = rank_stability_df[f'rank_{col}']
        valid = ~(ref_rank.isna() | alt_rank.isna())
        if valid.sum() >= 3:
            rho, p = spearmanr(ref_rank[valid], alt_rank[valid])
            weight_rank_corr.append({
                'Scenario': col, 'Spearman_rho_vs_Equal': round(rho, 3),
                'p_value': round(p, 4), 'n_LCZ': int(valid.sum())
            })
    weight_rank_corr_df = pd.DataFrame(weight_rank_corr)

    # Bootstrap PCA weight
    pca_weight_samples = []
    for _ in range(N_BOOTSTRAP_PCA):
        idx = rng.integers(0, len(pca_input), len(pca_input))
        Zb = StandardScaler().fit_transform(pca_input.iloc[idx].values)
        pc = PCA(n_components=1, random_state=RANDOM_STATE).fit(Zb)
        l = np.abs(pc.components_[0]); w = l / l.sum()
        pca_weight_samples.append(w[0])
    pca_w_lo, pca_w_hi = np.percentile(pca_weight_samples, [2.5, 97.5])

    # -------------------------------------------------------------------------
    # STEP 5: Radius sensitivity rankings
    # -------------------------------------------------------------------------
    print(f"[{year}] Step 5/6: Radius sensitivity of LCZ rankings...")
    lcz_rank_table = pd.DataFrame()
    for r in SENSITIVITY_RADII:
        lcz_col  = f'LCZ_{r}{suffix}'
        chvi_col = f'CHVI_{r}{suffix}'
        summary_r = (gdf_dhs.dropna(subset=[lcz_col, chvi_col])
                     .groupby(lcz_col)[chvi_col].mean().reset_index())
        summary_r.columns = ['LCZ', f'CHVI_{r}m_mean']
        summary_r[f'Rank_{r}m'] = summary_r[f'CHVI_{r}m_mean'].rank(ascending=False)
        lcz_rank_table = summary_r if lcz_rank_table.empty else lcz_rank_table.merge(
            summary_r[['LCZ', f'CHVI_{r}m_mean', f'Rank_{r}m']], on='LCZ', how='outer')

    ref_ranks = lcz_rank_table[f'Rank_{REFERENCE_RADIUS}m']
    rank_corr_rows = []
    for r in SENSITIVITY_RADII:
        col = f'Rank_{r}m'
        if col not in lcz_rank_table.columns:
            continue
        other = lcz_rank_table[col]
        valid = ~(ref_ranks.isna() | other.isna())
        if valid.sum() >= 3:
            rho, p = spearmanr(ref_ranks[valid], other[valid])
            rank_corr_rows.append({
                'Radius_m': r, 'Spearman_rho_vs_500m': round(rho, 3),
                'p_value': round(p, 4), 'n_LCZ': int(valid.sum())
            })
    rank_corr_df = pd.DataFrame(rank_corr_rows)

    # -------------------------------------------------------------------------
    # STEP 6: Figures, Excel, PDF
    # -------------------------------------------------------------------------
    print(f"[{year}] Step 6/6: Generating figures, Excel & PDF...")
    boundary_gdf = gpd.read_file(BOUNDARY_SHP).to_crs(epsg=4326)

    # --- A. LST map
    xmin, ymin, xmax, ymax = boundary_gdf.total_bounds
    grid_x, grid_y = np.mgrid[xmin:xmax:800j, ymin:ymax:800j]
    z_lst = df_env[cfg['lst_col']].values
    grid_z = griddata((df_env['POINT_X'], df_env['POINT_Y']), z_lst, (grid_x, grid_y), method='cubic')
    grid_z_lin = griddata((df_env['POINT_X'], df_env['POINT_Y']), z_lst, (grid_x, grid_y), method='linear')
    grid_z = np.where(np.isnan(grid_z), grid_z_lin, grid_z)
    boundary_poly = boundary_gdf.geometry.union_all()
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            if not boundary_poly.contains(Point(grid_x[i, j], grid_y[i, j])):
                grid_z[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(10, 8), dpi=600)
    c1 = ax.pcolormesh(grid_x, grid_y, grid_z, cmap=plt.get_cmap('RdYlBu_r'), shading='auto')
    boundary_gdf.boundary.plot(ax=ax, color='black', linewidth=1.5, label='Bamako Boundary')
    ax.scatter(gdf_dhs['LONGITUDE'], gdf_dhs['LATITUDE'], color='black', marker='^', s=25, alpha=0.8, label='DHS Clusters')
    plt.title(f"Land Surface Temperature (LST) - Bamako {year}", fontsize=14, fontweight='bold')
    plt.xlabel("Longitude", fontsize=12); plt.ylabel("Latitude", fontsize=12)
    cbar1 = plt.colorbar(c1, ax=ax, fraction=0.036, pad=0.04); cbar1.set_label('LST (°C)', fontsize=12)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'Thermal_Map_LST_{tag}.png'), dpi=600)
    plt.close()

    # --- B. CHVI point map
    fig, ax = plt.subplots(figsize=(10, 8), dpi=600)
    boundary_gdf.boundary.plot(ax=ax, color='black', linewidth=1.5)
    scatter = ax.scatter(gdf_dhs['LONGITUDE'], gdf_dhs['LATITUDE'], c=gdf_dhs['CHVI'], cmap='YlOrRd', s=80, edgecolor='k', alpha=0.9)
    plt.title(f"Composite Heat Vulnerability Index (CHVI) - Bamako {year}", fontsize=14, fontweight='bold')
    plt.xlabel("Longitude", fontsize=12); plt.ylabel("Latitude", fontsize=12)
    cbar2 = plt.colorbar(scatter, ax=ax, fraction=0.036, pad=0.04); cbar2.set_label('CHVI [0 - 1]', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'CHVI_Vulnerability_Map_{tag}.png'), dpi=600)
    plt.close()

    # --- C. Radius sensitivity plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=600)
    plot_df = lcz_rank_table.melt(
        id_vars=['LCZ'], value_vars=[f'CHVI_{r}m_mean' for r in SENSITIVITY_RADII if f'CHVI_{r}m_mean' in lcz_rank_table.columns],
        var_name='Radius', value_name='Mean_CHVI')
    plot_df['Radius'] = plot_df['Radius'].str.extract(r'(\d+)').astype(int)
    sns.lineplot(data=plot_df, x='Radius', y='Mean_CHVI', hue='LCZ', marker='o', ax=ax)
    ax.set_xlabel('Buffer radius (m)', fontsize=12); ax.set_ylabel('Mean CHVI', fontsize=12)
    ax.set_title(f'CHVI Sensitivity to Buffer Radius ({year})', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.5); ax.legend(title='Dominant LCZ', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'CHVI_Sensitivity_Radius_{tag}.png'), dpi=600)
    plt.close()

    # --- D. Bootstrap forest plot
    fig, ax = plt.subplots(figsize=(9, 6), dpi=600)
    bp = boot_df.dropna(subset=['Mean_CHVI']).sort_values('Mean_CHVI')
    y_pos = np.arange(len(bp))
    colors = bp['Reliability_flag'].map({
        'HIGH (n>=15)': '#2c7bb6', 'MODERATE (5<=n<15)': '#fdae61',
        'LOW (2<=n<5)': '#d7191c', 'SINGLETON (n=1) — no CI': '#999999'})
    ax.errorbar(bp['Mean_CHVI'], y_pos, xerr=[bp['Mean_CHVI'] - bp['CI95_low'], bp['CI95_high'] - bp['Mean_CHVI']],
                fmt='o', ecolor='black', capsize=4, markersize=8, markerfacecolor='white', markeredgewidth=2, zorder=3)
    for yi, (_, row) in zip(y_pos, bp.iterrows()):
        ax.scatter(row['Mean_CHVI'], yi, color=colors.iloc[yi], s=90, zorder=4)
        ax.text(row['CI95_high'] + 0.01, yi, f"n={int(row['n_clusters'])}", va='center', fontsize=8)
    ax.set_yticks(y_pos); ax.set_yticklabels(bp['LCZ'])
    ax.set_xlabel('Mean CHVI (500 m buffer) with bootstrap 95% CI', fontsize=11)
    ax.set_title(f'CHVI by LCZ — Bootstrap Uncertainty ({year})', fontsize=12, fontweight='bold')
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'CHVI_Bootstrap_CI_by_LCZ_{tag}.png'), dpi=600)
    plt.close()

    # --- E. Weight sensitivity heatmap
    fig, ax = plt.subplots(figsize=(10, 6), dpi=600)
    rank_heat = rank_stability_df.set_index('LCZ')[[f'rank_{c}' for c in weight_cols]].copy()
    rank_heat.columns = [c.replace('rank_CHVI_w_', '').replace('rank_CHVI', 'Equal') for c in rank_heat.columns]
    sns.heatmap(rank_heat, annot=True, fmt='.0f', cmap='RdYlGn_r', cbar_kws={'label': 'Rank (1 = most vulnerable)'}, ax=ax)
    ax.set_title(f'LCZ Vulnerability Ranking — CHVI Weight Sensitivity ({year})', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'CHVI_Weight_Sensitivity_Ranks_{tag}.png'), dpi=600)
    plt.close()

    # -------------------------------------------------------------------------
    # --- G. MULTI-VARIABLE RASTER MAPS (SVI, TSI, CHVI) - TWO PANEL + GeoTIFF
    # -------------------------------------------------------------------------
    print(f"[{year}] Generating SVI, TSI & CHVI 500 m raster maps (classified, two-panel) & GeoTIFFs...")

    boundary_utm = boundary_gdf.to_crs(epsg=32629)
    xmin_u, ymin_u, xmax_u, ymax_u = boundary_utm.total_bounds
    pad, grid_res = 500.0, 100
    gx = np.arange(xmin_u - pad, xmax_u + pad, grid_res)
    gy = np.arange(ymin_u - pad, ymax_u + pad, grid_res)
    grid_x_u, grid_y_u = np.meshgrid(gx, gy)

    boundary_poly_utm = boundary_utm.geometry.union_all()
    pts = np.column_stack([grid_x_u.ravel(), grid_y_u.ravel()])
    inside = np.zeros(len(pts), dtype=bool)
    if boundary_poly_utm.geom_type == 'Polygon':
        inside = MplPath(np.array(boundary_poly_utm.exterior.coords)).contains_points(pts)
    elif boundary_poly_utm.geom_type == 'MultiPolygon':
        for poly in boundary_poly_utm.geoms:
            inside |= MplPath(np.array(poly.exterior.coords)).contains_points(pts)

    try:
        gdf_communes = gpd.read_file(COMMUNES_SHP).to_crs(epsg=32629)
        name_field = None
        for f in ['NOM', 'NAME', 'Commune', 'COMMUNE', 'nom', 'name', 'commune']:
            if f in gdf_communes.columns:
                name_field = f
                break
        if name_field is None:
            name_field = [c for c in gdf_communes.columns if gdf_communes[c].dtype == object][0]
    except Exception as e:
        print(f"   (Communes shapefile not found: {e})")
        gdf_communes = None
        name_field = None

    def plot_and_export_raster(var_col, title_text, filename_prefix):
        # 1. Interpolation
        gdf_var = gdf_dhs.dropna(subset=[var_col]).copy()
        gdf_var_utm = gdf_var.to_crs(epsg=32629)
        x_var = gdf_var_utm.geometry.x.values
        y_var = gdf_var_utm.geometry.y.values
        z_var = gdf_var_utm[var_col].values

        grid_var = griddata((x_var, y_var), z_var, (grid_x_u, grid_y_u), method='cubic')
        grid_var_lin = griddata((x_var, y_var), z_var, (grid_x_u, grid_y_u), method='linear')
        grid_var = np.where(np.isnan(grid_var), grid_var_lin, grid_var)
        grid_var_nn = griddata((x_var, y_var), z_var, (grid_x_u, grid_y_u), method='nearest')
        grid_var = np.where(np.isnan(grid_var), grid_var_nn, grid_var)

        # 2. Application du masque spatial
        grid_var_flat = grid_var.ravel().copy()
        grid_var_flat[~inside] = np.nan
        grid_var = grid_var_flat.reshape(grid_var.shape)

        # 3. Création du GeoTIFF
        try:
            from rasterio.transform import from_origin
            import rasterio
            transform = from_origin(grid_x_u.min(), grid_y_u.max(), grid_res, grid_res)
            with rasterio.open(
                os.path.join(output_dir, f'{filename_prefix}_{tag}.tif'),
                'w', driver='GTiff', height=grid_var.shape[0], width=grid_var.shape[1],
                count=1, dtype='float32', crs='EPSG:32629', transform=transform, nodata=np.nan
            ) as dst:
                dst.write(grid_var.astype('float32'), 1)
            print(f"   -> {filename_prefix} GeoTIFF exported for {year}.")
        except Exception as e:
            print(f"   -> Error exporting {filename_prefix} GeoTIFF: {e}")

        # 4. Figure 2 panneaux
        bounds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
        colors = ['#1f78b4', '#41b6c4', '#a1d99b', '#e5f5a0', '#ffff99', '#fdb863', '#e66101', '#b30000']
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(bounds, ncolors=len(colors), clip=True)

        fig, axes = plt.subplots(1, 2, figsize=(18, 9), dpi=600, sharex=True, sharey=True)
        panels = [(axes[0], 'a)'), (axes[1], 'b)')]

        for ax, panel_label in panels:
            ax.pcolormesh(grid_x_u, grid_y_u, grid_var, cmap=cmap, norm=norm, shading='auto')
            boundary_utm.boundary.plot(ax=ax, color='black', linewidth=1.4)

            if gdf_communes is not None:
                gdf_communes.boundary.plot(ax=ax, color='gray', linewidth=0.8)
                for _, row in gdf_communes.iterrows():
                    if name_field and pd.notna(row[name_field]):
                        centroid = row.geometry.centroid
                        ax.text(centroid.x, centroid.y, str(row[name_field]), fontsize=7, ha='center', va='center', color='black', fontweight='bold')

            ax.set_aspect('equal')
            ax.tick_params(labelsize=9)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}"))
            ax.set_xlabel("Easting (m, UTM 29N)", fontsize=10)
            ax.set_ylabel("Northing (m, UTM 29N)", fontsize=10)

            x_arrow = xmax_u - 0.10 * (xmax_u - xmin_u)
            y_arrow = ymax_u - 0.08 * (ymax_u - ymin_u)
            ax.annotate('', xy=(x_arrow, y_arrow + 0.06 * (ymax_u - ymin_u)), xytext=(x_arrow, y_arrow),
                        arrowprops=dict(facecolor='black', width=6, headwidth=18, edgecolor='black'))
            ax.text(x_arrow, y_arrow + 0.075 * (ymax_u - ymin_u), 'N', ha='center', va='bottom', fontsize=12, fontweight='bold')

            sb_len_m = 6000
            sb_x0 = xmin_u + 0.06 * (xmax_u - xmin_u)
            sb_y0 = ymin_u + 0.05 * (ymax_u - ymin_u)
            ax.plot([sb_x0, sb_x0 + sb_len_m], [sb_y0, sb_y0], color='black', linewidth=4, solid_capstyle='butt', zorder=6)
            for xx in [sb_x0, sb_x0 + sb_len_m/2, sb_x0 + sb_len_m]:
                ax.plot([xx, xx], [sb_y0 - 0.012 * (ymax_u - ymin_u), sb_y0 + 0.012 * (ymax_u - ymin_u)], color='black', linewidth=2, zorder=6)
            ax.text(sb_x0, sb_y0 - 0.03 * (ymax_u - ymin_u), '0', ha='center', fontsize=8, fontweight='bold')
            ax.text(sb_x0 + sb_len_m/2, sb_y0 - 0.03 * (ymax_u - ymin_u), '3', ha='center', fontsize=8, fontweight='bold')
            ax.text(sb_x0 + sb_len_m, sb_y0 - 0.03 * (ymax_u - ymin_u), '6 km', ha='center', fontsize=8, fontweight='bold')

            ax.text(0.02, 0.97, panel_label, transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.85))

        fig.suptitle(f"{title_text} — Bamako {year}", fontsize=16, fontweight='bold', y=0.98)
        legend_labels = ['0.0–0.1', '0.1–0.2', '0.2–0.3', '0.3–0.4', '0.4–0.5', '0.5–0.6', '0.6–0.7', '>0.7']
        legend_handles = [Patch(facecolor=c, edgecolor='black', label=l) for c, l in zip(colors, legend_labels)]
        fig.legend(handles=legend_handles, loc='lower center', ncol=8, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02), title=f'{title_text.split("(")[0].strip()} classes')

        plt.tight_layout(rect=[0, 0.04, 1, 0.96])
        plt.savefig(os.path.join(output_dir, f'{filename_prefix}_{year}.png'), dpi=600, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f'{filename_prefix}_{year}.pdf'), dpi=600, bbox_inches='tight')
        plt.close()
        print(f"   -> Two-panel {title_text} map saved for {year}.")

    # Appel de la fonction pour les 3 indicateurs (générera les images ET les .tif)
    plot_and_export_raster('SVI_500', 'Socio-Economic Vulnerability Index (SVI, 500 m)', 'SVI_500m_Raster_Map')
    plot_and_export_raster('TSI_500', 'Thermal Stress Index (TSI, 500 m)', 'TSI_500m_Raster_Map')
    plot_and_export_raster('CHVI_500', 'Composite Heat Vulnerability Index (CHVI, 500 m)', 'CHVI_500m_Raster_Map')


    # --- H. Excel workbook & CSVs
    excel_out = os.path.join(output_dir, f'Q1_Thermal_Vulnerability_Tables_{tag}.xlsx')
    df_export = pd.DataFrame(gdf_dhs.drop(columns='geometry', errors='ignore'))

    with pd.ExcelWriter(excel_out, engine='openpyxl') as writer:
        df_export.to_excel(writer, sheet_name='Cluster_Level_Data', index=False)
        lcz_summary = df_export.groupby('LCZ').agg({
            'LST': ['count', 'mean', 'std', 'min', 'max'], 'NDVI': 'mean', 'NDBI': 'mean',
            'SVI': 'mean', 'Thermal_Stress_Index': 'mean', 'CHVI': 'mean'
        })
        lcz_summary.to_excel(writer, sheet_name='LCZ_Summary_Statistics')
        corr_cols = ['LST', 'NDVI', 'NDBI', 'Wealth_Index', 'Age_Vulnerability_Pct', 'Housing_Deprivation_Index', 'SVI', 'Thermal_Stress_Index', 'CHVI']
        corr_df = df_export[corr_cols].corr(method='pearson')
        corr_df.to_excel(writer, sheet_name='Correlation_Matrix')
        lcz_rank_table.to_excel(writer, sheet_name='Sensitivity_LCZ_Rankings', index=False)
        rank_corr_df.to_excel(writer, sheet_name='Sensitivity_Rank_Correlation', index=False)
        boot_df.to_excel(writer, sheet_name='Bootstrap_CI_by_LCZ', index=False)
        rank_stability_df.to_excel(writer, sheet_name='Weight_Rank_Stability', index=False)
        weight_rank_corr_df.to_excel(writer, sheet_name='Weight_Rank_Correlation', index=False)

    # --- I. Per-year executive PDF
    pdf_path = os.path.join(output_dir, f'CHVI_Executive_Report_{tag}.pdf')
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 11)); ax.axis('off')
        ax.text(0.5, 0.95, f"COMPOSITE HEAT VULNERABILITY REPORT (BAMAKO {year})", fontsize=16, fontweight='bold', ha='center')
        ax.text(0.5, 0.92, "500m Buffer Neighborhood + Multi-Radius Sensitivity + Bootstrap", fontsize=11, ha='center', style='italic')
        summary_txt = (
            "1. METHODOLOGY:\n"
            "   - Reference buffer: 500 m around each DHS cluster.\n"
            "   - Sensitivity: 250, 500, 1000, 1500, 2000 m (MAUP).\n"
            "   - SVI = Wealth + Age + Housing deprivation.\n"
            "   - TSI = min-max(LST + NDBI - NDVI).\n"
            "   - CHVI = (SVI + TSI) / 2.\n\n"
            "2. ROBUSTNESS (Spearman rho vs 500m):\n"
        )
        ax.text(0.08, 0.86, summary_txt, fontsize=10, verticalalignment='top')
        if not rank_corr_df.empty:
            data = [rank_corr_df.columns.tolist()] + rank_corr_df.round(3).values.tolist()
            t = ax.table(cellText=data, loc='upper center', bbox=[0.08, 0.58, 0.84, 0.14]); t.auto_set_font_size(False); t.set_fontsize(8.5)
        ax.text(0.08, 0.54, "3. SUMMARY BY LCZ (main analysis):", fontsize=10)
        lcz_simple = df_export.groupby('LCZ')[['LST', 'SVI', 'Thermal_Stress_Index', 'CHVI']].mean().reset_index()
        lcz_simple.columns = ['LCZ', 'Mean LST (°C)', 'Mean SVI', 'Mean TSI', 'Mean CHVI']
        data = [lcz_simple.columns.tolist()] + lcz_simple.round(3).values.tolist()
        t = ax.table(cellText=data, loc='center', bbox=[0.05, 0.15, 0.90, 0.32]); t.auto_set_font_size(False); t.set_fontsize(8.5)
        pdf.savefig(fig, bbox_inches='tight'); plt.close()

    return {
        'year': year, 'gdf_dhs': gdf_dhs, 'boot_df': boot_df,
        'rank_stability_df': rank_stability_df, 'weight_rank_corr_df': weight_rank_corr_df,
        'lcz_rank_table': lcz_rank_table, 'rank_corr_df': rank_corr_df,
        'lcz_summary': lcz_summary, 'w_pca': w_pca, 'pca_w_lo': pca_w_lo, 'pca_w_hi': pca_w_hi
    }


# =============================================================================
# 3. RUN BOTH YEARS
# =============================================================================
print("=================================================================")
print("  COMBINED LCZ-LST-DHS HEAT VULNERABILITY WORKFLOW (2018 & 2024)")
print("=================================================================")

results = {}
for year, cfg in YEAR_CONFIG.items():
    results[year] = run_year_workflow(year, cfg, OUTPUT_DIR)


# =============================================================================
# 4. INTER-ANNUAL COMPARISON (2018 vs 2024)
# =============================================================================
print("\n" + "#" * 78)
print("#  INTER-ANNUAL COMPARISON (2018 vs 2024)")
print("#" * 78)

comparison_rows = []
for year, res in results.items():
    g = res['gdf_dhs']
    comparison_rows.append({
        'Year': year, 'N_clusters': len(g),
        'Mean_LST': round(g['LST'].mean(), 3), 'Mean_SVI': round(g['SVI'].mean(), 3),
        'Mean_TSI': round(g['Thermal_Stress_Index'].mean(), 3), 'Mean_CHVI': round(g['CHVI'].mean(), 3)
    })
comparison_df = pd.DataFrame(comparison_rows)
print("\n Summary comparison:")
print(comparison_df.to_string(index=False))

lcz_compare = pd.DataFrame()
for year, res in results.items():
    b = res['boot_df'][['LCZ', 'Mean_CHVI']].rename(columns={'Mean_CHVI': f'CHVI_{year}'})
    lcz_compare = b if lcz_compare.empty else lcz_compare.merge(b, on='LCZ', how='outer')

if 'CHVI_2018' in lcz_compare.columns and 'CHVI_2024' in lcz_compare.columns:
    lcz_compare['Delta_CHVI'] = lcz_compare['CHVI_2024'] - lcz_compare['CHVI_2018']
    lcz_compare = lcz_compare.sort_values('Delta_CHVI', ascending=False)

print("\n Per-LCZ mean CHVI comparison:")
print(lcz_compare.to_string(index=False))

comparison_xlsx = os.path.join(OUTPUT_DIR, 'CHVI_InterAnnual_Comparison_2018_2024.xlsx')
with pd.ExcelWriter(comparison_xlsx, engine='openpyxl') as writer:
    comparison_df.to_excel(writer, sheet_name='Global_Summary', index=False)
    lcz_compare.to_excel(writer, sheet_name='By_LCZ_Delta', index=False)

# Comparison figure: mean CHVI by LCZ, 2018 vs 2024
if not lcz_compare.empty and 'CHVI_2018' in lcz_compare.columns and 'CHVI_2024' in lcz_compare.columns:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=600)
    lcz_plot = lcz_compare.dropna(subset=['CHVI_2018', 'CHVI_2024']).sort_values('CHVI_2024')
    y = np.arange(len(lcz_plot))
    ax.barh(y - 0.2, lcz_plot['CHVI_2018'], height=0.4, color='#2c7bb6', label='2018')
    ax.barh(y + 0.2, lcz_plot['CHVI_2024'], height=0.4, color='#d7191c', label='2024')
    ax.set_yticks(y); ax.set_yticklabels(lcz_plot['LCZ'])
    ax.set_xlabel('Mean CHVI (500 m buffer)', fontsize=11)
    ax.set_title('Mean CHVI by LCZ — 2018 vs 2024 (Bamako)', fontsize=12, fontweight='bold')
    ax.grid(axis='x', linestyle=':', alpha=0.5); ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'CHVI_Comparison_By_LCZ_2018_2024.png'), dpi=600)
    plt.close()

# =============================================================================
# 5. COMPARAISON GLOBALE (LST, SVI, TSI, CHVI) ET RAPPORT PDF FINAL
# =============================================================================
print("\n" + "#" * 78)
print("#  FINAL COMPARISON FIGURES & INTER-ANNUAL PDF REPORT")
print("#" * 78)

# --- A. Figure comparative : Métriques globales (LST, SVI, TSI, CHVI)
metrics_to_plot = ['Mean_LST', 'Mean_SVI', 'Mean_TSI', 'Mean_CHVI']
fig, ax = plt.subplots(figsize=(10, 6), dpi=600)
comp_plot = comparison_df.set_index('Year')[metrics_to_plot]
comp_plot.plot(kind='bar', ax=ax, color=['#e66101', '#5e3c99', '#b2abd2', '#d7191c'], edgecolor='black', width=0.7)
ax.set_ylabel('Mean value', fontsize=12)
ax.set_title('Global indicators — 2018 vs 2024 (Bamako)', fontsize=14, fontweight='bold')
ax.set_xticklabels(comp_plot.index, rotation=0, fontsize=11)
ax.grid(axis='y', linestyle=':', alpha=0.5)
ax.legend(title='Metric', loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'CHVI_Comparison_Global_2018_2024.png'), dpi=600)
plt.savefig(os.path.join(OUTPUT_DIR, 'CHVI_Comparison_Global_2018_2024.pdf'), dpi=600)
plt.close()

# --- B. Rapport PDF Inter-annuel (2 pages)
comp_pdf_path = os.path.join(OUTPUT_DIR, 'CHVI_InterAnnual_Report_2018_2024.pdf')
with PdfPages(comp_pdf_path) as pdf:
    # Page 1
    fig, ax = plt.subplots(figsize=(8.5, 11)); ax.axis('off')
    ax.text(0.5, 0.95, "INTER-ANNUAL HEAT VULNERABILITY REPORT — BAMAKO", fontsize=16, fontweight='bold', ha='center', fontname='Arial')
    ax.text(0.5, 0.92, "Comparison 2018 vs 2024 (500 m buffer, equal weights)", fontsize=11, ha='center', style='italic', fontname='Arial')
    ax.text(0.08, 0.86, "1. GLOBAL SUMMARY", fontsize=11, fontweight='bold')
    tdata = [comparison_df.columns.tolist()] + comparison_df.round(3).values.tolist()
    t = ax.table(cellText=tdata, loc='upper center', bbox=[0.05, 0.65, 0.90, 0.18])
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.3)
    ax.text(0.08, 0.60, "2. PER-LCZ MEAN CHVI & DELTA", fontsize=11, fontweight='bold')
    if not lcz_compare.empty:
        lc = lcz_compare.copy()
        cols = ['LCZ', 'CHVI_2018', 'CHVI_2024', 'Delta_CHVI']
        lc = lc[[c for c in cols if c in lc.columns]]
        t2 = ax.table(cellText=[lc.columns.tolist()] + lc.round(3).values.tolist(), loc='upper center', bbox=[0.05, 0.20, 0.90, 0.38])
        t2.auto_set_font_size(False); t2.set_fontsize(8); t2.scale(1, 1.3)
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # Page 2
    fig, ax = plt.subplots(figsize=(8.5, 11)); ax.axis('off')
    ax.text(0.5, 0.95, "METHODOLOGICAL NOTES & INTERPRETATION", fontsize=14, fontweight='bold', ha='center', fontname='Arial')
    notes = (
        "1. COMPARABILITY\n"
        "   - Both years use identical buffer radii (250/500/1000/1500/2000 m).\n"
        "   - LCZ mapping harmonized across epochs (2014 vs 2018-2024 codes).\n"
        "   - SVI and TSI computed with the same formulas and equal weights.\n\n"
        "2. UNCERTAINTY\n"
        "   - Bootstrap 95% CIs reported per LCZ (2000 resamples).\n"
        "   - Classes with n < 5 carry LOW reliability (wide CIs).\n"
        "   - Singletons (n=1) are descriptive only and excluded from ranking.\n\n"
        "3. WEIGHT SENSITIVITY\n"
        "   - Equal (0.5/0.5), TSI-dominant (0.3/0.7), SVI-dominant (0.7/0.3).\n"
        "   - PCA-derived weights and their bootstrap 95% CI reported.\n"
        "   - Spearman rho >= 0.8 indicates robust LCZ ordering.\n\n"
        "4. MAUP ROBUSTNESS\n"
        "   - Spearman rank correlations of LCZ CHVI across radii vs 500 m.\n"
        "   - High rho (>0.8) indicates rankings are stable to buffer choice.\n\n"
        "5. INTERPRETATION\n"
        "   - Positive Delta_CHVI indicates increased vulnerability in 2024.\n"
        "   - Negative Delta_CHVI indicates improvement (e.g., greening)."
    )
    ax.text(0.08, 0.88, notes, fontsize=10, verticalalignment='top', fontname='Arial')
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

print("\n=================================================================")
print("  ALL PROCESSES SUCCESSFULLY COMPLETED FOR 2018 AND 2024!")
print(f"  Outputs saved in: {OUTPUT_DIR}")
print("=================================================================")
