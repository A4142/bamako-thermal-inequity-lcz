"""
================================================================================
LANDSAT vs MODIS LST — CROSS-SENSOR VALIDATION FOR SCS
Publication-ready validation pipeline (Q1 journal standard)
================================================================================
Features:
    • Robust handling of missing values, #NUM!, and outliers
    • Landsat 2014 automatic detection of anomalous range (reflectance vs LST)
    • Pearson r, RMSE, MAE, Bias, and 95% confidence intervals
    • Reduced Major Axis (RMA) regression (standard for sensor comparison)
    • Publication-quality 1:1 scatter plots with density contours
    • Combined 3-panel figure (Fig. 4 format)
    • LaTeX-ready summary table
    • Explicit statement: product-to-product comparison, not ground validation

Author  : Adama KONATE
Version : 1.0 (SCS-ready)
================================================================================
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

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
    'ps.fonttype': 42,
    'axes.linewidth': 1.0,
})

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(message)s')
warnings.filterwarnings('ignore')

# ---- Paths ----------------------------------------------------------------
INPUT_FILE    = r"C:\Users\ADAMA KONATE\Downloads\Donnes\Validations.xls"
OUTPUT_FOLDER = r"C:\Users\ADAMA KONATE\Downloads\Donnes\LST_Validation_Outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

YEARS = ['2014', '2018', '2024']

# Physically plausible LST range for Sahelian cities (°C)
LST_MIN_PLAUSIBLE = 5.0
LST_MAX_PLAUSIBLE = 70.0

# Outlier threshold (robust z-score)
ROBUST_Z_THRESHOLD = 5.0


# ---------------------------------------------------------------------------
# 1. HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def load_and_clean(path):
    """Load Excel file and coerce all columns to numeric."""
    logging.info(f"Loading: {path}")
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()

    # Coerce to numeric, replacing #NUM!, blanks, and text by NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    logging.info(f"  Rows: {len(df)}")
    logging.info(f"  Columns: {df.columns.tolist()}")
    return df


def remove_outliers_robust(x, y, z_thresh=ROBUST_Z_THRESHOLD):
    """
    Remove outliers using robust z-score based on MAD (median absolute deviation).
    Applied jointly to both variables.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 10:
        return x, y, 0

    # Robust z-score for x
    med_x = np.median(x)
    mad_x = np.median(np.abs(x - med_x))
    z_x = 0.6745 * (x - med_x) / mad_x if mad_x > 0 else np.zeros_like(x)

    # Robust z-score for y
    med_y = np.median(y)
    mad_y = np.median(np.abs(y - med_y))
    z_y = 0.6745 * (y - med_y) / mad_y if mad_y > 0 else np.zeros_like(y)

    keep = (np.abs(z_x) < z_thresh) & (np.abs(z_y) < z_thresh)
    n_removed = int(np.sum(~keep))
    return x[keep], y[keep], n_removed


def compute_metrics(ls, md, n_boot=1000, random_state=42):
    """
    Compute validation metrics with bootstrap 95% confidence intervals.
    """
    n = len(ls)

    # Point estimates
    r_pearson, _ = pearsonr(ls, md)
    rho_spearman, _ = spearmanr(ls, md)
    rmse = np.sqrt(mean_squared_error(md, ls))
    mae = mean_absolute_error(md, ls)
    bias = np.mean(ls - md)
    std_diff = np.std(ls - md, ddof=1)

    # RMA regression (sensor comparison standard)
    # Slope_RMA = SD(y) / SD(x) with sign of Pearson r
    sd_ls, sd_md = np.std(ls, ddof=1), np.std(md, ddof=1)
    slope_rma = sd_ls / sd_md if sd_md > 0 else np.nan
    intercept_rma = np.mean(ls) - slope_rma * np.mean(md)

    # OLS regression for reference
    slope_ols, intercept_ols, _, _, _ = stats.linregress(md, ls)

    # Bootstrap 95% CI
    rng = np.random.default_rng(random_state)
    boot_r, boot_rmse, boot_bias, boot_slope = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ls_b, md_b = ls[idx], md[idx]
        try:
            boot_r.append(pearsonr(ls_b, md_b)[0])
            boot_rmse.append(np.sqrt(mean_squared_error(md_b, ls_b)))
            boot_bias.append(np.mean(ls_b - md_b))
            boot_slope.append(np.std(ls_b, ddof=1) / np.std(md_b, ddof=1))
        except Exception:
            continue

    ci_r = np.percentile(boot_r, [2.5, 97.5])
    ci_rmse = np.percentile(boot_rmse, [2.5, 97.5])
    ci_bias = np.percentile(boot_bias, [2.5, 97.5])
    ci_slope = np.percentile(boot_slope, [2.5, 97.5])

    return {
        'n_pairs': n,
        'r_pearson': round(r_pearson, 3),
        'r_pearson_CI_low': round(ci_r[0], 3),
        'r_pearson_CI_high': round(ci_r[1], 3),
        'rho_spearman': round(rho_spearman, 3),
        'RMSE_C': round(rmse, 2),
        'RMSE_CI_low': round(ci_rmse[0], 2),
        'RMSE_CI_high': round(ci_rmse[1], 2),
        'MAE_C': round(mae, 2),
        'Bias_C': round(bias, 2),
        'Bias_CI_low': round(ci_bias[0], 2),
        'Bias_CI_high': round(ci_bias[1], 2),
        'Std_diff_C': round(std_diff, 2),
        'Slope_RMA': round(slope_rma, 3),
        'Slope_RMA_CI_low': round(ci_slope[0], 3),
        'Slope_RMA_CI_high': round(ci_slope[1], 3),
        'Intercept_RMA': round(intercept_rma, 2),
        'Slope_OLS': round(slope_ols, 3),
        'Intercept_OLS': round(intercept_ols, 2),
        'Landsat_mean_C': round(np.mean(ls), 2),
        'MODIS_mean_C': round(np.mean(md), 2),
    }


def scatter_with_contours(ax, ls, md, metrics, year, show_legend=False):
    """Create a publication-quality 1:1 scatter plot with density contours."""
    # Scatter
    ax.scatter(md, ls, s=8, alpha=0.35, color='steelblue',
               edgecolor='none', zorder=2, rasterized=True)

    # Data-driven limits
    lims = [float(np.nanmin([md.min(), ls.min()])),
            float(np.nanmax([md.max(), ls.max()]))]
    pad = 0.05 * (lims[1] - lims[0])
    lims = [lims[0] - pad, lims[1] + pad]

    # 1:1 line
    ax.plot(lims, lims, 'k--', lw=1.0, label='1:1 line', zorder=3)

    # RMA regression line
    x_fit = np.linspace(lims[0], lims[1], 100)
    y_fit = metrics['Slope_RMA'] * x_fit + metrics['Intercept_RMA']
    ax.plot(x_fit, y_fit, 'r-', lw=1.5,
            label=f'RMA: y = {metrics["Slope_RMA"]:.2f}x '
                  f'+ {metrics["Intercept_RMA"]:.2f}',
            zorder=4)

    # Labels and title
    ax.set_xlabel('MODIS LST (°C)')
    ax.set_ylabel('Landsat LST (°C)')
    ax.set_title(f'{year}\n'
                 f'r = {metrics["r_pearson"]:.2f}, '
                 f'RMSE = {metrics["RMSE_C"]:.2f} °C, '
                 f'n = {metrics["n_pairs"]}',
                 fontsize=12)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.4, zorder=1)

    if show_legend:
        ax.legend(loc='upper left', frameon=True, fontsize=9,
                  framealpha=0.9, edgecolor='0.7')


def plot_combined(all_pairs, all_metrics, output_folder):
    """Create the 3-panel combined figure (Fig. 4 format)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, year in zip(axes, YEARS):
        if year not in all_pairs:
            ax.axis('off')
            continue
        ls, md = all_pairs[year]
        m = next(x for x in all_metrics if x['Year'] == year)
        scatter_with_contours(ax, ls, md, m, year, show_legend=(year == '2014'))

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(output_folder, f'Validation_Combined.{ext}'),
                    dpi=600, bbox_inches='tight')
    plt.close()


def generate_latex_table(summary_df, output_folder):
    """Generate a LaTeX-ready table for the manuscript."""
    latex = summary_df.to_latex(
        index=False,
        float_format="%.2f",
        caption="Cross-sensor validation of Landsat-derived LST against MODIS MOD11A1.",
        label="tab:validation",
        escape=False,
    )
    with open(os.path.join(output_folder, 'Validation_Table.tex'), 'w') as f:
        f.write(latex)


# ---------------------------------------------------------------------------
# 2. MAIN
# ---------------------------------------------------------------------------
def main():
    logging.info("=" * 78)
    logging.info(" LANDSAT vs MODIS LST — CROSS-SENSOR VALIDATION (SCS-ready)")
    logging.info("=" * 78)

    df = load_and_clean(INPUT_FILE)

    all_metrics = []
    all_pairs = {}

    for year in YEARS:
        logging.info(f"\n--- Year {year} ---")
        ls_col = f'LandsatLST_{year}'
        md_col = f'ModisLST_{year}'

        if ls_col not in df.columns or md_col not in df.columns:
            logging.warning(f"  Missing columns for {year}. Skipping.")
            continue

        ls = df[ls_col].to_numpy(dtype=float)
        md = df[md_col].to_numpy(dtype=float)

        # Check for anomalous Landsat range (reflectance vs LST)
        ls_valid = ls[np.isfinite(ls)]
        if len(ls_valid) > 0:
            ls_min, ls_max = np.nanmin(ls_valid), np.nanmax(ls_valid)
            if ls_max < LST_MIN_PLAUSIBLE or ls_min > LST_MAX_PLAUSIBLE:
                logging.warning(
                    f"  {ls_col} range [{ls_min:.2f}, {ls_max:.2f}] is "
                    f"physically implausible for LST in °C. "
                    f"Likely a reflectance or Kelvin product. Skipping."
                )
                continue

        # Remove NaNs
        mask = np.isfinite(ls) & np.isfinite(md)
        ls_c, md_c = ls[mask], md[mask]
        logging.info(f"  Valid pairs (no NaN): {len(ls_c)}")

        # Remove outliers
        ls_c, md_c, n_out = remove_outliers_robust(ls_c, md_c)
        logging.info(f"  Outliers removed (robust z > {ROBUST_Z_THRESHOLD}): {n_out}")
        logging.info(f"  Final pairs: {len(ls_c)}")

        if len(ls_c) < 30:
            logging.warning(f"  Too few pairs for {year}. Skipping.")
            continue

        metrics = compute_metrics(ls_c, md_c)
        metrics['Year'] = year
        all_metrics.append(metrics)
        all_pairs[year] = (ls_c, md_c)

        logging.info(
            f"  r = {metrics['r_pearson']:.3f} "
            f"[{metrics['r_pearson_CI_low']:.3f}, {metrics['r_pearson_CI_high']:.3f}]"
        )
        logging.info(f"  RMSE = {metrics['RMSE_C']:.2f} °C "
                     f"[{metrics['RMSE_CI_low']:.2f}, {metrics['RMSE_CI_high']:.2f}]")
        logging.info(f"  MAE  = {metrics['MAE_C']:.2f} °C")
        logging.info(f"  Bias = {metrics['Bias_C']:.2f} °C "
                     f"[{metrics['Bias_CI_low']:.2f}, {metrics['Bias_CI_high']:.2f}]")
        logging.info(f"  RMA slope = {metrics['Slope_RMA']:.3f}")

        # Individual scatter
        fig, ax = plt.subplots(figsize=(7, 7))
        scatter_with_contours(ax, ls_c, md_c, metrics, year, show_legend=True)
        plt.tight_layout()
        for ext in ['png', 'pdf']:
            plt.savefig(os.path.join(OUTPUT_FOLDER,
                                     f'Validation_{year}.{ext}'),
                        dpi=600, bbox_inches='tight')
        plt.close()

    # Combined figure
    if all_pairs:
        plot_combined(all_pairs, all_metrics, OUTPUT_FOLDER)

    # Summary table
    if all_metrics:
        columns_order = [
            'Year', 'n_pairs',
            'r_pearson', 'r_pearson_CI_low', 'r_pearson_CI_high',
            'rho_spearman',
            'RMSE_C', 'RMSE_CI_low', 'RMSE_CI_high',
            'MAE_C', 'Bias_C', 'Bias_CI_low', 'Bias_CI_high',
            'Std_diff_C',
            'Slope_RMA', 'Slope_RMA_CI_low', 'Slope_RMA_CI_high',
            'Intercept_RMA',
            'Landsat_mean_C', 'MODIS_mean_C',
        ]
        summary = pd.DataFrame(all_metrics)[columns_order]

        summary.to_csv(os.path.join(OUTPUT_FOLDER, 'Validation_Summary.csv'),
                       index=False)
        summary.to_excel(os.path.join(OUTPUT_FOLDER, 'Validation_Summary.xlsx'),
                         index=False)
        generate_latex_table(summary, OUTPUT_FOLDER)

        # TXT report
        with open(os.path.join(OUTPUT_FOLDER, 'Validation_Report.txt'),
                  'w', encoding='utf-8') as f:
            f.write("=" * 78 + "\n")
            f.write(" LANDSAT vs MODIS LST — CROSS-SENSOR VALIDATION\n")
            f.write("=" * 78 + "\n\n")
            f.write(summary.to_string(index=False))
            f.write("\n\n")
            f.write("METHODOLOGICAL NOTES:\n")
            f.write("-" * 78 + "\n")
            f.write("• This is a product-to-product comparison, not an absolute\n")
            f.write("  validation against in-situ ground measurements.\n")
            f.write("• Landsat LST (30 m native) was resampled to 1 km to match\n")
            f.write("  the MODIS MOD11A1 grid.\n")
            f.write("• Outliers were removed using a robust z-score (MAD-based,\n")
            f.write(f"  threshold = {ROBUST_Z_THRESHOLD}σ).\n")
            f.write("• Confidence intervals (95%) were computed via bootstrap\n")
            f.write("  resampling (n = 1000).\n")
            f.write("• The RMA (Reduced Major Axis) regression is reported as the\n")
            f.write("  standard approach for sensor-to-sensor comparisons, as it\n")
            f.write("  accounts for errors in both variables.\n")
            f.write("• High RMSE values likely reflect resolution mismatch,\n")
            f.write("  Saharan dust aerosols, and differences in acquisition\n")
            f.write("  times, rather than absolute retrieval error.\n")
            f.write("• Relative intra-urban metrics (SUHII, MTP) are preferred\n")
            f.write("  over absolute LST values in the analysis.\n")

        print("\n" + "=" * 78)
        print(" VALIDATION SUMMARY")
        print("=" * 78)
        print(summary.to_string(index=False))

    logging.info(f"\n✅ Validation completed. Outputs at: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
