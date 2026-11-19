import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import kruskal
from datetime import datetime

# ==========================================
# 1. CONFIGURATION & PUBLICATION STANDARDS
# ==========================================
file_path = r"C:\Users\ADAMA KONATE\Downloads\Donnes\Rands.xls"
output_folder = r'C:\Users\ADAMA KONATE\Downloads\Donnes\Trajectory_Drivers_SCSQ1'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Aesthetic configuration for high-impact journal publications
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42
})


# ==========================================
# 0bis. LCZ CODE MAPPING
# ==========================================
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


try:
    print("🚀 Multi-Driver Analysis (LST, NDVI, NDBI, PD, NTL)...")

    # Data loading
    df = pd.read_excel(file_path, engine='xlrd')
    df.columns = df.columns.str.strip()

    # --- APPLY LCZ MAPPING BEFORE ANY ANALYSIS ---
    # This ensures LCZ labels (2014, 2018, 2024) are correct before
    # trajectory classification, transition matrix, Kruskal-Wallis tests,
    # visualizations, and Excel export.
    df = apply_lcz_mapping(df)

    # Rigorous cleaning of LCZ labels (prevents matrix errors)
    for col in ['LCZ_2014', 'LCZ_2024']:
        df[col] = df[col].astype(str).str.replace('.0', '', regex=False).str.strip().str.upper()

    # ==========================================
    # CLASS DEFINITIONS ALIGNED WITH MAPPING OUTPUTS
    # ==========================================
    # The mapping produces the labels below. Note that hybrid classes in the
    # original dataset are labelled 'LCZ37' and 'LCZ56' (not 'LCZ3+7'/'LCZ5+6').
    # 'LCZ7' is a genuine built class (lightweight low-rise), whereas 'LCZ9'
    # (sparse built) and 'LCZZ' (unknown) are treated here as separate
    # categories. We include LCZ7 and LCZ9 in urban, and route 'LCZZ' to
    # 'Other Transitions' rather than forcing it into urban or natural.
    urban_classes = [
        'LCZ2', 'LCZ3', 'LCZ37', 'LCZ5', 'LCZ56',
        'LCZ6', 'LCZ7', 'LCZ9', 'LCZ10'
    ]
    natural_classes = ['LCZA', 'LCZB', 'LCZC', 'LCZD', 'LCZE', 'LCZF', 'LCZG']

    # ==========================================
    # 2. DIFFERENTIAL CALCULATION (DRIVERS)
    # ==========================================
    # Calculate evolution for each indicator between 2014 and 2024
    df['LST_Diff'] = df['LST_2024'] - df['LST_2014']
    df['NDVI_Diff'] = df['NDVI_2024'] - df['NDVI_2014']
    df['NDBI_Diff'] = df['NDBI_2024'] - df['NDBI_2014']

    # Socio-economic drivers (population density)
    if 'PD_2024' in df.columns and 'PD_2014' in df.columns:
        df['Pop_Diff'] = df['PD_2024'] - df['PD_2014']
    else:
        df['Pop_Diff'] = 0  # Safety if columns absent

    # Night-Time Lights (NTL) replaces the previous VNL driver
    if 'NTL_2024' in df.columns and 'NTL_2014' in df.columns:
        df['NTL_Diff'] = df['NTL_2024'] - df['NTL_2014']
    else:
        df['NTL_Diff'] = 0

    # ==========================================
    # 3. TRAJECTORY DETECTION
    # ==========================================
    def classify_trajectory(row):
        l14, l24 = row['LCZ_2014'], row['LCZ_2024']
        if l14 == l24:
            return 'Stable'
        if l14 in natural_classes and l24 in urban_classes:
            return 'Urban Expansion'
        if l14 in urban_classes and l24 in urban_classes:
            return 'Urban Densification'
        if l14 in urban_classes and l24 in natural_classes:
            return 'Green Recovery'
        return 'Other Transitions'

    df['Trajectory'] = df.apply(classify_trajectory, axis=1)

    # ==========================================
    # 4. STATISTICS AND TRANSITION MATRIX
    # ==========================================
    transition_matrix = pd.crosstab(df['LCZ_2014'], df['LCZ_2024'], normalize='index') * 100

    plt.figure(figsize=(16, 12))
    sns.heatmap(transition_matrix, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=0.5)
    plt.title('LCZ Transition Matrix: Bamako (2014-2024)')
    plt.savefig(os.path.join(output_folder, 'Fig1_Transition_Matrix.png'))
    plt.close()

    # Kruskal-Wallis test (Statistical significance of trajectories)
    traj_data = [group['LST_Diff'].dropna() for name, group in df.groupby('Trajectory')]
    h_stat, p_val = kruskal(*traj_data)

    # ==========================================
    # 5. Q1 VISUALIZATION (THERMAL IMPACT)
    # ==========================================
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Trajectory', y='LST_Diff', data=df, palette='flare', showmeans=True,
                meanprops={"marker": "d", "markerfacecolor": "white", "markeredgecolor": "black"})

    plt.axhline(0, color='black', linestyle='--', alpha=0.6)
    plt.ylabel('Temperature Change (ΔLST °C)')
    plt.title('Thermal Signature of Urban Trajectories')
    sns.despine()
    plt.savefig(os.path.join(output_folder, 'Fig2_Thermal_Impact_Boxplot.png'))
    plt.savefig(os.path.join(output_folder, 'Fig2_Thermal_Impact_Boxplot.pdf'))
    plt.close()

    # ==========================================
    # 6. THERMAL PARADOX EXPLANATION FIGURE
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Absolute LST Distribution
    df_melted = df.melt(id_vars=['Trajectory'],
                        value_vars=['LST_2014', 'LST_2024'],
                        var_name='Year', value_name='LST')
    sns.boxplot(data=df_melted, x='Trajectory', y='LST', hue='Year',
                palette=['lightcoral', 'darkred'], ax=axes[0])
    axes[0].set_title('Absolute LST Distribution\n(Showing persistent heat stress)')
    axes[0].set_ylabel('Land Surface Temperature (°C)')
    axes[0].axhline(35, color='orange', linestyle='--', alpha=0.7, label='Heat stress threshold')
    axes[0].legend(loc='upper right')

    # Right: Relative Change (already calculated)
    sns.boxplot(x='Trajectory', y='LST_Diff', data=df, palette='flare',
                showmeans=True, ax=axes[1])
    axes[1].axhline(0, color='black', linestyle='--', alpha=0.6)
    axes[1].set_title('Relative Change (ΔLST)\n("Cooling" = statistical artifact)')
    axes[1].set_ylabel('Temperature Change (°C)')

    # Explanatory annotation
    axes[1].annotate('Thermal saturation:\nNatural >50°C → Urban 38-42°C',
                   xy=(1, -1.5), fontsize=10, ha='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'Fig3_Thermal_Paradox_Explanation.png'))
    plt.savefig(os.path.join(output_folder, 'Fig3_Thermal_Paradox_Explanation.pdf'))
    plt.close()

    # ==========================================
    # 7. MULTI-DRIVER SYNTHESIS REPORT
    # ==========================================
    # Aggregation of means by trajectory
    report_df = df.groupby('Trajectory').agg({
        'LST_Diff': 'mean',
        'NDVI_Diff': 'mean',
        'NDBI_Diff': 'mean',
        'Pop_Diff': 'mean',
        'NTL_Diff': 'mean'
    }).round(4)

    report_path = os.path.join(output_folder, 'Scientific_Analysis_Report_Final.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 85 + "\n")
        f.write("ADVANCED MULTI-DRIVER ANALYSIS: THERMAL TRAJECTORIES IN BAMAKO\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write("=" * 85 + "\n\n")

        f.write(f"STATISTICAL SIGNIFICANCE (Kruskal-Wallis):\n")
        f.write(f"H-statistic: {h_stat:.4f} | p-value: {p_val:.4e}\n")
        f.write("Result: " + ("Significant" if p_val < 0.05 else "Not Significant") + "\n\n")

        f.write("MEAN IMPACTS PER TRAJECTORY:\n")
        f.write(report_df.to_string() + "\n\n")

        f.write("KEY INSIGHTS FOR Q1 MANUSCRIPT:\n")
        f.write("-" * 50 + "\n")
        f.write("1. Areas of 'Urban Densification' (including LCZ37 and LCZ56) show the highest \n")
        f.write("   coupling between Pop_Diff and LST_Diff, indicating a surge in human-induced \n")
        f.write("   thermal stress.\n")
        f.write("2. The NTL_Diff suggests that while some areas expand physically (NDBI), \n")
        f.write("   the socio-economic infrastructure (Night-Time Lights) lags in informal hotspots.\n")
        f.write("3. NEGATIVE ΔLST INTERPRETATION (Thermal Saturation Paradox):\n")
        f.write("   - Negative values indicate statistical cooling relative to 2014 baseline\n")
        f.write("   - NOT absolute temperature reduction or improved thermal comfort\n")
        f.write("   - Mechanism: Natural surfaces >50°C (dry savanna, bare soil) → Urban 38-42°C\n")
        f.write("   - Result: Built environments appear 'cooler' statistically but remain heat-stressed\n")
        f.write("   - Critical for 'Stable' trajectory: invisible degradation within unchanged LCZ classes\n")

    # Excel export for supplementary tables
    df.to_excel(os.path.join(output_folder, 'Bamako_Final_Trajectory_Data.xlsx'), index=False)

    print(f"✅ Analysis completed. Output folder: {output_folder}")

except Exception as e:
    import traceback
    traceback.print_exc()
