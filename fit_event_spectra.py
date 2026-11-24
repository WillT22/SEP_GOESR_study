#%% Import necessary libraries
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon, mannwhitneyu, spearmanr

FONT_SIZE = 20
plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': FONT_SIZE + 2,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'legend.fontsize': FONT_SIZE,
    'figure.titlesize': FONT_SIZE + 4,
    'legend.markerscale': 1.0
})

BASE_DIR = Path(r"C:\Users\wzt0020\Box\SEP_Box\SGPS_data")

# Import event peak data
df_event_peaks = pd.read_parquet(BASE_DIR / "SEP_event_peaks_summary.parquet")

# Import updated SEP event catalog
catalog_path = Path(r"C:\Users\wzt0020\Box\SEP_Box\SGPS_data\updated_sep_catalog.csv")
sep_catalog = pd.read_csv(catalog_path, encoding='utf-8-sig')
sep_catalog.columns = sep_catalog.columns.str.strip()

# Ensure Begin_Time_UTC is UTC-aware and create the matching folder key
sep_catalog['Begin_Time_UTC'] = pd.to_datetime(sep_catalog['Begin_Time_UTC'], utc=True)
sep_catalog['Event_Folder'] = 'Event_' + sep_catalog['Begin_Time_UTC'].dt.strftime('%Y%m%d_%H%M')

# Locate the '>10 MeV Maximum (pfu)' column regardless of trailing arrow characters
pfu_matches = [c for c in sep_catalog.columns if '10 MeV Maximum' in c and 'pfu' in c.lower()]
if not pfu_matches:
    raise KeyError(f"PFU column not found. Available columns: {list(sep_catalog.columns)}")

raw_pfu_col = pfu_matches[0]
sep_catalog['Max_PFU_10MeV'] = pd.to_numeric(
    sep_catalog[raw_pfu_col].astype(str).str.replace(',', '', regex=False).str.strip(),
    errors='coerce'
)

# Define S-scale
def classify_s_scale(pfu):
    """Classifies the event into the NOAA Solar Radiation Storm Scale based on peak >10 MeV pfu."""
    if pd.isna(pfu):
        return "Unknown"
    if pfu >= 100000:
        return "S5"
    elif pfu >= 10000:
        return "S4"
    elif pfu >= 1000:
        return "S3"
    elif pfu >= 100:
        return "S2"
    elif pfu >= 10:
        return "S1"
    return "Below S1"

# Fast O(1) lookup from Event_Folder -> Max_PFU_10MeV
pfu_lookup = (
    sep_catalog.drop_duplicates(subset='Event_Folder')
    .set_index('Event_Folder')['Max_PFU_10MeV']
    .to_dict()
)

#%% Create Ellison & Ramaty fit
# Original formulation: https://scixplorer.org/abs/1985ApJ...298..400E/abstract?fromADS=true&ads_compat=1
# GLE Study: https://link.springer.com/article/10.1007/s11214-012-9884-2
# =============================================================================
#    Ellison & Ramaty (1985): j(E) = j0 * E^(-gamma) * exp(-E / E0)
# =============================================================================
def ellison_ramaty_log10(E, log10_j0, gamma, E0):
    """Returns log10(Flux) for the Ellison & Ramaty model."""
    E = np.asarray(E, dtype=float)
    return log10_j0 - gamma * np.log10(E) - (E / E0) * np.log10(np.e)

def ellison_ramaty(E, log10_j0, gamma, E0):
    """Returns linear Flux for the Ellison & Ramaty model."""
    return 10.0 ** ellison_ramaty_log10(E, log10_j0, gamma, E0)


#%% Create Band Function Fit
# Original formulation: https://scixplorer.org/abs/1993ApJ...413..281B/abstract?fromADS=true&ads_compat=1
# SEPEM Study https://www.aanda.org/articles/aa/full_html/2024/11/aa50046-24/aa50046-24.html
# =============================================================================
#    Band (1993) Double Power Law
#    Uses delta_gamma = (gamma_b - gamma_a) > 0 to guarantee gamma_b > gamma_a
# =============================================================================
def band_log10(E, log10_j0, gamma_a, delta_gamma, E0):
    """
    Returns log10(Flux) for the Band function.
    Parameters:
        log10_j0    : log10 normalization constant
        gamma_a     : low-energy spectral index
        delta_gamma : gamma_b - gamma_a (must be > 0 so high-energy slope is steeper)
        E0          : rollover energy (MeV); break energy is E_b = delta_gamma * E0
    """
    E = np.asarray(E, dtype=float)
    gamma_b = gamma_a + delta_gamma
    E_break = delta_gamma * E0

    low_E = log10_j0 - gamma_a * np.log10(E) - (E / E0) * np.log10(np.e)
    high_E = (log10_j0 - gamma_b * np.log10(E) + delta_gamma * np.log10(E_break) - delta_gamma * np.log10(np.e))

    return np.where(E <= E_break, low_E, high_E)

def band(E, log10_j0, gamma_a, delta_gamma, E0):
    """Returns linear Flux for the Band function."""
    return 10.0 ** band_log10(E, log10_j0, gamma_a, delta_gamma, E0)


#%% Fit Absolute Max Spectra (Weighted Chi-Squared GOF + Scale-Invariant AICc/BIC)
APPLY_UNCERT_FLOOR = False  # Set True to enforce a minimum fractional uncertainty
UNCERT_FLOOR_FRAC = 0.10    # 10% floor when APPLY_UNCERT_FLOOR is True

def compute_chi2_gof(residuals, sigma, k):
    """Returns (chi2_val, chi2_red, p_value) for weighted residuals."""
    n = len(residuals)
    dof = n - k
    if dof <= 0 or np.any(sigma <= 0):
        return np.nan, np.nan, np.nan
    chi2_val = np.sum((residuals / sigma) ** 2)
    chi2_red = chi2_val / dof
    p_val = chi2.sf(chi2_val, dof)
    return chi2_val, chi2_red, p_val

def compute_bic_scale_invariant(chi2_val, n, k):
    """Scale-invariant weighted BIC: n*ln(chi2/n) + k*ln(n)."""
    if pd.isna(chi2_val) or chi2_val <= 0 or n <= k:
        return np.nan
    return n * np.log(chi2_val / n) + k * np.log(n)

def compute_aicc_scale_invariant(chi2_val, n, k):
    """Scale-invariant weighted AICc: n*ln(chi2/n) + 2k + 2k(k+1)/(n-k-1)."""
    if pd.isna(chi2_val) or chi2_val <= 0 or n <= k + 1:
        return np.nan
    aic = n * np.log(chi2_val / n) + 2 * k
    return aic + (2 * k * (k + 1)) / (n - k - 1)


# --- Mode Toggle ---
GROUP_BY_ORIENTATION_ONLY = False  # True = 2 spectra/event (East & West pooled); False = per (Sat, Orientation)
mode_tag = "grouped_EW" if GROUP_BY_ORIENTATION_ONLY else "per_sensor"

fit_records = []
peak_type, uncert_col = 'TotalMaxFlux', 'TotalMaxUncert'

group_cols = ['Event', 'Orientation'] if GROUP_BY_ORIENTATION_ONLY else ['Event', 'Satellite', 'Orientation']

for group_key, df_group in df_event_peaks.groupby(group_cols):
    if GROUP_BY_ORIENTATION_ONLY:
        event_name, orientation = group_key
        sat = 'ALL'
    else:
        event_name, sat, orientation = group_key

    # Pull and classify the pfu s-scale
    event_pfu = pfu_lookup.get(event_name, np.nan)
    event_s_scale = classify_s_scale(event_pfu)
    
    # Exclude background channels so flat noise floors don't skew the high-energy tail
    df_real = df_group[~df_group['IsBackground'].fillna(False)].sort_values('EffectiveEnergy')

    df_valid = df_real.dropna(subset=['EffectiveEnergy', peak_type, uncert_col])
    df_valid = df_valid[
        (df_valid['EffectiveEnergy'] > 0) &
        (df_valid[peak_type] > 0) &
        (df_valid[uncert_col] > 0)
    ]

    E = df_valid['EffectiveEnergy'].to_numpy(dtype=float)
    flux = df_valid[peak_type].to_numpy(dtype=float)
    uncert = df_valid[uncert_col].to_numpy(dtype=float)
    n_pts = len(E)

    if APPLY_UNCERT_FLOOR:
        uncert = np.maximum(uncert, UNCERT_FLOOR_FRAC * flux)

    row = {
        'Event': event_name,
        'Satellite': sat,
        'Orientation': orientation,
        'Max_PFU_10MeV': event_pfu,
        'S_Scale': event_s_scale,
        'PeakType': peak_type,
        'N_Points': n_pts,
        # Ellison & Ramaty (k=3)
        'ER_log10_j0': np.nan, 'ER_gamma': np.nan, 'ER_E0': np.nan,
        'ER_rmse': np.nan, 'ER_chi2': np.nan, 'ER_chi2_red': np.nan, 'ER_pval': np.nan,
        'ER_bic': np.nan, 'ER_aicc': np.nan,
        # Band (k=4)
        'Band_log10_j0': np.nan, 'Band_gamma_a': np.nan, 'Band_gamma_b': np.nan,
        'Band_E0': np.nan, 'Band_E_break': np.nan,
        'Band_rmse': np.nan, 'Band_chi2': np.nan, 'Band_chi2_red': np.nan, 'Band_pval': np.nan,
        'Band_bic': np.nan, 'Band_aicc': np.nan,
        # Model comparison
        'BestModel_BIC': None, 'BestModel_AICc': None,
    }

    if n_pts < 4:
        fit_records.append(row)
        continue

    log_flux = np.log10(flux)
    sigma_log = uncert / (flux * np.log(10.0))
    er_success = False
    band_success = False

    # 1. Fit Ellison & Ramaty (k=3 -> needs >= 4 points)
    try:
        p0_er = [log_flux[0], 2.0, 30.0]
        bounds_er = ([-10.0, 0.0, 0.5], [15.0, 10.0, 2000.0])
        popt_er, _ = curve_fit(
            ellison_ramaty_log10, E, log_flux,
            sigma=sigma_log, absolute_sigma=True,
            p0=p0_er, bounds=bounds_er, maxfev=10000
        )
        resid_er = log_flux - ellison_ramaty_log10(E, *popt_er)
        er_chi2, er_chi2_red, er_pval = compute_chi2_gof(resid_er, sigma_log, k=3)

        row['ER_log10_j0'], row['ER_gamma'], row['ER_E0'] = popt_er
        row['ER_rmse'] = np.sqrt(np.mean(resid_er ** 2))
        row['ER_chi2'] = er_chi2
        row['ER_chi2_red'] = er_chi2_red
        row['ER_pval'] = er_pval
        row['ER_bic'] = compute_bic_scale_invariant(er_chi2, n_pts, k=3)
        row['ER_aicc'] = compute_aicc_scale_invariant(er_chi2, n_pts, k=3)
        er_success = True
    except RuntimeError:
        pass

    # 2. Fit Band (k=4 -> needs >= 5 points)
    if n_pts >= 5:
        try:
            p0_band = [log_flux[0], 1.5, 2.0, 30.0]
            bounds_band = ([-10.0, 0.0, 0.05, 0.5], [15.0, 8.0, 12.0, 2000.0])
            popt_b, _ = curve_fit(
                band_log10, E, log_flux,
                sigma=sigma_log, absolute_sigma=True,
                p0=p0_band, bounds=bounds_band, maxfev=20000
            )
            resid_b = log_flux - band_log10(E, *popt_b)
            band_chi2, band_chi2_red, band_pval = compute_chi2_gof(resid_b, sigma_log, k=4)
            log10_j0, gamma_a, delta_gamma, E0 = popt_b

            row['Band_log10_j0'] = log10_j0
            row['Band_gamma_a'] = gamma_a
            row['Band_gamma_b'] = gamma_a + delta_gamma
            row['Band_E0'] = E0
            row['Band_E_break'] = delta_gamma * E0
            row['Band_rmse'] = np.sqrt(np.mean(resid_b ** 2))
            row['Band_chi2'] = band_chi2
            row['Band_chi2_red'] = band_chi2_red
            row['Band_pval'] = band_pval
            row['Band_bic'] = compute_bic_scale_invariant(band_chi2, n_pts, k=4)
            row['Band_aicc'] = compute_aicc_scale_invariant(band_chi2, n_pts, k=4)
            band_success = True
        except RuntimeError:
            pass

    # 3. Select Best Model by lower BIC
    er_bic, band_bic = row['ER_bic'], row['Band_bic']
    if pd.notna(er_bic) and pd.notna(band_bic):
        row['BestModel_BIC'] = 'Band' if band_bic < er_bic else 'EllisonRamaty'
    elif er_success:
        row['BestModel_BIC'] = 'EllisonRamaty'
    elif band_success:
        row['BestModel_BIC'] = 'Band'

    # 4. Select Best Model by lower AICc
    er_aicc, band_aicc = row['ER_aicc'], row['Band_aicc']
    if pd.notna(er_aicc) and pd.notna(band_aicc):
        row['BestModel_AICc'] = 'Band' if band_aicc < er_aicc else 'EllisonRamaty'
    elif er_success:
        row['BestModel_AICc'] = 'EllisonRamaty'
    elif band_success:
        row['BestModel_AICc'] = 'Band'

    fit_records.append(row)

# Convert to DataFrame and save
df_fits = pd.DataFrame(fit_records)
df_fits.to_parquet(BASE_DIR / f"SEP_spectral_fits_{mode_tag}.parquet", index=False)
df_fits.to_csv(BASE_DIR / f"SEP_spectral_fits_{mode_tag}.csv", index=False)

print(f"Completed fits for {len(df_fits)} Absolute Max spectra (Uncert floor={APPLY_UNCERT_FLOOR}).")
print("\n--- Best Model Counts ---")
print(df_fits['BestModel_BIC'].value_counts(dropna=False))
print(df_fits['BestModel_AICc'].value_counts(dropna=False))
print("\n--- Median Reduced Chi-Squared ---")
print(f"ER median chi2_red:   {df_fits['ER_chi2_red'].median():.3f} (p >= 0.05: {(df_fits['ER_pval'].dropna() >= 0.05).mean():.1%})")
print(f"Band median chi2_red: {df_fits['Band_chi2_red'].median():.3f} (p >= 0.05: {(df_fits['Band_pval'].dropna() >= 0.05).mean():.1%})")

print("\n--- Event Counts by S-Scale (unique events) ---")
print(df_fits.drop_duplicates(subset='Event')['S_Scale'].value_counts())

#%% Plot All Absolute Max Energy Spectra with Fits (Color = East/West, Shape = Satellite)
PLOT_BOOL = True
if PLOT_BOOL:
    PLOT_BEST_FIT_ONLY = False  # Set True to only draw the AICc-winning fit

    absolute_dir = BASE_DIR / "Event_Spectra_Fits_Plots" / f"Absolute_Peak_Spectra_{mode_tag}"
    absolute_dir.mkdir(parents=True, exist_ok=True)

    orient_colors = {'East': '#1f77b4', 'West': '#ff7f0e'}  # Blue = East, Orange = West
    sat_markers = {
        'GOES16': 'o',  # Circle
        'GOES17': 's',  # Square
        'GOES18': '^',  # Triangle
        'GOES19': 'D',  # Diamond
    }
    sat_size_offset = {
        'GOES16': 0,
        'GOES17': 0,
        'GOES18': 2.5,  # Boost triangle size to match circles/squares
        'GOES19': 0,
    }
    SGPS_CHANNELS = ['P1', 'P2A', 'P2B', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8A', 'P8B', 'P8C', 'P9', 'P10']

    fit_linestyles = {
        'ER':   '-',
        'Band': '--',
    }

    fits_lookup = df_fits.set_index(['Event', 'Satellite', 'Orientation'])

    event_meta = (
        df_fits.drop_duplicates(subset='Event')
        .set_index('Event')[['Max_PFU_10MeV', 'S_Scale']]
        .to_dict('index')
    )

    print(f"Generating Absolute Peak energy spectra plots for {df_event_peaks['Event'].nunique()} events...")

    for event_name, df_event in df_event_peaks.groupby('Event'):
        print(f"  -> Plotting spectra for {event_name}...")

        fig, ax = plt.subplots(figsize=(20, 8))
        present_sats = sorted(df_event['Satellite'].unique())

        # A. Plot data points (Color = Orientation, Shape = Satellite)
        for (sat, orientation), df_group in df_event.groupby(['Satellite', 'Orientation']):
            df_group = df_group.sort_values('ChannelNum')
            color = orient_colors.get(orientation, 'black')
            marker = sat_markers.get(sat, 'o')
            ms_boost = sat_size_offset.get(sat, 0)

            is_bg = df_group['IsBackground'].fillna(False)
            df_real = df_group[~is_bg]
            df_bg = df_group[is_bg]

            if not df_bg.empty:
                ax.plot(
                    df_bg['ChannelNum'], df_bg['TotalMaxFlux'],
                    color=color, linestyle='none', marker=marker, markersize=10 + ms_boost,
                    markerfacecolor='lightgray', markeredgecolor=color, markeredgewidth=1.4,
                    alpha=0.5, zorder=5
                )

            if not df_real.empty:
                ax.plot(
                    df_real['ChannelNum'], df_real['TotalMaxFlux'],
                    color=color, linestyle='none', marker=marker, markersize=13 + ms_boost,
                    markeredgecolor='black', markeredgewidth=1.2,
                    alpha=0.85, zorder=6
                )

        # B. Overlay fits (Color = Orientation, Linestyle = Model)
        fit_group_cols = ['Orientation'] if GROUP_BY_ORIENTATION_ONLY else ['Satellite', 'Orientation']

        for fkey, df_fgrp in df_event.groupby(fit_group_cols):
            if GROUP_BY_ORIENTATION_ONLY:
                orientation = fkey[0] if isinstance(fkey, tuple) else fkey
                sat_key = 'ALL'
            else:
                sat_key, orientation = fkey

            fit_color = orient_colors.get(orientation, 'black')
            fit_key = (event_name, sat_key, orientation)
            df_real_f = df_fgrp[~df_fgrp['IsBackground'].fillna(False)].dropna(subset=['EffectiveEnergy'])

            if fit_key not in fits_lookup.index or df_real_f.empty:
                continue

            fit_row = fits_lookup.loc[fit_key]
            if isinstance(fit_row, pd.DataFrame):
                fit_row = fit_row.iloc[0]

            df_curve = (
                df_real_f.groupby('ChannelNum', as_index=False)['EffectiveEnergy']
                .mean()
                .sort_values('ChannelNum')
            )
            ch_pts = df_curve['ChannelNum'].to_numpy()
            E_pts = df_curve['EffectiveEnergy'].to_numpy(dtype=float)
            best_model = fit_row.get('BestModel_AICc', None)

            if pd.notna(fit_row['ER_log10_j0']):
                is_best = (best_model == 'EllisonRamaty')
                if not PLOT_BEST_FIT_ONLY or is_best:
                    er_flux = ellison_ramaty(E_pts, fit_row['ER_log10_j0'], fit_row['ER_gamma'], fit_row['ER_E0'])
                    ax.plot(
                        ch_pts, er_flux, color=fit_color, linestyle=fit_linestyles['ER'],
                        linewidth=2.6 if is_best else 1.5, alpha=0.9 if is_best else 0.35,
                        zorder=4 if is_best else 3
                    )

            if pd.notna(fit_row['Band_log10_j0']):
                is_best = (best_model == 'Band')
                if not PLOT_BEST_FIT_ONLY or is_best:
                    delta_gamma = fit_row['Band_gamma_b'] - fit_row['Band_gamma_a']
                    band_flux = band(E_pts, fit_row['Band_log10_j0'], fit_row['Band_gamma_a'], delta_gamma, fit_row['Band_E0'])
                    ax.plot(
                        ch_pts, band_flux, color=fit_color, linestyle=fit_linestyles['Band'],
                        linewidth=2.6 if is_best else 1.5, alpha=0.9 if is_best else 0.35,
                        zorder=4 if is_best else 3
                    )

        energy_labels = [
            f"{df_event.loc[df_event['Channel'] == ch, 'EffectiveEnergy'].mean():.1f}"
            if not df_event[df_event['Channel'] == ch].empty and 'EffectiveEnergy' in df_event.columns
            else "-"
            for ch in SGPS_CHANNELS
        ]

        ax.set_yscale('log')
        ax.set_xlim(-0.5, len(SGPS_CHANNELS) - 0.5)
        ax.set_xticks(range(len(SGPS_CHANNELS)))
        ax.set_xticklabels(energy_labels, fontweight='bold', color='black')

        for i, ch_name in enumerate(SGPS_CHANNELS):
            ax.text(i, -0.08, ch_name, transform=ax.get_xaxis_transform(),
                    ha='center', va='top', color='dimgray', fontweight='bold')

        ax.set_xlabel('Effective Energy (MeV)', fontweight='bold', labelpad=30)
        ax.set_ylabel('Proton Flux (#/cm²/s/sr/MeV)', fontweight='bold', labelpad=10)
        ax.set_title(f"Absolute (Total Max) Peak Energy Spectrum\nEvent: {event_name.replace('Event_', '')}",
                     fontweight='bold', pad=15)

        ax.grid(True, which="major", ls="--", alpha=0.4)

        # Clean, structured legend: Orientation colors + Satellite shapes + Fit styles
        handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=orient_colors['East'],
                   markeredgecolor='black', markersize=12, label='East'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=orient_colors['West'],
                   markeredgecolor='black', markersize=12, label='West'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
                   markeredgecolor='gray', markeredgewidth=1.4, markersize=10,
                   alpha=0.6, label='Background'),
            Line2D([0], [0], color='none', label=''),  # Spacer
        ]
        for sat in present_sats:
            handles.append(
                Line2D([0], [0], marker=sat_markers.get(sat, 'o'), color='w',
                       markerfacecolor='white', markeredgecolor='black',
                       markeredgewidth=1.5, markersize=11 + sat_size_offset.get(sat, 0),
                       label=sat)
            )
        handles.extend([
            Line2D([0], [0], color='none', label=''),  # Spacer
            Line2D([0], [0], color='black', linestyle=fit_linestyles['ER'],
                   linewidth=2.4, label='ER Fit'),
            Line2D([0], [0], color='black', linestyle=fit_linestyles['Band'],
                   linewidth=2.4, label='Band Fit'),
        ])

        ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc='upper left',
                  borderaxespad=0., title="Legend")

        # PFU & NOAA S-Scale annotation box (upper-right)
        meta = event_meta.get(event_name, {})
        pfu_val = meta.get('Max_PFU_10MeV', np.nan)
        s_scale = meta.get('S_Scale', 'Unknown')

        pfu_str = f"{pfu_val:,.0f} pfu" if pd.notna(pfu_val) and pfu_val >= 10 else (
            f"{pfu_val:.1f} pfu" if pd.notna(pfu_val) else "N/A"
        )
        info_text = f">10 MeV Max: {pfu_str}\nNOAA Scale: {s_scale}"

        ax.text(
            0.98, 0.96, info_text,
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=FONT_SIZE - 2, fontweight='bold',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='white',
                edgecolor='gray',
                linewidth=1.2,
                alpha=0.9
            ),
            zorder=10
        )

        plt.tight_layout()
        fig.savefig(absolute_dir / f"{event_name}_Absolute_Spectrum.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"\n{'-'*50}")
    print("ALL ABSOLUTE PEAK SPECTRA PLOTS GENERATED AND SAVED SAFELY.")
    print(f"{'-'*50}")

#%% 1. Parameter Covariance & Correlation Matrices (Catalog-Level)
corr_dir = BASE_DIR / "Correlation_Heatmaps"
corr_dir.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {
    'ER_log10_j0':   r'$j_0$',
    'ER_gamma':      r'$\gamma$',
    'ER_E0':         '$E_0$\n(MeV)',
    'Band_log10_j0': r'$j_0$',
    'Band_gamma_a':  r'$\gamma_a$',
    'Band_gamma_b':  r'$\gamma_b$',
    'Band_E0':       '$E_0$\n(MeV)',
    'Band_E_break':  '$E_{\\mathrm{break}}$\n(MeV)',
}

def compute_spearman_with_p(df_sub, cols, title):
    """Compute and print Spearman rank correlation (rho) and two-sided p-value matrices."""
    df_mat = df_sub[cols].dropna()
    n = len(df_mat)
    print(f"\n{'='*65}")
    print(f"{title} — Spearman Correlation & p-values (n={n})")
    print(f"{'='*65}")
    if n < 3:
        print("Not enough valid fits to compute correlation.")
        return None, None

    res = spearmanr(df_mat.to_numpy(dtype=float))
    corr_df = pd.DataFrame(res.statistic, index=cols, columns=cols)
    pval_df = pd.DataFrame(res.pvalue, index=cols, columns=cols)

    print("\n--- Spearman rho ---")
    print(corr_df.to_string(float_format=lambda x: f"{x:8.3f}"))
    print("\n--- Two-Sided p-values ---")
    print(pval_df.to_string(float_format=lambda x: f"{x:8.4g}"))

    return corr_df, pval_df


def p_to_stars(p):
    """Convert p-value to standard significance stars."""
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def plot_corr_heatmap(corr_mat, pval_mat, title, save_path, figsize=(8, 7)):
    """Plot a diverging heatmap of Spearman rho annotated with significance stars."""
    if corr_mat is None or corr_mat.empty:
        return

    vals = corr_mat.to_numpy(dtype=float)
    pvals = pval_mat.to_numpy(dtype=float)
    labels = [LABEL_MAP.get(c, c) for c in corr_mat.columns]
    n = len(labels)

    m_tests = n * (n - 1) // 2
    pvals_adj = np.minimum(pvals * m_tests, 1.0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(vals, cmap='RdBu_r', vmin=-1.0, vmax=1.0, aspect='equal')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'Spearman Correlation ($\rho$)', labelpad=10)
    cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=0, ha='center')
    ax.set_yticklabels(labels)

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', bottom=False, left=False)

    for i in range(n):
        for j in range(n):
            v = vals[i, j]
            if np.isnan(v):
                continue
            stars = "" if i == j else p_to_stars(pvals_adj[i, j])  # <-- update this line
            cell_text = f"{v:.2f}\n{stars}" if stars else f"{v:.2f}"
            txt_color = 'white' if abs(v) > 0.55 else 'black'
            ax.text(
                j, i, cell_text,
                ha='center', va='center', linespacing=0.95,
                color=txt_color, fontweight='bold', fontsize=FONT_SIZE - 3
            )

    ax.set_title(title, fontweight='bold', pad=15)
    fig.text(
        0.5, 0.02,
        f"* $p_{{\\mathrm{{adj}}}}$ < 0.05   |   ** $p_{{\\mathrm{{adj}}}}$ < 0.01   |   *** $p_{{\\mathrm{{adj}}}}$ < 0.001  (Bonferroni, m={m_tests})",
        ha='center', fontsize=FONT_SIZE - 5, color='dimgray'
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# 1. Ellison-Ramaty
er_corr, er_pval = compute_spearman_with_p(
    df_fits,
    cols=['ER_log10_j0', 'ER_gamma', 'ER_E0'],
    title="Ellison-Ramaty Fit Parameters"
)
plot_corr_heatmap(
    er_corr, er_pval,
    title="Ellison-Ramaty Parameter Correlations",
    save_path=corr_dir / f"ER_Spearman_Correlation_Heatmap_{mode_tag}.png",
    figsize=(7.5, 6.5)
)

# 2. Band
band_corr, band_pval = compute_spearman_with_p(
    df_fits,
    cols=['Band_log10_j0', 'Band_gamma_a', 'Band_gamma_b', 'Band_E0', 'Band_E_break'],
    title="Band Fit Parameters"
)
plot_corr_heatmap(
    band_corr, band_pval,
    title="Band Parameter Correlations",
    save_path=corr_dir / f"Band_Spearman_Correlation_Heatmap_{mode_tag}.png",
    figsize=(9.5, 8.5)
)


#%% 2. Per-Satellite East vs. West Statistical Comparison
if not GROUP_BY_ORIENTATION_ONLY:
    params_to_test = [
        ('ER_log10_j0',   'Ellison-Ramaty', False),
        ('ER_gamma',      'Ellison-Ramaty', False),
        ('ER_E0',         'Ellison-Ramaty', True),
        ('Band_log10_j0', 'Band',           False),
        ('Band_gamma_a',  'Band',           False),
        ('Band_gamma_b',  'Band',           False),
        ('Band_E0',       'Band',           True),
        ('Band_E_break',  'Band',           True),
    ]

    per_sat_rows = []

    for sat, df_sat in df_fits.groupby('Satellite'):
        for col, model, is_log in params_to_test:
            # 1. Paired by Event within this single satellite
            piv = df_sat.pivot_table(
                index='Event',
                columns='Orientation',
                values=col,
                aggfunc='first'
            )

            if {'East', 'West'}.issubset(piv.columns):
                paired = piv[['East', 'West']].dropna()
                if is_log:
                    paired = paired[(paired['East'] > 0) & (paired['West'] > 0)]
            else:
                paired = pd.DataFrame(columns=['East', 'West'])

            n_pairs = len(paired)
            wilcoxon_p = np.nan
            med_delta = np.nan

            if n_pairs >= 5:
                if is_log:
                    diff = np.log10(paired['West'] / paired['East'])
                    med_delta = (paired['West'] / paired['East']).median()
                else:
                    diff = paired['West'] - paired['East']
                    med_delta = diff.median()

                if not np.allclose(diff, 0):
                    _, wilcoxon_p = wilcoxon(diff, alternative='two-sided')

            # 2. Unpaired Mann-Whitney U within this single satellite
            east_all = df_sat.loc[df_sat['Orientation'] == 'East', col].dropna()
            west_all = df_sat.loc[df_sat['Orientation'] == 'West', col].dropna()
            if is_log:
                east_all = east_all[east_all > 0]
                west_all = west_all[west_all > 0]

            mwu_p = np.nan
            if len(east_all) >= 3 and len(west_all) >= 3:
                _, mwu_p = mannwhitneyu(east_all, west_all, alternative='two-sided')

            per_sat_rows.append({
                'Satellite': sat,
                'Model': model,
                'Parameter': col,
                'Scale': 'Ratio (W/E)' if is_log else 'Diff (W-E)',
                'N_Pairs': n_pairs,
                'East_Med': paired['East'].median() if n_pairs > 0 else np.nan,
                'West_Med': paired['West'].median() if n_pairs > 0 else np.nan,
                'Med_W_vs_E': med_delta,
                'Wilcoxon_p': wilcoxon_p,
                'Sig_05': (wilcoxon_p < 0.05) if pd.notna(wilcoxon_p) else False,
                'MWU_p_Unpaired': mwu_p,
            })

    df_ew_per_sat = pd.DataFrame(per_sat_rows)
    df_ew_per_sat.to_csv(BASE_DIR / f"East_West_Parameter_Stats_Per_Satellite.csv", index=False)

    for sat, df_out in df_ew_per_sat.groupby('Satellite'):
        print(f"\n{'='*90}")
        print(f"East vs. West Stats — {sat}")
        print(f"{'='*90}")
        print(df_out.drop(columns=['Satellite']).to_string(index=False, float_format=lambda x: f"{x:.4g}"))


#%% Statistical Comparison: East vs. West Fit Parameters
# Parameters to test: (column_name, model_name, is_log_scale)
params_to_test = [
    ('ER_log10_j0',   'Ellison-Ramaty', False),
    ('ER_gamma',      'Ellison-Ramaty', False),
    ('ER_E0',         'Ellison-Ramaty', True),
    ('Band_log10_j0', 'Band',           False),
    ('Band_gamma_a',  'Band',           False),
    ('Band_gamma_b',  'Band',           False),
    ('Band_E0',       'Band',           True),
    ('Band_E_break',  'Band',           True),
]

stat_rows = []

for col, model, is_log in params_to_test:
    # 1. Paired data: match simultaneous East & West fits for the same (Event, Satellite)
    piv = df_fits.pivot_table(
        index=['Event', 'Satellite'],
        columns='Orientation',
        values=col,
        aggfunc='first'
    )

    if {'East', 'West'}.issubset(piv.columns):
        paired = piv[['East', 'West']].dropna()
        if is_log:
            paired = paired[(paired['East'] > 0) & (paired['West'] > 0)]
    else:
        paired = pd.DataFrame(columns=['East', 'West'])

    n_pairs = len(paired)
    wilcoxon_stat, wilcoxon_p = np.nan, np.nan
    med_delta = np.nan

    if n_pairs >= 5:
        if is_log:
            # Test fractional difference: log10(West / East)
            diff = np.log10(paired['West'] / paired['East'])
            med_delta = (paired['West'] / paired['East']).median()  # Median W/E ratio
        else:
            diff = paired['West'] - paired['East']
            med_delta = diff.median()                               # Median W - E difference

        if not np.allclose(diff, 0):
            wilcoxon_stat, wilcoxon_p = wilcoxon(diff, alternative='two-sided')

    # 2. Unpaired data (Mann-Whitney U on full East vs. West distributions for reference)
    east_all = df_fits.loc[df_fits['Orientation'] == 'East', col].dropna()
    west_all = df_fits.loc[df_fits['Orientation'] == 'West', col].dropna()
    if is_log:
        east_all = east_all[east_all > 0]
        west_all = west_all[west_all > 0]

    mwu_p = np.nan
    if len(east_all) >= 3 and len(west_all) >= 3:
        _, mwu_p = mannwhitneyu(east_all, west_all, alternative='two-sided')

    stat_rows.append({
        'Model': model,
        'Parameter': col,
        'Scale': 'Ratio (W/E)' if is_log else 'Diff (W-E)',
        'N_Pairs': n_pairs,
        'East_Med': paired['East'].median() if n_pairs > 0 else np.nan,
        'West_Med': paired['West'].median() if n_pairs > 0 else np.nan,
        'Med_W_vs_E': med_delta,
        'Wilcoxon_p': wilcoxon_p,
        'Sig_Paired_05': (wilcoxon_p < 0.05) if pd.notna(wilcoxon_p) else False,
        'MWU_p_Unpaired': mwu_p,
    })

df_ew_stats = pd.DataFrame(stat_rows)
df_ew_stats.to_csv(BASE_DIR / f"East_West_Parameter_Stats_{mode_tag}.csv", index=False)

print(df_ew_stats.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


#%% Plot ER and Band Fit Parameter Histograms (Absolute Peak, East vs. West)
hist_dir = BASE_DIR / "Fit_Parameter_Histograms"
hist_dir.mkdir(parents=True, exist_ok=True)

orient_colors = {'East': '#1f77b4', 'West': '#ff7f0e'}

er_params = [
    ('ER_log10_j0', r'$\log_{10}(j_0)$', False),
    ('ER_gamma',    r'$\gamma$',         False),
    ('ER_E0',       r'$E_0$ (MeV)',      True),
]

band_params = [
    ('Band_log10_j0', r'$\log_{10}(j_0)$',          False),
    ('Band_gamma_a',  r'$\gamma_a$',                False),
    ('Band_gamma_b',  r'$\gamma_b$',                False),
    ('Band_E0',       r'$E_0$ (MeV)',               True),
    ('Band_E_break',  r'$E_{\mathrm{break}}$ (MeV)', True),
]

def plot_param_histograms(df_sub, params, model_name, save_path, bins=20):
    n_params = len(params)
    ncols = 3
    nrows = int(np.ceil(n_params / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (col, xlabel, is_log) in zip(axes, params):
        vals_east = df_sub.loc[df_sub['Orientation'] == 'East', col].dropna()
        vals_west = df_sub.loc[df_sub['Orientation'] == 'West', col].dropna()

        if is_log:
            vals_east = vals_east[vals_east > 0]
            vals_west = vals_west[vals_west > 0]

        combined = pd.concat([vals_east, vals_west])
        if combined.empty:
            ax.set_title(f"{xlabel} (No Data)")
            continue

        if is_log:
            bin_edges = np.logspace(np.log10(combined.min()), np.log10(combined.max()), bins + 1)
            ax.set_xscale('log')
        else:
            bin_edges = np.linspace(combined.min(), combined.max(), bins + 1)

        for orient, vals in [('East', vals_east), ('West', vals_west)]:
            if vals.empty:
                continue
            color = orient_colors[orient]
            med = vals.median()
            ax.hist(
                vals, bins=bin_edges, alpha=0.5, color=color, edgecolor='black',
                linewidth=0.8, label=f"{orient} (n={len(vals)}, med={med:.2f})"
            )
            ax.axvline(med, color=color, linestyle='--', linewidth=1.8)

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize=14)

    for ax in axes[n_params:]:
        ax.set_visible(False)

    fig.suptitle(f"{model_name} Fit Parameters — Absolute Peak (East vs. West)", y=1)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

plot_param_histograms(
    df_fits, er_params, 'Ellison-Ramaty',
    hist_dir / f"ER_Parameters_TotalMaxFlux_Hist_{mode_tag}.png"
)
plot_param_histograms(
    df_fits, band_params, 'Band',
    hist_dir / f"Band_Parameters_TotalMaxFlux_Hist_{mode_tag}.png"
)

print(f"Saved parameter histograms to {hist_dir}")

