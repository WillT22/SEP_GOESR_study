# Import necessary libraries
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FONT_SIZE = 20
plt.rcParams.update({
    'font.size': FONT_SIZE,          # Base font size
    'axes.titlesize': FONT_SIZE + 2, # Subplot titles
    'axes.labelsize': FONT_SIZE,     # Axis labels (L*, Time, Flux)
    'xtick.labelsize': FONT_SIZE,    # X-axis tick numbers
    'ytick.labelsize': FONT_SIZE,    # Y-axis tick numbers
    'legend.fontsize': FONT_SIZE,    # Legend text
    'figure.titlesize': FONT_SIZE + 4, # Main suptitle
    'legend.markerscale': 1.0
})

# Import event peak data
df_event_peaks = pd.read_parquet(r"C:\Users\wzt0020\Box\SEP_Box\SGPS_data\SEP_event_peaks_summary.parquet")

# Import SEP catalog

# Define S-scale
def classify_s_scale(pfu):
    """
    Classifies the event into the NOAA Solar Radiation Storm Scale based on peak >10 MeV pfu.
    """
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

#%% Fit the spectra for each event
# --- Helpers: Compute AICc and BIC from log10 residuals ---
def compute_bic(residuals, k):
    n = len(residuals)
    rss = np.sum(residuals ** 2)
    if rss <= 0 or n <= k:
        return np.nan
    return n * np.log(rss / n) + k * np.log(n)
def compute_aicc(residuals, k):
    n = len(residuals)
    rss = np.sum(residuals ** 2)
    if rss <= 0 or n <= k + 1:
        return np.nan
    aic = n * np.log(rss / n) + 2 * k
    return aic + (2 * k * (k + 1)) / (n - k - 1)

# --- Batch Fitting Loop ---
fit_records = []

# Loop over every unique (Event, Satellite, Orientation) combination
for (event_name, sat, orientation), df_group in df_event_peaks.groupby(['Event', 'Satellite', 'Orientation']):
    
    # Exclude background channels so flat noise floors don't skew the high-energy tail
    df_real = df_group[~df_group['IsBackground'].fillna(False)].sort_values('EffectiveEnergy')
    
    # Fit both Onset and Absolute peak spectra
    for peak_type in ['OnsetFlux', 'TotalMaxFlux']:
        
        df_valid = df_real.dropna(subset=['EffectiveEnergy', peak_type])
        df_valid = df_valid[(df_valid['EffectiveEnergy'] > 0) & (df_valid[peak_type] > 0)]
        
        E = df_valid['EffectiveEnergy'].to_numpy(dtype=float)
        flux = df_valid[peak_type].to_numpy(dtype=float)
        n_pts = len(E)
        
        # Base row with NaNs for failed/skipped fits
        row = {
            'Event': event_name,
            'Satellite': sat,
            'Orientation': orientation,
            'PeakType': peak_type,
            'N_Points': n_pts,
            # Ellison & Ramaty (k=3)
            'ER_log10_j0': np.nan, 'ER_gamma': np.nan, 'ER_E0': np.nan,
            'ER_rmse': np.nan, 'ER_bic': np.nan, 'ER_aicc': np.nan,
            # Band (k=4)
            'Band_log10_j0': np.nan, 'Band_gamma_a': np.nan, 'Band_gamma_b': np.nan,
            'Band_E0': np.nan, 'Band_E_break': np.nan,
            'Band_rmse': np.nan, 'Band_bic': np.nan, 'Band_aicc': np.nan,
            # Model comparison
            'BestModel_BIC': None, 'BestModel_AICc': None
        }
        
        if n_pts < 4:
            fit_records.append(row)
            continue
            
        log_flux = np.log10(flux)
        
        # 1. Fit Ellison & Ramaty (3 parameters -> needs >= 4 points)
        try:
            p0_er = [log_flux[0], 2.0, 30.0]
            bounds_er = ([-10.0, 0.0, 0.5], [15.0, 10.0, 2000.0])
            popt_er, _ = curve_fit(
                ellison_ramaty_log10, E, log_flux,
                p0=p0_er, bounds=bounds_er, maxfev=10000
            )
            resid_er = log_flux - ellison_ramaty_log10(E, *popt_er)
            
            row['ER_log10_j0'], row['ER_gamma'], row['ER_E0'] = popt_er
            row['ER_rmse'] = np.sqrt(np.mean(resid_er ** 2))
            row['ER_bic'] = compute_bic(resid_er, k=3)
            row['ER_aicc'] = compute_aicc(resid_er, k=3)
        except RuntimeError:
            pass
            
        # 2. Fit Band (4 parameters -> needs >= 5 points)
        if n_pts >= 5:
            try:
                p0_band = [log_flux[0], 1.5, 2.0, 30.0]
                bounds_band = ([-10.0, 0.0, 0.05, 0.5], [15.0, 8.0, 12.0, 2000.0])
                popt_b, _ = curve_fit(
                    band_log10, E, log_flux,
                    p0=p0_band, bounds=bounds_band, maxfev=20000
                )
                resid_b = log_flux - band_log10(E, *popt_b)
                log10_j0, gamma_a, delta_gamma, E0 = popt_b
                
                row['Band_log10_j0'] = log10_j0
                row['Band_gamma_a'] = gamma_a
                row['Band_gamma_b'] = gamma_a + delta_gamma
                row['Band_E0'] = E0
                row['Band_E_break'] = delta_gamma * E0
                row['Band_rmse'] = np.sqrt(np.mean(resid_b ** 2))
                row['Band_bic'] = compute_bic(resid_b, k=4)
                row['Band_aicc'] = compute_aicc(resid_b, k=4)
            except RuntimeError:
                pass
        
        # 3. Select the best model 
        # Pick Best Model by lower BIC
        er_bic, band_bic = row['ER_bic'], row['Band_bic']
        if pd.notna(er_bic) and pd.notna(band_bic):
            row['BestModel_BIC'] = 'Band' if band_bic < er_bic else 'EllisonRamaty'
        elif pd.notna(er_bic):
            row['BestModel_BIC'] = 'EllisonRamaty'
        elif pd.notna(band_bic):
            row['BestModel_BIC'] = 'Band'

        # Pick Best Model by lower AICc
        er_aicc, band_aicc = row['ER_aicc'], row['Band_aicc']
        if pd.notna(er_bic) and pd.notna(band_bic):
            row['BestModel_AICc'] = 'Band' if band_aicc < er_aicc else 'EllisonRamaty'
        elif pd.notna(er_aicc):
            row['BestModel_AICc'] = 'EllisonRamaty'
        elif pd.notna(band_aicc):
            row['BestModel_AICc'] = 'Band'
            
        fit_records.append(row)

# Convert to DataFrame and save
df_fits = pd.DataFrame(fit_records)

output_dir = Path(r"C:\Users\wzt0020\Box\SEP_Box\SGPS_data")
df_fits.to_parquet(output_dir / "SEP_spectral_fits.parquet", index=False)
df_fits.to_csv(output_dir / "SEP_spectral_fits.csv", index=False)

print(f"Completed fits for {len(df_fits)} spectra.")
print(df_fits['BestModel_BIC'].value_counts(dropna=False).values)
print(df_fits['BestModel_AICc'].value_counts(dropna=False).values)

#%% Plot ER and Band fit parameter histograms (East vs. West)
hist_dir = Path(r"C:\Users\wzt0020\Box\SEP_Box\SGPS_data\Fit_Parameter_Histograms")
hist_dir.mkdir(parents=True, exist_ok=True)

orient_colors = {'East': '#1f77b4', 'West': '#ff7f0e'}

er_params = [
    ('ER_log10_j0', r'$\log_{10}(j_0)$', False),
    ('ER_gamma',    r'$\gamma$',         False),
    ('ER_E0',       r'$E_0$ (MeV)',      True),
]

band_params = [
    ('Band_log10_j0', r'$\log_{10}(j_0)$',    False),
    ('Band_gamma_a',  r'$\gamma_a$',          False),
    ('Band_gamma_b',  r'$\gamma_b$',          False),
    ('Band_E0',       r'$E_0$ (MeV)',         True),
    ('Band_E_break',  r'$E_{\mathrm{break}}$ (MeV)', True),
]

def plot_param_histograms(df_sub, params, model_name, peak_label, save_path, bins=20):
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

        # Shared bin edges so East and West bars align directly
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
        ax.legend(fontsize = 14)

    # Hide any unused subplot slots (e.g., 6th slot in the 2x3 Band grid)
    for ax in axes[n_params:]:
        ax.set_visible(False)

    fig.suptitle(f"{model_name} Fit Parameters — {peak_label} (East vs. West)", y=1)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# Generate histograms for both Onset and Absolute Max
for peak_type, peak_label in [('OnsetFlux', 'Onset Peak'), ('TotalMaxFlux', 'Absolute Peak')]:
    df_peak = df_fits[df_fits['PeakType'] == peak_type]
    #df_peak = df_peak[df_peak['BestModel_AICc'] == 'EllisonRamaty']
    #df_peak = df_peak[df_peak['BestModel_AICc'] == 'Band']
    
    plot_param_histograms(
        df_peak, er_params, 'Ellison-Ramaty', peak_label,
        hist_dir / f"ER_Parameters_{peak_type}_Hist.png"
    )
    plot_param_histograms(
        df_peak, band_params, 'Band', peak_label,
        hist_dir / f"Band_Parameters_{peak_type}_Hist.png"
    )

print(f"Saved parameter histograms to {hist_dir}")

#%% Plot all energy spectra with fits
# --- Configuration & Folder Setup ---
base_plot_dir = Path(r"C:\Users\wzt0020\Box\SEP_Box\SGPS_data\Event_Spectra_Fits_Plots")
onset_dir = base_plot_dir / "Onset_Peak_Spectra"
absolute_dir = base_plot_dir / "Absolute_Peak_Spectra"

onset_dir.mkdir(parents=True, exist_ok=True)
absolute_dir.mkdir(parents=True, exist_ok=True)

# Consistent styling
sat_colors = {'GOES16': '#1f77b4', 'GOES17': '#ff7f0e', 'GOES18': '#2ca02c', 'GOES19': '#d62728'}
SGPS_CHANNELS = ['P1', 'P2A', 'P2B', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8A', 'P8B', 'P8C', 'P9', 'P10']

# Create linestyles
fit_linestyles = {
    ('ER',   'West'): '-',
    ('ER',   'East'): (0, (8, 2, 1.5, 2)),  # Long dash-dot
    ('Band', 'West'): (0, (5, 3)),          # Medium dash
    ('Band', 'East'): (0, (1.5, 2.5)),      # Dotted
}

# Index df_fits for fast lookup by (Event, Satellite, Orientation, PeakType)
fits_lookup = df_fits.set_index(['Event', 'Satellite', 'Orientation', 'PeakType'])

print(f"Generating energy spectra plots for {df_event_peaks['Event'].nunique()} events...")

# Loop through each distinct event in your peaks dataframe
for event_name, df_event in df_event_peaks.groupby('Event'):
    
    print(f"  -> Plotting spectra for {event_name}...")
    
    for peak_type, save_folder, title_prefix in [
        ('OnsetFlux', onset_dir, 'Onset (Prompt) Peak'),
        ('TotalMaxFlux', absolute_dir, 'Absolute (Total Max) Peak')
    ]:
        
        fig, ax = plt.subplots(figsize=(20, 8))
        
        # Group directly by Satellite and the pre-calculated Orientation
        for (sat, orientation), df_group in df_event.groupby(['Satellite', 'Orientation']):
            
            df_group = df_group.sort_values('ChannelNum')
            
            color = sat_colors.get(sat, 'black')
            marker = 'o' if orientation == 'West' else 's'
            label = f"{sat} ({orientation})"
            
            # Split into real peaks vs. background
            is_bg = df_group['IsBackground'].fillna(False)
            df_real = df_group[~is_bg]
            df_bg = df_group[is_bg]

            # 1. Faint connecting line across ALL channels
            ax.plot(df_group['ChannelNum'], df_group[peak_type],
                    color=color, linestyle='none', linewidth=1.0, alpha=0.2, zorder=2)

            # 2. Real peaks: full color, solid fill, high visibility
            if not df_real.empty:
                ax.plot(df_real['ChannelNum'], df_real[peak_type],
                        color=color, linestyle='none', marker=marker, markersize=10,
                        linewidth=1.5, label=label, alpha=0.9, zorder=4)
                real_series = df_group[peak_type].where(~is_bg)
                ax.plot(df_group['ChannelNum'], real_series,
                        color=color, linestyle='none', linewidth=1.4, alpha=0.5, zorder=3)

            # 3. Background points: gray-filled, faded satellite-colored border
            if not df_bg.empty:
                bg_label = label if df_real.empty else None
                ax.plot(df_bg['ChannelNum'], df_bg[peak_type],
                        color=color, linestyle='none', marker=marker, markersize=9,
                        markerfacecolor='lightgray', markeredgecolor=color, markeredgewidth=1.2,
                        alpha=0.4, label=bg_label, zorder=3)

            # 4. Overlay Ellison-Ramaty and Band Fits (evaluated across real channels)
            fit_key = (event_name, sat, orientation, peak_type)
            if fit_key in fits_lookup.index and not df_real.empty:
                fit_row = fits_lookup.loc[fit_key]
                if isinstance(fit_row, pd.DataFrame):
                    fit_row = fit_row.iloc[0]

                df_fit_pts = df_real.dropna(subset=['EffectiveEnergy'])
                E_pts = df_fit_pts['EffectiveEnergy'].to_numpy(dtype=float)
                ch_pts = df_fit_pts['ChannelNum'].to_numpy()
                best_model = fit_row.get('BestModel_AICc', None)

                # Ellison & Ramaty Fit
                if pd.notna(fit_row['ER_log10_j0']):
                    er_flux = ellison_ramaty(
                        E_pts, fit_row['ER_log10_j0'], fit_row['ER_gamma'], fit_row['ER_E0']
                    )
                    is_best = (best_model == 'EllisonRamaty')
                    ax.plot(ch_pts, er_flux, color=color,
                            linestyle=fit_linestyles.get(('ER', orientation), '-'),
                            linewidth=2.5,
                            alpha=0.95 if is_best else 0.45, zorder=5)

                # Band Fit
                if pd.notna(fit_row['Band_log10_j0']):
                    delta_gamma = fit_row['Band_gamma_b'] - fit_row['Band_gamma_a']
                    band_flux = band(
                        E_pts, fit_row['Band_log10_j0'], fit_row['Band_gamma_a'],
                        delta_gamma, fit_row['Band_E0']
                    )
                    is_best = (best_model == 'Band')
                    ax.plot(ch_pts, band_flux, color=color,
                            linestyle=fit_linestyles.get(('Band', orientation), '--'),
                            linewidth=2.5,
                            alpha=0.95 if is_best else 0.45, zorder=5)

        energy_labels = []
        for ch in SGPS_CHANNELS:
            ch_data = df_event[df_event['Channel'] == ch]
            if not ch_data.empty and 'EffectiveEnergy' in ch_data.columns:
                mean_energy = ch_data['EffectiveEnergy'].mean()
                energy_labels.append(f"{mean_energy:.1f}")
            else:
                energy_labels.append("-")

        # --- Formatting ---
        ax.set_yscale('log')
        ax.set_xlim(-0.5, len(SGPS_CHANNELS) - 0.5)
        
        ax.set_xticks(range(len(SGPS_CHANNELS)))
        ax.set_xticklabels(energy_labels, fontweight='bold', color='black')
        
        for i, ch_name in enumerate(SGPS_CHANNELS):
            ax.text(i, -0.08, ch_name, transform=ax.get_xaxis_transform(), 
                    ha='center', va='top', color='dimgray', fontweight='bold')
        
        ax.set_xlabel('Effective Energy (MeV)', fontweight='bold', labelpad=30)
        ax.set_ylabel('Proton Flux (#/cm²/s/sr/MeV)', fontweight='bold', labelpad=10)
        ax.set_title(f"{title_prefix} Energy Spectrum\nEvent: {event_name.replace('Event_', '')}", 
                     fontweight='bold', pad=15)
        
        ax.grid(True, which="major", ls="--", alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()
        handles.extend([
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
                   markeredgecolor='gray', markeredgewidth=1.2, markersize=9,
                   alpha=0.5, label='Background'),
            Line2D([0], [0], color='black', linestyle=fit_linestyles[('ER', 'West')],
                   linewidth=2.5, label='ER Fit (West)'),
            Line2D([0], [0], color='black', linestyle=fit_linestyles[('ER', 'East')],
                   linewidth=2.5, label='ER Fit (East)'),
            Line2D([0], [0], color='black', linestyle=fit_linestyles[('Band', 'West')],
                   linewidth=2.5, label='Band Fit (West)'),
            Line2D([0], [0], color='black', linestyle=fit_linestyles[('Band', 'East')],
                   linewidth=2.5, label='Band Fit (East)'),
        ])
        ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc='upper left',
                  borderaxespad=0., title="Satellites & Fits")
        
        plt.tight_layout()
        
        filename = f"{event_name}_{title_prefix.split()[0]}_Spectrum.png"
        full_path = save_folder / filename
        
        fig.savefig(full_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

print(f"\n{'-'*50}")
print("ALL SPECTRA PLOTS GENERATED AND SAVED SAFELY.")
print(f"{'-'*50}")