#%% Import Libraries and SEP Catalog
import pandas as pd
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
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

SGPS_CHANNELS = ['P1', 'P2A', 'P2B', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8A', 'P8B', 'P8C', 'P9', 'P10']
CHANNEL_TO_NUM = {ch: i for i, ch in enumerate(SGPS_CHANNELS)}
CHANNEL_THRESHOLDS = {
    'P1': 1e-1, 'P2A': 1e-1, 'P2B': 1e-1,
    'P3': 1e-2, 'P4': 1e-2, 'P5': 1e-2,
    'P6': 1e-3, 'P7': 1e-3, 'P8A': 1e-3,
    'P8B': 1e-3, 'P8C': 1e-3, 'P9': 1e-3, 'P10': 1e-3
}


# Import updated SEP event catalog
catalog_path = Path(r"C:\Users\wzt0020\Box\SEP_Box\SGPS_data\updated_sep_catalog.csv")  # Adjust path if needed
sep_catalog = pd.read_csv(catalog_path)
sep_catalog.columns = sep_catalog.columns.str.strip()

# Ensure Begin_Time_UTC is UTC-aware and create the matching folder key
sep_catalog['Begin_Time_UTC'] = pd.to_datetime(sep_catalog['Begin_Time_UTC'], utc=True)
sep_catalog['Event_Folder'] = 'Event_' + sep_catalog['Begin_Time_UTC'].dt.strftime('%Y%m%d_%H%M')


# Import quiet lookup data
df_quiet = pd.read_csv(r"C:\Users\wzt0020\Box\SEP_Box\SGPS_data\quiet_background_stats.csv")
quiet_lookup = {
    (row.Satellite, row.Sensor, row.Channel, row.YearMonth): (row.QuietMedian, row.QuietSigma)
    for row in df_quiet.itertuples(index=False)
}


#%% From GOES data, import the relevant dates for each satellite
# 1. Define the base directory where the event data was saved
data_input_base = Path(r"C:\Users\wzt0020\Box\SEP_Box\SGPS_data\Event_Data")

# 2. Initialize a nested dictionary to hold all the event data
# Structure: loaded_events['Event_YYYYMMDD_HHMM']['goes16'][0] = DataFrame
loaded_events = defaultdict(lambda: defaultdict(dict))

print(f"Scanning for event data in: {data_input_base}")

# 3. Iterate through all 'Event_' folders
for event_dir in data_input_base.glob("Event_*"):
    if not event_dir.is_dir():
        continue
        
    event_name = event_dir.name  # e.g., "Event_20230501_1430"
    
    # 4. Iterate through all Parquet files inside this specific event's folder
    for pq_file in event_dir.glob("*_event_data.parquet"):
        
        # Extract satellite and sensor from the filename 
        # Example filename: "goes16_sensor0_event_data.parquet"
        parts = pq_file.stem.split('_')
        sat_name = parts[0]
        
        # Safely extract the integer sensor number
        try:
            sensor_num = int(parts[1].replace('sensor', ''))
        except ValueError:
            print(f"Warning: Could not parse sensor number from {pq_file.name}")
            continue
            
        # 5. Load the Parquet file into a Pandas DataFrame
        df_event = pd.read_parquet(pq_file)
        
        # Store it in the nested dictionary
        loaded_events[event_name][sat_name][sensor_num] = df_event

print(f"Successfully loaded {len(loaded_events)} events into memory.")


#%% Determine the maximum flux for each channel as a function of the effective energy for a given sensor
# Before begin time difference
time_difference = pd.Timedelta(hours=3)

# --- Tunable Noise-Rejection Parameters ---
N_SIGMA = 3.0               # Peak must be at least 3-sigma above pre-event background
MIN_RATIO = 2.0             # Peak must be at least 2x the background median
MIN_POINTS_ABOVE_BG = 3     # Minimum points above (bg + 3*sigma) to reject single-point spikes

summary_data = []

# Iterate through every event loaded into memory
for event_name, satellites in loaded_events.items(): 
    
    # Look up the matching event row in the catalog
    event_match = sep_catalog[sep_catalog['Event_Folder'] == event_name]
    if event_match.empty:
        print(f"Warning: {event_name} not found in sep_catalog. Skipping.")
        continue
    event_row = event_match.iloc[0]
    
    # Universal start of the search window: Begin_Time_UTC - 6 hours
    slice_start = event_row['Begin_Time_UTC'] - time_difference
    
    for sat_name, sensors in satellites.items():
        for sensor, df_event in sensors.items():
            
            # --- Determine the dominant physical orientation for this event ---
            state = 0  # Default to upright
            if 'yaw_flip_flag' in df_event.columns:
                physical_states = df_event['yaw_flip_flag'][df_event['yaw_flip_flag'].isin([0, 2])]
                if not physical_states.empty:
                    state = physical_states.mode().iloc[0]
            
            if state == 0:
                orientation = 'West' if sensor == 0 else 'East'
            else:  # state == 2 (Inverted)
                orientation = 'East' if sensor == 0 else 'West'
            
            # Loop through all possible SGPS channels
            for ch in SGPS_CHANNELS:
                flux_col = f'{ch}_Flux'
                if flux_col not in df_event.columns:
                    continue
                
                # Pull this channel's specific end time from the catalog
                end_col = f'{ch}_End_Time'
                slice_end_raw = event_row.get(end_col, pd.NaT)
                if pd.isna(slice_end_raw):
                    continue
                    
                slice_end = pd.to_datetime(slice_end_raw)
                if slice_end.tzinfo is None:
                    slice_end = slice_end.tz_localize('UTC')

                # Slice the buffered Parquet data to [Begin_Time_UTC - 6h, ch_End_Time]
                df_window = df_event.loc[slice_start:slice_end]
                flux_series = df_window[flux_col].replace(0, np.nan).dropna()
                flux_series = flux_series[flux_series > 0.0]
                
                if flux_series.empty:
                    continue

                # --- 1. Estimate Background Noise Level ---
                # Use the pre-event buffer (-24h to -6h); fall back to the event window if empty
                bg_series = df_event.loc[:slice_start, flux_col].replace(0, np.nan).dropna()
                bg_series = bg_series[bg_series > 0.0]
                if len(bg_series) < 2:
                    bg_series = flux_series

                bg_median = bg_series.median()
                # Robust sigma via MAD; fall back to std() if quantized counts make MAD == 0
                mad = (bg_series - bg_median).abs().median()
                bg_sigma = 1.4826 * mad if mad > 0 else bg_series.std()
                if pd.isna(bg_sigma):
                    bg_sigma = 0.0

                # --- 2. Test Whether a Discernible Peak Exists ---
                total_max_time = flux_series.idxmax()
                total_max_flux = flux_series.max()
                event_ym = event_row['Begin_Time_UTC'].strftime('%Y-%m')
                ch_threshold = CHANNEL_THRESHOLDS.get(ch, 1e-3)

                q_median, q_sigma = quiet_lookup.get(
                    (sat_name.upper(), sensor, ch, event_ym),
                    (ch_threshold, 0.0)
                )

                # Use whichever baseline is higher: your physical threshold or that month's actual quiet median
                baseline = max(ch_threshold, q_median)
                has_real_peak = total_max_flux > (baseline + N_SIGMA * q_sigma)

                energy_col = f'{ch}_Energy'
                uncert_col = f'{ch}_Uncert'

                eff_energy = df_window.loc[total_max_time, energy_col] if energy_col in df_window.columns else np.nan
                total_max_uncert = df_window.loc[total_max_time, uncert_col] if uncert_col in df_window.columns else np.nan

                if has_real_peak:
                    # Calculate the Onset (Prompt) Peak
                    peaks, _ = find_peaks(flux_series.values, prominence=total_max_flux * 0.1)
                    
                    if len(peaks) > 0:
                        onset_idx = peaks[0]
                        onset_time = flux_series.index[onset_idx]
                        onset_flux = flux_series.iloc[onset_idx]
                        onset_uncert = df_window.loc[onset_time, uncert_col] if uncert_col in df_window.columns else np.nan
                    else:
                        onset_time = total_max_time
                        onset_flux = total_max_flux
                        onset_uncert = total_max_uncert
                    
                    is_single_peak = (onset_time == total_max_time)
                    is_background = False
                else:
                    # No discernible peak: keep the raw max, but flag as background
                    onset_flux = total_max_flux
                    onset_time = total_max_time
                    onset_uncert = total_max_uncert
                    is_single_peak = True
                    is_background = True
                
                summary_data.append({
                    'Event': event_name,
                    'Satellite': sat_name.upper(),
                    'Sensor': sensor,
                    'Orientation': orientation,
                    'ChannelNum': CHANNEL_TO_NUM.get(ch, 99),  
                    'Channel': ch,  
                    'EffectiveEnergy': eff_energy,
                    'OnsetFlux': onset_flux,
                    'OnsetUncert': onset_uncert,
                    'OnsetTime': onset_time,
                    'TotalMaxFlux': total_max_flux,
                    'TotalMaxUncert': total_max_uncert,
                    'TotalMaxTime': total_max_time,
                    'IsSinglePeak': is_single_peak,
                    'IsBackground': is_background
                })

# Convert the results to a clean DataFrame and sort chronologically and by channel
df_event_peaks = pd.DataFrame(summary_data)
df_event_peaks = df_event_peaks.sort_values(['Event', 'Satellite', 'Orientation', 'ChannelNum']).reset_index(drop=True)

# --- Save Summary Data ---
output_dir = Path(r"C:\Users\wzt0020\Box\SEP_Box\SGPS_data")
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Parquet (recommended for reloading into Python/pandas with exact dtypes & UTC timezones)
parquet_path = output_dir / "SEP_event_peaks_summary.parquet"
df_event_peaks.to_parquet(parquet_path, index=False)

# 2. CSV (for quick visual inspection in Excel or sharing)
csv_path = output_dir / "SEP_event_peaks_summary.csv"
df_event_peaks.to_csv(csv_path, index=False)

print(f"Saved {len(df_event_peaks)} rows to:\n  - {parquet_path}\n  - {csv_path}")

#%% Plot the peak flux for each event
# --- Configuration & Folder Setup ---
PLOT_UNCERTAINTY = True  # Toggle flux error bars on/off

base_plot_dir = Path(r"C:\Users\wzt0020\Box\SEP_Box\SGPS_data\Event_Spectra_Plots")
onset_dir = base_plot_dir / "Onset_Peak_Spectra"
absolute_dir = base_plot_dir / "Absolute_Peak_Spectra"

onset_dir.mkdir(parents=True, exist_ok=True)
absolute_dir.mkdir(parents=True, exist_ok=True)

# Consistent styling
sat_colors = {'GOES16': '#1f77b4', 'GOES17': '#ff7f0e', 'GOES18': '#2ca02c', 'GOES19': '#d62728'}
SGPS_CHANNELS = ['P1', 'P2A', 'P2B', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8A', 'P8B', 'P8C', 'P9', 'P10']

def log_safe_yerr(flux, uncert):
    """Clips lower error bars so (flux - yerr_low) stays > 0 on a log y-axis."""
    f = np.asarray(flux, dtype=float)
    u = np.nan_to_num(np.asarray(uncert, dtype=float), nan=0.0)
    u = np.maximum(u, 0.0)
    lower = np.minimum(u, f * 0.99)
    return [lower, u]

print(f"Generating energy spectra plots for {df_event_peaks['Event'].nunique()} events...")

# Loop through each distinct event in your peaks dataframe
for event_name, df_event in df_event_peaks.groupby('Event'):
    
    print(f"  -> Plotting spectra for {event_name}...")
    
    # Loop over both peak types and their corresponding uncertainty columns
    for peak_type, uncert_col, save_folder, title_prefix in [
        ('OnsetFlux', 'OnsetUncert', onset_dir, 'Onset (Prompt) Peak'),
        ('TotalMaxFlux', 'TotalMaxUncert', absolute_dir, 'Absolute (Total Max) Peak')
    ]:
        
        fig, ax = plt.subplots(figsize=(20, 8))
        use_err = PLOT_UNCERTAINTY and (uncert_col in df_event.columns)
        
        # Group directly by Satellite and the pre-calculated Orientation
        for (sat, orientation), df_group in df_event.groupby(['Satellite', 'Orientation']):
            
            df_group = df_group.sort_values('ChannelNum')
            
            color = sat_colors.get(sat, 'black')
            linestyle = '-' if orientation == 'West' else '--'
            marker = 'o' if orientation == 'West' else 's'
            label = f"{sat} ({orientation})"
            
            # Split into real peaks vs. background
            is_bg = df_group['IsBackground'].fillna(False)
            df_real = df_group[~is_bg]
            df_bg = df_group[is_bg]

            # 1. Faint connecting line across ALL channels (keeps the spectrum continuous)
            ax.plot(df_group['ChannelNum'], df_group[peak_type],
                    color=color, linestyle=linestyle, linewidth=1.2, alpha=0.25, zorder=2)

            # 2. Real peaks: full color, solid fill, optional error bars
            if not df_real.empty:
                if use_err:
                    ax.errorbar(
                        df_real['ChannelNum'], df_real[peak_type],
                        yerr=log_safe_yerr(df_real[peak_type], df_real[uncert_col]),
                        color=color, linestyle='none', marker=marker, markersize=10,
                        elinewidth=1.4, capsize=8, capthick=2,
                        label=label, alpha=0.9, zorder=4
                    )
                else:
                    ax.plot(
                        df_real['ChannelNum'], df_real[peak_type],
                        color=color, linestyle='none', marker=marker, markersize=10,
                        linewidth=1.5, label=label, alpha=0.9, zorder=4
                    )
                # Connect consecutive real peaks with a full-strength line
                real_series = df_group[peak_type].where(~is_bg)
                ax.plot(df_group['ChannelNum'], real_series,
                        color=color, linestyle=linestyle, linewidth=1.8, alpha=0.85, zorder=3)

            # 3. Background points: gray-filled, faded satellite-colored border, optional faded error bars
            if not df_bg.empty:
                bg_label = label if df_real.empty else None
                if use_err:
                    ax.errorbar(
                        df_bg['ChannelNum'], df_bg[peak_type],
                        yerr=log_safe_yerr(df_bg[peak_type], df_bg[uncert_col]),
                        color=color, linestyle='none', marker=marker, markersize=9,
                        markerfacecolor='lightgray', markeredgecolor=color, markeredgewidth=1.2,
                        elinewidth=1.0, capsize=8, capthick=2, alpha=0.4, label=bg_label, zorder=3
                    )
                else:
                    ax.plot(
                        df_bg['ChannelNum'], df_bg[peak_type],
                        color=color, linestyle='none', marker=marker, markersize=9,
                        markerfacecolor='lightgray', markeredgecolor=color, markeredgewidth=1.2,
                        alpha=0.4, label=bg_label, zorder=3
                    )

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
        handles.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
                   markeredgecolor='gray', markeredgewidth=1.2, markersize=9,
                   alpha=0.5, label='Background')
        )
        ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc='upper left',
                  borderaxespad=0., title="Satellites")
        
        plt.tight_layout()
        
        filename = f"{event_name}_{title_prefix.split()[0]}_Spectrum.png"
        full_path = save_folder / filename
        
        fig.savefig(full_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

print(f"\n{'-'*50}")
print("ALL SPECTRA PLOTS GENERATED AND SAVED SAFELY.")
print(f"{'-'*50}")

#%% Plot the effective energies as a function of time
# --- Custom Stateful Date Formatting Logic ---
class SparseDateFormatter(ticker.Formatter):
    def __call__(self, x, pos=None):
        
        # --- FIX: Fetch tick locations directly from the axis ---
        locs = self.axis.get_majorticklocs() if self.axis is not None else []
        
        if pos is None or len(locs) == 0:
            return ''
            
        dt = mdates.num2date(x)
        
        if pos == 0:
            year_str = dt.strftime('%Y')
            date_str = dt.strftime('%m/%d')
        else:
            # Use the dynamically fetched locs list to check the previous tick
            prev_dt = mdates.num2date(locs[pos - 1])
            year_str = dt.strftime('%Y') if dt.year != prev_dt.year else ' '
            date_str = dt.strftime('%m/%d') if dt.date() != prev_dt.date() else ' '
            
        time_str = dt.strftime('%H:%M')
        
        return f"{time_str}\n{date_str}\n{year_str}"


# Define your base output directory for plots
plot_output_base = Path(r"C:\Users\wzt0020\Box\SEP_Box\SGPS_data\Energy_Time_Plots")

# --- Plot Configuration ---
plot_configs = {
    'Absolute Max': {'time_col': 'TotalMaxTime', 'file_suffix': 'Absolute_Max'},
    'Onset Max':    {'time_col': 'OnsetTime',    'file_suffix': 'Onset_Max'}
}

markers = {'East': 'o', 'West': 's'}  # Circle for East, Square for West

print("Generating timing plots for all events...")

# 1. Loop through each unique event
for event_name, df_event in df_event_peaks.groupby('Event'):
    
    # Create the dedicated folder for this event
    event_save_dir = plot_output_base / event_name
    event_save_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Loop through each satellite active during this event
    for sat_name, df_sat in df_event.groupby('Satellite'):
        
        # 3. Generate both Absolute and Onset plots for this Sat/Event combo
        for plot_title, config in plot_configs.items():
            time_col = config['time_col']
            file_suffix = config['file_suffix']
            
            # Drop rows where the time might be missing (NaT) or energy is NaN
            df_plot = df_sat.dropna(subset=[time_col, 'EffectiveEnergy'])
            if df_plot.empty:
                continue

            fig, ax = plt.subplots(figsize=(14, 6))

            # Scatter points grouped by physical orientation
            for orientation in ['East', 'West']:
                df_sens = df_plot[df_plot['Orientation'] == orientation]
                if df_sens.empty:
                    continue

                is_bg = df_sens['IsBackground'].fillna(False) if 'IsBackground' in df_sens.columns else pd.Series(False, index=df_sens.index)
                df_real = df_sens[~is_bg]
                df_bg = df_sens[is_bg]
                m = markers.get(orientation, 'o')

                # 1. Real peaks: full turbo color, black border, high visibility
                if not df_real.empty:
                    ax.scatter(
                        df_real[time_col],
                        df_real['EffectiveEnergy'],
                        c=df_real['ChannelNum'],
                        cmap='turbo',
                        vmin=0, vmax=12,
                        s=150,
                        marker=m,
                        edgecolor='black',
                        linewidth=1.5,
                        alpha=0.9,
                        zorder=4,
                        label=f'{orientation}'
                    )

                # 2. Background points: same turbo channel colors, but faded/see-through
                if not df_bg.empty:
                    bg_label = f'{orientation}' if df_real.empty else None
                    ax.scatter(
                        df_bg[time_col],
                        df_bg['EffectiveEnergy'],
                        c=df_bg['ChannelNum'],
                        cmap='turbo',
                        vmin=0, vmax=12,
                        s=130,
                        marker=m,
                        edgecolor='black',
                        linewidth=1.2,
                        alpha=0.3,
                        zorder=2,
                        label=bg_label
                    )

            # Annotate each point with its channel name: East to the right, West to the left
            for idx, row in df_plot.iterrows():
                is_west = row['Orientation'] == 'West'
                x_offset = -10 if is_west else 10
                h_align = 'right' if is_west else 'left'
                is_bg_row = bool(row.get('IsBackground', False))

                ax.annotate(
                    row['Channel'],
                    (row[time_col], row['EffectiveEnergy']),
                    xytext=(x_offset, 0),
                    textcoords='offset points',
                    ha=h_align,
                    va='center',
                    color='gray' if is_bg_row else 'black',
                    alpha=0.5 if is_bg_row else 1.0,
                    weight='normal' if is_bg_row else 'bold'
                )

            # Format Titles and Labels
            # event_name is usually something like "Event_20230501_1430", so we can strip the prefix for a cleaner title
            clean_event_title = event_name.replace('Event_', '')
            fig.suptitle(f'{sat_name} Time of {plot_title} Flux vs. Effective Energy\nEvent: {clean_event_title}', 
                         weight='bold', y=0.95)

            ax.set_xlabel(f'{plot_title} Time (UTC)')
            ax.set_ylabel('Effective Energy (MeV)')
            ax.set_yscale('log')

            # Set explicit linear-style tick labels on the log y-axis
            y_ticks = [1, 3, 10, 30, 100, 300]
            ax.set_yticks(y_ticks)
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
            ax.set_ylim(0.8,500)

            ax.xaxis.set_major_formatter(SparseDateFormatter())
            plt.xticks(rotation=0, ha='center')

            ax.legend(title='Sensor', bbox_to_anchor=(1.02, 1), loc='upper left')

            ax.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.7, zorder=0)
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4, zorder=0)

            plt.tight_layout()
            
            # Save the plot into the specific event's folder
            save_path = event_save_dir / f"{sat_name}_{file_suffix}_Timing.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig) 
            
    print(f"Processed plots for {event_name}")

print("All event timing plots completed.")


#%% Recreate event plots with circled flux maxima
def plot_event_flux_with_peaks(df_event, sat_name, sensor, df_peaks, event_name,
                               start_date=None, end_date=None, averaging='5min', 
                               save_dir=None, show_plot=True, clean_yaw=True,
                               vline_start=None, vline_end=None):
    """
    Plots channel proton fluxes directly from a Parquet-loaded event DataFrame 
    and overlays highly visible circle markers on the absolute and onset maxima.
    """
    if df_event.empty:
        return

    # --- 1. Extract Flux and Yaw Data ---
    # Find all columns that end in '_Flux' (e.g., 'P1_Flux', 'P2A_Flux')
    flux_cols = [col for col in df_event.columns if col.endswith('_Flux')]
    if not flux_cols:
        return
        
    df_flux_plot = df_event[flux_cols].replace(0, np.nan).copy()
    
    # Rename columns to just the channel name (e.g., 'P1') for a clean legend and easy matching
    rename_dict = {col: col.replace('_Flux', '') for col in flux_cols}
    df_flux_plot = df_flux_plot.rename(columns=rename_dict)

    has_yaw = 'yaw_flip_flag' in df_event.columns
    if has_yaw:
        df_yaw = df_event[['yaw_flip_flag']].copy()

    # --- 2. Apply Date Filters ---
    if start_date:
        df_flux_plot = df_flux_plot.loc[start_date:]
        if has_yaw: df_yaw = df_yaw.loc[start_date:]
    if end_date:
        df_flux_plot = df_flux_plot.loc[:end_date]
        if has_yaw: df_yaw = df_yaw.loc[:end_date]
        
    if df_flux_plot.empty:
        return

    # --- 3. Time Resampling ---
    if averaging != '5min':
        df_flux_plot = df_flux_plot.resample(averaging).mean()

    avg_label = {'5min': '5-Minute', '1H': 'Hourly', '1D': 'Daily'}.get(averaging, f'{averaging}')

    # --- 4. Generate the Base Plot ---
    fig, ax = plt.subplots(figsize=(18, 7))

    # Plot the lines and extract their assigned colors
    lines = df_flux_plot.plot(ax=ax, linewidth=1.5, colormap='turbo', alpha=0.8)
    ax.set_yscale('log')
    line_colors = {line.get_label(): line.get_color() for line in ax.get_lines()}

    # --- 5. Add Peak Overlay Markers ---
    df_sat_peaks = df_peaks[(df_peaks['Event'] == event_name) & 
                            (df_peaks['Satellite'] == sat_name.upper()) & 
                            (df_peaks['Sensor'] == sensor)]

    for _, peak_row in df_sat_peaks.iterrows():
        ch = peak_row['Channel']
        
        if ch in df_flux_plot.columns:
            color = line_colors.get(ch, 'black')
            is_bg = bool(peak_row.get('IsBackground', False))
            
            # Styling based on whether the channel is a real peak or background
            marker_alpha = 0.35 if is_bg else 1.0
            edge_lw = 1.2 if is_bg else 1.5
            z_marker = 4 if is_bg else 6
            
            # Absolute Max Peak
            abs_time = peak_row['TotalMaxTime']
            abs_flux = peak_row['TotalMaxFlux']
            if pd.notna(abs_time) and (df_flux_plot.index.min() <= abs_time <= df_flux_plot.index.max()):
                if not is_bg:
                    ax.scatter(abs_time, abs_flux, s=220, color='white', alpha=0.9, zorder=5)
                ax.scatter(abs_time, abs_flux, s=95 if is_bg else 110, marker='o', 
                           facecolors=color, edgecolors='black', linewidths=edge_lw, 
                           alpha=marker_alpha, zorder=z_marker)

            # Onset Max Peak (only if distinct)
            if not peak_row['IsSinglePeak']:
                onset_time = peak_row['OnsetTime']
                onset_flux = peak_row['OnsetFlux']
                if pd.notna(onset_time) and (df_flux_plot.index.min() <= onset_time <= df_flux_plot.index.max()):
                    if not is_bg:
                        ax.scatter(onset_time, onset_flux, s=200, marker='D', color='white', alpha=0.9, zorder=5)
                    ax.scatter(onset_time, onset_flux, s=85 if is_bg else 100, marker='D', 
                               facecolors=color, edgecolors='black', linewidths=edge_lw, 
                               alpha=marker_alpha, zorder=z_marker)

    # --- 6. Add Yaw Flip Lines & Orientation Labels ---
    if has_yaw and not df_yaw.empty:
        if clean_yaw:
            df_yaw['yaw_flip_flag'] = df_yaw['yaw_flip_flag'].ffill().bfill()
            is_spike = (df_yaw['yaw_flip_flag'] != df_yaw['yaw_flip_flag'].shift(1)) & \
                       (df_yaw['yaw_flip_flag'] != df_yaw['yaw_flip_flag'].shift(-1))
            df_yaw.loc[is_spike, 'yaw_flip_flag'] = np.nan
            df_yaw['yaw_flip_flag'] = df_yaw['yaw_flip_flag'].ffill().bfill()
        else:
            df_yaw['yaw_flip_flag'] = df_yaw['yaw_flip_flag'].fillna(1)
        
        df_yaw['block'] = (df_yaw['yaw_flip_flag'] != df_yaw['yaw_flip_flag'].shift(1)).cumsum()
            
        for block_id, group in df_yaw.groupby('block'):
            state = group['yaw_flip_flag'].iloc[0]
            start_t = group.index[0]
            
            orientation = None
            y_pos = 0.98
            if state == 0:
                orientation = 'West' if sensor == 0 else 'East'
            elif state == 2:
                orientation = 'East' if sensor == 0 else 'West'
            elif not clean_yaw:
                orientation = 'Maneuvering' if state == 1 else 'Unknown'
                y_pos = 0.88 if state == 1 else 0.83
            
            if start_t > df_flux_plot.index.min():
                ax.axvline(start_t, color='black', linestyle='--', linewidth=1.5, alpha=0.8, zorder=3)
            if orientation:
                ax.text(start_t, y_pos, f" {orientation} ", 
                        transform=ax.get_xaxis_transform(), color='black',
                        verticalalignment='top', horizontalalignment='left', zorder=4)

    # --- 7. Shade the Background ---
    if vline_start and vline_end:
        dt_start = pd.to_datetime(vline_start)
        dt_end = pd.to_datetime(vline_end)
        if dt_start.tzinfo is None: dt_start = dt_start.tz_localize('UTC')
        if dt_end.tzinfo is None: dt_end = dt_end.tz_localize('UTC')
        ax.axvspan(dt_start, dt_end, color='lightgray', alpha=0.3, label='Event Duration')

    # --- 8. Formatting & Legends ---
    min_date = df_flux_plot.index.min().strftime('%Y-%m-%d')
    max_date = df_flux_plot.index.max().strftime('%Y-%m-%d')
    clean_event_title = event_name.replace('Event_', '')
    
    ax.set_title(f'{sat_name.upper()} Proton Flux Peaks (Sensor {sensor}) | {avg_label} Average\nEvent: {clean_event_title}', pad=20)
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Flux (#/cm²/s/sr/MeV)')

    # Identify channels that actually have at least one valid data point in the plotted window
    valid_channels = set(df_flux_plot.dropna(axis=1, how='all').columns)

    # Keep only valid channels and non-channel items (like 'Event Duration')
    handles, labels = ax.get_legend_handles_labels()
    legend_elements = [
        h for h, lbl in zip(handles, labels)
        if (lbl in valid_channels) or (lbl not in df_flux_plot.columns)
    ]

    # Append custom peak markers
    legend_elements.extend([
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=9, 
               markeredgecolor='black', markeredgewidth=1.5, label='Absolute Max', alpha=0.9),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', markersize=8, 
               markeredgecolor='black', markeredgewidth=1.5, label='Onset Max', alpha=0.9),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, 
               markeredgecolor='black', markeredgewidth=1.2, label='Background', alpha=0.5)
    ])

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    ax.margins(x=0, y=0)
    
    # Set the y-axis upper bound to 2x the absolute max across all plotted channels
    y_max = df_flux_plot.max().max()
    if pd.notna(y_max) and y_max > 0:
        ax.set_ylim(top=y_max * 2)

    ax.grid(True, which="major", ls="--", alpha=0.5)
    plt.tight_layout()

    # --- 9. Save and Show ---
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        filename = f"{sat_name}_sensor{sensor}_peaks_{averaging}.png"
        full_filepath = save_path / filename
        fig.savefig(full_filepath, dpi=300, bbox_inches='tight')
        print(f"  -> Saved {filename}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


#%% Call function, plot maxima
plot_output_base = Path(r"C:\Users\wzt0020\Box\SEP_Box\SGPS_data\Event_Plots_Maxima")
plot_output_base.mkdir(parents=True, exist_ok=True)

pre_event_buffer = pd.Timedelta(hours=24)
post_event_buffer = pd.Timedelta(hours=24)
plot_averaging = '5min'

print(f"Generating flux plots with peak overlays for {len(loaded_events)} events...")

for event_name, satellites in loaded_events.items():
    
    # Retrieve the exact boundaries from the catalog for shading and slicing
    event_row = sep_catalog[sep_catalog['Event_Folder'] == event_name]
    if event_row.empty:
        continue
    event_row = event_row.iloc[0]
    
    start_raw = event_row['Begin_Time_UTC']
    end_raw = event_row['End_Time_UTC']
    if pd.isna(start_raw) or pd.isna(end_raw):
        continue

    event_start = pd.to_datetime(start_raw)
    event_end = pd.to_datetime(end_raw)
    if event_start.tzinfo is None: event_start = event_start.tz_localize('UTC')
    if event_end.tzinfo is None: event_end = event_end.tz_localize('UTC')
    
    event_start_str = (event_start - time_difference).strftime('%Y-%m-%d %H:%M')
    event_end_str = event_end.strftime('%Y-%m-%d %H:%M')
    
    # Calculate visual bounds for the plot axis (buffered)
    plot_start = event_start - pre_event_buffer
    plot_end = event_end + post_event_buffer
    
    # Shift plot start back by threshold to accommodate the onset window
    plot_start_str = (plot_start - time_difference).strftime('%Y-%m-%d %H:%M')
    plot_end_str = plot_end.strftime('%Y-%m-%d %H:%M')
    
    event_save_dir = plot_output_base / event_name
    print(f"\nProcessing {event_name}...")
    
    for sat_name, sensors in satellites.items():
        for sensor, df_event in sensors.items():
            
            # Call the new DataFrame-based plotting function
            plot_event_flux_with_peaks(
                df_event=df_event,
                sat_name=sat_name,
                sensor=sensor,
                df_peaks=df_event_peaks,
                event_name=event_name,
                start_date=plot_start_str,
                end_date=plot_end_str,
                averaging=plot_averaging,
                save_dir=event_save_dir,
                show_plot=False,
                clean_yaw=True,
                vline_start=event_start_str,
                vline_end=event_end_str
            )

print("Peak overlay plotting complete.")
