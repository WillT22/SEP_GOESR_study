#%% Import necessary libraries
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from scipy.signal import find_peaks

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

#%% Import GOES-R Data
# The strict ordered list of instrument channels
SGPS_CHANNELS = ['P1', 'P2A', 'P2B', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8A', 'P8B', 'P8C', 'P9', 'P10']
# Generate reverse mapping automatically for strict sorting in other scripts: {'P1': 0, 'P2A': 1...}
CHANNEL_TO_NUM = {ch: i for i, ch in enumerate(SGPS_CHANNELS)}

DURATION_CHANNELS = ['P4', 'P5', 'P6', 'P7', 'P8A', 'P8B', 'P8C', 'P9', 'P10']
LOW_ENERGY_CHANNELS = ['P1', 'P2A', 'P2B', 'P3']

# Define the base directory containing the satellite folders
BASE_DATA_DIR = Path(r"C:\Users\wzt0020\Box\SEP_Box\SGPS_data")

SATELLITES = ['goes16', 'goes17', 'goes18', 'goes19']

data_columns = [
    'DiffProtonLowerEnergy', 'DiffProtonUpperEnergy', 
    'DiffProtonEffectiveEnergy', 'AvgDiffProtonFlux', 'AvgDiffProtonFluxUncert', 'yaw_flip_flag'
]

# Initialize the nested dictionary
goes_data = defaultdict(lambda: defaultdict(dict))

for sat_name in SATELLITES:
    print(f"{'-'*50}")
    
    # Define the path to this specific satellite's Parquet file
    parquet_file = BASE_DATA_DIR / f"{sat_name}_master_data.parquet"
    
    if not parquet_file.exists():
        print(f"No Parquet file found for {sat_name.upper()}. Skipping...")
        continue
        
    print(f"Loading {sat_name.upper()}...")
    df_sat = pd.read_parquet(parquet_file)
    
    available_data_columns = [col for col in data_columns if col in df_sat.columns]
    
    # Group the single satellite's data
    grouped = df_sat.groupby(['sensor_units', 'diff_channels'])
    
    for (sensor, channel), group_df in grouped:
        s_key = int(sensor)
        c_key = int(channel)
        
        try:
            channel_name = SGPS_CHANNELS[c_key]
        except IndexError:
            channel_name = f"Unknown_{c_key}"
        
        # Set time as index, sort chronologically, and subset columns
        indexed_df = group_df.set_index('time').sort_index()
        goes_data[sat_name][s_key][channel_name] = indexed_df[available_data_columns]
        
    print(f"{sat_name.upper()} dictionary populated successfully.")

print(f"\nAll available satellite data loaded into goes_data.")

print("Applying historical yaw flip correction to goes_data...")

for sat_name, sensors in goes_data.items():
    for sensor, channels in sensors.items():
        for channel, df_channel in channels.items():
            
            if 'yaw_flip_flag' in df_channel.columns and not df_channel.empty:
                first_valid_idx = df_channel['yaw_flip_flag'].first_valid_index()
                
                if first_valid_idx is not None:
                    # Fill everything before the first official recorded flag with 0 (Upright)
                    df_channel.loc[:first_valid_idx, 'yaw_flip_flag'] = df_channel.loc[:first_valid_idx, 'yaw_flip_flag'].fillna(0)
                else:
                    # If there is no valid yaw data at all (e.g., historical GOES-16/17), assume 0 (Upright)
                    df_channel['yaw_flip_flag'] = df_channel['yaw_flip_flag'].fillna(0)

print("Historical yaw correction complete.")

print(f"Dictionary successfully loaded. Available satellites: {list(goes_data.keys())}")

#%% Plot all data for all channels
def plot_goes_flux(goes_data, sat_name, sensor, start_date=None, end_date=None, 
                   averaging='5min', save_dir=None, show_plot=True, clean_yaw=True,
                   vline_start=None, vline_end=None):
    """
    Plots all channel proton fluxes for a specific GOES satellite and sensor.
    
    Parameters:
    -----------
    goes_data : dict
        The nested master dictionary containing the satellite data.
    sat_name : str
        The satellite key (e.g., 'goes19').
    sensor : int
        The sensor number (e.g., 0 or 1).
    start_date : str or None
        Start date string (e.g., '2025-05-31 17:00').
    end_date : str or None
        End date string (e.g., '2025-06-05').
    averaging : str
        Pandas offset string ('5min', '1H', '1D').
    save_dir : str
        Directory to save figure. Disabled if 'None'.
    show_plot : bool
        Boolean of showing the plot in python.
    clean_yaw : bool
        If True, despikes yaw telemetry, fills gaps, and cleanly staggers labels. 
        If False, plots raw telemetry gaps, spikes, and Maneuvering/Unknown states.
    vline_start : str or None
        Actual event start date string (e.g., '2025-05-31 17:00').
    vline_end : str or None
        Actual event end date string (e.g., '2025-06-05').
    """
    
    if sat_name not in goes_data or sensor not in goes_data[sat_name]:
        print(f"Error: No data found for {sat_name.upper()} Sensor {sensor}.")
        return

    # Catch completely empty sensors
    if not goes_data[sat_name][sensor]:
        print(f"No channels populated for {sat_name.upper()} Sensor {sensor}. Skipping...")
        return

    # --- 1. Extract Flux and Yaw Data ---
    flux_dict = {}
    
    first_ch = list(goes_data[sat_name][sensor].keys())[0]
    has_yaw = 'yaw_flip_flag' in goes_data[sat_name][sensor][first_ch].columns
    
    if has_yaw:
        df_yaw = goes_data[sat_name][sensor][first_ch][['yaw_flip_flag']].copy()

    for ch_name in goes_data[sat_name][sensor].keys():
        col_name = f'S{sensor}_{ch_name}'
        flux_dict[col_name] = goes_data[sat_name][sensor][ch_name]['AvgDiffProtonFlux']

    df_flux_plot = pd.DataFrame(flux_dict).replace(0, np.nan)

    # --- Filter out massive multi-month data gaps (Pre-operational testing anomalies) ---
    if not df_flux_plot.empty and len(df_flux_plot) > 2:
        time_diffs = df_flux_plot.index.to_series().diff()
        max_gap = time_diffs.max()
        
        if pd.notna(max_gap) and max_gap > pd.Timedelta(days=90):
            gap_idx = time_diffs.argmax()
            points_before = gap_idx
            points_after = len(df_flux_plot) - gap_idx
            
            if points_before < points_after:
                df_flux_plot = df_flux_plot.iloc[gap_idx:]
            else:
                df_flux_plot = df_flux_plot.iloc[:gap_idx]
                
            if has_yaw and not df_yaw.empty:
                df_yaw = df_yaw.loc[df_flux_plot.index.min() : df_flux_plot.index.max()]

    # --- 2. Apply Date Filters ---
    if start_date:
        df_flux_plot = df_flux_plot.loc[start_date:]
        if has_yaw: df_yaw = df_yaw.loc[start_date:]
    if end_date:
        df_flux_plot = df_flux_plot.loc[:end_date]
        if has_yaw: df_yaw = df_yaw.loc[:end_date]
        
    if df_flux_plot.empty:
        print(f"No data available for {sat_name.upper()} Sensor {sensor} in the specified date range.")
        return

    # --- 3. Time Resampling ---
    if averaging != '5min':
        df_flux_plot = df_flux_plot.resample(averaging).mean()

    avg_label = {'5min': '5-Minute', '1H': 'Hourly', '1D': 'Daily'}.get(averaging, f'{averaging}')

    # --- 4. Generate the Base Plot ---
    fig, ax = plt.subplots(figsize=(18, 7))

    df_flux_plot.plot(ax=ax, linewidth=1.5, colormap='turbo')
    ax.set_yscale('log')

    # --- 5. Add Yaw Flip Lines & Orientation Labels ---
    if has_yaw and not df_yaw.empty:

        # Apply cleanup logic ONLY if clean_yaw is True
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
            end_t = group.index[-1]

            if clean_yaw:
                # CLEAN VERSION: Centered text, ignore state 1
                orientation = None 
                if state == 0:
                    orientation = 'West' if sensor == 0 else 'East'
                    y_pos = 0.98
                elif state == 2:
                    orientation = 'East' if sensor == 0 else 'West'
                    y_pos = 0.98
                
                if start_t > df_flux_plot.index.min():
                    ax.axvline(start_t, color='black', linestyle='--', linewidth=1.5, alpha=0.8, zorder=3)
                
                if orientation:
                    ax.text(start_t, y_pos, f" {orientation} ", 
                            transform=ax.get_xaxis_transform(), color='black',
                            verticalalignment='top', horizontalalignment='left', zorder=4)
            else:
                # MESSY VERSION: Left-aligned text, show all states, raw heights
                if state == 0:
                    orientation = 'West' if sensor == 0 else 'East'
                    y_pos = 0.98
                elif state == 2:
                    orientation = 'East' if sensor == 0 else 'West'
                    y_pos = 0.98
                elif state == 1:
                    orientation = 'Maneuvering'
                    y_pos = 0.88
                else:
                    orientation = 'Unknown'
                    y_pos = 0.83
                
                if start_t > df_flux_plot.index.min():
                    ax.axvline(start_t, color='black', linestyle='--', linewidth=1.5, alpha=0.8, zorder=3)
                
                ax.text(start_t, y_pos, f" {orientation} ", 
                        transform=ax.get_xaxis_transform(), color='black',
                        verticalalignment='top', horizontalalignment='left', zorder=4)
    else:
        # Static label fallback
        orientation = 'West' if sensor == 0 else 'East'
        ax.text(start_t, 0.98, f" {orientation} ", 
                transform=ax.get_xaxis_transform(), color='black',
                verticalalignment='top', zorder=4)

    # --- Shade the background during the exact event duration ---
    if vline_start and vline_end:
        dt_start = pd.to_datetime(vline_start)
        dt_end = pd.to_datetime(vline_end)
        
        # Localize to UTC to match the index, if necessary
        if dt_start.tzinfo is None: dt_start = dt_start.tz_localize('UTC')
        if dt_end.tzinfo is None: dt_end = dt_end.tz_localize('UTC')
        
        # axvspan creates a shaded rectangle between two x-values
        # alpha=0.3 makes it transparent so the grid and data lines still show through
        ax.axvspan(dt_start, dt_end, color='lightgray', alpha=0.3, label='Event Duration')

    # --- 6. Formatting & Legends ---
    min_date = df_flux_plot.index.min().strftime('%Y-%m-%d')
    max_date = df_flux_plot.index.max().strftime('%Y-%m-%d')
    ax.set_title(f'{sat_name.upper()} Proton Flux Data (All Channels, Sensor {sensor}) | {avg_label} Average\n({min_date} to {max_date})', pad=20)

    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Flux (#/cm²/s/sr/MeV)')

    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    ax.margins(x=0, y=0)
    plt.tight_layout()

    # --- 7. Save and Show ---
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Append state suffix so they don't overwrite each other
        state_suffix = "clean" if clean_yaw else "messy"
        filename = f"{sat_name}_sensor{sensor}_{averaging}_{state_suffix}.png"
        
        full_filepath = save_path / filename
        fig.savefig(full_filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# Plot all satellite data for operaiton period
PLOT_FULL = False
if PLOT_FULL:    
    for satellite in goes_data.keys():
        for sensor in goes_data[satellite].keys():
            plot_goes_flux(goes_data, satellite, sensor, 
                        save_dir=r'C:\Users\wzt0020\Box\SEP_Box\SGPS_data\data_figures',
                        clean_yaw=False)


#%% Import SEP Catalog
# =======================================================
#         Event Selection / Catalog Processing
# =======================================================

url = "https://www.ngdc.noaa.gov/stp/space-weather/interplanetary-data/solar-proton-events/SEP%20page%20code.html"
sep_tables = pd.read_html(url)
sep_catalog = sep_tables[0]

# Overwrite the original columns with the converted timezone-aware datetimes
sep_catalog['Begin Time Yr M/D (UTC)'] = pd.to_datetime(
    sep_catalog['Begin Time Yr M/D (UTC)'], 
    format='%Y %m/%d %H%M', 
    utc=True, 
    errors='coerce'
)

# Note that 1979 07/07/1010 is a typo in the NOAA catalog that will become NaT after conversion.
sep_catalog['Maximum Time Yr M/D (UTC)'] = pd.to_datetime(
    sep_catalog['Maximum Time Yr M/D (UTC)'], 
    format='%Y %m/%d %H%M', 
    utc=True, 
    errors='coerce'
)

# Rename the headers to your preferred uniform format
sep_catalog = sep_catalog.rename(columns={
    'Begin Time Yr M/D (UTC)': 'Begin_Time_UTC',
    'Maximum Time Yr M/D (UTC)': 'Maximum_Time_UTC'
})

#%% Find the end time of each event
# Implement channel-based threshold
CHANNEL_THRESHOLDS = {
    'P1':  1e-1,    'P2A': 1e-1,    'P2B': 1e-1,
    'P3':  1e-2,    'P4':  1e-2,    'P5':  1e-2,
    'P6':  1e-3,    'P7':  1e-3,    'P8A': 1e-3,
    'P8B': 1e-3,    'P8C': 1e-3,    'P9':  1e-3,
    'P10': 1e-3
}
threshold_array = np.array([CHANNEL_THRESHOLDS[ch] for ch in SGPS_CHANNELS])

# At a 5-minute cadence, 6 points = 30 minutes, 12 points = 1 hour.
REQUIRED_CONSECUTIVE_POINTS = 3
col_name = 'End_Time_UTC'

# Clean the column names by removing any hidden leading/trailing spaces
sep_catalog.columns = sep_catalog.columns.str.strip()

# 1. Insert the new End Time column into the main catalog for the CSV
if col_name not in sep_catalog.columns:
    sep_catalog.insert(1, col_name, pd.Series(pd.NaT, index=sep_catalog.index, dtype='datetime64[ns, UTC]'))
else:
    sep_catalog[col_name] = pd.Series(pd.NaT, index=sep_catalog.index, dtype='datetime64[ns, UTC]')

# List to hold the comprehensive data for the Parquet file
multi_channel_records = []

# Ensure the catalog is strictly chronological, then shift the start times up by 1
sep_catalog['Begin_Time_UTC'] = pd.to_datetime(sep_catalog['Begin_Time_UTC'])
sep_catalog = sep_catalog.sort_values('Begin_Time_UTC').reset_index(drop=True)
sep_catalog['Next_Event_Start_UTC'] = sep_catalog['Begin_Time_UTC'].shift(-1)

# 2. Iterate through the DataFrame rows
for idx, row in sep_catalog.iterrows():
    
    event_start = pd.to_datetime(row['Begin_Time_UTC'])
    if event_start.tzinfo is None:
        event_start = event_start.tz_localize('UTC')
        
    min_end_time = event_start + pd.Timedelta(hours=6)
    
    # Enforce a strict maximum physical duration (e.g., 10 days)
    max_physical_end = event_start + pd.Timedelta(days=10)
    next_event_start = row['Next_Event_Start_UTC']
    
    if pd.notna(next_event_start):
        max_end_time = pd.to_datetime(next_event_start)
        if max_end_time.tzinfo is None: max_end_time = max_end_time.tz_localize('UTC')
        
        # Cap it so events don't stretch into a 10-day abyss if the next event is far away
        if max_end_time > max_physical_end:
            max_end_time = max_physical_end
            
        # Safeguard if events are back-to-back
        if max_end_time < min_end_time:
            min_end_time = max_end_time
    else:
        max_end_time = max_physical_end
    
    event_record = {'Begin_Time_UTC': event_start}
    valid_duration_end_times = []
    
    # 3. Calculate actual decay only for the core ~>10 MeV channels
    for ch in DURATION_CHANNELS:
        ch_threshold = CHANNEL_THRESHOLDS.get(ch, 1e-3)
        potential_end_times = []
        
        for satellite in goes_data.keys():
            for sensor in goes_data[satellite].keys():
                
                if ch in goes_data[satellite][sensor]:
                    df_ch = goes_data[satellite][sensor][ch]
                    
                    df_ch = df_ch[~df_ch.index.duplicated(keep='last')].sort_index()
                    active_mask = (df_ch.index >= event_start) & (df_ch.index <= min_end_time)
                    
                    if not df_ch[active_mask].empty:
                        time_mask = (df_ch.index > min_end_time) & (df_ch.index <= max_end_time)
                        df_window = df_ch[time_mask]
                        
                        if not df_window.empty:
                            below_thresh = df_window['AvgDiffProtonFlux'] < ch_threshold
                            rolling_count = below_thresh.rolling(window=REQUIRED_CONSECUTIVE_POINTS).sum()
                            sustained_dips = rolling_count[rolling_count == REQUIRED_CONSECUTIVE_POINTS]
                            
                            if not sustained_dips.empty:
                                end_of_block_time = sustained_dips.index[0]
                                end_iloc = df_window.index.get_loc(end_of_block_time)
                                first_dip_time = df_window.index[end_iloc - REQUIRED_CONSECUTIVE_POINTS + 1]
                                potential_end_times.append(first_dip_time)
                                
        
        if potential_end_times:
            final_ch_end_time = max(potential_end_times)
            valid_duration_end_times.append(final_ch_end_time)
        else:
            final_ch_end_time = pd.NaT
            
        event_record[f'{ch}_End_Time'] = final_ch_end_time

    # 4. Establish the Master End Time from the core channels
    if valid_duration_end_times:
        # Trust the max of the healthy channels that actually decayed
        master_end_time = max(valid_duration_end_times)
    else:
        # Bleed-over fallback: if NO channels dipped, use the boundary cutoff
        master_end_time = max_end_time
        
    # Save to the main catalog for CSV export and plot shading
    sep_catalog.loc[idx, col_name] = master_end_time

    # 5. Force ANY channel that failed to dip (or was skipped) to inherit the Master End Time ---
    for ch in SGPS_CHANNELS:
        # If the channel is currently NaT (because it didn't dip, or is P1/P2A/P2B/P3)
        if pd.isna(event_record.get(f'{ch}_End_Time')):
            event_record[f'{ch}_End_Time'] = master_end_time

    multi_channel_records.append(event_record)

print("Event end times calculated successfully.")

# =============================================================================
# EXPORT DATA
# =============================================================================
base_out_dir = Path(r"C:\Users\wzt0020\Box\SEP_Box\SGPS_data")

# Export 1: The standard CSV with only the P4 end time
csv_output_file = base_out_dir / "updated_sep_catalog.csv"
sep_catalog.to_csv(csv_output_file, index=False)
print(f"Updated SEP catalog (P4 End Time) saved to: {csv_output_file}")

# Export 2: The comprehensive Parquet file with all channel end times
df_all_channels = pd.DataFrame(multi_channel_records)
parquet_output_file = base_out_dir / "sep_catalog_all_channels.parquet"
df_all_channels.to_parquet(parquet_output_file)
print(f"Multi-channel end time catalog saved to: {parquet_output_file}")

#%% Plot all events
# --- Configuration ---
# Set the base directory where you want the plot folders to go
plot_output_base = Path(r"C:\Users\wzt0020\Box\SEP_Box\SGPS_data\Event_Plots")
plot_output_base.mkdir(parents=True, exist_ok=True)

# Define a visual buffer so the plot doesn't start/end exactly on the event boundaries
pre_event_buffer = pd.Timedelta(hours=24)
post_event_buffer = pd.Timedelta(hours=24)

# Set the averaging for the event plots ('5min', '1H', etc.)
plot_averaging = '5min'

# --- Plotting Loop ---
print(f"Starting batch plotting for {len(sep_catalog)} catalog entries...")

for idx, row in sep_catalog.iterrows():
    start_raw = row['Begin_Time_UTC']
    end_raw = row['End_Time_UTC']
    
    # 1. Skip events that don't have both a start and end time
    if pd.isna(start_raw) or pd.isna(end_raw):
        continue
        
    # Ensure they are timezone-aware pandas datetimes
    event_start = pd.to_datetime(start_raw)
    event_end = pd.to_datetime(end_raw)
    if event_start.tzinfo is None: event_start = event_start.tz_localize('UTC')
    if event_end.tzinfo is None: event_end = event_end.tz_localize('UTC')
    event_start_str = event_start.strftime('%Y-%m-%d %H:%M')
    event_end_str = event_end.strftime('%Y-%m-%d %H:%M')
    
    # 2. Apply the visual buffer and format into clean strings for the plotting function
    plot_start = event_start - pre_event_buffer
    plot_end = event_end + post_event_buffer
    
    plot_start_str = plot_start.strftime('%Y-%m-%d %H:%M')
    plot_end_str = plot_end.strftime('%Y-%m-%d %H:%M')
    
    # Create a clean folder name for this specific event (e.g., "Event_20230501_1430")
    event_folder_name = f"Event_{event_start.strftime('%Y%m%d_%H%M')}"
    event_save_dir = plot_output_base / event_folder_name
    
    print(f"\nProcessing {event_folder_name} (Index {idx})...")
    
    # 3. Loop through all active satellites and sensors
    for satellite in goes_data.keys():
        for sensor in goes_data[satellite].keys():
            
            # Check if this specific satellite/sensor has data in this time window
            # (Grabbing the first available channel just to check the index bounds)
            first_ch = list(goes_data[satellite][sensor].keys())[0]
            df_check = goes_data[satellite][sensor][first_ch]
            
            active_mask = (df_check.index >= plot_start) & (df_check.index <= plot_end)
            
            if not df_check[active_mask].empty:
                print(f"  -> Plotting {satellite.upper()} Sensor {sensor}...")
                
                # 4. Call plotting function
                plot_goes_flux(
                    goes_data=goes_data,
                    sat_name=satellite,
                    sensor=sensor,
                    start_date=plot_start_str,
                    end_date=plot_end_str,
                    averaging=plot_averaging,
                    save_dir=event_save_dir,   # Save into this event's dedicated folder
                    show_plot=False,           # CRITICAL: Keep False so it doesn't crash your browser/RAM
                    clean_yaw=True,             # Use the clean staggered E/W text labels
                    vline_start=event_start_str,
                    vline_end=event_end_str
                )

print(f"\n{'-'*50}")
print("ALL EVENT PLOTS GENERATED AND SAVED.")
print(f"{'-'*50}")

#%% Save all event data excluding missing or 1 yaw_flip_flags
data_output_base = Path(r"C:\Users\wzt0020\Box\SEP_Box\SGPS_data\Event_Data")
data_output_base.mkdir(parents=True, exist_ok=True)

print(f"Extracting raw event data for {len(df_all_channels)} catalog entries...")

for idx, row in df_all_channels.iterrows():
    
    # Get the global start time for the event
    global_start = pd.to_datetime(row['Begin_Time_UTC'])
    if pd.isna(global_start):
        continue
        
    if global_start.tzinfo is None: 
        global_start = global_start.tz_localize('UTC')
        
    event_str = global_start.strftime('%Y%m%d_%H%M')
    
    for sat in goes_data.keys():
        for sensor in goes_data[sat].keys():
            
            # Dictionary to hold the sliced columns before combining
            sensor_event_data = {}
            yaw_extracted = False
            
            for ch in SGPS_CHANNELS:
                if ch in goes_data[sat][sensor]:
                    df_ch = goes_data[sat][sensor][ch]
                    
                    # 1. Determine Start Time (Fallback to global if a channel-specific start isn't defined)
                    ch_start_col = f'{ch}_Begin_Time'
                    if ch_start_col in row and pd.notna(row[ch_start_col]):
                        start_t = pd.to_datetime(row[ch_start_col])
                    else:
                        start_t = global_start
                        
                    # 2. Determine End Time
                    ch_end_col = f'{ch}_End_Time'
                    if ch_end_col in row and pd.notna(row[ch_end_col]):
                        end_t = pd.to_datetime(row[ch_end_col])
                    else:
                        # Skip this channel entirely if it has no valid end time for this event
                        continue
                        
                    if start_t.tzinfo is None: start_t = start_t.tz_localize('UTC')
                    if end_t.tzinfo is None: end_t = end_t.tz_localize('UTC')
                    
                    # 3. Slice the data precisely to this channel's start/end window
                    df_sliced = df_ch.loc[start_t:end_t]
                    
                    if not df_sliced.empty:
                        # Grab the flux and uncert for this specific channel
                        sensor_event_data[f'{ch}_Flux'] = df_sliced['AvgDiffProtonFlux']
                        sensor_event_data[f'{ch}_Energy'] = df_sliced['DiffProtonEffectiveEnergy']
                        
                        if 'AvgDiffProtonFluxUncert' in df_sliced.columns:
                            sensor_event_data[f'{ch}_Uncert'] = df_sliced['AvgDiffProtonFluxUncert']
                        
                        # Grab the yaw flag just once for this sensor, since it applies to all channels
                        if not yaw_extracted and 'yaw_flip_flag' in df_sliced.columns:
                            sensor_event_data['yaw_flip_flag'] = df_sliced['yaw_flip_flag']
                            yaw_extracted = True
            
            # 4. If we successfully extracted data for this sat/sensor, combine and save
            if sensor_event_data:
                # pd.DataFrame automatically aligns all the differently-sized series by their DatetimeIndex!
                df_combined = pd.DataFrame(sensor_event_data)
                
                # Filter out missing or state 1 (Maneuvering) yaw_flip_flags
                if 'yaw_flip_flag' in df_combined.columns:
                    # Keeping only 0 (Upright) and 2 (Inverted) inherently excludes 1 and NaN
                    df_combined = df_combined[df_combined['yaw_flip_flag'].isin([0, 2])]
                
                # Ensure the DataFrame didn't become completely empty after filtering
                if not df_combined.empty:
                    # Create a subfolder for the event to keep things organized
                    event_folder = data_output_base / f"Event_{event_str}"
                    event_folder.mkdir(parents=True, exist_ok=True)
                    
                    # Save to Parquet
                    out_file = event_folder / f"{sat}_sensor{sensor}_event_data.parquet"
                    df_combined.to_parquet(out_file)

print(f"\n{'-'*50}")
print("ALL EVENT PARQUET FILES SAVED SUCCESSFULLY.")
print(f"{'-'*50}")
# %%
