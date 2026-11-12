#%% Import necessary libraries
import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

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

# SGPS Paper https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021SW002750
# SEPEM Study Paper https://www.aanda.org/articles/aa/full_html/2024/11/aa50046-24/aa50046-24.html

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# The strict ordered list of instrument channels
SGPS_CHANNELS = ['P1', 'P2A', 'P2B', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8A', 'P8B', 'P8C', 'P9', 'P10']
# Generate reverse mapping automatically for strict sorting in other scripts: {'P1': 0, 'P2A': 1...}
CHANNEL_TO_NUM = {ch: i for i, ch in enumerate(SGPS_CHANNELS)}

# Define the base directory containing the satellite folders
BASE_DATA_DIR = Path(r"C:\Users\wzt0020\Box\SEP_Box\SGPS_data")

# --- DIAGNOSTIC MODE SETTINGS ---
# Set to a specific string to isolate the run, or set to None to process all
DIAGNOSTIC_SATELLITE = None  # e.g., 'goes19' or None
DIAGNOSTIC_YEAR = None        # e.g., '2026' or None

SATELLITES = [DIAGNOSTIC_SATELLITE] if DIAGNOSTIC_SATELLITE else ['goes16', 'goes17', 'goes18', 'goes19']

# Initialize the master data structures
goes_data = defaultdict(lambda: defaultdict(dict))
yaw_ledgers = {}  # Will store ledgers only for GOES-17 and GOES-19

#%% Generate Functions
# =============================================================================
# FUNCTIONS
# =============================================================================
def generate_yaw_ledger(df):
    """
    Extracts a chronological ledger of spacecraft yaw maneuvers by using 
    the telemetry yaw_flip_flag (0 = Upright, 1 = Maneuvering, 2 = Inverted).
    Data gaps > 45 minutes are treated as a '1' (Maneuvering).
    """
    # 1. Isolate unique timestamps and the yaw flag, sort chronologically
    df_time = df[['time', 'yaw_flip_flag']].drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)
    
    ledger_data = []
    
    # Trackers for the loop
    in_maneuver = False
    flip_start = None
    
    # Find the initial stable state to prevent logging the start of the dataset as a flip
    stable_states = df_time[df_time['yaw_flip_flag'].isin([0, 2])]
    if stable_states.empty:
        return pd.DataFrame()
    
    last_stable_flag = stable_states.iloc[0]['yaw_flip_flag']
    
    # 2. Iterate through the timeline to detect state changes
    for i in range(1, len(df_time)):
        prev_time = df_time.loc[i-1, 'time']
        curr_time = df_time.loc[i, 'time']
        
        curr_flag = df_time.loc[i, 'yaw_flip_flag']
        
        # Check for gaps > 45 minutes (which we treat as a maneuver / flag 1)
        dt = curr_time - prev_time
        is_gap = dt > pd.Timedelta(minutes=45)
        
        if not in_maneuver:
            # A maneuver begins if we hit a 1, a gap, or a direct jump to a new stable state
            if curr_flag == 1 or is_gap or (curr_flag in [0, 2] and curr_flag != last_stable_flag):
                in_maneuver = True
                flip_start = prev_time
        
        if in_maneuver:
            # The maneuver concludes when the telemetry settles on a stable 0 or 2
            if curr_flag in [0, 2]:
                
                # Only record to the ledger if the state actually changed (ignores random telemetry drops)
                if curr_flag != last_stable_flag:
                    flip_end = curr_time
                    duration = flip_end - flip_start
                    config = 'Inverted (S0=East, S1=West)' if curr_flag == 2 else 'Upright (S0=West, S1=East)'
                    
                    ledger_data.append({
                        'Flip Start (UTC)': flip_start,
                        'Flip End (UTC)': flip_end,
                        'Duration': duration,
                        'New Configuration': config,
                        'New State Flag': int(curr_flag)
                    })
                    
                    # Update the stable tracker
                    last_stable_flag = curr_flag
                    
                # Reset maneuver tracker
                in_maneuver = False
                
    return pd.DataFrame(ledger_data)

def get_satellite_energy_ledger(goes_dict, satellite, sensors=[0, 1]):
    """
    Scans the nested dictionary for calibration changes across all sensors 
    and channels, returning a single consolidated DataFrame.
    """
    # Reordered to strictly follow: Low, High, Effective
    energy_cols = ['DiffProtonLowerEnergy', 'DiffProtonUpperEnergy', 'DiffProtonEffectiveEnergy']
    col_names = ['Lower', 'Upper', 'Effective']
    
    all_changes = []
    
    for sensor in sensors:
        if sensor not in goes_dict[satellite]:
            continue
            
        sensor_dict = goes_dict[satellite][sensor]
        
        for channel_key, df_channel in sensor_dict.items():
            # 1. Isolate the 3 energy columns, drop empty rows, round strictly to 2 decimals
            df_en = df_channel[energy_cols].dropna().round(2)
            
            if df_en.empty:
                continue
                
            # Rename columns for the final clean output
            df_en.columns = col_names
                
            # 2. Strip duplicate overlapping timestamps caused by midnight NetCDF boundaries
            df_en = df_en[~df_en.index.duplicated(keep='first')].sort_index()
            
            # 3. Extract rows where AT LEAST ONE of the three columns changed
            changes_mask = (df_en != df_en.shift(1)).any(axis=1)
            df_changed = df_en[changes_mask].copy()
            
            # 4. Insert Sensor and Channel as the first two columns
            df_changed.insert(0, 'Sensor', sensor)
            df_changed.insert(1, 'Channel', f"{channel_key}")
            
            all_changes.append(df_changed)
            
    if not all_changes:
        return pd.DataFrame()
        
    # 5. Concatenate all DataFrames
    df_energy_ledger = pd.concat(all_changes)
    
    # Define the exact sorting order for the new instrument channel names
    channel_order = {
        'P1': 0, 'P2A': 1, 'P2B': 2, 'P3': 3, 'P4': 4, 'P5': 5, 
        'P6': 6, 'P7': 7, 'P8A': 8, 'P8B': 9, 'P8C': 10, 'P9': 11, 'P10': 12
    }

    # 6. Map the channel strings to their numerical rank for strict sorting
    df_energy_ledger['ChannelNum'] = df_energy_ledger['Channel'].map(channel_order)
    
    # Ensure the index has a name so we can reference it easily during the sort
    idx_name = df_energy_ledger.index.name if df_energy_ledger.index.name else 'Timestamp'
    df_energy_ledger.index.name = idx_name
    
    # Reset the index to sort by Timestamp -> Sensor -> ChannelNum, then put the index back
    df_energy_ledger = df_energy_ledger.reset_index()
    df_energy_ledger = df_energy_ledger.sort_values(by=[idx_name, 'Sensor', 'ChannelNum'])
    df_energy_ledger = df_energy_ledger.set_index(idx_name)
    
    # Drop the temporary sorting column
    df_energy_ledger = df_energy_ledger.drop(columns=['ChannelNum'])
    
    return df_energy_ledger

#%% Import SEP Catalog
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


#%% Import all GOES data files and populate the dictionary
# Note: Sensor 0 is -X (West) and Sensor 1 is +X (East). This is reversed when yaw_flip_flag is 2.
# =============================================================================
# DATA EXTRACTION & PROCESSING LOOP
# =============================================================================
vars_to_keep_base = [
    'sensor_units', 'yaw_flip_flag', 'diff_channels',
    'DiffProtonLowerEnergy', 'DiffProtonUpperEnergy', 
    'DiffProtonEffectiveEnergy', 'AvgDiffProtonFlux', 'AvgDiffProtonFluxUncert'
]

data_columns = [
    'DiffProtonLowerEnergy', 'DiffProtonUpperEnergy', 
    'DiffProtonEffectiveEnergy', 'AvgDiffProtonFlux', 'AvgDiffProtonFluxUncert', 'yaw_flip_flag'
]

for sat_name in SATELLITES:
    print(f"\n{'='*85}")
    print(f"STARTING IMPORT FOR: {sat_name.upper()}")
    print(f"{'='*85}")
    
    # Build dynamic path
    sat_dir = BASE_DATA_DIR / sat_name / "sgps-l2-avg5m"
    
    if not sat_dir.exists():
        print(f"Directory not found: {sat_dir}. Skipping {sat_name.upper()}...")
        continue

    all_file_paths = sorted(list(sat_dir.rglob("*.nc")))
    print(f"Found {len(all_file_paths)} total files across all years.")

    # Group files by year
    files_by_year = defaultdict(list)
    for file in all_file_paths:
        try:
            year = file.name.split('_d')[1][:4]
            files_by_year[year].append(file)
        except IndexError:
            continue

    all_daily_dfs = []

    # Apply Diagnostic Year Filter
    years_to_run = [DIAGNOSTIC_YEAR] if DIAGNOSTIC_YEAR else sorted(files_by_year.keys())

    # 1. Load Data
    for year in years_to_run:
        if year not in files_by_year:
            print(f"\n--- No data found for Year: {year} ---")
            continue
            
        year_files = files_by_year[year]
        print(f"\n--- Loading Data for Year: {year} ---")
        
        for file in tqdm(year_files, desc=f"Processing {year}", unit="file"):
            with xr.open_dataset(file, engine="netcdf4") as dataset:
                time_col = 'L2_SciData_TimeStamp' if 'L2_SciData_TimeStamp' in dataset.variables else 'time'
                    
                vars_in_file = [v for v in vars_to_keep_base if v in dataset.variables] + [time_col]
                
                df_daily = dataset[vars_in_file].to_dataframe().reset_index()
                
                if time_col == 'L2_SciData_TimeStamp' and 'L2_SciData_TimeStamp' in df_daily.columns:
                    df_daily = df_daily.rename(columns={'L2_SciData_TimeStamp': 'time'})
                    
                all_daily_dfs.append(df_daily)

    if not all_daily_dfs:
        print(f"No valid data extracted for {sat_name.upper()}. Skipping...")
        continue

    # 2. Concatenate and Clean
    print(f"\nConcatenating {sat_name.upper()} files...")
    df_master = pd.concat(all_daily_dfs, ignore_index=True)
    df_master = df_master.dropna(subset=['time'])
    df_master['time'] = pd.to_datetime(df_master['time'], utc=True)

    energy_cols = [c for c in ['DiffProtonLowerEnergy', 'DiffProtonUpperEnergy', 'DiffProtonEffectiveEnergy'] if c in df_master.columns]
    df_master[energy_cols] = np.round(df_master[energy_cols] / 1000.0, decimals=2)

    flux_cols = [c for c in ['AvgDiffProtonFlux', 'AvgDiffProtonFluxUncert'] if c in df_master.columns]
    df_master[flux_cols] = df_master[flux_cols] * 1000.0

    # 3. Group and Populate Main Dictionary
    print(f"Grouping {sat_name.upper()} data and populating dictionary...")
    grouped = df_master.groupby(['sensor_units', 'diff_channels'])

    available_data_columns = [col for col in data_columns if col in df_master.columns]

    for (sensor, channel), group_df in grouped:
        s_key = int(sensor)
        c_key = int(channel)
        
        try:
            channel_name = SGPS_CHANNELS[c_key]
        except IndexError:
            channel_name = f"Unknown_{c_key}"
        
        indexed_df = group_df.set_index('time').sort_index()
        goes_data[sat_name][s_key][channel_name] = indexed_df[available_data_columns]

    print(f"{sat_name.upper()} dictionary populated successfully.")

    # 4. Generate Yaw Ledger conditionally
    if sat_name in ['goes17', 'goes19']:
        print(f"\nGenerating Yaw Flip Ledger for {sat_name.upper()}...")
        
        df_yaw = generate_yaw_ledger(df_master)
        yaw_ledgers[sat_name] = df_yaw
        
        print(f"\n{'-'*85}")
        print(f"{sat_name.upper()} YAW FLIP LEDGER")
        print(f"{'-'*85}")
        if df_yaw.empty:
            print("No yaw flips detected.")
        else:
            print(df_yaw.to_string(index=False))

print(f"\n{'='*85}")
print("ALL PROCESSING COMPLETE.")
print(f"{'='*85}")

#%% Print Unique Energy Values for Each Sensor
# Set Pandas display option to strictly format floats to 2 decimals visually
pd.set_option('display.float_format', '{:.2f}'.format)

# Define where you want to save the text file (saves in the current working directory by default)
output_filepath = Path("satellite_energy_ledgers.txt")

# Open the file in write mode ('w')
with open(output_filepath, 'w') as f:
    
    # Loop through all satellites that were successfully loaded into the dictionary
    for sat_name in goes_data.keys():
        
        # Generate the consolidated ledger
        df_ledger = get_satellite_energy_ledger(goes_data, sat_name, sensors=[0, 1])
        
        # Construct the header string
        header = f"\n{'-'*60}\nSatellite: {sat_name.upper()}\n{'-'*60}\n"
        
        # 1. Print to the console so you can still see it in your notebook/terminal
        print(header)
        print(df_ledger)
        
        # 2. Write the exact same output to the text file
        f.write(header)
        # .to_string() converts the dataframe to a clean text table honoring the float format
        f.write(df_ledger.to_string() + "\n")

print(f"\n[{'-'*40}]")
print(f"Success! Output saved to: {output_filepath.absolute()}")
print(f"[{'-'*40}]")

# Reset the float format back to default so it doesn't affect unrelated scripts
pd.reset_option('display.float_format')


#%% Find the mode of each satellite channel data
from collections import defaultdict
# Initialize the nested dictionary to store the modes dynamically
sat_modes = defaultdict(lambda: defaultdict(dict))

# Optional: Create a flat list to easily convert the results into a clean table later
mode_records = []

# Define which column you are finding the mode for
target_col = 'AvgDiffProtonFlux'

for sat_name, sensors in goes_data.items():
    for sensor, channels in sensors.items():
        for channel, df_channel in channels.items():
            
            # Ensure the column exists and the dataframe isn't empty to avoid errors
            if target_col in df_channel.columns and not df_channel.empty:
                
                # Filter out zeros, NaNs, and negative fill/error values 
                # (Physical flux cannot be negative, so > 0 isolates strict valid data)
                valid_data = df_channel[df_channel[target_col] > 0.0][target_col].dropna()
                
                if not valid_data.empty:
                    # .mode() returns a Series (in case of multi-modal ties), so extract the first value
                    modes = valid_data.mode()
                    
                    if not modes.empty:
                        mode_val = modes.iloc[0]
                        
                        # 1. Save to your nested dictionary
                        sat_modes[sat_name][sensor][channel] = mode_val
                        
                        # 2. Append to the flat list for the tabular output
                        mode_records.append({
                            'Satellite': sat_name.upper(),
                            'Sensor': sensor,
                            'Channel': channel,
                            'AvgDiffProtonFlux_Mode': mode_val
                        })

# Convert to a DataFrame for optimized viewing and exporting
df_modes = pd.DataFrame(mode_records)

# Sort strictly by Satellite, Sensor, and then the numeric channel order using your existing dictionary
df_modes['Channel_Num'] = df_modes['Channel'].map(CHANNEL_TO_NUM)
df_modes = df_modes.sort_values(['Satellite', 'Sensor', 'Channel_Num']).drop(columns=['Channel_Num'])

print(f"\n{'-'*65}")
print("MODES OF AVG DIFF PROTON FLUX BY SATELLITE & SENSOR")
print(f"{'-'*65}")
print(df_modes.to_string(index=False))


#%% Find the end time of each event
import pandas as pd

threshold = 1e-2
col_name = 'End_Time_UTC'

# Clean the column names by removing any hidden leading/trailing spaces
sep_catalog.columns = sep_catalog.columns.str.strip()

# 1. Insert the new column, explicitly setting the dtype to timezone-aware UTC
if col_name not in sep_catalog.columns:
    sep_catalog.insert(1, col_name, pd.Series(pd.NaT, index=sep_catalog.index, dtype='datetime64[ns, UTC]'))
else:
    sep_catalog[col_name] = pd.Series(pd.NaT, index=sep_catalog.index, dtype='datetime64[ns, UTC]')

# 2. Iterate through the DataFrame rows
for idx, row in sep_catalog.iterrows():
    
    event_start = pd.to_datetime(row['Begin_Time_UTC'])
    if event_start.tzinfo is None:
        event_start = event_start.tz_localize('UTC')
        
    # The event cannot end until at least 24 hours have passed
    min_end_time = event_start + pd.Timedelta(hours=24)
    
    # SAFEGUARD 1: Cap the search window to prevent runaway events across massive data gaps
    max_end_time = event_start + pd.Timedelta(days=30) 
    
    potential_end_times = []
    
    for satellite in goes_data.keys():
        for sensor in goes_data[satellite].keys():
            
            if 'P4' in goes_data[satellite][sensor]:
                df_p4 = goes_data[satellite][sensor]['P4']
                
                # SAFEGUARD 2: Was this satellite active when the event started?
                # Check if it logged any data between the event start and the 24-hour mark
                active_mask = (df_p4.index >= event_start) & (df_p4.index <= min_end_time)
                
                if not df_p4[active_mask].empty:
                    
                    # The satellite is confirmed active for this event. Find the end time.
                    time_logic = (df_p4.index > min_end_time) & (df_p4.index <= max_end_time)
                    flux_logic = df_p4['AvgDiffProtonFlux'] < threshold
                    
                    event_ended_df = df_p4[time_logic & flux_logic]
                    
                    if not event_ended_df.empty:
                        first_dip_time = event_ended_df.index[0]
                        potential_end_times.append(first_dip_time)
    
    if potential_end_times:
        sep_catalog.loc[idx, col_name] = max(potential_end_times)

print("Event end times calculated successfully.")


#%% Plot the flux data for one energy channel as a function of time
# --- Configuration ---
sat_name = 'goes16'
target_channel = 'P9'
threshold = 1e-3  # Threshold for plotting horizontal line

# Set to a date string (e.g., '2023-03-01') or leave as None or '' to plot all data
start_date = None #'2025-05-31' # '2023-07-17 20:00'  # '2023-04-21 20:00', '2023-03-13'
end_date =  None #'2025-06-10' # '2023-07-22'  # '2023-04-24 16:00', '2023-03-24'    

# Averaging flag (Pandas offset strings: '5min' for raw, '1H' for hourly, '1D' for daily)
averaging = '5min'  

# --- Data Extraction ---
# Extract the flux columns from the nested dictionary
s0_flux = goes_data[sat_name][0][target_channel]['AvgDiffProtonFlux']
s1_flux = goes_data[sat_name][1][target_channel]['AvgDiffProtonFlux']

# Combine into a single DataFrame for easy plotting and drop zeros
df_flux_plot = pd.DataFrame({
    f'S0_{target_channel}': s0_flux,
    f'S1_{target_channel}': s1_flux
}).replace(0, np.nan)

# --- Date Filtering ---
# Slice the dataframe if date variables are provided
if start_date:
    df_flux_plot = df_flux_plot.loc[start_date:]
if end_date:
    df_flux_plot = df_flux_plot.loc[:end_date]

# --- Time Resampling ---
# If anything other than 5min is selected, resample the data and take the mean
if averaging != '5min':
    df_flux_plot = df_flux_plot.resample(averaging).mean()

# Create a clean label for the plot title
avg_label = {'5min': '5-Minute', '1H': 'Hourly', '1D': 'Daily'}.get(averaging, f'{averaging}')

# --- Generate the Plot ---
fig, ax = plt.subplots(figsize=(18, 4))
df_flux_plot.plot(ax=ax, linewidth=2)
ax.set_yscale('log')

# Dynamically update the title to show the date range and cadence being plotted
min_date = df_flux_plot.index.min().strftime('%Y-%m-%d')
max_date = df_flux_plot.index.max().strftime('%Y-%m-%d')
ax.set_title(f'{sat_name.upper()} Proton Flux Data (S0 & S1 Channel {target_channel}) | {avg_label} Average | {min_date} to {max_date}')

ax.set_xlabel('Time (UTC)')
ax.set_ylabel('Flux (#/cm²/s/sr/MeV)')

# Plot statistic lines
ax.axhline(threshold, linestyle='-.', color='b', label=f'Threshold: {threshold:.1e}')

ax.legend()
plt.show()

#%% Plot all channels for both sensors in a single figure for an event
# --- Configuration ---
sat_name = 'goes19'
sensor = 0  # Sensor 0 or 1
threshold = 1e-2  # Threshold for plotting horizontal line

# Set to a date string (e.g., '2023-03-01') or leave as None or '' to plot all data
start_date = None # '2023-07-17 20:00'  
end_date = None # '2023-07-22'          

# Averaging flag (Pandas offset strings: '5min' for raw, '1H' for hourly, '1D' for daily)
averaging = '5min'  

# --- Data Extraction ---
flux_dict = {}

# Loop over the official channel names instead of integers
for ch_name in goes_data[sat_name][sensor].keys():
    # Check if the channel exists in the dictionary to prevent KeyErrors
    if ch_name in goes_data[sat_name][sensor]:
        col_name = f'S{sensor}_{ch_name}'
        flux_dict[col_name] = goes_data[sat_name][sensor][ch_name]['AvgDiffProtonFlux']

# Combine into a single DataFrame for easy plotting and drop zeros
df_flux_plot = pd.DataFrame(flux_dict).replace(0, np.nan)

# --- Date Filtering ---
if start_date:
    df_flux_plot = df_flux_plot.loc[start_date:]
if end_date:
    df_flux_plot = df_flux_plot.loc[:end_date]

# --- Time Resampling ---
if averaging != '5min':
    df_flux_plot = df_flux_plot.resample(averaging).mean()

avg_label = {'5min': '5-Minute', '1H': 'Hourly', '1D': 'Daily'}.get(averaging, f'{averaging}')

# --- Generate the Plot ---
fig, ax = plt.subplots(figsize=(18, 7))

# Plot all columns using a sequential rainbow colormap ('turbo' or 'rainbow')
df_flux_plot.plot(ax=ax, linewidth=1.5, colormap='turbo')
ax.set_yscale('log')

# Dynamically update the title
min_date = df_flux_plot.index.min().strftime('%Y-%m-%d')
max_date = df_flux_plot.index.max().strftime('%Y-%m-%d')
ax.set_title(f'{sat_name.upper()} Proton Flux Data (All Channels, Sensor {sensor}) | {avg_label} Average', pad=20)

ax.set_xlabel('Time (UTC)')
ax.set_ylabel('Flux (#/cm²/s/sr/MeV)')

# Move the legend outside the plot area since it contains 27 items
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)

# Adjust layout so the legend isn't cut off when saving or viewing
plt.tight_layout()
plt.show()


#%% Plot the maximum flux for each channel as a function of the effective energy for a given sensor
# --- Configuration ---
sat_name = 'goes19'
FONT_SIZE = 12

# Set to a date string (e.g., '2023-03-01') or leave as None to plot all data
start_date = '2025-05-31 17:00'  
end_date = '2025-06-05'   

# --- Data Extraction & Max Flux Calculation ---
summary_data = []

# Loop over both sensors
for sensor in [0, 1]:
    for ch in goes_data[sat_name][sensor].keys():
        df_channel = goes_data[sat_name][sensor][ch]
        
        # Isolate the flux data and apply date filters
        flux_series = df_channel['AvgDiffProtonFlux'].replace(0, np.nan).dropna()
        if start_date:
            flux_series = flux_series.loc[start_date:]
        if end_date:
            flux_series = flux_series.loc[:end_date]
            
        if not flux_series.empty:
            ch_max_time = flux_series.idxmax()
            ch_max_flux = flux_series.max()
            
            # Check if the maximum flux meets the threshold before saving
            if ch_max_flux > 1e-3:
                
                # Extract corresponding effective energy
                if ch_max_time in df_channel.index:
                    eff_energy = df_channel.loc[ch_max_time, 'DiffProtonEffectiveEnergy']
                    
                    summary_data.append({
                        'Sensor': sensor,  # Kept as integer so the plot loop filtering works
                        'ChannelNum': CHANNEL_TO_NUM.get(ch, 0),  # Map the string (e.g., 'P1') back to an integer for the colormap
                        'Channel': ch,  # Store the string name for the plot labels
                        'EffectiveEnergy': eff_energy,
                        'MaxFlux': ch_max_flux,
                        'MaxTime': ch_max_time  
                    })

df_summary = pd.DataFrame(summary_data).dropna()

fig, ax = plt.subplots(figsize=(14, 6))

markers = {0: 'o', 1: '^'}  

# 1. Plot the scatter points
for sensor_id in [0, 1]:
    df_sens = df_summary[df_summary['Sensor'] == sensor_id]
    if df_sens.empty:
        continue
        
    ax.scatter(
        df_sens['EffectiveEnergy'], 
        df_sens['MaxFlux'], 
        c=df_sens['ChannelNum'],  
        cmap='turbo',
        vmin=0, vmax=12,  # Adjusted to match the 0-12 mapping of the new channel list
        s=150,
        marker=markers[sensor_id],
        edgecolor='black', 
        linewidth=1.5, 
        alpha=0.9,
        zorder=3,
        label=f'Sensor {sensor_id}'
    )

# 2. Add a single label per channel (placed above the highest point for that channel)
for ch_num, group in df_summary.groupby('ChannelNum'):
    label_x = group['EffectiveEnergy'].mean()
    label_y = group['MaxFlux'].max()
    ch_name = group['Channel'].iloc[0]
    
    ax.annotate(
        ch_name, 
        (label_x, label_y),
        xytext=(0, 12),             
        textcoords='offset points',
        ha='center',                
        va='bottom',
        weight='bold'
    )

t_start = start_date if start_date else "Dataset Start"
t_end = end_date if end_date else "Dataset End"

fig.suptitle(f'{sat_name.upper()} Peak Flux Spectrum (Sensors 0 & 1) | {t_start} to {t_end}')

ax.set_xlabel('Effective Energy (MeV)')
ax.set_ylabel('Maximum Flux (#/cm²/s/sr/MeV)')

ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20))
ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

if not df_summary.empty:
    ax.set_ylim(bottom=1e-3, top=df_summary['MaxFlux'].max() * 2)

ax.set_xscale('log')
ax.set_yscale('log')

ax.legend(title='Instrument')

ax.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.7, zorder=0)
ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4, zorder=0)

plt.tight_layout()
plt.show()

# --- Generate the Time vs. Energy Plot ---
fig, ax = plt.subplots(figsize=(14, 6))

for sensor_id in [0, 1]:
    df_sens = df_summary[df_summary['Sensor'] == sensor_id]
    if df_sens.empty:
        continue
        
    ax.scatter(
        df_sens['MaxTime'], 
        df_sens['EffectiveEnergy'], 
        c=df_sens['ChannelNum'],  
        cmap='turbo',
        vmin=0, vmax=12,  # Adjusted to match the 0-12 mapping
        s=150,
        marker=markers[sensor_id],
        edgecolor='black', 
        linewidth=1.5, 
        alpha=0.9,
        zorder=3,
        label=f'Sensor {sensor_id}'
    )

# Annotate each point with its channel name
for idx, row in df_summary.iterrows():
    y_offset = -12 if row['Sensor'] == 1 else 10 
    
    ax.annotate(
        row['Channel'], 
        (row['MaxTime'], row['EffectiveEnergy']),
        xytext=(0, y_offset), 
        textcoords='offset points',
        ha='center',
        weight='bold'
    )

fig.suptitle(f'{sat_name.upper()} Time of Peak Flux vs. Effective Energy (Sensors 0 & 1)', weight='bold', y=0.92)

ax.set_xlabel('Peak Time (UTC)')
ax.set_ylabel('Effective Energy (MeV)')

ax.set_yscale('log')

# --- Custom Stateful Date Formatting Logic ---
class SparseDateFormatter(ticker.Formatter):
    def __call__(self, x, pos=None):
        if pos is None or len(self.locs) == 0:
            return ''
            
        dt = mdates.num2date(x)
        
        if pos == 0:
            year_str = dt.strftime('%Y')
            date_str = dt.strftime('%m/%d')
        else:
            prev_dt = mdates.num2date(self.locs[pos - 1])
            year_str = dt.strftime('%Y') if dt.year != prev_dt.year else ' '
            date_str = dt.strftime('%m/%d') if dt.date() != prev_dt.date() else ' '
            
        time_str = dt.strftime('%H:%M')
        
        return f"{time_str}\n{date_str}\n{year_str}"

ax.xaxis.set_major_formatter(SparseDateFormatter())
plt.xticks(rotation=0, ha='center')

ax.legend(title='Instrument')

ax.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.7, zorder=0)
ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4, zorder=0)

plt.tight_layout()
plt.show()