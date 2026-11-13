#%% Import necessary libraries
import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

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

# Export yaw ledgers to text file
if yaw_ledgers:
    # Define the output path in your base directory
    ledger_output_path = BASE_DATA_DIR / "combined_yaw_flip_ledger.txt"
    
    print(f"\nSaving combined yaw ledger to {ledger_output_path}...")
    
    with open(ledger_output_path, "w") as f:
        f.write(f"{'='*85}\n")
        f.write("COMBINED GOES YAW FLIP LEDGER\n")
        f.write(f"{'='*85}\n\n")
        
        for sat, df_yaw in yaw_ledgers.items():
            f.write(f"{'-'*85}\n")
            f.write(f"{sat.upper()} YAW FLIP LEDGER\n")
            f.write(f"{'-'*85}\n")
            
            if df_yaw.empty:
                f.write("No yaw flips detected.\n\n")
            else:
                # Write the dataframe formatted exactly like the console output
                f.write(df_yaw.to_string(index=False))
                f.write("\n\n")
                
    print("Yaw ledger saved successfully.")

# Save the cleaned, flat dataframe before the grouping loop
df_master.to_parquet(r"C:\Users\wzt0020\Box\SEP_Box\SGPS_data\goes_master_data.parquet")
print("Data saved to Parquet.")

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