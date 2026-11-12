#%% Import necessary libraries
import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

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


#%% Import a singular data file (diagnostics)
# Load the NetCDF file into an xarray Dataset and convert to DataFrame
file_path = "C:\\Users\\Will\\Box\\SEP_Box\\SGPS_data\\goes18\\sgps-l2-avg5m\\2023\\sci_sgps-l2-avg5m_g18_d20230329_v3-0-0.nc"
dataset = xr.open_dataset(file_path, engine="netcdf4")
df = dataset.to_dataframe().reset_index()

# 1. Format the timestamp before splitting the data
df['time'] = pd.to_datetime(df['time'], utc=True)

# 2. Initialize the nested dictionary structure
goes_data = {
    'goes18': {
        0: {},  # Sensor 0
        1: {}   # Sensor 1
    }
}

# 3. Define exactly which physics columns you want in the final DataFrames
data_columns = [
    'DiffProtonLowerEnergy', 
    'DiffProtonUpperEnergy', 
    'DiffProtonEffectiveEnergy', 
    'AvgDiffProtonFlux', 
    'AvgDiffProtonFluxUncert'
]

# 4. Group the flat dataframe by sensor and channel
grouped = df.groupby(['sensor_units', 'diff_channels'])

# 5. Loop through the groups and populate the dictionary
for (sensor, channel), group_df in grouped:
    # Cast to standard integers for clean dictionary keys
    s_key = int(sensor)
    c_key = int(channel)
    
    # Set the timestamp as the index
    indexed_df = group_df.set_index('time')
    
    # Keep only the requested data columns
    final_df = indexed_df[data_columns]
    
    # Assign the dataframe to its specific location in the nested dictionary
    goes_data['goes18'][s_key][c_key] = final_df

print("Dictionary populated successfully.")

#%% Import all GOES-18 data files and populate the dictionary
# 1. Point to the base directory (Year removed)
data_dir = Path(r"C:\Users\Will\Box\SEP_Box\SGPS_data\goes18\sgps-l2-avg5m\2023")
all_file_paths = sorted(list(data_dir.rglob("*.nc")))
print(f"Found {len(all_file_paths)} total files. Grouping by year...")

# Group files by year by parsing the GOES filename (e.g., sci_sgps-l2-avg5m_g18_d20220101_v1.nc)
files_by_year = defaultdict(list)
for file in all_file_paths:
    # Extracts the 4-digit year right after the '_d' in the filename
    year = file.name.split('_d')[1][:4]
    files_by_year[year].append(file)

# Initialize a list to hold the daily dataframes
all_daily_dfs = []

# Define exactly which columns we want to extract
vars_to_keep_base = [
    'sensor_units', 'diff_channels',
    'DiffProtonLowerEnergy', 'DiffProtonUpperEnergy', 
    'DiffProtonEffectiveEnergy', 'AvgDiffProtonFlux', 'AvgDiffProtonFluxUncert'
]

# 2. Loop through each year, and run a progress bar for the files in that year
for year in sorted(files_by_year.keys()):
    year_files = files_by_year[year]
    print(f"\n--- Loading Data for Year: {year} ---")
    
    # Wrap the file list in tqdm to generate the progress bar
    for file in tqdm(year_files, desc=f"Processing {year}", unit="file"):
        with xr.open_dataset(file, engine="netcdf4") as dataset:
            
            if 'L2_SciData_TimeStamp' in dataset.variables:
                time_col = 'L2_SciData_TimeStamp'
            else:
                time_col = 'time'
                
            vars_to_keep = vars_to_keep_base + [time_col]
            
            # Extract only the necessary variables to a dataframe
            df_daily = dataset[vars_to_keep].to_dataframe().reset_index()
            
            if time_col == 'L2_SciData_TimeStamp' and 'L2_SciData_TimeStamp' in df_daily.columns:
                df_daily = df_daily.rename(columns={'L2_SciData_TimeStamp': 'time'})
                
            all_daily_dfs.append(df_daily)

# 3. Concatenate all daily dataframes into one massive dataframe
print("\nConcatenating all files...")
df_master = pd.concat(all_daily_dfs, ignore_index=True)

# Format the timestamp and drop any empty rows generated by dimension alignment
df_master = df_master.dropna(subset=['time'])
df_master['time'] = pd.to_datetime(df_master['time'], utc=True)

# Convert energies from keV to MeV
energy_cols = ['DiffProtonLowerEnergy', 'DiffProtonUpperEnergy', 'DiffProtonEffectiveEnergy']
df_master[energy_cols] = np.round(df_master[energy_cols] / 1000.0, decimals=2)

# Convert differential fluxes from per keV to per MeV
flux_cols = ['AvgDiffProtonFlux', 'AvgDiffProtonFluxUncert']
df_master[flux_cols] = df_master[flux_cols] * 1000.0

# 4. Initialize the nested dictionary structure
goes_data = {
    'goes18': {
        0: {},  # Sensor 0
        1: {}   # Sensor 1
    }
}

data_columns = [
    'DiffProtonLowerEnergy', 
    'DiffProtonUpperEnergy', 
    'DiffProtonEffectiveEnergy', 
    'AvgDiffProtonFlux', 
    'AvgDiffProtonFluxUncert'
]

# 5. Group the massive flat dataframe by sensor and channel
print("Grouping data and populating dictionary...")
grouped = df_master.groupby(['sensor_units', 'diff_channels'])

# 6. Loop through the groups and populate the dictionary
for (sensor, channel), group_df in grouped:
    s_key = int(sensor)
    c_key = int(channel)
    
    indexed_df = group_df.set_index('time').sort_index()
    final_df = indexed_df[data_columns]
    
    goes_data['goes18'][s_key][c_key] = final_df

print("Dictionary populated successfully with all files.")

#%% Print Unique Energy Values for Each Sensor
def get_channel_changes(goes_dict, satellite, sensor):
    """
    Scans the nested dictionary for calibration changes and returns a dictionary 
    where keys are channels (e.g., 'Ch1') and values are DataFrames of the changes.
    """
    sensor_dict = goes_dict[satellite][sensor]
    
    # Reordered to strictly follow: Low, High, Effective
    energy_cols = ['DiffProtonLowerEnergy', 'DiffProtonUpperEnergy', 'DiffProtonEffectiveEnergy']
    col_names = ['Lower', 'Upper', 'Effective']
    
    channel_dict = {}
    
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
        
        # Save the isolated DataFrame directly to our dictionary
        channel_dict[f"Ch{channel_key}"] = df_en[changes_mask]
        
    return channel_dict

# --- Apply the function and print formatted output ---
sat_name = 'goes18'
sensors = [0, 1]

# Set Pandas display option to strictly format floats to 2 decimals visually
pd.set_option('display.float_format', '{:.2f}'.format)

for sensor in sensors:
    # Call your function (assuming you named it get_channel_changes)
    ledger_dict = get_channel_changes(goes_data, sat_name, sensor)
    
    # Sort the dictionary keys numerically so Ch10 comes after Ch9, not Ch1
    sorted_channels = sorted(ledger_dict.keys(), key=lambda x: int(x.replace('Ch', '')))
    
    for ch_key in sorted_channels:
        print(f"\n{'-'*60}")
        print(f"Satellite: {sat_name.upper()} | Sensor: {sensor} | Channel: {ch_key}")
        print(f"{'-'*60}")
        print(ledger_dict[ch_key])

# Reset the float format back to default so it doesn't affect unrelated scripts
pd.reset_option('display.float_format')


#%% Plot the flux data for one energy channel as a function of time
# --- Configuration ---
sat_name = 'goes18'
target_channel = 4
threshold = 1e-2  # Threshold for plotting horizontal line

# Set to a date string (e.g., '2023-03-01') or leave as None or '' to plot all data
start_date = '2023-07-17 20:00'  # '2023-04-21 20:00', '2023-03-13'
end_date = '2023-07-22'  # '2023-04-24 16:00', '2023-03-24'   

# Averaging flag (Pandas offset strings: '5min' for raw, '1H' for hourly, '1D' for daily)
averaging = '5min'  

# --- Data Extraction ---
# Extract the flux columns from the nested dictionary
s0_flux = goes_data[sat_name][0][target_channel]['AvgDiffProtonFlux']
s1_flux = goes_data[sat_name][1][target_channel]['AvgDiffProtonFlux']

# Combine into a single DataFrame for easy plotting and drop zeros
df_flux_plot = pd.DataFrame({
    f'S0_Ch{target_channel}': s0_flux,
    f'S1_Ch{target_channel}': s1_flux
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

# Calculate overall statistics based ONLY on the filtered/resampled data
overall_mean = df_flux_plot.stack().mean()
overall_mode = df_flux_plot.stack().mode()[0] if not df_flux_plot.stack().mode().empty else np.nan
overall_max = df_flux_plot.max().max()

# Plot statistic lines
ax.axhline(threshold, linestyle='-.', color='b', label=f'Threshold: {threshold:.1e}')
#ax.axhline(overall_mean, linestyle='--', color='k', label=f'Mean: {overall_mean:.2e}')
#ax.axhline(overall_mode, linestyle='-.', color='b', label=f'Mode: {overall_mode:.2e}')
#ax.axhline(overall_max, linestyle=':', color='r', label=f'Max: {overall_max:.2e}')

ax.legend()
plt.show()

#%% Find the beginning flux for each SEP event in the catalog
# --- Configuration ---
sat_name = 'goes18'
sensor = 0

# 1. Combine all channels (1-13) into a single DataFrame for the target sensor
flux_dict = {}
for ch in range(1, 14):
    if ch in goes_data[sat_name][sensor]:
        flux_dict[f'Ch{ch}'] = goes_data[sat_name][sensor][ch]['AvgDiffProtonFlux']

df_all_flux = pd.DataFrame(flux_dict).dropna(how='all')

# 2. Ensure the catalog times are formatted as UTC datetime objects to match GOES data
catalog_times = pd.to_datetime(sep_catalog['Begin_Time_UTC'], utc=True)

# 3. Extract the flux at the exact (or closest) timestamps
# Tolerance ensures it doesn't pull a value from days ago if there is a data gap
event_flux_df = df_all_flux.reindex(catalog_times, method='nearest', tolerance=pd.Timedelta('5min'))

# for ch in range(1, 12):
#     min_flux = event_flux_df[f'Ch{ch}'].dropna().values.min()
#     max_flux = event_flux_df[f'Ch{ch}'].dropna().values.max()
#     print(f"Channel {ch}: Min Flux = {min_flux:.2e}, Max Flux = {max_flux:.2e}")
