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
file_path = "C:\\Users\\wzt0020\\Box\\SEP_Box\\SGPS_data\\goes18\\sgps-l2-avg5m\\2023\\sci_sgps-l2-avg5m_g18_d20230329_v3-0-0.nc"
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
data_dir = Path(r"C:\Users\wzt0020\Box\SEP_Box\SGPS_data\goes18\sgps-l2-avg5m\2023")
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
            df_changed.insert(1, 'Channel', f"Ch{channel_key}")
            
            all_changes.append(df_changed)
            
    if not all_changes:
        return pd.DataFrame()
        
    # 5. Concatenate all DataFrames
    df_final = pd.concat(all_changes)
    
    # 6. Extract the numeric channel value for strict numerical sorting
    df_final['ChannelNum'] = df_final['Channel'].str.replace('Ch', '').astype(int)
    
    # Ensure the index has a name so we can reference it easily during the sort
    idx_name = df_final.index.name if df_final.index.name else 'Timestamp'
    df_final.index.name = idx_name
    
    # Reset the index to sort by Timestamp -> Sensor -> ChannelNum, then put the index back
    df_final = df_final.reset_index()
    df_final = df_final.sort_values(by=[idx_name, 'Sensor', 'ChannelNum'])
    df_final = df_final.set_index(idx_name)
    
    # Drop the temporary sorting column
    df_final = df_final.drop(columns=['ChannelNum'])
    
    return df_final


# --- Apply the function and print formatted output ---
sat_name = 'goes18'

# Set Pandas display option to strictly format floats to 2 decimals visually
pd.set_option('display.float_format', '{:.2f}'.format)

# Generate the consolidated ledger
df_ledger = get_satellite_energy_ledger(goes_data, sat_name, sensors=[0, 1])

# Print the single consolidated header
print(f"\n{'-'*60}")
print(f"Satellite: {sat_name.upper()}")
print(f"{'-'*60}")
print(df_ledger)

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

# Plot statistic lines
ax.axhline(threshold, linestyle='-.', color='b', label=f'Threshold: {threshold:.1e}')

ax.legend()
plt.show()

#%% Plot all channels for both sensors in a single figure for an event
# --- Configuration ---
sat_name = 'goes18'
sensor = 1  # Sensor 0 or 1
threshold = 1e-2  # Threshold for plotting horizontal line

# Set to a date string (e.g., '2023-03-01') or leave as None or '' to plot all data
start_date = '2023-07-17 20:00'  
end_date = '2023-07-22'          

# Averaging flag (Pandas offset strings: '5min' for raw, '1H' for hourly, '1D' for daily)
averaging = '5min'  

# --- Data Extraction ---
flux_dict = {}

# Loop over both sensors and all 12 channels
for ch in range(1, 13):
    # Check if the channel exists in the dictionary to prevent KeyErrors
    if ch in goes_data[sat_name][sensor]:
        col_name = f'S{sensor}_Ch{ch}'
        flux_dict[col_name] = goes_data[sat_name][sensor][ch]['AvgDiffProtonFlux']

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
sat_name = 'goes18'

# Set to a date string (e.g., '2023-03-01') or leave as None to plot all data
start_date = '2023-07-17 20:00'  
end_date = '2023-07-22'   

# --- Data Extraction & Max Flux Calculation ---
summary_data = []

# Loop over both sensors
for sensor in [0, 1]:
    for ch in range(1, 14):
        if ch in goes_data[sat_name][sensor]:
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
                            'Sensor': sensor,
                            'ChannelNum': ch,  # Used to lock the colormap sequence
                            'Channel': f'Ch{ch}',
                            'EffectiveEnergy': eff_energy,
                            'MaxFlux': ch_max_flux,
                            'MaxTime': ch_max_time  
                        })

df_summary = pd.DataFrame(summary_data).dropna()

# --- Generate the Plot ---
fig, ax = plt.subplots(figsize=(14, 6))

markers = {0: 'o', 1: '^'}  # Circles for Sensor 0, Triangles for Sensor 1

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
        vmin=1, vmax=13,  
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
    # Average the X position slightly if the energies differ marginally
    label_x = group['EffectiveEnergy'].mean()
    # Find the maximum Y position between Sensor 0 and Sensor 1 so the text clears both
    label_y = group['MaxFlux'].max()
    ch_name = group['Channel'].iloc[0]
    
    ax.annotate(
        ch_name, 
        (label_x, label_y),
        xytext=(0, 12),             # Shift text 12 points straight up
        textcoords='offset points',
        fontsize=FONT_SIZE - 2,
        ha='center',                # Center align the text over the point
        va='bottom',
        weight='bold'
    )

# Formatting Dates dynamically for Title
t_start = start_date if start_date else "Dataset Start"
t_end = end_date if end_date else "Dataset End"

fig.suptitle(f'{sat_name.upper()} Peak Flux Spectrum (Sensors 0 & 1) | {t_start} to {t_end}', fontsize=FONT_SIZE + 2, weight='bold')

ax.set_xlabel('Effective Energy (MeV)')
ax.set_ylabel('Maximum Flux (#/cm²/s/sr/MeV)')

ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20))
ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
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
markers = {0: 'o', 1: '^'} 

for sensor_id in [0, 1]:
    df_sens = df_summary[df_summary['Sensor'] == sensor_id]
    if df_sens.empty:
        continue
        
    # Plot Peak Time on X and Effective Energy on Y
    ax.scatter(
        df_sens['MaxTime'], 
        df_sens['EffectiveEnergy'], 
        c=df_sens['ChannelNum'],  
        cmap='turbo',
        vmin=1, vmax=13,  
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
    # Adding a slight vertical offset for S1 to prevent text overlap if times are identical
    y_offset = -12 if row['Sensor'] == 1 else 10 
    
    ax.annotate(
        row['Channel'], 
        (row['MaxTime'], row['EffectiveEnergy']),
        xytext=(0, y_offset), 
        textcoords='offset points',
        fontsize=FONT_SIZE - 2,
        ha='center',
        weight='bold'
    )

fig.suptitle(f'{sat_name.upper()} Time of Peak Flux vs. Effective Energy (Sensors 0 & 1)', weight='bold', y=0.92)

ax.set_xlabel('Peak Time (UTC)')
ax.set_ylabel('Effective Energy (MeV)')

# Set Y-axis to logarithmic to properly space the energy bands
ax.set_yscale('log')

# --- Custom Stateful Date Formatting Logic ---
class SparseDateFormatter(ticker.Formatter):
    def __call__(self, x, pos=None):
        # Matplotlib sometimes calls this without a position during setup
        if pos is None or len(self.locs) == 0:
            return ''
            
        dt = mdates.num2date(x)
        
        # Always populate the very first tick
        if pos == 0:
            year_str = dt.strftime('%Y')
            date_str = dt.strftime('%m/%d')
        else:
            # Look at the previous tick to see if the year or day changed
            prev_dt = mdates.num2date(self.locs[pos - 1])
            year_str = dt.strftime('%Y') if dt.year != prev_dt.year else ' '
            date_str = dt.strftime('%m/%d') if dt.date() != prev_dt.date() else ' '
            
        time_str = dt.strftime('%H:%M')
        
        # Return a 3-line string, preserving vertical spacing with spaces
        return f"{time_str}\n{date_str}\n{year_str}"

# Apply the custom formatter
ax.xaxis.set_major_formatter(SparseDateFormatter())
plt.xticks(rotation=0, ha='center')

ax.legend(title='Instrument')

ax.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.7, zorder=0)
ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4, zorder=0)

plt.tight_layout()
plt.show()