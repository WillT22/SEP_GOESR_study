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

# Define the base directory containing the satellite folders
BASE_DATA_DIR = Path(r"C:\Users\Will\Box\SEP_Box\SGPS_data")

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

print(f"Dictionary successfully loaded. Available satellites: {list(goes_data.keys())}")

#%% Plot all data for all channels
def plot_goes_flux(goes_data, sat_name, sensor, start_date=None, end_date=None, averaging='5min', save_dir=None, show_plot=True):
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
    """
    
    if sat_name not in goes_data or sensor not in goes_data[sat_name]:
            print(f"Error: No data found for {sat_name.upper()} Sensor {sensor}.")
            return

    # --- 1. Extract Flux and Yaw Data ---
    flux_dict = {}
    
    # Grab the first available channel to safely extract the yaw flag (since it's identical across channels)
    first_ch = list(goes_data[sat_name][sensor].keys())[0]
    has_yaw = 'yaw_flip_flag' in goes_data[sat_name][sensor][first_ch].columns
    
    if has_yaw:
        df_yaw = goes_data[sat_name][sensor][first_ch][['yaw_flip_flag']].copy()

    for ch_name in goes_data[sat_name][sensor].keys():
        col_name = f'S{sensor}_{ch_name}'
        flux_dict[col_name] = goes_data[sat_name][sensor][ch_name]['AvgDiffProtonFlux']

    df_flux_plot = pd.DataFrame(flux_dict).replace(0, np.nan)

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
        # Note: We do not resample df_yaw, as we want the exact 5-min timestamp of the physical flip

    avg_label = {'5min': '5-Minute', '1H': 'Hourly', '1D': 'Daily'}.get(averaging, f'{averaging}')

    # --- 4. Generate the Base Plot ---
    fig, ax = plt.subplots(figsize=(18, 7))

    df_flux_plot.plot(ax=ax, linewidth=1.5, colormap='turbo')
    ax.set_yscale('log')

    # --- 5. Add Yaw Flip Lines & Orientation Labels ---
    if has_yaw and not df_yaw.empty:
        # Drop missing data to cleanly find the state blocks
        df_yaw_clean = df_yaw.dropna(subset=['yaw_flip_flag']).copy()
        
        if not df_yaw_clean.empty:
            # Group into contiguous blocks of the same yaw flag
            df_yaw_clean['block'] = (df_yaw_clean['yaw_flip_flag'] != df_yaw_clean['yaw_flip_flag'].shift(1)).cumsum()
            
            for block_id, group in df_yaw_clean.groupby('block'):
                state = group['yaw_flip_flag'].iloc[0]
                start_t = group.index[0]
                
                # Determine physical orientation
                if state == 0:
                    orientation = 'West' if sensor == 0 else 'East'
                elif state == 2:
                    orientation = 'East' if sensor == 0 else 'West'
                elif state == 1:
                    orientation = 'Maneuvering'
                else:
                    orientation = 'Unknown'
                
                # Draw vertical dashed line at the transition (skip if it's the exact start of the plot to avoid overlapping the y-axis)
                if start_t > df_flux_plot.index.min():
                    ax.axvline(start_t, color='black', linestyle='--', linewidth=1.5, alpha=0.8, zorder=3)
                
                # Add text label at the top of the plot for this time block
                # get_xaxis_transform() means X is in Datetime, but Y is a percentage (0.98 = 98% height)
                ax.text(start_t, 0.98, f" {orientation} ", 
                        transform=ax.get_xaxis_transform(), 
                        fontsize=11, fontweight='bold', color='black',
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.2'),
                        zorder=4)
    else:
        # Static label fallback for satellites without yaw flip capabilities (e.g., GOES-16 is always upright)
        orientation = 'West' if sensor == 0 else 'East'
        ax.text(df_flux_plot.index.min(), 0.98, f" {orientation} ", 
                transform=ax.get_xaxis_transform(), 
                fontsize=11, fontweight='bold', color='black',
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.2'),
                zorder=4)

    # --- 6. Formatting & Legends ---
    min_date = df_flux_plot.index.min().strftime('%Y-%m-%d')
    max_date = df_flux_plot.index.max().strftime('%Y-%m-%d')
    ax.set_title(f'{sat_name.upper()} Proton Flux Data (All Channels, Sensor {sensor}) | {avg_label} Average\n({min_date} to {max_date})', pad=20)

    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Flux (#/cm²/s/sr/MeV)')

    # Move the legend outside the plot area
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)

    # Adjust layout so the legend isn't cut off when saving or viewing
    ax.margins(x=0, y=0)
    plt.tight_layout()

    # Save the figure (optional)
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True) # Creates the folder if it doesn't exist
        
        # Create a clean filename dynamically
        safe_start = start_date.replace(':', '').replace(' ', '_') if start_date else min_date
        filename = f"{sat_name}_sensor{sensor}_{averaging}_{safe_start}.png"
        full_filepath = save_path / filename
        
        # Save at high resolution (dpi=300) with no cropped edges
        fig.savefig(full_filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")

    # Show the plot (optional)
    if show_plot:
        plt.show()
    else:
        plt.close(fig) # Closes the figure in the background to free up RAM


# Plot all satellite data for operaiton period
for satellite in goes_data.keys():
    for sensor in goes_data[satellite].keys():
        plot_goes_flux(goes_data, satellite, sensor, save_dir=r'C:\Users\Will\Box\SEP_Box\SGPS_data\data_figures')



#%% Import SEP Catalog
# =======================================================
#
#         Event Selection / Catalog Processing
#
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
# Implement channel-based thresholed
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

# Define the output path (adjust the folder/filename as needed)
output_file = Path(r"C:\Users\Will\Box\SEP_Box\SGPS_data\updated_sep_catalog.csv")
# Export the DataFrame to a CSV file
sep_catalog.to_csv(output_file, index=False)
print(f"Updated SEP catalog saved successfully to: {output_file}")

#%% Find the mode of each satellite channel data
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

# Set to a date string (e.g., '2023-03-01') or leave as None or '' to plot all data
start_date = None #'2025-05-31 17:00'  
end_date = None #'2025-06-05'            

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
sat_name = 'goes18'

# Set to a date string (e.g., '2023-03-01') or leave as None to plot all data
start_date = '2025-05-31 17:00'  
end_date = '2025-06-05'   

# --- Data Extraction & Dual-Peak Calculation ---
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
            
            # 1. Calculate the Total Maximum Peak
            total_max_time = flux_series.idxmax()
            total_max_flux = flux_series.max()
            
            # Check if the maximum flux meets the threshold before saving
            if total_max_flux > 1e-3:
                
                # 2. Calculate the Onset (Prompt) Peak
                # Prominence of 10% of the total max filters out minor statistical noise
                peaks, _ = find_peaks(flux_series.values, prominence=total_max_flux * 0.10)
                
                if len(peaks) > 0:
                    onset_idx = peaks[0]
                    onset_time = flux_series.index[onset_idx]
                    onset_flux = flux_series.iloc[onset_idx]
                else:
                    # if no distinct early peak is found
                    onset_time = total_max_time
                    onset_flux = total_max_flux
                
                # Extract corresponding effective energy
                if total_max_time in df_channel.index:
                    # Note that the effective energy may be different between onset and total max if temporally position just right
                    eff_energy = df_channel.loc[total_max_time, 'DiffProtonEffectiveEnergy']
                    
                    # 3. Create the flag comparing the two timestamps
                    is_single_peak = (onset_time == total_max_time)
                    
                    summary_data.append({
                        'Sensor': sensor,  
                        'ChannelNum': CHANNEL_TO_NUM.get(ch, 0),  
                        'Channel': ch,  
                        'EffectiveEnergy': eff_energy,
                        'OnsetFlux': onset_flux,
                        'OnsetTime': onset_time,
                        'TotalMaxFlux': total_max_flux,
                        'TotalMaxTime': total_max_time,
                        'IsSinglePeak': is_single_peak  # True if onset and total max are the exact same point
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
        df_sens['TotalMaxFlux'], 
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
    label_y = group['TotalMaxFlux'].max()
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
    ax.set_ylim(bottom=1e-3, top=df_summary['TotalMaxFlux'].max() * 2)

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
        df_sens['TotalMaxTime'], 
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
        (row['TotalMaxTime'], row['EffectiveEnergy']),
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