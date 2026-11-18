#%% Import Libraries and SEP Catalog
import pandas as pd
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.signal import find_peaks
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

SGPS_CHANNELS = ['P1', 'P2A', 'P2B', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8A', 'P8B', 'P8C', 'P9', 'P10']
CHANNEL_TO_NUM = {ch: i for i, ch in enumerate(SGPS_CHANNELS)}
CHANNEL_THRESHOLDS = {
    'P1': 1e-1, 'P2A': 1e-1, 'P2B': 1e-1,
    'P3': 1e-2, 'P4': 1e-2, 'P5': 1e-2,
    'P6': 1e-3, 'P7': 1e-3, 'P8A': 1e-3,
    'P8B': 1e-3, 'P8C': 1e-3, 'P9': 1e-3, 'P10': 1e-3
}

# Path to the saved catalog
catalog_file = Path(r"C:\Users\wzt0020\Box\SEP_Box\SGPS_data\updated_sep_catalog.csv")

# Load the catalog and parse the time columns back into datetime objects
sep_catalog = pd.read_csv(
    catalog_file, 
    parse_dates=['Begin_Time_UTC', 'End_Time_UTC']
)

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


#%% Plot the maximum flux for each channel as a function of the effective energy for a given sensor
summary_data = []

# Iterate through every event loaded into memory
for event_name, satellites in loaded_events.items():
    for sat_name, sensors in satellites.items():
        for sensor, df_event in sensors.items():
            
            # --- Determine the dominant physical orientation for this event ---
            state = 0 # Default to upright
            if 'yaw_flip_flag' in df_event.columns:
                # Isolate valid physical states and find the most common one during the event
                physical_states = df_event['yaw_flip_flag'][df_event['yaw_flip_flag'].isin([0, 2])]
                if not physical_states.empty:
                    state = physical_states.mode().iloc[0]
            
            # Map the physical state and sensor number to the actual pointing direction
            if state == 0:
                orientation = 'West' if sensor == 0 else 'East'
            else: # state == 2 (Inverted)
                orientation = 'East' if sensor == 0 else 'West'
            
            # Loop through all possible SGPS channels
            for ch in SGPS_CHANNELS:
                flux_col = f'{ch}_Flux'
                
                # Check if this specific channel's data exists for this event/satellite
                if flux_col in df_event.columns:
                    
                    flux_series = df_event[flux_col].replace(0, np.nan).dropna()
                    flux_series = flux_series[flux_series > 0.0]
                    
                    if not flux_series.empty:
                        
                        # 1. Calculate the Total Maximum Peak
                        total_max_time = flux_series.idxmax()
                        total_max_flux = flux_series.max()
                        
                        # Grab the correct threshold for this channel
                        ch_threshold = CHANNEL_THRESHOLDS.get(ch, 1e-3)
                        
                        # Check if the maximum flux meets this specific channel's threshold
                        if total_max_flux > ch_threshold:
                            
                            # 2. Calculate the Onset (Prompt) Peak
                            peaks, _ = find_peaks(flux_series.values, prominence=total_max_flux * 0.1)
                            
                            if len(peaks) > 0:
                                onset_idx = peaks[0]
                                onset_time = flux_series.index[onset_idx]
                                onset_flux = flux_series.iloc[onset_idx]
                            else:
                                onset_time = total_max_time
                                onset_flux = total_max_flux
                            
                            # 3. Flag if the onset and max peak are identical
                            is_single_peak = (onset_time == total_max_time)

                            energy_col = f'{ch}_Energy'
                            if energy_col in df_event.columns:
                                eff_energy = df_event.loc[total_max_time, energy_col]
                            else:
                                eff_energy = np.nan
                            
                            summary_data.append({
                                'Event': event_name,
                                'Satellite': sat_name.upper(),
                                'Sensor': sensor,
                                'Orientation': orientation,
                                'ChannelNum': CHANNEL_TO_NUM.get(ch, 99),  
                                'Channel': ch,  
                                'EffectiveEnergy': eff_energy,
                                'OnsetFlux': onset_flux,
                                'OnsetTime': onset_time,
                                'TotalMaxFlux': total_max_flux,
                                'TotalMaxTime': total_max_time,
                                'IsSinglePeak': is_single_peak 
                            })

# Convert the results to a clean DataFrame and sort chronologically and by channel
df_event_peaks = pd.DataFrame(summary_data)
df_event_peaks = df_event_peaks.sort_values(['Event', 'Satellite', 'Orientation', 'ChannelNum']).reset_index(drop=True)

#%% Plot the peak flux for each event
# --- Configuration & Folder Setup ---
base_plot_dir = Path(r"C:\Users\wzt0020\Box\SEP_Box\SGPS_data\Event_Spectra_Plots")
onset_dir = base_plot_dir / "Onset_Peak_Spectra"
absolute_dir = base_plot_dir / "Absolute_Peak_Spectra"

onset_dir.mkdir(parents=True, exist_ok=True)
absolute_dir.mkdir(parents=True, exist_ok=True)

# Consistent styling
sat_colors = {'GOES16': '#1f77b4', 'GOES17': '#ff7f0e', 'GOES18': '#2ca02c', 'GOES19': '#d62728'}
SGPS_CHANNELS = ['P1', 'P2A', 'P2B', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8A', 'P8B', 'P8C', 'P9', 'P10']

print(f"Generating energy spectra plots for {df_event_peaks['Event'].nunique()} events...")

# Loop through each distinct event in your peaks dataframe
for event_name, df_event in df_event_peaks.groupby('Event'):
    
    print(f"  -> Plotting spectra for {event_name}...")
    
    # We will loop twice: once for 'OnsetFlux' and once for 'TotalMaxFlux'
    for peak_type, save_folder, title_prefix in [
        ('OnsetFlux', onset_dir, 'Onset (Prompt) Peak'),
        ('TotalMaxFlux', absolute_dir, 'Absolute (Total Max) Peak')
    ]:
        
        fig, ax = plt.subplots(figsize=(20, 8))
        
        # Group directly by Satellite and the pre-calculated Orientation
        for (sat, orientation), df_group in df_event.groupby(['Satellite', 'Orientation']):
            
            # Sort just to guarantee the line draws left-to-right correctly
            df_group = df_group.sort_values('ChannelNum')
            
            # Style: Match satellite color. Solid line/Circle for West, Dashed line/Square for East
            color = sat_colors.get(sat, 'black')
            linestyle = '-' if orientation == 'West' else '--'
            marker = 'o' if orientation == 'West' else 's'
            
            label = f"{sat} ({orientation})"
            
            ax.plot(df_group['ChannelNum'], df_group[peak_type], 
                    color=color, linestyle=linestyle, marker=marker, markersize=10,
                      linewidth=1.5, label=label, alpha=0.8)

        energy_labels = []
        for ch in SGPS_CHANNELS:
            ch_data = df_event[df_event['Channel'] == ch]
            if not ch_data.empty and 'EffectiveEnergy' in ch_data.columns:
                # Average the energy slightly in case different satellites have minor calibration offsets
                mean_energy = ch_data['EffectiveEnergy'].mean()
                energy_labels.append(f"{mean_energy:.1f}")
            else:
                energy_labels.append("-")

        # --- Formatting ---
        ax.set_yscale('log')
        
        # Lock the X-axis to span the entire 0-12 range
        ax.set_xlim(-0.5, len(SGPS_CHANNELS) - 0.5)
        
        # 1. Primary X-Axis Labels (Effective Energy in Black)
        # Setting the ticks and labels sequentially satisfies the Matplotlib warning
        ax.set_xticks(range(len(SGPS_CHANNELS)))
        ax.set_xticklabels(energy_labels, fontweight='bold', color='black')
        
        # 2. Secondary X-Axis Labels (Channel Names in Gray)
        for i, ch_name in enumerate(SGPS_CHANNELS):
            # Placing text just below the x-axis spine (y = -0.05 in axes fraction coordinates)
            ax.text(i, -0.08, ch_name, transform=ax.get_xaxis_transform(), 
                    ha='center', va='top', color='dimgray', fontweight='bold')
        
        # Push the main X-axis title down (labelpad=25) so it clears the dual labels
        ax.set_xlabel('Effective Energy (MeV)', fontweight='bold', labelpad=30)
        ax.set_ylabel('Proton Flux (#/cm²/s/sr/MeV)', fontweight='bold', labelpad=10)
        
        ax.set_title(f"{title_prefix} Energy Spectrum\nEvent: {event_name.replace('Event_', '')}", 
                     fontweight='bold', pad=15)
        
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title="Satellites")
        
        plt.tight_layout()
        
        filename = f"{event_name}_{title_prefix.split()[0]}_Spectrum.png"
        full_path = save_folder / filename
        
        fig.savefig(full_path, dpi=300, bbox_inches='tight')
        
        # Close the figure so Matplotlib drops it from RAM
        plt.close(fig)

print(f"\n{'-'*50}")
print("ALL SPECTRA PLOTS GENERATED AND SAVED SAFELY.")
print(f"{'-'*50}")

#%%
# # --- Generate the Time vs. Energy Plot ---
# fig, ax = plt.subplots(figsize=(14, 6))

# for sensor_id in [0, 1]:
#     df_sens = df_summary[df_summary['Sensor'] == sensor_id]
#     if df_sens.empty:
#         continue
        
#     ax.scatter(
#         df_sens['TotalMaxTime'], 
#         df_sens['EffectiveEnergy'], 
#         c=df_sens['ChannelNum'],  
#         cmap='turbo',
#         vmin=0, vmax=12,  # Adjusted to match the 0-12 mapping
#         s=150,
#         marker=markers[sensor_id],
#         edgecolor='black', 
#         linewidth=1.5, 
#         alpha=0.9,
#         zorder=3,
#         label=f'Sensor {sensor_id}'
#     )

# # Annotate each point with its channel name
# for idx, row in df_summary.iterrows():
#     y_offset = -12 if row['Sensor'] == 1 else 10 
    
#     ax.annotate(
#         row['Channel'], 
#         (row['TotalMaxTime'], row['EffectiveEnergy']),
#         xytext=(0, y_offset), 
#         textcoords='offset points',
#         ha='center',
#         weight='bold'
#     )

# fig.suptitle(f'{sat_name.upper()} Time of Peak Flux vs. Effective Energy (Sensors 0 & 1)', weight='bold', y=0.92)

# ax.set_xlabel('Peak Time (UTC)')
# ax.set_ylabel('Effective Energy (MeV)')

# ax.set_yscale('log')

# # --- Custom Stateful Date Formatting Logic ---
# class SparseDateFormatter(ticker.Formatter):
#     def __call__(self, x, pos=None):
#         if pos is None or len(self.locs) == 0:
#             return ''
            
#         dt = mdates.num2date(x)
        
#         if pos == 0:
#             year_str = dt.strftime('%Y')
#             date_str = dt.strftime('%m/%d')
#         else:
#             prev_dt = mdates.num2date(self.locs[pos - 1])
#             year_str = dt.strftime('%Y') if dt.year != prev_dt.year else ' '
#             date_str = dt.strftime('%m/%d') if dt.date() != prev_dt.date() else ' '
            
#         time_str = dt.strftime('%H:%M')
        
#         return f"{time_str}\n{date_str}\n{year_str}"

# ax.xaxis.set_major_formatter(SparseDateFormatter())
# plt.xticks(rotation=0, ha='center')

# ax.legend(title='Instrument')

# ax.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.7, zorder=0)
# ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4, zorder=0)

# plt.tight_layout()
# plt.show()