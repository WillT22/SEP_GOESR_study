#%% Import Libraries and SEP Catalog
import pandas as pd
from pathlib import Path

# Path to the saved catalog
catalog_file = Path(r"C:\Users\wzt0020\Box\SEP_Box\SGPS_data\updated_sep_catalog.csv")

# Load the catalog and parse the time columns back into datetime objects
sep_catalog = pd.read_csv(
    catalog_file, 
    parse_dates=['Begin_Time_UTC', 'End Time Yr M/D (UTC)']
)

#%% From GOES data, import the relevant dates for each satellite



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
            
            # Check if the maximum flux meets the thresholds before saving
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