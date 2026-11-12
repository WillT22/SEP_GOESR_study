import numpy as np
import xarray as xr
import pandas as pd
from collections import defaultdict

# Define the channel names for mapping
SGPS_CHANNELS = ['P1', 'P2A', 'P2B', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8A', 'P8B', 'P8C', 'P9', 'P10']

# Define the base variables to extract from the NetCDF
vars_to_keep_base = [
    'sensor_units', 'diff_channels',
    'DiffProtonLowerEnergy', 'DiffProtonUpperEnergy', 
    'DiffProtonEffectiveEnergy', 'AvgDiffProtonFlux', 'AvgDiffProtonFluxUncert'
]

# Load the NetCDF file into an xarray Dataset
file_path = "C:\\Users\\wzt0020\\Box\\SEP_Box\\SGPS_data\\goes16\\sgps-l2-avg5m\\2025\\sci_sgps-l2-avg5m_g16_d20250101_v3-0-2.nc"
dataset = xr.open_dataset(file_path, engine="netcdf4")

# Format the timestamp column name
time_col = 'L2_SciData_TimeStamp' if 'L2_SciData_TimeStamp' in dataset.variables else 'time'

# Subset the dataset BEFORE converting it to a DataFrame
vars_in_file = [v for v in vars_to_keep_base if v in dataset.variables] + [time_col]
df = dataset[vars_in_file].to_dataframe().reset_index()

# Set timestamp and clean NA
df['time'] = pd.to_datetime(df[time_col], utc=True)
df = df.dropna(subset=['time'])

# 2. Perform unit conversions on the massive flat dataframe BEFORE grouping (much faster)
energy_cols = ['DiffProtonLowerEnergy', 'DiffProtonUpperEnergy', 'DiffProtonEffectiveEnergy']
df[energy_cols] = np.round(df[energy_cols] / 1000.0, decimals=2)

flux_cols = ['AvgDiffProtonFlux', 'AvgDiffProtonFluxUncert']
df[flux_cols] = df[flux_cols] * 1000.0

# 3. Define exactly which physics columns you want in the final DataFrames
data_columns = [
    'DiffProtonLowerEnergy', 
    'DiffProtonUpperEnergy', 
    'DiffProtonEffectiveEnergy', 
    'AvgDiffProtonFlux', 
    'AvgDiffProtonFluxUncert'
]

# Initialize the dictionary
goes_data = defaultdict(lambda: defaultdict(dict))

# 4. Group the flat dataframe by sensor and channel
grouped = df.groupby(['sensor_units', 'diff_channels'])

# 5. Loop through the groups and populate the dictionary
for (sensor, channel), group_df in grouped:
    s_key = int(sensor)
    c_key = int(channel)
    
    try:
        channel_name = SGPS_CHANNELS[c_key]
    except IndexError:
        channel_name = f"Unknown_{c_key}"
    
    # Set the timestamp as the index and sort chronologically
    indexed_df = group_df.set_index('time').sort_index()

    # Keep only the requested data columns and assign to dictionary
    goes_data['goes16'][s_key][channel_name] = indexed_df[data_columns]

print("Dictionary populated successfully.")