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
